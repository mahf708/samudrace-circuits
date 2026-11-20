"""Shared pieces for the heat-flux witness on the real SamudrACE ocean (C1A0, bare network, no corrector).

- load_ocean(): the Samudra U-Net, its in/out names and the normalizer (means/stds) from the checkpoint
- states(): the captured ocean network inputs (normalized, packed in in_names order)
- Graph: an exact re-implementation of Samudra.forward that exposes every edge of the U-Net graph,
  so a receiver can read a sender's value from another run's cache (clean-cache convention, Vi-CD Eq. 3)
- masks, area weights, mixed-layer depth from the input temperature profile, and the slab law.

Edges (sender -> receiver):
  D0..D3  encoder level k -> next level down (through AvgPool)
  B       bottleneck (layers[8]) -> first upsample (layers[9])
  U3..U1  decoder ConvNeXt at level k -> upsample into level k-1
  S0..S3  encoder level k -> the additive skip `fts += skip` at decoder level k
Level 0 is full resolution (180x360), level 3 is 22x45, the bottleneck is 11x22.
"""
import os, json, hashlib
import numpy as np, torch
import torch.nn.functional as F

from hc import paths

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = paths.CKPT["C1A0"]
CKPT_MD5 = paths.CKPT_MD5["C1A0"]
STATES = os.environ.get("HC_STATES", os.path.join(paths.OUT, "states", "ic1985"))
DEV = torch.device(os.environ.get("DEV", "cuda" if torch.cuda.is_available() else "cpu"))
IDEPTH = np.array([0, 20, 30, 40, 50, 80, 110, 140, 170, 230, 410, 530, 1020, 1080, 1720, 1980, 2820, 3380, 4620, 6380.0])
ZMID = 0.5 * (IDEPTH[1:] + IDEPTH[:-1])
RHO_CP, DT = 4.1e6, 5 * 86400.0
SIGMA = 5.670374e-8
# +dQ warms the ocean. FSNS, FLDS positive down; FLUS, LHFLX, SHFLX positive up.
COMPS = {"FSNS": +1, "FLDS": +1, "FLUS": -1, "LHFLX": -1, "SHFLX": -1}


def ckpt_md5(path=None):
    path = path or CKPT
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def load_ocean(ckpt=None):
    from fme.coupled.inference.inference import load_stepper
    from hc import set_numerics
    set_numerics()
    st = load_stepper(ckpt or CKPT)
    so = st.ocean._step_obj
    norm = so.normalizer
    ocn = st.ocean.modules[0].module.to(DEV).eval()
    ocn.checkpoint_strategy = None
    for layer in ocn.layers:
        if hasattr(layer, "checkpoint_strategy"):
            layer.checkpoint_strategy = None
    means = {k: float(v) for k, v in norm.means.items()}
    stds = {k: float(v) for k, v in norm.stds.items()}
    return ocn, list(so.in_names), list(so.out_names), means, stds


def load_ocean_any(path):
    """Ocean network + names + normalizer from a coupled checkpoint OR an ocean-only Stepper checkpoint."""
    from hc import set_numerics
    set_numerics()
    import torch as _t
    ck = _t.load(path, map_location="cpu", weights_only=False, mmap=True)
    coupled = "ocean" in ck["stepper"]["config"]
    del ck
    if coupled:
        from fme.coupled.inference.inference import load_stepper
        st = load_stepper(path).ocean
    else:
        from fme.ace.stepper.single_module import load_stepper
        st = load_stepper(path)
    so = st._step_obj
    ocn = st.modules[0].module.to(DEV).eval()
    ocn.checkpoint_strategy = None
    for layer in ocn.layers:
        if hasattr(layer, "checkpoint_strategy"):
            layer.checkpoint_strategy = None
    means = {k: float(v) for k, v in so.normalizer.means.items()}
    stds = {k: float(v) for k, v in so.normalizer.stds.items()}
    return ocn, list(so.in_names), list(so.out_names), means, stds, st.timestep if hasattr(st, "timestep") else None


def state_files(states=None):
    """Captured ocean inputs: x_XXX.npy (hc.capture) or ocn_in_XXX.npy (laptop capture_states.py)."""
    d = states or STATES
    fs = sorted(f for f in os.listdir(d) if f.startswith("x_") and f.endswith(".npy"))
    return fs or sorted(f for f in os.listdir(d) if f.startswith("ocn_in_") and f.endswith(".npy"))


def load_state(k, states=None):
    d = states or STATES
    p = os.path.join(d, f"x_{k:03d}.npy")
    if not os.path.exists(p):
        p = os.path.join(d, f"ocn_in_{k:03d}.npy")
    return np.nan_to_num(np.load(p).astype("f4"))


def fixed_ocean_mask(states=None):
    """Ocean = cells where the official run's sst output is finite (meta.npz from hc.capture).
    Use this, not sea_surface_fraction > 0.5 on land-filled inputs (that marks 96% of the grid)."""
    d = states or STATES
    return np.load(os.path.join(d, "meta.npz"))["out_mask"][0]


class Grid:
    lat = np.linspace(-89.5, 89.5, 180)  # Gaussian grid ~ this to <0.3 deg; weights only
    lon = np.arange(0.5, 360, 1.0)
    w = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, 360))


def phys(x, ins, means, stds, name):
    return x[ins.index(name)] * stds[name] + means[name]


def masks(x, ins, means, stds):
    ssf = phys(x, ins, means, stds, "sea_surface_fraction")
    sic = phys(x, ins, means, stds, "ocean_sea_ice_fraction")
    ocean = ssf > 0.99
    openw = ocean & (sic < 0.01)
    lat, lon = Grid.lat, Grid.lon
    return {
        "global_open_ocean": openw,
        "nino34": openw & (np.abs(lat)[:, None] <= 5) & ((lon >= 190) & (lon <= 240))[None, :],
        "tropics_20": openw & (np.abs(lat)[:, None] <= 20),
        "southern_ocean_40_60S": openw & ((lat >= -60) & (lat <= -40))[:, None],
        "north_atlantic_40_60N": openw & ((lat >= 40) & (lat <= 60))[:, None] & ((lon >= 300) | (lon <= 10))[None, :],
    }


def wmean(a, m):
    w = Grid.w
    return float((a * w)[m].sum() / w[m].sum())


def mixed_layer_depth(x, ins, means, stds, dT=0.2):
    """Depth (m) where layer temperature first falls dT below the top layer, linear between layer
    centres; at least the top layer (20 m). NaN-free inputs are zero over land (not used there)."""
    T = np.stack([phys(x, ins, means, stds, f"temperatureCoarsened_{i}") for i in range(19)])
    target = T[0] - dT
    h = np.full(T.shape[1:], IDEPTH[-1])
    found = np.zeros(T.shape[1:], bool)
    for i in range(1, 19):
        cross = (~found) & (T[i] < target)
        frac = np.clip((T[i - 1] - target) / np.maximum(T[i - 1] - T[i], 1e-6), 0, 1)
        h = np.where(cross, ZMID[i - 1] + frac * (ZMID[i] - ZMID[i - 1]), h)
        found |= cross
    return np.maximum(h, IDEPTH[1])


def sw_absorbed_fraction(h, R=0.58, z1=0.35, z2=23.0):
    """Paulson & Simpson (1977) Jerlov type I: fraction of FSNS absorbed above depth h."""
    return 1 - (R * np.exp(-h / z1) + (1 - R) * np.exp(-h / z2))


def slab_law(dQ, h, comp):
    """dSST (K) of a slab of depth h over one 5-day step for a net-flux change dQ (W/m^2)."""
    f = sw_absorbed_fraction(h) if comp == "FSNS" else 1.0
    return f * dQ * DT / (RHO_CP * h)


class Graph:
    """Samudra.forward, one edge at a time. run(x, src=None, keep=False):
    src maps edge name -> sender tensor to read instead (optionally (tensor, channel_mask) to swap
    only some channels). keep=True returns the cache of every sender value too."""
    ENC = (0, 2, 4, 6)
    EDGES = ("D0", "D1", "D2", "D3", "B", "U3", "U2", "U1", "S0", "S1", "S2", "S3")

    def __init__(self, ocn):
        self.ocn = ocn
        self.L = ocn.layers
        self.pad = ocn.pad
        self.npad = ocn.N_pad

    def _pad_to(self, u, like):
        pads = np.array(like.shape[2:]) - np.array(u.shape[2:])
        u = F.pad(u, (int(pads[1] // 2), int(pads[1] - pads[1] // 2), 0, 0), mode=self.pad)
        return F.pad(u, (0, 0, int(pads[0] // 2), int(pads[0] - pads[0] // 2)), mode="constant")

    @torch.no_grad()
    def run(self, x, src=None, keep=False):
        src = src or {}
        cache = {}

        def e(name, v):
            cache[name] = v
            if name not in src:
                return v
            s = src[name]
            if isinstance(s, tuple):
                t, cm = s
                return torch.where(cm[None, :, None, None], t, v)
            return s

        L = self.L
        e0 = L[0](x)
        e1 = L[2](L[1](e("D0", e0)))
        e2 = L[4](L[3](e("D1", e1)))
        e3 = L[6](L[5](e("D2", e2)))
        b = L[8](L[7](e("D3", e3)))
        s3 = self._pad_to(L[9](e("B", b)), e3) + e("S3", e3)
        c3 = L[10](s3)
        s2 = self._pad_to(L[11](e("U3", c3)), e2) + e("S2", e2)
        c2 = L[12](s2)
        s1 = self._pad_to(L[13](e("U2", c2)), e1) + e("S1", e1)
        c1 = L[14](s1)
        up0 = self._pad_to(L[15](e("U1", c1)), e0)
        cache["up0"] = up0
        y = self.head(up0, e("S0", e0))
        return (y, cache) if keep else y

    @torch.no_grad()
    def head(self, up0, skip0):
        """Level-0 sum, last ConvNeXt and the output conv: all an S0- or U1-only patch changes."""
        L = self.L
        s0 = up0 + skip0
        c0 = L[16](s0)
        y = F.pad(c0, (self.npad, self.npad, 0, 0), mode=self.pad)
        y = F.pad(y, (0, 0, self.npad, self.npad), mode="constant")
        return L[17](y)


def dump(obj, path):
    def conv(o):
        if isinstance(o, dict):
            return {str(k): conv(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [conv(v) for v in o]
        if isinstance(o, (np.floating, float)):
            return float(o) if np.isfinite(o) else None
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.ndarray):
            return conv(o.tolist())
        return o
    json.dump(conv(obj), open(path, "w"), indent=1)
