"""exege ``Intervenable`` for SamudrACE C1A0's ocean (Samudra U-Net), replayed from cached inputs.

What it is
----------
``capture.py`` saves the exact fp32 tensor that entered Samudra at each 5-day ocean step of an
official ``fme.coupled.inference`` run (normalized, packed in ``in_names`` order, land filled
with 0). This adapter loads the checkpoint once and replays ocean steps from those tensors
with its own copy of ``Samudra.forward`` (checked bitwise against the module), so that every
place in the U-Net can be read or replaced:

    layer  0          network input, 91 channels, in PHYSICAL units (x * std + mean); an edit
                      is applied as a delta in normalized space, so an identity edit is exact
    layer  1 + k      output of ``Samudra.layers[k]``, k = 0..17 (ConvNeXt / AvgPool /
                      upsample / final Conv2d). For an upsample layer the place is AFTER the
                      pad and BEFORE the skip is added: it is the deep path's contribution only
    layer 19 + j      skip edge j: encoder level j (output of layers[2j]) as read by the
                      decoder where it is added (after layers[15 - 2j]). Replacing it leaves
                      the main path that leaves layers[2j] untouched

Only layers 0 (input), 1 (layers.0), 16 (layers.15), 17 (layers.16), 18 (layers.17) and 19
(skip.0) live on the 180x360 grid; the rest are ``off_grid_layers``. Hooks may still target
them: ``edit`` then sees ``(H_k * W_k, C)``. Layer shapes are in ``self.shapes``.

Outputs (``Rollout.fields``) are the 80 prognostic outputs denormalized to physical units,
NaN where the official run writes NaN (masks taken from the official prediction file).
The checkpoint's ocean corrector is re-implemented in numpy: salinity and ice volume forced
>= 0, ``ocean_sea_ice_fraction`` clamped to [0, 1], ``iceVolumeTotal`` zeroed where ice-free.
It has no heat-budget correction (surface_energy_flux_correction and
ocean_heat_content_correction are null), so the law tested here is not hard-coded, and sst
and temperature pass through untouched (validate.py checks against the official run).

Time
----
``initial_state(k)`` is cached input k (1985-01-01 + 5k days). ``run(state, steps)`` with
``steps > 1`` is an OCEAN-ONLY replay: the prognostic channels are fed back from the model's
own output, but the atmosphere fluxes and forcings come from cached input k + t of the CONTROL
coupled run (teacher-forced atmosphere). That matches the official control trajectory exactly,
but a perturbed arm sees no atmospheric feedback. The ocean is deterministic: ``noise_seed``
is accepted and ignored.

Two ways in
-----------
``run`` is the exege protocol (numpy in, numpy out). ``forward_batch`` is the torch-native
path the experiments use for speed: a batch of arms, each with its own {place: fn(tensor)}
patches, in one forward. Samples in a batch are independent (InstanceNorm is per sample).
"""

from __future__ import annotations

import glob
import os
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch
import yaml

from exege.core.errors import RequestError
from exege.latents.grid import Grid
from exege.latents.source import LatentInfo, LayerInfo
from exege.latents.steering import Hook, Rollout

from hc import paths

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = paths.CKPT["C1A0"]
CFG = paths.CKPT_CFG_C1A0
N_LAYERS = 18
SKIP0 = 1 + N_LAYERS  # layer index of skip edge 0
STEP_SECONDS = 5 * 86400


def place_name(layer: int) -> str:
    if layer == 0:
        return "input"
    if 1 <= layer <= N_LAYERS:
        return f"layers.{layer - 1}"
    if SKIP0 <= layer < SKIP0 + 4:
        return f"skip.{layer - SKIP0}"
    raise RequestError(f"no place {layer}: 0 = input, 1..18 = layers.0..17, 19..22 = skip.0..3")


def _device() -> torch.device:
    if os.environ.get("FME_FORCE_CPU"):
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SamudraReplay:
    """``Intervenable`` over one ocean step of SamudrACE C1A0, from cached coupled inputs."""

    def __init__(self, cache: str = os.path.join(paths.OUT, "states", "ic1985"), *, device: str | None = None,
                 ckpt: str | None = None, start_date: str | None = None):
        self.cache = os.path.abspath(cache)
        self.ckpt = ckpt or CKPT
        sd = os.path.join(self.cache, "start_date.txt")
        self.start_date = start_date or (open(sd).read().strip() if os.path.exists(sd) else "1985-01-01")
        self.dev = torch.device(device) if device else _device()
        ck = yaml.safe_load(open(CFG))["ocean"]["stepper"]["step"]["config"]
        self.in_names: list[str] = list(ck["in_names"])
        self.out_names: list[str] = list(ck["out_names"])
        nrm = ck["normalization"]["network"]
        self.in_mean = np.array([nrm["means"][n] for n in self.in_names], np.float32)
        self.in_std = np.array([nrm["stds"][n] for n in self.in_names], np.float32)
        self.out_mean = np.array([nrm["means"][n] for n in self.out_names], np.float32)
        self.out_std = np.array([nrm["stds"][n] for n in self.out_names], np.float32)
        self.prog_in = np.array([self.in_names.index(n) for n in self.out_names])  # out -> in slots
        self.files = sorted(glob.glob(os.path.join(self.cache, "x_*.npy")))
        if not self.files:
            raise RequestError(f"no cached inputs (x_*.npy) in {self.cache}: run capture.py")
        meta = np.load(os.path.join(self.cache, "meta.npz"))
        self.lat, self.lon = meta["lat"], meta["lon"]
        self.out_mask = meta["out_mask"]  # (80, 180, 360) True where the official output is finite
        self.landfrac_in = self.in_names.index("LANDFRAC")
        from fme.coupled.inference.inference import load_stepper

        from hc import set_numerics
        self.tf32 = set_numerics()
        st = load_stepper(self.ckpt)
        self.net = st.ocean.modules[0].module.to(self.dev).eval()
        self.net.checkpoint_strategy = None
        for layer in self.net.layers:
            if hasattr(layer, "checkpoint_strategy"):
                layer.checkpoint_strategy = None
        self.H, self.W = self.lat.size, self.lon.size
        lat2, lon2 = np.meshgrid(self.lat, self.lon, indexing="ij")
        self._grid = Grid(lat=lat2.ravel(), lon=lon2.ravel(), shape=(self.H, self.W),
                          mask=self.out_mask[0].ravel())
        x0 = self.load_x(0)
        rec: dict[int, torch.Tensor] = {}
        self.forward_batch(x0[None], [{}], record=range(0, SKIP0 + 4), out=rec)
        self.shapes = {k: tuple(v.shape[1:]) for k, v in rec.items()}  # (C, H, W)
        on, off = [], []
        for k in range(SKIP0 + 4):
            c, h, w = self.shapes[k]
            li = LayerInfo(index=k, label=place_name(k), n_channels=c)
            (on if (h, w) == (self.H, self.W) else off).append(li)
        self._info = LatentInfo(
            source=self.cache, times=tuple(self.time_label(i) for i in range(len(self.files))),
            layers=tuple(on), n_nodes=self.H * self.W, model="SamudrACE C1A0 ocean (Samudra U-Net)",
            component="ocean", checkpoint="mahf708/e3smv3-lr-hist-samudrace@rc1 checkpoints/C1A0.tar",
            calendar="noleap", timestep_seconds=STEP_SECONDS, off_grid_layers=tuple(off),
            options={"device": str(self.dev)},
            experiment={"replay": "ocean-only from cached coupled inputs; teacher-forced atmosphere"},
        )

    # -- exege protocol --------------------------------------------------------
    def info(self) -> LatentInfo:
        return self._info

    def grid(self) -> Grid:
        return self._grid

    def initial_state(self, start: str | int = 0) -> int:
        k = self._info.times.index(start) if isinstance(start, str) else int(start)
        if not 0 <= k < len(self.files):
            raise RequestError(f"start {start!r} not in 0..{len(self.files) - 1}")
        return k

    def run(self, initial_state: Any, steps: int, hooks: Sequence[Hook] = (), *,
            noise_seed: int | None = None, record: Sequence[tuple[int, int]] = ()) -> Rollout:
        k0 = int(initial_state)
        if steps < 1 or k0 + steps > len(self.files):
            raise RequestError(f"steps must be 1..{len(self.files) - k0} from start {k0}")
        for h in hooks:
            place_name(h.layer)
            if not 0 <= h.time < steps:
                raise RequestError(f"hook time {h.time} outside 0..{steps - 1}")
        for layer, t in record:
            place_name(layer)
        fields = {n: np.empty((steps, self.H * self.W)) for n in self.out_names}
        latents: dict[tuple[int, int], np.ndarray] = {}
        x = self.load_x(k0)
        for t in range(steps):
            patches = {h.layer: self._np_edit(h.layer, h.edit) for h in hooks if h.time == t}
            rec: dict[int, torch.Tensor] = {}
            want = [layer for layer, tt in record if tt == t]
            y = self.forward_batch(x[None], [patches], record=want, out=rec)[0]
            phys = self.to_physical(y)
            for i, n in enumerate(self.out_names):
                fields[n][t] = phys[i].ravel()
            for layer in want:
                latents[(layer, t)] = self._to_nodes(rec[layer][0])
            if t + 1 < steps:
                x = self.next_input(x, phys, k0 + t + 1)
        return Rollout(fields=fields, latents=latents)

    # -- torch-native path -------------------------------------------------------
    def load_x(self, k: int) -> torch.Tensor:
        return torch.from_numpy(np.load(self.files[k])).to(self.dev)

    def time_label(self, k: int) -> str:
        import cftime  # noleap calendar, as the model
        y, m, d = (int(v) for v in self.start_date[:10].split("-"))
        t = cftime.DatetimeNoLeap(y, m, d) + __import__("datetime").timedelta(days=5 * k)
        return t.strftime("%Y-%m-%d")

    @torch.no_grad()
    def forward_batch(self, x: torch.Tensor, patches: Sequence[dict[int, Callable]], *,
                      record: Sequence[int] = (), out: dict | None = None) -> torch.Tensor:
        """Samudra.forward on a batch ``x`` (B, 91, H, W), sample b patched by ``patches[b]``
        ({place: fn(tensor (C,h,w)) -> tensor}). ``record`` places are written to ``out``
        as (B, C, h, w) tensors, taken after any patch."""
        net, B = self.net, x.shape[0]
        record = set(record)

        def at(place: int, t: torch.Tensor) -> torch.Tensor:
            fns = [p.get(place) for p in patches]
            if any(f is not None for f in fns):
                t = torch.stack([f(t[b]) if f is not None else t[b] for b, f in enumerate(fns)])
            if out is not None and place in record:
                out[place] = t.clone()
            return t

        fts = at(0, x.clone())
        temp: list[torch.Tensor] = []
        count, ns = 0, net.num_steps
        for k, layer in enumerate(net.layers):
            if isinstance(layer, torch.nn.Conv2d):
                fts = torch.nn.functional.pad(fts, (net.N_pad, net.N_pad, 0, 0), mode=net.pad)
                fts = torch.nn.functional.pad(fts, (0, 0, net.N_pad, net.N_pad), mode="constant")
            fts = layer(fts)
            is_up = type(layer).__name__ in ("BilinearUpsample", "ZonallyPeriodicBilinearUpsample")
            if count < ns:
                fts = at(1 + k, fts)
                if type(layer).__name__ == "ConvNeXtBlock":
                    temp.append(fts)
                    count += 1
            else:
                if is_up:
                    j = int(2 * ns - count - 1)
                    shape, crop = np.array(temp[j].shape[2:]), np.array(fts.shape[2:])
                    pads = shape - crop
                    fts = torch.nn.functional.pad(fts, (pads[1] // 2, pads[1] - pads[1] // 2, 0, 0), mode=net.pad)
                    fts = torch.nn.functional.pad(fts, (0, 0, pads[0] // 2, pads[0] - pads[0] // 2), mode="constant")
                    fts = at(1 + k, fts)
                    fts = fts + at(SKIP0 + j, temp[j])
                    count += 1
                else:
                    fts = at(1 + k, fts)
        return fts

    # -- physics glue -------------------------------------------------------------
    def to_physical(self, y: torch.Tensor) -> np.ndarray:
        """(80, H, W) normalized network output -> physical units, official NaN mask, and the
        corrector's sea-ice / positivity rules (approximate; sst untouched)."""
        p = y.detach().float().cpu().numpy() * self.out_std[:, None, None] + self.out_mean[:, None, None]
        o = self.out_names
        for n in o:  # ForcePositive, then SeaIceFractionCorrection (remove_negative_ocean_fraction=False)
            if n.startswith("salinityCoarsened") or n == "iceVolumeTotal":
                p[o.index(n)] = np.maximum(p[o.index(n)], 0)
        i_sif, i_vol = o.index("ocean_sea_ice_fraction"), o.index("iceVolumeTotal")
        p[i_sif] = np.clip(p[i_sif], 0.0, 1.0)
        p[i_vol] = np.where(p[i_sif] == 0, 0.0, p[i_vol])
        return np.where(self.out_mask, p, np.nan)

    def next_input(self, x: torch.Tensor, phys: np.ndarray, k_next: int) -> torch.Tensor:
        """Next cached input with its prognostic channels replaced by this step's output."""
        xn = self.load_x(k_next).clone()
        norm = (np.nan_to_num(phys) - self.out_mean[:, None, None]) / self.out_std[:, None, None]
        norm = np.where(self.out_mask, norm, 0).astype(np.float32)
        xn[self.prog_in] = torch.from_numpy(norm).to(self.dev)
        return xn

    def _to_nodes(self, t: torch.Tensor) -> np.ndarray:
        c = t.shape[0]
        return t.detach().float().cpu().numpy().reshape(c, -1).T.copy()  # (nodes, channels)

    def _np_edit(self, layer: int, edit: Callable[[np.ndarray], np.ndarray]) -> Callable:
        scale = torch.from_numpy(self.in_std).to(self.dev)[:, None, None]
        shift = torch.from_numpy(self.in_mean).to(self.dev)[:, None, None]

        def fn(t: torch.Tensor) -> torch.Tensor:
            c, h, w = t.shape
            if layer == 0:
                phys = t * scale + shift
                a = phys.detach().cpu().numpy().reshape(c, -1).T
                e = np.asarray(edit(a.copy()), np.float32)
                if e.shape != a.shape:
                    raise RequestError(f"edit at input returned {e.shape}, expected {a.shape}")
                delta = torch.from_numpy((e - a).T.reshape(c, h, w)).to(self.dev)
                return t + delta / scale
            a = t.detach().cpu().numpy().reshape(c, -1).T
            e = np.asarray(edit(a.copy()), np.float32)
            if e.shape != a.shape:
                raise RequestError(f"edit at {place_name(layer)} returned {e.shape}, expected {a.shape}")
            return torch.from_numpy(e.T.reshape(c, h, w).copy()).to(self.dev)

        return fn


def write_meta(cache: str, predictions_nc: str) -> None:
    """lat, lon and per-output finite masks from an official ocean prediction file."""
    import xarray as xr

    ck = yaml.safe_load(open(CFG))["ocean"]["stepper"]["step"]["config"]
    ds = xr.open_dataset(predictions_nc).isel(sample=0, time=-1)
    mask = np.stack([np.isfinite(ds[n].values) for n in ck["out_names"]])
    np.savez(os.path.join(cache, "meta.npz"), lat=ds.lat.values, lon=ds.lon.values, out_mask=mask)
