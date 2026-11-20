"""Idea 3: one-step (6 h) gain per spherical-harmonic degree band of the C1A0 atmosphere network, by JVP.
PREREG 3. Network map (pre-corrector), linearised at truth states captured as the first atmosphere call of
truth-started evaluator runs (exact fme packing). Noise held fixed: we call conditional_model with our own
seeded isotropic noise draw (same generator as fme's NoiseConditionedModel), so primal and tangent see one draw.

Gain(band) = sqrt( sum_l in band P_out(l) / sum_l in band P_in(l) ), physical units (K, m/s, Pa), probes =
band-limited random fields with flat power over the band (complex Gaussian coefficients, all m <= l), 8 per band.
For every input channel probe we keep the response spectrum of every tracked output channel (diagonal and
cross pairs from the same JVPs). Noise gain: JVP w.r.t. the noise tensor (band-limited probe in the 32 noise
channels, unit RMS), response in physical units. Linearity: finite differences at eps in {0.01, 0.1} x the
state's band RMS on state 0, seed 0.
usage: python gain_atm.py CAPTURE_DIR OUT_TAG [state_index ...]   -> idea3_gain/results/gain_atm_TAG.npz
"""
import json
import os
import sys
import time

import numpy as np
import torch

from hc import paths, set_numerics
from hc.spectra import _sht

CAP, TAG = os.path.abspath(sys.argv[1]), sys.argv[2]
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RES, exist_ok=True)
set_numerics()
DEV = torch.device("cuda")
torch.backends.cudnn.benchmark = False

from fme.coupled.inference.inference import load_stepper  # noqa: E402
from fme.core.models.conditional_sfno.sfnonet import Context  # noqa: E402

st = load_stepper(paths.CKPT[os.environ.get("HC_CKPT", "C1A0")])
atm = st.atmosphere.modules[0].module.to(DEV).eval()
net = atm.conditional_model
atm.requires_grad_(False)  # forward-mode only: no backward graph
net.checkpointing = 0
for b in net.blocks:
    if hasattr(b, "mlp") and hasattr(b.mlp, "checkpointing"):
        b.mlp.checkpointing = 0
sa = st.atmosphere._step_obj
INN, OUTN = list(sa.in_names), list(sa.out_names)
MU = {k: float(v) for k, v in sa.normalizer.means.items()}
SD = {k: float(v) for k, v in sa.normalizer.stds.items()}
X = np.load(os.path.join(CAP, "atm_x_00.npy"))  # (n_states, 44, 180, 360), normalized
states = [int(s) for s in sys.argv[3:]] or list(range(X.shape[0]))
SEEDS = [0, 1, 2]
IN_CH = ["T_1", "T_4", "T_7", "U_1", "U_4", "U_7", "V_4", "V_7", "PS", "TS"]
OUT_CH = ["T_1", "T_4", "T_7", "U_1", "U_4", "U_7", "V_4", "V_7", "PS", "TS"]
BANDS = [(0, 2), (3, 5)] + [(l, l + 3) for l in range(6, 90, 4)]
NPROBE = 8
if os.environ.get("GAIN_TEST"):
    IN_CH, BANDS, SEEDS = ["T_7"], [(0, 2), (30, 33)], [0]
sht, isht = _sht(DEV, "s"), _sht(DEV, "is")
LMAX, MMAX = sht.lmax, sht.mmax
o_idx = torch.tensor([OUTN.index(c) for c in OUT_CH], device=DEV)
o_sd = torch.tensor([SD[c] for c in OUT_CH], device=DEV)[None, :, None, None]


def power(f):  # (..., H, W) -> (..., L) power per degree
    c = sht(f.float())
    p = c.abs() ** 2
    p[..., 1:] *= 2
    return p.sum(-1)


def band_probe(lo, hi, n, gen):
    c = torch.zeros(n, LMAX, MMAX, dtype=torch.complex64, device=DEV)
    for l in range(lo, hi + 1):
        m = min(l, MMAX - 1) + 1
        z = torch.randn(n, m, 2, generator=gen, device=DEV)
        z = torch.complex(z[..., 0], z[..., 1]) / np.sqrt(2.0)
        z[:, 0] = z[:, 0].real.to(torch.complex64) * np.sqrt(2.0)
        c[:, l, :m] = z / np.sqrt(2 * l + 1)  # flat power per degree
    f = isht(c)
    lat = torch.tensor(np.cos(np.deg2rad(np.linspace(-89.5, 89.5, 180))), device=DEV, dtype=torch.float32)
    rms = torch.sqrt((f ** 2 * lat[:, None]).sum((-2, -1)) / (lat.sum() * 360))
    return f / rms[:, None, None]  # unit area-weighted RMS


def make_noise(seed, n):
    from fme.ace.registry.stochastic_sfno import isotropic_noise

    torch.manual_seed(seed)
    if atm._inverse_sht is not None:
        return isotropic_noise((n, atm.embed_dim), atm._lmax, atm._mmax, atm._inverse_sht, device=DEV)
    return torch.randn(n, atm.embed_dim, 180, 360, device=DEV)


def ctx(noise):
    pos = atm.pos_embed.repeat(noise.shape[0], 1, 1, 1) if atm.pos_embed is not None else None
    return Context(embedding_scalar=None, embedding_pos=pos, labels=None, noise=noise)


def f_out(x, noise):  # physical-unit tracked outputs
    y = net(x, ctx(noise))
    return y[:, o_idx] * o_sd  # deviations: means cancel in differences/JVPs


res = {"bands": BANDS, "in_ch": IN_CH, "out_ch": OUT_CH, "seeds": SEEDS, "states": states,
       "P_in": [], "P_resp": [], "P_noise_resp": [], "P_seeddiff": [], "P_state_out": []}
lin = {}
t0 = time.time()
B = int(os.environ.get("GAIN_BATCH", "2"))
PARTS = os.path.join(RES, "gain_parts")
os.makedirs(PARTS, exist_ok=True)
for si in states:
    x0 = torch.tensor(X[si:si + 1], device=DEV)
    for seed in SEEDS:
        part = os.path.join(PARTS, f"state{si:02d}_seed{seed}.npz")
        if os.path.exists(part) and not os.environ.get("GAIN_TEST"):  # resume: this (state, seed) is done
            z = np.load(part)
            for k in ("P_in", "P_resp", "P_noise_resp", "P_seeddiff", "P_state_out"):
                res[k].append(z[k])
            print(f"state {si} seed {seed} resumed", flush=True)
            continue
        nz = make_noise(1000 * si + seed, 1)
        gen = torch.Generator(device=DEV).manual_seed(10_000 + 100 * si + seed)
        Pin = np.zeros((len(IN_CH), len(BANDS), LMAX))
        Pr = np.zeros((len(IN_CH), len(BANDS), len(OUT_CH), LMAX))
        for ii, cin in enumerate(IN_CH):
            ci = INN.index(cin)
            for bi, (lo, hi) in enumerate(BANDS):
                d = band_probe(lo, hi, NPROBE, gen)  # physical units
                for s in range(0, NPROBE, B):
                    dd = d[s:s + B]
                    n = dd.shape[0]
                    v = torch.zeros(n, *x0.shape[1:], device=DEV)
                    v[:, ci] = dd / SD[cin]
                    _, jv = torch.func.jvp(lambda xx: f_out(xx, nz.repeat(n, 1, 1, 1)),
                                           (x0.repeat(n, 1, 1, 1),), (v,))
                    Pin[ii, bi] += power(dd).sum(0).double().cpu().numpy()
                    Pr[ii, bi] += power(jv).sum(0).double().cpu().numpy()
                if si == states[0] and seed == 0 and cin in ("T_7", "U_7", "PS") and bi % 4 == 0:
                    with torch.no_grad():
                        y0 = f_out(x0, nz)
                        xs = power(x0[0, ci] * SD[cin])[lo:hi + 1].sum().sqrt() / np.sqrt(4 * np.pi)
                        for eps in (0.01, 0.1):
                            sc = float(eps * xs / 1.0)
                            v = torch.zeros(NPROBE, *x0.shape[1:], device=DEV)
                            v[:, ci] = d * sc / SD[cin]
                            ys = torch.cat([f_out(x0 + v[j:j + 1], nz) for j in range(NPROBE)]) - y0
                            pf = power(ys / sc).sum(0)[:, lo:hi + 1].sum(-1)
                            pj = torch.tensor(Pr[ii, bi][:, lo:hi + 1].sum(-1), device=DEV)
                            lin[f"{cin}|band{lo}-{hi}|eps{eps}"] = (pf.sqrt() / pj.sqrt()).cpu().numpy().tolist()
        res["P_in"].append(Pin)
        res["P_resp"].append(Pr)
        # noise gain: tangent in the noise input
        Pn = np.zeros((len(BANDS), len(OUT_CH), LMAX))
        for bi, (lo, hi) in enumerate(BANDS):
            d = band_probe(lo, hi, 4, gen)  # (4,H,W) unit RMS, applied to all noise channels
            vn = d[:, None].expand(-1, nz.shape[1], -1, -1) / np.sqrt(nz.shape[1])
            _, jn = torch.func.jvp(lambda q: f_out(x0.repeat(4, 1, 1, 1), q), (nz.repeat(4, 1, 1, 1),), (vn.contiguous(),))
            Pn[bi] = power(jn).sum(0).double().cpu().numpy() / 4
        res["P_noise_resp"].append(Pn)
        with torch.no_grad():
            ya, yb = f_out(x0, nz), f_out(x0, make_noise(1000 * si + seed + 500, 1))
            res["P_seeddiff"].append(power(ya - yb)[0].double().cpu().numpy())
            res["P_state_out"].append(power(ya)[0].double().cpu().numpy())
        if not os.environ.get("GAIN_TEST"):
            np.savez(part, **{k: res[k][-1] for k in ("P_in", "P_resp", "P_noise_resp", "P_seeddiff", "P_state_out")})
            json.dump(lin, open(part.replace(".npz", "_lin.json"), "w"))
        print(f"state {si} seed {seed} {time.time() - t0:.0f}s", flush=True)

np.savez_compressed(os.path.join(RES, f"gain_atm_{TAG}.npz"),
                    **{k: np.array(v) for k, v in res.items() if k.startswith("P_")},
                    bands=np.array(BANDS), states=np.array(states))
json.dump({"in_ch": IN_CH, "out_ch": OUT_CH, "seeds": SEEDS, "states": states, "bands": BANDS,
           "nprobe": NPROBE, "linearity_fd_over_jvp": lin, "capture": CAP, "seconds": time.time() - t0},
          open(os.path.join(RES, f"gain_atm_{TAG}.json"), "w"), indent=1)
print("done", time.time() - t0)
