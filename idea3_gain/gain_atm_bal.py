"""Idea 3, EXPLORATORY (PREREG change log 2026-11-20): balanced-probe one-step gain of the C1A0 atmosphere.

Tangent = truth 5-day state difference d = x(t) - x(t - 5 d) over all prognostic input channels together
(forcings / fractions held fixed at 0), band-passed to one SH band per JVP (physical units per channel).
Gain_v(band) = sqrt( in-band output power of v / in-band input power of v ), the analogue of the truth per-degree
AR(1) transfer. Noise held fixed as in gain_atm.py (own seeded draw), 3 seeds, 12 states.
usage: python gain_atm_bal.py CAP_T CAP_TM5 TAG -> results/gain_atm_bal_TAG.json/.npz
"""
import json
import os
import sys
import time

import numpy as np
import torch

from hc import paths, set_numerics
from hc.spectra import _sht

CAP, CAPM, TAG = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2]), sys.argv[3]
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
set_numerics()
DEV = torch.device("cuda")
from fme.coupled.inference.inference import load_stepper  # noqa: E402
from fme.core.models.conditional_sfno.sfnonet import Context  # noqa: E402
from fme.ace.registry.stochastic_sfno import isotropic_noise  # noqa: E402

st = load_stepper(paths.CKPT["C1A0"])
atm = st.atmosphere.modules[0].module.to(DEV).eval()
atm.requires_grad_(False)
net = atm.conditional_model
net.checkpointing = 0
for b in net.blocks:
    if hasattr(b, "mlp") and hasattr(b.mlp, "checkpointing"):
        b.mlp.checkpointing = 0
sa = st.atmosphere._step_obj
INN, OUTN = list(sa.in_names), list(sa.out_names)
SD = {k: float(v) for k, v in sa.normalizer.stds.items()}
PROG = [n for n in INN if n in OUTN]  # prognostic inputs (also outputs)
TRACK = ["T_1", "T_4", "T_7", "U_1", "U_4", "U_7", "V_4", "V_7", "PS", "TS"]
BANDS = [(0, 2), (3, 5)] + [(l, l + 3) for l in range(6, 90, 4)]
X, XM = np.load(os.path.join(CAP, "atm_x_00.npy")), np.load(os.path.join(CAPM, "atm_x_00.npy"))
sht, isht = _sht(DEV, "s"), _sht(DEV, "is")
o_idx = torch.tensor([OUTN.index(c) for c in TRACK], device=DEV)
o_sd = torch.tensor([SD[c] for c in TRACK], device=DEV)[None, :, None, None]
pin = torch.tensor([INN.index(c) for c in PROG], device=DEV)
pin_sd = torch.tensor([SD[c] for c in PROG], device=DEV)[:, None, None]
trk_in = [PROG.index(c) for c in TRACK]


def power(f):
    c = sht(f.float())
    p = c.abs() ** 2
    p[..., 1:] *= 2
    return p.sum(-1)


def bandpass(f, lo, hi):
    c = sht(f.float())
    keep = torch.zeros(c.shape[-2], device=DEV)
    keep[lo:hi + 1] = 1
    return isht(c * keep[:, None])


def f_out(x, noise):
    pos = atm.pos_embed.repeat(noise.shape[0], 1, 1, 1) if atm.pos_embed is not None else None
    return net(x, Context(embedding_scalar=None, embedding_pos=pos, labels=None, noise=noise))[:, o_idx] * o_sd


Pin = np.zeros((X.shape[0], 3, len(BANDS), len(TRACK), sht.lmax))
Pout = np.zeros_like(Pin)
Pcross = np.zeros_like(Pin)  # Re sum_m conj(c_in) c_out per l (m>0 doubled): projection numerator


def cross(a, b):
    ca, cb = sht(a.float()), sht(b.float())
    p = (ca.conj() * cb).real
    p[..., 1:] *= 2
    return p.sum(-1)
t0 = time.time()
for si in range(X.shape[0]):
    x0 = torch.tensor(X[si:si + 1], device=DEV)
    d_phys = (torch.tensor(X[si] - XM[si], device=DEV)[pin]) * pin_sd  # (n_prog, H, W) physical
    for k, seed in enumerate((0, 1, 2)):
        torch.manual_seed(1000 * si + seed)
        nz = isotropic_noise((1, atm.embed_dim), atm._lmax, atm._mmax, atm._inverse_sht, device=DEV)
        for bi, (lo, hi) in enumerate(BANDS):
            db = bandpass(d_phys, lo, hi)  # (n_prog, H, W)
            v = torch.zeros_like(x0)
            v[0, pin] = db / pin_sd
            _, jv = torch.func.jvp(lambda xx: f_out(xx, nz), (x0,), (v.contiguous(),))
            Pin[si, k, bi] = power(db[trk_in]).double().cpu().numpy()
            Pout[si, k, bi] = power(jv[0]).double().cpu().numpy()
            Pcross[si, k, bi] = cross(db[trk_in], jv[0]).double().cpu().numpy()
    print(f"state {si} {time.time() - t0:.0f}s", flush=True)
g = {}
for vi, c in enumerate(TRACK):
    a = np.zeros((X.shape[0] * 3, len(BANDS)))
    for bi, (lo, hi) in enumerate(BANDS):
        a[:, bi] = np.sqrt(Pout[:, :, bi, vi, lo:hi + 1].sum(-1) / Pin[:, :, bi, vi, lo:hi + 1].sum(-1)).ravel()
    pr = np.zeros_like(a)
    for bi, (lo, hi) in enumerate(BANDS):
        pr[:, bi] = (Pcross[:, :, bi, vi, lo:hi + 1].sum(-1) / Pin[:, :, bi, vi, lo:hi + 1].sum(-1)).ravel()
    g[c] = {"mean": a.mean(0).tolist(), "p05": np.percentile(a, 5, 0).tolist(), "p95": np.percentile(a, 95, 0).tolist(),
            "proj_mean": pr.mean(0).tolist(), "proj_p05": np.percentile(pr, 5, 0).tolist(), "proj_p95": np.percentile(pr, 95, 0).tolist()}
np.savez_compressed(os.path.join(RES, f"gain_atm_bal_{TAG}.npz"), P_in=Pin, P_out=Pout, P_cross=Pcross, bands=np.array(BANDS))
json.dump({"track": TRACK, "bands": BANDS, "gain": g, "prognostic_channels_perturbed": PROG, "seconds": time.time() - t0,
           "label": "EXPLORATORY (PREREG change log 2026-11-20)"}, open(os.path.join(RES, f"gain_atm_bal_{TAG}.json"), "w"), indent=1)
print(json.dumps({c: [round(x, 2) for x in g[c]["mean"][::4]] for c in TRACK}))
