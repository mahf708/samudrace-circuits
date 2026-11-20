"""Idea 2 starter: do explanations keep the emulator's REAL errors? (PREREG §2; Nov 4 paper)

One-step (6 h) C1A0 atmosphere forecasts from E3SM truth states, scored against E3SM truth at t + 6 h.
- States: the first atmosphere call of truth-started runs (`hc.evaluate --capture-atm 1`), e.g. data/states/atm12
  (Jan 16 / Apr 16 / Jul 15 / Oct 13, 00 UTC of 1996, 2042, 2047). Fit dates = years 1996 and 2042, score
  dates = 2047 (PREREG fit/score blocks).
- Model: the network map (PRE-corrector), noise held fixed per seed (own seeded isotropic draw), seeds 0..3.
  The reference forecast is seed 0.
- Error e = model(seed 0) − truth per node and variable. Tolerance tau = max(K · sigma_seed, 1% of the field's
  spatial SD), sigma_seed = SD over the 4 seeds at that node (same IC). Node is an "error" if |e| > tau.
- An explanation produces y_expl. Scores (area-weighted, cos lat):
    A_ok  = P(|y_expl − y_model| <= tau | ok)      A_err = P(|y_expl − y_model| <= tau | error)
    three-way split on error nodes: reproduces (|y_expl − y_model| <= tau), corrects (not reproduces and
    |y_expl − truth| <= tau), other.
- Explanations shipped (add yours in EXPLANATIONS):
    seed1           another seed's forecast (the seed ceiling)
    pca{k}_L{l}     SFNO block-l output replaced by its projection on the top-k PCs (fit dates, seed 0); k = 0
                    is the empty explanation (the fit-date mean latent)
    offset          scalar-offset control: persistence (the input value) + a per-variable constant fitted on fit dates
    affine          output-level control: a + g * input, per variable, weighted LSQ on fit dates
Not included yet: corrector in/out, block bootstrap, the ocean, pre-corrector strata (PREREG 2.4).
usage: python idea2_errors/strata.py [CAPTURE=data/states/atm12] [--layer 7] [--k 0 8 32 128] [--K 2]
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from hc import paths, set_numerics, truth_atm, truth_ocean

ap = argparse.ArgumentParser()
ap.add_argument("capture", nargs="?", default="data/states/atm12")
ap.add_argument("--layer", type=int, default=7, help="SFNO block output (1..8)")
ap.add_argument("--k", type=int, nargs="+", default=[0, 8, 32, 128])
ap.add_argument("--K", type=float, default=2.0, help="tau = max(K sigma_seed, 1% spatial SD)")
ap.add_argument("--dates", nargs="+", default=[f"{y}-{md}" for y in (1996, 2042, 2047) for md in ("01-16", "04-16", "07-15", "10-13")])
ap.add_argument("--score-years", type=int, nargs="+", default=[2047])
a = ap.parse_args()
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RES, exist_ok=True)
set_numerics()
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from fme.coupled.inference.inference import load_stepper  # noqa: E402
from fme.ace.registry.stochastic_sfno import isotropic_noise  # noqa: E402
from fme.core.models.conditional_sfno.sfnonet import Context  # noqa: E402

st = load_stepper(paths.CKPT["C1A0"])
atm = st.atmosphere.modules[0].module.to(DEV).eval()
atm.requires_grad_(False)
net = atm.conditional_model
net.checkpointing = 0
sa = st.atmosphere._step_obj
INN, OUTN = list(sa.in_names), list(sa.out_names)
MU = {k: float(v) for k, v in sa.normalizer.means.items()}
SD = {k: float(v) for k, v in sa.normalizer.stds.items()}
X = np.load(os.path.join(a.capture, "atm_x_00.npy"))
assert X.shape[0] == len(a.dates), f"{X.shape[0]} captured states vs {len(a.dates)} dates"
HEAD = ["TS", "T_7", "U_7", "PS", "surface_precipitation_rate", "FLUT", "FSUTOA"]
w = np.cos(np.deg2rad(np.linspace(-89.5, 89.5, 180)))[:, None] * np.ones((1, 360))
score_idx = [i for i, d in enumerate(a.dates) if int(d[:4]) in a.score_years]
fit_idx = [i for i in range(len(a.dates)) if i not in score_idx]

_patch = {}


def hook(mod, args, out):
    f = _patch.get("fn")
    return f(out) if f is not None else out


net.blocks[a.layer - 1].register_forward_hook(hook)


def noise(si, seed):
    torch.manual_seed(10_000 * si + seed)
    return isotropic_noise((1, atm.embed_dim), atm._lmax, atm._mmax, atm._inverse_sht, device=DEV)


@torch.no_grad()
def forward(si, seed, fn=None, record=None):
    x = torch.tensor(X[si:si + 1], device=DEV)
    pos = atm.pos_embed if atm.pos_embed is not None else None
    if record is not None:
        _patch["fn"] = lambda o: (record.append(o.detach()), o)[1]
    else:
        _patch["fn"] = fn
    y = net(x, Context(embedding_scalar=None, embedding_pos=pos, labels=None, noise=noise(si, seed)))
    _patch["fn"] = None
    return {n: (y[0, i] * SD[n] + MU[n]).double().cpu().numpy() for i, n in enumerate(OUTN)}


t0 = time.time()
# PCA on fit dates (seed 0): channels of the block output, samples = nodes x dates
lat_rec = []
for si in fit_idx:
    forward(si, 0, record=lat_rec)
Z = torch.cat([r.flatten(2).transpose(1, 2).reshape(-1, r.shape[1]) for r in lat_rec]).double()
mean = Z.mean(0)
cov = (Z - mean).T @ (Z - mean) / Z.shape[0]
evals, evecs = torch.linalg.eigh(cov)
U = evecs[:, torch.argsort(evals, descending=True)].float()
mean = mean.float()
explained = (evals.sort(descending=True).values.cumsum(0) / evals.sum()).cpu().numpy()
del Z, lat_rec


def pca_fn(k):
    Uk = U[:, :k]

    def fn(o):
        B, C = o.shape[:2]
        z = o.flatten(2).transpose(1, 2) - mean
        zr = (z @ Uk) @ Uk.T if k > 0 else torch.zeros_like(z)
        return (zr + mean).transpose(1, 2).reshape(o.shape)
    return fn


def truth_at(si):
    y, m, d = (int(v) for v in a.dates[si].split("-"))
    return truth_atm.fields(truth_ocean.stamp(y, m, d) + 0.25, OUTN)


# forecasts for every date
F = {}
for si in range(len(a.dates)):
    F[si] = {"model": [forward(si, s) for s in range(4)], "truth": truth_at(si)}
    for k in a.k:
        F[si][f"pca{k}"] = forward(si, 0, fn=pca_fn(k))
    F[si]["persist"] = {n: X[si, INN.index(n)] * SD[n] + MU[n] for n in OUTN if n in INN}
print(f"forecasts done {time.time() - t0:.0f}s", flush=True)

# controls fitted on fit dates
off = {n: float(np.mean([np.average(F[i]["model"][0][n] - F[i]["persist"][n], weights=w) for i in fit_idx]))
       for n in OUTN if n in INN}
aff = {}  # per-variable affine map of persistence: y = a + g * input, weighted LSQ on fit dates
for n in off:
    xs = np.concatenate([F[i]["persist"][n].ravel() for i in fit_idx])
    ys = np.concatenate([F[i]["model"][0][n].ravel() for i in fit_idx])
    ws = np.concatenate([w.ravel()] * len(fit_idx))
    A = np.stack([np.ones_like(xs), xs], 1) * np.sqrt(ws)[:, None]
    coef = np.linalg.lstsq(A, ys * np.sqrt(ws), rcond=None)[0]
    aff[n] = (float(coef[0]), float(coef[1]))


def expl(si, name, n):
    if name == "seed1":
        return F[si]["model"][1][n]
    if name.startswith("pca"):
        return F[si][name][n]
    if name == "offset":
        return F[si]["persist"][n] + off[n] if n in off else None
    if name == "affine":
        return aff[n][0] + aff[n][1] * F[si]["persist"][n] if n in aff else None
    raise KeyError(name)


EXPLANATIONS = ["seed1"] + [f"pca{k}" for k in a.k] + ["offset", "affine"]
res = {"capture": a.capture, "layer": a.layer, "K": a.K, "fit_dates": [a.dates[i] for i in fit_idx],
       "score_dates": [a.dates[i] for i in score_idx], "pca_explained": {k: float(explained[k - 1]) if k else 0.0 for k in a.k},
       "variables": {}}
for n in HEAD:
    acc = {e: {"ok": 0.0, "err": 0.0, "rep": 0.0, "cor": 0.0, "oth": 0.0} for e in EXPLANATIONS}
    tot_ok = tot_err = 0.0
    for si in score_idx:
        ym = F[si]["model"][0][n]
        tr = F[si]["truth"][n]
        sig = np.std(np.stack([F[si]["model"][s][n] for s in range(4)]), axis=0, ddof=1)
        tau = np.maximum(a.K * sig, 0.01 * np.sqrt(np.average((tr - np.average(tr, weights=w)) ** 2, weights=w)))
        err = np.abs(ym - tr) > tau
        tot_ok += np.sum(w * ~err)
        tot_err += np.sum(w * err)
        for e in EXPLANATIONS:
            ye = expl(si, e, n)
            if ye is None:
                continue
            agree = np.abs(ye - ym) <= tau
            cor = (~agree) & (np.abs(ye - tr) <= tau)
            acc[e]["ok"] += np.sum(w * (agree & ~err))
            acc[e]["err"] += np.sum(w * (agree & err))
            acc[e]["rep"] += np.sum(w * (agree & err))
            acc[e]["cor"] += np.sum(w * (cor & err))
            acc[e]["oth"] += np.sum(w * (~agree & ~cor & err))
    v = {"error_area_frac": float(tot_err / (tot_ok + tot_err))}
    for e in EXPLANATIONS:
        if e in ("offset", "affine") and n not in off:
            continue  # output-level controls need the variable among the inputs (prognostic)
        v[e] = {"A_ok": float(acc[e]["ok"] / tot_ok), "A_err": float(acc[e]["err"] / tot_err) if tot_err else None,
                "err_reproduces": float(acc[e]["rep"] / tot_err) if tot_err else None,
                "err_corrects": float(acc[e]["cor"] / tot_err) if tot_err else None,
                "err_other": float(acc[e]["oth"] / tot_err) if tot_err else None}
    res["variables"][n] = v
res["seconds"] = time.time() - t0
tag = f"L{a.layer}_K{a.K:g}"
json.dump(res, open(os.path.join(RES, f"strata_{tag}.json"), "w"), indent=1)
lines = [f"| variable | error area | " + " | ".join(f"{e} A_ok/A_err" for e in EXPLANATIONS) + " |",
         "|---|---|" + "---|" * len(EXPLANATIONS)]
for n, v in res["variables"].items():
    cells = [f"{v[e]['A_ok']:.2f}/{v[e]['A_err']:.2f}" if e in v and v[e]["A_err"] is not None else "–" for e in EXPLANATIONS]
    lines.append(f"| {n} | {v['error_area_frac']:.2f} | " + " | ".join(cells) + " |")
open(os.path.join(RES, f"strata_{tag}.md"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
