"""Idea 1 step 3 (F3, network arm): can FLUS be removed without breaking FLDS/FSNS?  PREREG 1.3 'Ablation'.

Place P (argument): 'l0' = layer-0 output (280 ch; feeds skip.0 and the deep path), or 'skipmod' / 'convblk'
(that edge's 280-ch contribution before the sum).
(ii) channel set: single-channel screen on FIT states (FLUS uniform, dQ=+25: response when only channel c of P is
     corrupted), top-32 by area-mean share, then greedy (on 6 fit states) to k in {10, 32}. Ablation = mean-ablate
     the set (replace P's channels by their per-node mean over fit states) in clean AND perturbed runs.
     Controls: 20 random size-matched sets at the same place.
(i)  input-direction ablation: FLUS input := 5.67e-8 * sst_in^4 (de-leak value) at ocean cells; FLUS perturbations
     are then overwritten (FLUS slope = 0 by construction); other fluxes re-scored; also the clean-output change.
Scored on SCORE states (uniform + PREREG boxes; random controls on every 3rd score state), all 5 fluxes, dQ in {-100,-25,25,100}, area-mean metric.
usage: python ablate.py PLACE FIT_DIR[,FIT_DIR] SCORE_DIR[,SCORE_DIR] TAG
"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hc import graph  # noqa: E402

PLACE = sys.argv[1]
FIT = [os.path.abspath(p) for p in sys.argv[2].split(",")]
SCORE = [os.path.abspath(p) for p in sys.argv[3].split(",")]
TAG = sys.argv[4]
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
ocn, ins, outs, means, stds = graph.load_ocean()
G = graph.Graph(ocn)
L0 = ocn.layers[0]
DEV, W = graph.DEV, graph.Grid.w
isst = outs.index("sst")
FLUX = {"FLUS": -1, "FLDS": 1, "FSNS": 1, "LHFLX": -1, "SHFLX": -1}
DQ = [-100.0, -25.0, 25.0, 100.0]
LAT_C, LON_C = [-45, -25, -5, 15, 35], [30, 110, 190, 270]
rng = np.random.default_rng(0)


def states(dirs, stride=4):
    out = []
    for d in dirs:
        fixed = graph.fixed_ocean_mask(d)
        for k in range(0, len(graph.state_files(d)), stride):
            x = graph.load_state(k, d)
            ice = x[ins.index("ocean_sea_ice_fraction")] * stds["ocean_sea_ice_fraction"] + means["ocean_sea_ice_fraction"]
            out.append((d, k, x, fixed & (ice < 0.01), fixed))
    return out


def boxes(openw):
    lat, lon = graph.Grid.lat, graph.Grid.lon
    res = {}
    for la in LAT_C:
        for lo in LON_C:
            m = (np.abs(lat - la)[:, None] <= 5) & (np.abs(((lon - lo + 180) % 360) - 180)[None, :] <= 5)
            if (W * (m & openw)).sum() >= 0.95 * (W * m).sum():
                res[f"box{la:+d}_{lo:03d}"] = m & openw
    return res


@torch.no_grad()
def parts(x):
    sk = L0.skip_module(x)
    l0 = L0(x)
    return sk, l0 - sk


@torch.no_grad()
def run_with(x, mod=None):
    """Forward with P's channels modified by mod(P_tensor)->P_tensor (or None)."""
    sk, cb = parts(x)
    if mod is not None:
        if PLACE == "l0":
            l0 = mod(sk + cb)
        elif PLACE == "skipmod":
            l0 = mod(sk) + cb
        else:
            l0 = sk + mod(cb)
    else:
        l0 = sk + cb
    return G.run(x, src={"D0": l0, "S0": l0})[:, isst] * stds["sst"]


def place_value(x):
    sk, cb = parts(x)
    return {"l0": sk + cb, "skipmod": sk, "convblk": cb}[PLACE]


def perturb(x0, name, dq, area):
    x = x0.clone()
    x[:, ins.index(name)] += FLUX[name] * dq * torch.tensor(area.astype("f4"), device=DEV) / stds[name]
    return x


t0 = time.time()
fit = states(FIT)
score = states(SCORE)
print(f"fit {len(fit)} score {len(score)} states", flush=True)

# mean of P over fit states (per node, per channel)
Pmean = None
for _, _, x, _, _ in fit:
    v = place_value(torch.tensor(x[None], device=DEV))
    Pmean = v if Pmean is None else Pmean + v
Pmean = Pmean / len(fit)

# (ii-a) single-channel screen on fit states: FLUS uniform +25
C = Pmean.shape[1]
share = np.zeros((len(fit), C))
for si, (_, _, x, openw, _) in enumerate(fit):
    x0 = torch.tensor(x[None], device=DEV)
    x1 = perturb(x0, "FLUS", 25.0, openw)
    p0, p1 = place_value(x0), place_value(x1)
    y0 = run_with(x0)
    full = float(((run_with(x1) - y0)[0].cpu().numpy() * W * openw).sum() / (W * openw).sum())
    for c0 in range(0, C, 8):
        cs = list(range(c0, min(C, c0 + 8)))
        xb = x0.expand(len(cs), -1, -1, -1)
        Pc = p0.expand(len(cs), -1, -1, -1).clone()
        for i, c in enumerate(cs):
            Pc[i, c] = p1[0, c]
        y = run_with(xb, mod=lambda _p: Pc)
        r = ((y - y0).cpu().numpy() * (W * openw)[None]).sum((1, 2)) / (W * openw).sum()
        share[si, cs] = r / full
    if si % 5 == 0:
        print(f"screen state {si} {time.time() - t0:.0f}s", flush=True)
mshare = share.mean(0)
top32 = list(np.argsort(-mshare)[:32])

# (ii-b) greedy on 6 fit states
gst = [fit[i] for i in np.linspace(0, len(fit) - 1, 6).astype(int)]
gcache = []
for _, _, x, openw, _ in gst:
    x0 = torch.tensor(x[None], device=DEV)
    x1 = perturb(x0, "FLUS", 25.0, openw)
    y0 = run_with(x0)
    full = float(((run_with(x1) - y0)[0].cpu().numpy() * W * openw).sum() / (W * openw).sum())
    gcache.append((x0, place_value(x0), place_value(x1), y0, full, openw))


def set_share(S):
    v = []
    for x0, p0, p1, y0, full, openw in gcache:
        Pc = p0.clone()
        Pc[:, S] = p1[:, S]
        y = run_with(x0, mod=lambda _p: Pc)
        v.append(float(((y - y0)[0].cpu().numpy() * W * openw).sum() / (W * openw).sum()) / full)
    return float(np.mean(v))


chosen, greedy_curve = [], []
for _ in range(32):
    best = max((c for c in top32 if c not in chosen), key=lambda c: set_share(chosen + [c]))
    chosen.append(int(best))
    greedy_curve.append(set_share(chosen))
print("greedy curve", np.round(greedy_curve, 3).tolist(), f"{time.time() - t0:.0f}s", flush=True)
sets = {"greedy10": chosen[:10], "greedy32": chosen[:32]}
for k in (10, 32):
    for j in range(20):
        sets[f"random{k}_{j}"] = sorted(rng.choice(C, k, replace=False).tolist())


def ablate_mod(S):
    S = torch.tensor(S, device=DEV)

    def mod(p):
        p = p.clone()
        p[:, S] = Pmean[:, S].expand(p.shape[0], -1, -1, -1)
        return p
    return mod


def deleak_input(x):
    x = x.clone()
    iF, iS = ins.index("FLUS"), ins.index("sst")
    sst = x[:, iS] * stds["sst"] + means["sst"]
    ocean = x[:, iS] != 0
    x[:, iF] = torch.where(ocean, (5.67e-8 * sst ** 4 - means["FLUS"]) / stds["FLUS"], x[:, iF])
    return x


# score
arms = {"none": None, **{k: ablate_mod(v) for k, v in sets.items()}}
rows = []
for si, (d, k, x, openw, fixed) in enumerate(score):
    x0 = torch.tensor(x[None], device=DEV)
    areas = {"uniform": openw} | boxes(openw)
    rec = {"dir": d, "k": k, "slopes": {}}
    use = [(a, m) for a, m in arms.items() if not a.startswith("random") or si % 3 == 0]
    for arm, mod in use + [("deleak_input", "INPUT")]:
        xin = deleak_input(x0) if mod == "INPUT" else x0
        y0 = run_with(xin, None if mod == "INPUT" else mod)
        if arm == "deleak_input":
            dy = (y0 - run_with(x0))[0].cpu().numpy()
            rec["deleak_input_clean_dSST"] = {"mean_open": float((dy * W * openw).sum() / (W * openw).sum()),
                                              "rms_open": float(np.sqrt((dy ** 2 * W * openw).sum() / (W * openw).sum()))}
        sl = {}
        for name in FLUX:
            for aname, am in areas.items():
                xs = torch.cat([perturb(x0, name, dq, am) for dq in DQ])
                if mod == "INPUT":
                    xs = deleak_input(xs)
                y = run_with(xs, None if mod == "INPUT" else mod)
                r = ((y - y0).cpu().numpy() * (W * am)[None]).sum((1, 2)) / (W * am).sum()
                sl[f"{name}|{aname}"] = 100 * float(np.dot(DQ, r) / np.dot(DQ, DQ))
        rec["slopes"][arm] = sl
    rows.append(rec)
    print(f"score state {si} {time.time() - t0:.0f}s", flush=True)

json.dump({"place": PLACE, "fit": FIT, "score": SCORE, "single_channel_share_mean": mshare.tolist(),
           "top32": [int(c) for c in top32], "greedy": chosen, "greedy_curve": greedy_curve, "sets": sets,
           "rows": rows, "seconds": time.time() - t0},
          open(os.path.join(RES, f"ablate_{PLACE}_{TAG}.json"), "w"))
print("done", time.time() - t0)
