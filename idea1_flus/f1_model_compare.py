"""F1 (c): model input-response slope maps vs the truth conditional slope maps. PREREG 1.1 (c) and verdict.

Model: bare C1A0 ocean (fp32), uniform open-ocean shifts dQ in {-100,-25,25,100} of warming flux for each of the
five fluxes, per-node slope through the origin (K per 100 W/m2), averaged over states (stride 4) of a capture dir.
Truth: f1_maps.npz '<block>|score|<flux>' coefficients (per W/m2 of the flux as signed in training) -> converted
to K per 100 W/m2 of warming flux. Pattern correlation is area-weighted over cells valid in both.
Verdict rules exactly as PREREG 1.1.
usage: python f1_model_compare.py STATES_DIR TRUTH_BLOCK TAG  -> results/f1_compare_TAG.json, f1_model_maps_TAG.npz
"""
import json
import os
import sys

import numpy as np
import torch

from hc import graph

S, BLK, TAG = os.path.abspath(sys.argv[1]), sys.argv[2], sys.argv[3]
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
ocn, ins, outs, means, stds = graph.load_ocean()
isst = outs.index("sst")
W = graph.Grid.w
FIX = graph.fixed_ocean_mask(S)
DQ = np.array([-100.0, -25.0, 25.0, 100.0])
WARM = {"FLUS": -1, "FLDS": 1, "FSNS": 1, "LHFLX": -1, "SHFLX": -1}
ks = list(range(0, len(graph.state_files(S)), 4))
acc = {c: np.zeros((180, 360)) for c in WARM}
cnt = np.zeros((180, 360))
glob = {c: [] for c in WARM}
for k in ks:
    x = graph.load_state(k, S)
    ice = x[ins.index("ocean_sea_ice_fraction")] * stds["ocean_sea_ice_fraction"] + means["ocean_sea_ice_fraction"]
    ow = FIX & (ice < 0.01)
    xs = [x]
    for c, sg in WARM.items():
        for dq in DQ:
            z = x.copy()
            z[ins.index(c)] += (sg * dq * ow / stds[c]).astype("f4")
            xs.append(z)
    ys = []
    with torch.no_grad():
        for i in range(0, len(xs), 7):
            ys.append(ocn(torch.tensor(np.stack(xs[i:i + 7]), device=graph.DEV))[:, isst].float().cpu().numpy() * stds["sst"])
    ys = np.concatenate(ys)
    for ci, c in enumerate(WARM):
        R = ys[1 + 4 * ci:5 + 4 * ci] - ys[0]
        node = 100 * np.tensordot(DQ, R, axes=(0, 0)) / (DQ @ DQ)
        acc[c] += np.where(ow, node, 0)
        glob[c].append(float((node * W)[ow].sum() / W[ow].sum()))
    cnt += ow
model = {c: np.where(cnt >= 0.9 * len(ks), acc[c] / np.maximum(cnt, 1), np.nan) for c in WARM}
tm = np.load(os.path.join(RES, "f1_maps.npz"))
out = {"states_dir": S, "truth_block": BLK, "n_states": len(ks), "per_flux": {}}
for c in WARM:
    truth = tm[f"{BLK}|score|{c}"].astype("f8") * 100 * WARM[c]
    v = np.isfinite(truth) & np.isfinite(model[c])
    w = W[v]
    a, b = model[c][v], truth[v]
    am, bm = (w * a).sum() / w.sum(), (w * b).sum() / w.sum()
    corr = float((w * (a - am) * (b - bm)).sum() / np.sqrt((w * (a - am) ** 2).sum() * (w * (b - bm) ** 2).sum()))
    nc = v & (graph.Grid.lat[:, None] > -70)  # sensitivity: exclude ice-shelf cavities (not open ocean)
    w2, a2, b2 = W[nc], model[c][nc], truth[nc]
    a2m, b2m = (w2 * a2).sum() / w2.sum(), (w2 * b2).sum() / w2.sum()
    corr_nc = float((w2 * (a2 - a2m) * (b2 - b2m)).sum() / np.sqrt((w2 * (a2 - a2m) ** 2).sum() * (w2 * (b2 - b2m) ** 2).sum()))
    out["per_flux"][c] = {"pattern_corr_excl_south_of_70S": corr_nc, "ratio_excl_south_of_70S": float(a2m / b2m),"model_global_mean": float(am), "truth_global_mean": float(bm), "ratio_model_over_truth": float(am / bm),
                          "pattern_corr": corr, "model_state_global_slopes": glob[c],
                          "model_frac_area_thermometer_sign": float((w * (a < 0)).sum() / w.sum()) if c == "FLUS" else None}
F = out["per_flux"]["FLUS"]
truth_frac = float(json.load(open(os.path.join(RES, "f1.json")))[f"regression|{BLK}|score"]["FLUS"]["frac_area_negative(thermometer_sign_for_FLUS)"])
crit = {"truth_thermometer_area_gt_0.75": truth_frac > 0.75, "ratio_in_[0.33,3]": 0.33 <= F["ratio_model_over_truth"] <= 3,
        "pattern_corr_gt_0.3": F["pattern_corr"] > 0.3}
if all(crit.values()):
    verdict = "leak supported"
elif truth_frac < 0.5:
    verdict = "leak refuted"
else:
    verdict = "something else to explain"
out["verdict"] = {"criteria": crit, "truth_thermometer_area": truth_frac, "verdict": verdict}
json.dump(out, open(os.path.join(RES, f"f1_compare_{TAG}.json"), "w"), indent=1)
np.savez_compressed(os.path.join(RES, f"f1_model_maps_{TAG}.npz"), **{c: model[c].astype("f4") for c in WARM})
print(json.dumps(out["verdict"], indent=1), json.dumps({c: {k: v for k, v in d.items() if k != "model_state_global_slopes"} for c, d in out["per_flux"].items()}, indent=1))
