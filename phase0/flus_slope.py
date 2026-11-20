"""Reproduce the laptop number: C1A0 uniform open-ocean FLUS slope, ~ -2.04 K per 100 W/m2 of warming flux
(73 states from 1985, range -2.08..-2.01). Core of cross-flux-witness-circuit/witness_states.py, unchanged
definitions: bare Samudra network (no corrector), dQ in {-100,-25,25,100} W/m2 of *warming* flux applied over
open water (sea_surface_fraction > 0.99 and ice < 0.01 in the input), FLUS shifted by -dQ, slope through the
origin of the area-weighted global open-ocean mean dSST, in K per 100 W/m2.
Also reported with the fixed ocean mask (official sst output finite) for the same perturbation.
usage: python phase0/flus_slope.py STATES_DIR [stride] -> phase0/flus_slope.json
"""
import json
import os
import sys
import time

import numpy as np
import torch

from hc.graph import COMPS, DEV, Grid, load_ocean, load_state, masks, state_files, fixed_ocean_mask

S = os.path.abspath(sys.argv[1])
stride = int(sys.argv[2]) if len(sys.argv) > 2 else 1
ocn, ins, outs, means, stds = load_ocean()
isst = outs.index("sst")
DQ = np.array([-100.0, -25.0, 25.0, 100.0])
ks = list(range(0, len(state_files(S)), stride))
fixed = fixed_ocean_mask(S)
w = Grid.w


@torch.no_grad()
def fwd(xs):
    xb = torch.tensor(np.stack(xs), device=DEV)
    return ocn(xb)[:, isst].float().cpu().numpy() * stds["sst"]


rows, t0 = [], time.time()
for k in ks:
    x0 = load_state(k, S)
    M = masks(x0, ins, means, stds)
    ow = M["global_open_ocean"]
    ice = x0[ins.index("ocean_sea_ice_fraction")] * stds["ocean_sea_ice_fraction"] + means["ocean_sea_ice_fraction"]
    ow_fixed = fixed & (ice < 0.01)
    xs, tags = [x0], [None]
    for c, sgn in COMPS.items():
        for dq in DQ:
            x = x0.copy()
            x[ins.index(c)] += (sgn * dq * ow / stds[c]).astype("f4")
            xs.append(x); tags.append((c, dq))
    ys = np.concatenate([fwd(xs[i:i + 7]) for i in range(0, len(xs), 7)])
    r = {"k": k, "open_frac_orig": float((w * ow).sum() / w.sum()), "open_frac_fixed": float((w * ow_fixed).sum() / w.sum())}
    for c in COMPS:
        R = np.stack([ys[tags.index((c, dq))] - ys[0] for dq in DQ])
        for mname, m in (("orig", ow), ("fixed", ow & ow_fixed)):
            vals = np.array([float((rr * w)[m].sum() / w[m].sum()) for rr in R])
            r[f"{c}_{mname}"] = 100 * float(DQ @ vals / (DQ @ DQ))
        node = 100 * np.tensordot(DQ, R, axes=(0, 0)) / (DQ @ DQ)
        if c == "FLUS":
            r["FLUS_wrong_sign_area_frac"] = float((w * (node < 0))[ow].sum() / w[ow].sum())
    rows.append(r)
    print(k, f"{time.time() - t0:.0f}s", " ".join(f"{c}={r[c + '_orig']:+.3f}" for c in COMPS), flush=True)

summ = {}
for key in rows[0]:
    if key == "k":
        continue
    v = np.array([r[key] for r in rows])
    summ[key] = {"mean": float(v.mean()), "min": float(v.min()), "max": float(v.max())}
out = {"states_dir": S, "n_states": len(rows), "stride": stride, "dQ": DQ.tolist(), "device": str(DEV),
       "laptop_reference": {"FLUS": -2.04, "range": [-2.08, -2.01], "wrong_sign_area": 0.998},
       "summary": summ, "rows": rows}
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "flus_slope.json"), "w"), indent=1)
print(json.dumps(summ, indent=1))
