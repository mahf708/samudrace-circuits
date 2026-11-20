"""15-minute walkthrough: from a truth-started run to two scored numbers.

Given a capture directory written by `python -m hc.evaluate RUN --start ... --steps N --capture`
(then `python -m hc.meta RUN`), this computes on the captured held-out states:
  1. C1A0's FLUS input-response slope (uniform open-ocean shift, K per 100 W/m2 of warming flux),
     with hc.graph (exact Samudra.forward);
  2. the share of that response carried by the full-resolution skip (skip.0), with exege's `samudra`
     Patchable adapter (EdgePatch semantics);
and writes RUN/walkthrough.json. Expected on 2045 states: slope ~ -1.8, skip.0 share ~ 0.85.
usage: python examples/walkthrough.py RUN [n_states=4]
"""
import json
import os
import sys
import time

import numpy as np
import torch

from hc import graph, paths, set_numerics

RUN = os.path.abspath(sys.argv[1])
N = int(sys.argv[2]) if len(sys.argv) > 2 else 4
t0 = time.time()
set_numerics()  # fp32: perturbation work must not use TF32

# --- 1. FLUS slope with the exact graph -------------------------------------------------------------
ocn, ins, outs, mu, sd = graph.load_ocean()
fixed = graph.fixed_ocean_mask(RUN)  # official sst output finite (NOT sea_surface_fraction > 0.5)
W = graph.Grid.w
ks = list(range(min(N, len(graph.state_files(RUN)))))
DQ = np.array([-100.0, -25.0, 25.0, 100.0])
slopes = []
for k in ks:
    x = graph.load_state(k, RUN)
    ice = x[ins.index("ocean_sea_ice_fraction")] * sd["ocean_sea_ice_fraction"] + mu["ocean_sea_ice_fraction"]
    ow = fixed & (ice < 0.01)  # open ocean
    xs = [x]
    for dq in DQ:  # +dq of warming flux = FLUS lowered by dq
        z = x.copy()
        z[ins.index("FLUS")] -= (dq * ow / sd["FLUS"]).astype("f4")
        xs.append(z)
    with torch.no_grad():
        y = ocn(torch.tensor(np.stack(xs), device=graph.DEV))[:, outs.index("sst")].float().cpu().numpy() * sd["sst"]
    r = [float(((yy - y[0]) * W * ow).sum() / (W * ow).sum()) for yy in y[1:]]
    slopes.append(100 * float(DQ @ r / (DQ @ DQ)))
del ocn
torch.cuda.empty_cache()

# --- 2. skip.0 share with the exege adapter ---------------------------------------------------------
from exege.circuits.contract import EdgePatch, open_patchable  # noqa: E402

meta = os.path.join(RUN, "meta.json")
if not os.path.exists(meta):
    json.dump({"in_names": ins, "out_names": outs, "means": mu, "stds": sd}, open(meta, "w"))
ad = open_patchable(RUN, "samudra", checkpoint=paths.CKPT["C1A0"], device=str(graph.DEV))
g = ad.graph()
shares = []
for k in ks:
    clean = ad.states([k])
    x = graph.load_state(k, RUN)
    ice = x[ins.index("ocean_sea_ice_fraction")] * sd["ocean_sea_ice_fraction"] + mu["ocean_sea_ice_fraction"]
    ow = fixed & (ice < 0.01)
    pert = ad.perturb(clean, {"FLUS": (25.0 * ow).ravel()})  # FLUS +25 W/m2 over open ocean
    base = ad.step(clean)["sst"][0]
    am = lambda y: float(((y - base).reshape(180, 360) * W * ow).sum() / (W * ow).sum())  # noqa: E731
    full = am(ad.step(pert)["sst"][0])
    m = g.mask(0.0)  # everything reads the clean source ...
    for s, r in [("input", "l0.skip_module"), ("input", "l0.convblock"), ("l0.skip_module", "out"), ("l0.convblock", "out")]:
        m[g.sender(s), g.receiver(r)] = 1.0  # ... except layer 0 and the skip.0 edges into the output
    shares.append(am(ad.step(pert, EdgePatch(clean, m))["sst"][0]) / full)

out = {"run": RUN, "states": ks, "flus_slope_K_per_100Wm2_warming": slopes, "flus_slope_mean": float(np.mean(slopes)),
       "skip0_share": shares, "skip0_share_mean": float(np.mean(shares)), "seconds": time.time() - t0}
json.dump(out, open(os.path.join(RUN, "walkthrough.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k.endswith("mean") or k == "seconds"}, indent=1))
