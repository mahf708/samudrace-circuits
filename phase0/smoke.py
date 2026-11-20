"""Phase 0 smoke test on one GPU.

Expects three capture dirs made by hc.capture (2 coupled steps from 1985-01-01, seed 0):
  A, B : control, same seed  -> same-seed bitwise reproducibility of fme steps
  D    : de-leaked (CombinedFluxConfig FLUS := 5.67e-8 * sst_start^4) -> switch works at load time
Checks:
  1. common.Graph (hc.graph.Graph) vs Samudra.forward, max |diff|
  2. SamudraReplay.forward_batch vs Samudra.forward, max |diff|, and vs captured fme output
  3. A vs B: every x/y tensor and every prediction-file variable bitwise equal
  4. D: ocean FLUS input at step 0 and 1 vs 5.67e-8 * sst^4 and vs control
usage: python phase0/smoke.py A B D  -> phase0/smoke.json
"""
import json
import os
import sys

import numpy as np
import torch
import xarray as xr

from hc import graph
from hc.samudra_adapter import SamudraReplay

A, B, D = (os.path.abspath(p) for p in sys.argv[1:4])
res = {"device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}

# 1. Graph
ocn, ins, outs, means, stds = graph.load_ocean()
g = graph.Graph(ocn)
x = torch.tensor(graph.load_state(0, A)[None], device=graph.DEV)
with torch.no_grad():
    y_ref = ocn(x.clone())
y = g.run(x.clone())
res["graph_vs_module_maxabs"] = float((y - y_ref).abs().max())
res["graph_vs_module_bitwise"] = bool(torch.equal(y, y_ref))
y_cap = torch.tensor(np.load(os.path.join(A, "y_000.npy")), device=graph.DEV)
res["module_vs_captured_fme_output_maxabs"] = float((y_ref[0] - y_cap).abs().max())
del ocn, g
torch.cuda.empty_cache()

# 2. SamudraReplay
ad = SamudraReplay(A)
xa = ad.load_x(0)
with torch.no_grad():
    ym = ad.net(xa[None].clone())
yr = ad.forward_batch(xa[None], [{}])
res["replay_vs_module_maxabs"] = float((yr - ym).abs().max())
res["replay_vs_module_bitwise"] = bool(torch.equal(yr, ym))
res["replay_vs_captured_fme_output_maxabs"] = float((yr[0] - y_cap).abs().max())
xb = xa[None].expand(4, -1, -1, -1)
res["replay_batch4_vs_single_maxabs"] = float((ad.forward_batch(xb, [{}] * 4)[2] - yr[0]).abs().max())

# 3. same-seed reproducibility
bit = {}
for f in sorted(os.listdir(A)):
    if f.endswith(".npy"):
        bit[f] = bool(np.array_equal(np.load(os.path.join(A, f)), np.load(os.path.join(B, f)), equal_nan=True))
for realm in ("ocean", "atmosphere"):
    pa = os.path.join(A, "fme", realm, "autoregressive_predictions.nc")
    if os.path.exists(pa):
        da, db = xr.open_dataset(pa), xr.open_dataset(os.path.join(B, "fme", realm, "autoregressive_predictions.nc"))
        bit[f"{realm}_predictions_all_vars"] = all(
            np.array_equal(da[v].values, db[v].values, equal_nan=True) for v in da.data_vars)
res["same_seed_bitwise"] = bit
res["same_seed_all_bitwise"] = all(bit.values())

# 4. de-leak switch
iF, iS = ins.index("FLUS"), ins.index("sst")
dl = {}
for k in (0, 1):
    xc = np.load(os.path.join(A, f"x_{k:03d}.npy"))
    xd = np.load(os.path.join(D, f"x_{k:03d}.npy"))
    sst = xc[iS] * stds["sst"] + means["sst"]  # start-of-step sst the ocean sees (same in both at k=0)
    sst_d = xd[iS] * stds["sst"] + means["sst"]
    fl_c = xc[iF] * stds["FLUS"] + means["FLUS"]
    fl_d = xd[iF] * stds["FLUS"] + means["FLUS"]
    ocean = xc[iS] != 0
    emis = 5.67e-8 * sst_d ** 4
    # FLUS is also scaled by the open-water fraction (open_water_flux_scaling) after combine
    r = fl_d[ocean] / np.maximum(emis[ocean], 1e-6)
    dl[f"step{k}"] = {
        "control_FLUS_mean_ocean": float(fl_c[ocean].mean()),
        "deleak_FLUS_mean_ocean": float(fl_d[ocean].mean()),
        "deleak_over_emission_median": float(np.median(r)),
        "deleak_over_emission_p05_p95": [float(np.percentile(r, 5)), float(np.percentile(r, 95))],
        "control_minus_deleak_rms_Wm2": float(np.sqrt(np.mean((fl_c - fl_d)[ocean] ** 2))),
        "sst_input_identical": bool(np.array_equal(xc[iS], xd[iS])),
        "other_channels_identical": bool(all(np.array_equal(xc[i], xd[i]) for i in range(len(ins)) if i != iF)),
    }
res["deleak"] = dl
json.dump(res, open(os.path.join(os.path.dirname(__file__), "smoke.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
