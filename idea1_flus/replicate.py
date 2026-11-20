"""Idea 1 step 4 (secondary, PREREG 1.4): the FLUS shortcut across ocean checkpoints.

Truth-forced 1995 inputs (19 states, every 4th 5-day window from 1995-01-01; built from E3SM output with each
checkpoint's own normalizer), bare ocean network, fp32. Uniform open-ocean and PREREG-lattice 10x10 box shifts,
dQ in {-100,-25,25,100} of warming flux, area-mean metric. Slopes in K per 100 W/m2 per model step (5 days).
'Has the shortcut' = FLUS slope with the thermometer sign over > 75% of open-ocean area (node slopes, uniform).
E17 (1-day ocean, daily-window training) is excluded: these are 5-day-window inputs.
usage: python replicate.py NAME=PATH [NAME=PATH ...]  -> results/replicate_<first NAME>.json (one file per call)
"""
import json
import os
import sys
import time

import numpy as np
import torch

from hc import graph, truth_ocean

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
W = graph.Grid.w
WARM = {"FLUS": -1, "FLDS": 1, "FSNS": 1, "LHFLX": -1, "SHFLX": -1}
DQ = np.array([-100.0, -25.0, 25.0, 100.0])
LAT_C, LON_C = [-45, -25, -5, 15, 35], [30, 110, 190, 270]
STAMPS = [truth_ocean.stamp(1995, 1, 1) + 20 * i for i in range(19)]


def boxes(openw):
    lat, lon = graph.Grid.lat, graph.Grid.lon
    out = {}
    for la in LAT_C:
        for lo in LON_C:
            m = (np.abs(lat - la)[:, None] <= 5) & (np.abs(((lon - lo + 180) % 360) - 180)[None, :] <= 5)
            if (W * (m & openw)).sum() >= 0.95 * (W * m).sum():
                out[f"box{la:+d}_{lo:03d}"] = m & openw
    return out


phys = {t: (truth_ocean.fields_at(t), truth_ocean.fields_at(t + 5)) for t in STAMPS}
results = {}
for arg in sys.argv[1:]:
    name, path = arg.split("=", 1)
    t0 = time.time()
    ocn, ins, outs, mu, sd, step = graph.load_ocean_any(path)
    isst = outs.index("sst")
    rows = []
    for t in STAMPS:
        a, b = phys[t]
        x = np.zeros((len(ins), 180, 360), "f4")
        land = ~np.isfinite(a["sst"])
        for i, n in enumerate(ins):
            v = b[n] if n in truth_ocean.FORCING else a[n]
            z = np.nan_to_num((v - mu[n]) / sd[n], nan=0.0)
            if n != "LANDFRAC":
                z[land] = 0.0
            x[i] = z
        openw = (~land) & (np.nan_to_num(a["ocean_sea_ice_fraction"], nan=1.0) < 0.01)
        areas = {"uniform": openw} | boxes(openw)
        x0 = torch.tensor(x[None], device=graph.DEV)
        with torch.no_grad():
            y0 = ocn(x0)[:, isst] * sd["sst"]
        rec = {"stamp": t}
        for c, sg in WARM.items():
            for an, am in areas.items():
                amt = torch.tensor(am.astype("f4"), device=graph.DEV)
                xs = x0.repeat(4, 1, 1, 1)
                for j, dq in enumerate(DQ):
                    xs[j, ins.index(c)] += float(sg * dq / sd[c]) * amt
                with torch.no_grad():
                    r = ((ocn(xs)[:, isst] * sd["sst"] - y0).cpu().numpy())
                wa = W * am
                vals = (r * wa[None]).sum((1, 2)) / wa.sum()
                rec[f"{c}|{an}"] = 100 * float(DQ @ vals / (DQ @ DQ))
                if an == "uniform" and c == "FLUS":
                    node = 100 * np.tensordot(DQ, r, axes=(0, 0)) / (DQ @ DQ)
                    rec["FLUS_thermometer_area"] = float((W * (node < 0))[openw].sum() / W[openw].sum())
        rows.append(rec)
    summ = {}
    for c in WARM:
        u = np.array([r[f"{c}|uniform"] for r in rows])
        bx = np.array([v for r in rows for k, v in r.items() if k.startswith(f"{c}|box")])
        summ[c] = {"uniform_mean": float(u.mean()), "uniform_min": float(u.min()), "uniform_max": float(u.max()),
                   "box_mean": float(bx.mean()), "box_p05": float(np.percentile(bx, 5)), "box_p95": float(np.percentile(bx, 95))}
    ta = np.array([r["FLUS_thermometer_area"] for r in rows])
    summ["FLUS_thermometer_area_mean"] = float(ta.mean())
    summ["has_shortcut"] = bool(ta.mean() > 0.75)
    summ["ratios_to_FLDS_uniform"] = {c: summ[c]["uniform_mean"] / summ["FLDS"]["uniform_mean"] for c in WARM}
    results[name] = {"path": path, "timestep": str(step), "summary": summ, "rows": rows, "seconds": time.time() - t0}
    print(name, json.dumps({c: round(summ[c]["uniform_mean"], 3) for c in WARM}), "box FLUS", round(summ["FLUS"]["box_mean"], 3),
          "thermo area", round(summ["FLUS_thermometer_area_mean"], 3), flush=True)
    del ocn
    torch.cuda.empty_cache()
first = sys.argv[1].split("=", 1)[0]
json.dump(results, open(os.path.join(RES, f"replicate_{first}.json"), "w"), indent=1)
