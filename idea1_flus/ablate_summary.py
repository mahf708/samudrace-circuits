"""Summarise ablate_<place>_<tag>.json against PREREG 1.3's success rule.
Success = FLUS slope flips to the lawful sign or |FLUS| < 25% of unablated, AND FLDS and FSNS within +-25% of
unablated, AND no random size-matched set achieves the same. Evaluated separately for uniform and box shifts
(box = mean over boxes and states). Ratios are per state (arm / none), then averaged.
usage: python ablate_summary.py ablate_l0_v1 -> results/<name>_summary.json
"""
import json
import os
import sys

import numpy as np

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
name = sys.argv[1]
d = json.load(open(os.path.join(RES, f"{name}.json")))
FL = ["FLUS", "FLDS", "FSNS", "LHFLX", "SHFLX"]


def ratios(arm, kind):
    out = {f: [] for f in FL}
    for r in d["rows"]:
        if arm not in r["slopes"]:
            continue
        s, s0 = r["slopes"][arm], r["slopes"]["none"]
        for f in FL:
            keys = [k for k in s if k.startswith(f + "|") and ((k == f + "|uniform") == (kind == "uniform"))]
            for k in keys:
                out[f].append(s[k] / s0[k])
    return {f: float(np.mean(v)) if v else None for f, v in out.items()}


def success(rt):
    flus = rt["FLUS"] is not None and (rt["FLUS"] < 0 or abs(rt["FLUS"]) < 0.25)
    keep = all(rt[f] is not None and abs(rt[f] - 1) <= 0.25 for f in ("FLDS", "FSNS"))
    return bool(flus and keep)


summ = {"place": d["place"], "greedy": d["greedy"][:32], "greedy_curve_fit_share": d["greedy_curve"],
        "single_channel_top10": sorted(range(len(d["single_channel_share_mean"])),
                                       key=lambda c: -d["single_channel_share_mean"][c])[:10]}
arms = sorted({a for r in d["rows"] for a in r["slopes"]})
for kind in ("uniform", "box"):
    res = {}
    for a in arms:
        if a == "none":
            continue
        rt = ratios(a, kind)
        res[a] = {"ratio_to_unablated": rt, "success": success(rt)}
    rnd = {k: [res[a]["ratio_to_unablated"]["FLUS"] for a in res if a.startswith(f"random{k}_")] for k in (10, 32)}
    summ[kind] = {
        "greedy10": res.get("greedy10"), "greedy32": res.get("greedy32"), "deleak_input": res.get("deleak_input"),
        "random10_FLUS_ratio": {"median": float(np.median(rnd[10])), "min": float(np.min(rnd[10])), "max": float(np.max(rnd[10]))},
        "random32_FLUS_ratio": {"median": float(np.median(rnd[32])), "min": float(np.min(rnd[32])), "max": float(np.max(rnd[32]))},
        "random_successes": {k: int(sum(res[a]["success"] for a in res if a.startswith(f"random{k}_"))) for k in (10, 32)},
    }
    for k in (10, 32):
        g = res.get(f"greedy{k}")
        summ[kind][f"greedy{k}_prereg_success"] = bool(g and g["success"] and summ[kind]["random_successes"][k] == 0)
clean = [r["deleak_input_clean_dSST"] for r in d["rows"]]
summ["deleak_input_clean_dSST"] = {"mean_open_K": float(np.mean([c["mean_open"] for c in clean])),
                                   "rms_open_K": float(np.mean([c["rms_open"] for c in clean]))}
json.dump(summ, open(os.path.join(RES, f"{name}_summary.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in summ.items() if k not in ("greedy",)}, indent=1))
