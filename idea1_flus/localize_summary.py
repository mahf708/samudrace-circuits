"""Summarise localize_TAG.json files: slopes, flux ratios, gains per MLD rule, and edge shares.

Slopes are through the origin over the 4 amplitudes, in K per 100 W/m2 of warming flux (fluxes) or K/K (sst
perturbations). Edge share = combo slope / full slope (perturbed-area mean metric) and the projection
coefficient (slope of the projection coefficients is not needed: proj is already a ratio; we average the 4).
Spread: over states (uniform) and over states x boxes (box). Mean, min, max and 5-95% reported.
usage: python localize_summary.py TAG [TAG...]  -> results/localize_summary_<TAGS>.json and .md
"""
import json
import os
import sys

import numpy as np

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
tags = sys.argv[1:]
rows = []
for t in tags:
    d = json.load(open(os.path.join(RES, f"localize_{t}.json")))
    for r in d["rows"]:
        r["tag"] = t
        rows.append(r)
combos = ["full", "s0only", "deeponly", "skipmod", "convblk", "convfrz"]


def slope(amps, vals, per=100.0):
    a, v = np.array(amps), np.array(vals)
    return per * float(a @ v / (a @ a))


def stats(v):
    v = np.array(v, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return None
    return {"mean": float(v.mean()), "min": float(v.min()), "max": float(v.max()), "p05": float(np.percentile(v, 5)),
            "p95": float(np.percentile(v, 95)), "n": int(v.size)}


out = {"tags": tags}
perts = sorted({r["pert"] for r in rows}, key=["FLUS", "FLDS", "FSNS", "LHFLX", "SHFLX", "sstT0", "sst"].index)
for tag in tags + ["all"]:
    R = [r for r in rows if tag == "all" or r["tag"] == tag]
    for kind in ("uniform", "box"):
        Rk = [r for r in R if (r["area"] == "uniform") == (kind == "uniform")]
        if not Rk:
            continue
        per = {}
        # per (state, area) FLDS slope for ratios
        flds = {(r["tag"], r["k"], r["area"]): slope(r["amps"], r["full_areamean"]) for r in Rk if r["pert"] == "FLDS"}
        for p in perts:
            Rp = [r for r in Rk if r["pert"] == p]
            scale = 1.0 if p.startswith("sst") else 100.0
            s_full = [slope(r["amps"], r["full_areamean"], scale) for r in Rp]
            e = {"slope": stats(s_full)}
            if not p.startswith("sst"):
                e["ratio_to_FLDS"] = stats([slope(r["amps"], r["full_areamean"]) / flds[(r["tag"], r["k"], r["area"])]
                                            for r in Rp])
                e["gain_vs_slab_law"] = {hn: stats([slope(r["amps"], r["full_areamean"]) / r["law_K_per_100"][hn]
                                                    for r in Rp]) for hn in Rp[0]["law_K_per_100"]}
            if p == "FLUS":
                e["frac_states_thermometer_sign"] = float(np.mean(np.array(s_full) < 0))
            sh = {}
            for c in combos[1:]:
                sa = [slope(r["amps"], r[c]["areamean"], scale) / slope(r["amps"], r["full_areamean"], scale) for r in Rp]
                sp = [float(np.mean(r[c]["proj"])) for r in Rp]
                sh[c] = {"areamean_share": stats(sa), "projection": stats(sp)}
            e["edge_shares"] = sh
            per[p] = e
        out[f"{tag}|{kind}"] = per

name = "_".join(tags)
json.dump(out, open(os.path.join(RES, f"localize_summary_{name}.json"), "w"), indent=1)
lines = [f"# Localisation summary ({', '.join(tags)})", ""]
for key, per in out.items():
    if key == "tags":
        continue
    lines += [f"## {key}", "", "| pert | slope mean [min, max] | /FLDS | s0only | deeponly | skipmod | convblk | convfrz (area-mean share; projection) |",
              "|---|---|---|---|---|---|---|---|"]
    for p, e in per.items():
        s = e["slope"]
        rt = e.get("ratio_to_FLDS")
        cells = []
        for c in combos[1:]:
            a, pj = e["edge_shares"][c]["areamean_share"], e["edge_shares"][c]["projection"]
            cells.append(f"{a['mean']:.2f}; {pj['mean']:.2f}")
        lines.append(f"| {p} | {s['mean']:+.3f} [{s['min']:+.3f}, {s['max']:+.3f}] | "
                     f"{'' if rt is None else format(rt['mean'], '+.2f')} | " + " | ".join(cells) + " |")
    lines.append("")
open(os.path.join(RES, f"localize_summary_{name}.md"), "w").write("\n".join(lines))
print("\n".join(lines))
