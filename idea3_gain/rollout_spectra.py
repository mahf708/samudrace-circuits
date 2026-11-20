"""Idea 3 / track T3 starter: do 1-year C1A0 rollouts lose small-scale variance where the one-step gain says?

Inputs:
  model: daily pre-corrector spectra recorded during 1-year coupled runs started from truth
         (`hc.evaluate --atm-spectra`): data/states/t1995, data/rollout1y/t2040, data/states/t2045 -> atm_spectra.npz
         (T_4, T_7: scalar SH power per l; UV_4, UV_7: vector-SHT power per l; one record per day, call 4k = day k+1,
         valid at 06 UTC).
  truth: idea3_gain/results/truth_spectra.npz (`truth_transfer.py transfer`): E3SM daily 00 UTC spectra of the same
         years, same transforms (T scalar, UV vector).
Output: ratio model/truth of mean power per l over days 181–365 (months 7–12) per variable and year, plus the
  balanced-probe projection curve (gain_atm_bal_v2.json) for comparison -> results/rollout_spectra.json, .png.
Caveats: model spectra are pre-corrector network outputs at 06 UTC, truth at 00 UTC (daily sampling, diurnal phase
  differs by 6 h); l is truncated at 89 for T (truth coefficients were stored to l = 89).
usage: python idea3_gain/rollout_spectra.py
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from hc import paths  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
ROOT = paths.ROOT
RUNS = {1995: "data/states/t1995", 2040: "data/rollout1y/t2040", 2045: "data/states/t2045"}
TRUTH = np.load(os.path.join(RES, "truth_spectra.npz"))
TV = ["T_1", "T_4", "T_7", "U_1", "U_4", "U_7", "V_4", "V_7", "PS", "TS"]  # truth_transfer.VARS order
DAYS = slice(180, 365)
out, ratios = {"days": "181-365", "years": list(RUNS)}, {}
for y, d in RUNS.items():
    m = np.load(os.path.join(ROOT, d, "atm_spectra.npz"))
    tdaily = TRUTH[f"truth_daily00Z_{y}"]  # (365, V, 90)
    for v in ("T_4", "T_7", "UV_4", "UV_7"):
        mod = m[v][:, 0] if m[v].ndim == 3 else m[v]  # (days, l) (batch of 1)
        if v.startswith("T"):
            tru = tdaily[:, TV.index(v)]
        else:
            tru = TRUTH[f"truth_daily00Z_{y}_{v.replace('_', '')}vec"]
        L = min(mod.shape[-1], tru.shape[-1], 90)
        n = min(len(mod), len(tru))
        r = mod[:n][DAYS, :L].mean(0) / tru[:n][DAYS, :L].mean(0)
        ratios[(v, y)] = r
        out[f"{v}|{y}"] = r.tolist()
bal = os.path.join(RES, "gain_atm_bal_v2.json")
gb = json.load(open(bal)) if os.path.exists(bal) else None
json.dump(out, open(os.path.join(RES, "rollout_spectra.json"), "w"))
fig, axs = plt.subplots(1, 4, figsize=(16, 3.6))
for ax, v in zip(axs, ("T_4", "T_7", "UV_4", "UV_7")):
    for y, col in zip(RUNS, ("C0", "C1", "C2")):
        r = ratios[(v, y)]
        ax.plot(np.arange(len(r)), r, color=col, label=f"rollout {y}, months 7–12")
    if gb:
        key = v.replace("UV", "U")
        b = gb["gain"].get(key)
        if b:
            mid = [(lo + hi) / 2 for lo, hi in gb["bands"]]
            ax2 = ax.twinx()
            ax2.plot(mid, b["proj_mean"], "k--", lw=1, label="balanced one-step projection")
            ax2.set_ylim(0, 1.2)
            ax2.set_ylabel("one-step projection", fontsize=8)
    ax.axhline(1, color="k", ls=":", lw=0.8)
    ax.set_yscale("log")
    ax.set_title(f"{v}: rollout / truth power")
    ax.set_xlabel("SH degree l")
axs[0].legend(fontsize=7, frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(RES, "rollout_spectra.png"))
for v in ("T_4", "T_7", "UV_4", "UV_7"):
    r = np.mean([ratios[(v, y)] for y in RUNS], axis=0)
    print(v, "ratio at l=", {l: round(float(r[l]), 2) for l in (2, 8, 16, 32, 48, 64, 80, 89) if l < len(r)})
