"""Summarise gain_atm_*.npz (+ truth_transfer.json): gain per SH band in physical units, spread over states and
seeds, with baselines: persistence (1), truth one-step transfer (score years), damped-linear AR(1) (fit years;
the same statistic as the truth transfer on other years -> a restatement, labelled), and the noise gain.
usage: python gain_summary.py TAG [TAG...] -> results/gain_atm_summary.json, web fig gain_atm.png (if WEBFIG set)
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
tags = sys.argv[1:]
P_in, P_r, P_n, P_sd, P_st, meta = [], [], [], [], [], None
for t in tags:
    z = np.load(os.path.join(RES, f"gain_atm_{t}.npz"))
    P_in.append(z["P_in"]); P_r.append(z["P_resp"]); P_n.append(z["P_noise_resp"]); P_sd.append(z["P_seeddiff"])
    P_st.append(z["P_state_out"])
    meta = json.load(open(os.path.join(RES, f"gain_atm_{t}.json")))
P_in, P_r, P_n = np.concatenate(P_in), np.concatenate(P_r), np.concatenate(P_n)
P_sd, P_st = np.concatenate(P_sd), np.concatenate(P_st)
IN, OUT, BANDS = meta["in_ch"], meta["out_ch"], [tuple(b) for b in meta["bands"]]
mid = np.array([(a + b) / 2 for a, b in BANDS])


def band_gain(i_in, i_out):
    """(n_samples, n_bands) in-band gain for input channel i_in -> output channel i_out."""
    g = np.zeros((P_r.shape[0], len(BANDS)))
    for bi, (lo, hi) in enumerate(BANDS):
        num = P_r[:, i_in, bi, i_out, lo:hi + 1].sum(-1)
        den = P_in[:, i_in, bi, lo:hi + 1].sum(-1)
        g[:, bi] = np.sqrt(num / den)
    return g


def leak(i_in):
    """fraction of the diagonal response power outside the probed band."""
    out = np.zeros((P_r.shape[0], len(BANDS)))
    for bi, (lo, hi) in enumerate(BANDS):
        tot = P_r[:, i_in, bi, i_in, :].sum(-1)
        out[:, bi] = 1 - P_r[:, i_in, bi, i_in, lo:hi + 1].sum(-1) / tot
    return out


tt = json.load(open(os.path.join(RES, "truth_transfer.json"))) if os.path.exists(os.path.join(RES, "truth_transfer.json")) else None
summ = {"bands": BANDS, "band_mid": mid.tolist(), "n_samples": int(P_r.shape[0]), "diag": {}, "cross": {},
        "noise_gain": {}, "baselines": {}}
for c in IN:
    if c in OUT:
        g = band_gain(IN.index(c), OUT.index(c))
        summ["diag"][c] = {"mean": g.mean(0).tolist(), "p05": np.percentile(g, 5, 0).tolist(),
                           "p95": np.percentile(g, 95, 0).tolist(), "min": g.min(0).tolist(), "max": g.max(0).tolist(),
                           "leak_mean": leak(IN.index(c)).mean(0).tolist()}
for a, b in [("T_7", "U_7"), ("U_7", "T_7"), ("PS", "U_7"), ("TS", "T_7")]:
    g = band_gain(IN.index(a), OUT.index(b))
    summ["cross"][f"{a}->{b}"] = {"mean": g.mean(0).tolist(), "p05": np.percentile(g, 5, 0).tolist(),
                                  "p95": np.percentile(g, 95, 0).tolist()}
for oi, c in enumerate(OUT):
    gn = np.zeros((P_n.shape[0], len(BANDS)))
    for bi, (lo, hi) in enumerate(BANDS):
        gn[:, bi] = np.sqrt(P_n[:, bi, oi, :].sum(-1))  # response power (all l) per unit-RMS noise probe in band
    summ["noise_gain"][c] = {"mean": gn.mean(0).tolist()}
    summ["noise_gain"][c]["seed_diff_power_per_l"] = P_sd[:, oi].mean(0).tolist()
if tt:
    V = tt["vars"]
    for blk in ("1990s", "2040s"):
        for key in ("truth_transfer_score", "damped_linear_fit"):
            arr = np.array(tt[f"{blk}|{key}"])  # (V, L<=90)
            bb = {}
            for c in OUT:
                if c in V:
                    v = arr[V.index(c)]
                    bb[c] = [float(np.mean(v[lo:min(hi, len(v) - 1) + 1])) for lo, hi in BANDS]
            summ["baselines"][f"{blk}|{key}"] = bb
json.dump(summ, open(os.path.join(RES, "gain_atm_summary.json"), "w"), indent=1)

fig, axs = plt.subplots(2, 5, figsize=(16, 6.4), sharex=True)
for ax, c in zip(axs.ravel(), OUT):
    d = summ["diag"][c]
    ax.fill_between(mid, d["p05"], d["p95"], color="C0", alpha=0.25, lw=0)
    ax.plot(mid, d["mean"], "C0-", lw=2, label="C1A0 JVP gain")
    ax.axhline(1, color="k", lw=1, ls=":", label="persistence")
    for blk, col in (("1990s", "C3"), ("2040s", "C1")):
        b = summ["baselines"].get(f"{blk}|truth_transfer_score", {}).get(c)
        if b:
            ax.plot(mid, b, color=col, lw=1.3, label=f"truth AR(1) transfer {blk}")
    ax.set_title(c)
    ax.set_ylim(0, 1.3)
for ax in axs[-1]:
    ax.set_xlabel("SH degree l (band centre)")
axs[0, 0].set_ylabel("gain (physical units)")
axs[0, 0].legend(fontsize=7, frameon=False)
fig.tight_layout()
web = os.environ.get("WEBFIG")
if web:
    fig.savefig(os.path.join(web, "gain_atm.png"))
fig.savefig(os.path.join(RES, "gain_atm.png"))
print(json.dumps({c: [round(x, 2) for x in summ["diag"][c]["mean"][::4]] for c in OUT}))
