"""F1: is the ocean's FLUS input a target leak in E3SM truth?  (PREREG 1.1 (a), (b))

Blocks (PREREG 0): 1990s fit 1995-97 / score 1998-99; 2040s fit 2040-44 / score 2045-49. Contrast (training
years, labelled): 1985-89. A sample is the transition into window w (the target): the ocean reads sst[w-1]
(previous-window mean) and the window-w fluxes; it predicts sst[w]. The sample's year is the year of window w.

(a) Alignment: FLUS[w] vs eps*sigma*T^4 for 5 SST definitions; eps fitted on fit years (open-ocean LSQ),
    scored on score years: RMS and R^2 (pooled over open-ocean nodes and windows) and per-cell R^2 maps.
(b) Per open-ocean cell OLS on score years: dSST = sst[w]-sst[w-1] ~ FLUS, FLDS, FSNS, LHFLX, SHFLX (window w),
    sst[w-1], annual+semiannual harmonics, intercept. Newey-West (lag 6) SEs. Slopes reported in K per 100 W/m2
    of WARMING flux (FLUS, LHFLX, SHFLX enter with -1). Open-ocean cell = ice < 0.01 in every window of the
    block's score (or fit) years. (b') Frisch-Waugh restatement of the FLUS coefficient (same number; check).
usage: python f1_analyze.py  -> idea1_flus/results/f1.json, f1_maps.npz
"""
import json
import os

import numpy as np

from hc import paths

D = os.path.join(paths.OUT, "f1")
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RES, exist_ok=True)
SIGMA = 5.670374e-8
FLUX = ["FLUS", "FLDS", "FSNS", "LHFLX", "SHFLX"]
WARM = {"FLUS": -1, "FLDS": 1, "FSNS": 1, "LHFLX": -1, "SHFLX": -1}
lat = np.linspace(-89.5, 89.5, 180)  # weights only (Gaussian grid within 0.3 deg)
W = (np.cos(np.deg2rad(lat))[:, None] * np.ones((1, 360))).astype("f8")
BLOCKS = {"1990s": {"fit": [1995, 1996, 1997], "score": [1998, 1999]},
          "2040s": {"fit": list(range(2040, 2045)), "score": list(range(2045, 2050))},
          "1980s_train_contrast": {"fit": [1985, 1986, 1987], "score": [1988, 1989]}}
KEYS = ["t", "sst", "ice", "TS_start", "TS_end", "T4_mean", "FLUS_eam"] + FLUX


def load(years):
    """Concatenate contiguous years; return arrays for transitions into windows of `years` (needs year-1)."""
    have_prev = os.path.exists(os.path.join(D, f"year_{years[0] - 1}.npz"))
    ys = ([years[0] - 1] if have_prev else []) + list(years)
    zs = [np.load(os.path.join(D, f"year_{y}.npz")) for y in ys]
    a = {k: np.concatenate([z[k] for z in zs]) for k in KEYS}
    n0 = len(zs[0]["t"]) if have_prev else 1  # without year-1 on disk, drop the block's first transition
    prev = {k: a[k][n0 - 1:-1] for k in ("sst", "ice")}
    cur = {k: a[k][n0:] for k in KEYS}
    return cur, prev


def sst_defs(cur, prev):
    return {"prev_window_mean_sst[w-1]": prev["sst"] ** 4, "target_window_mean_sst[w]": cur["sst"] ** 4,
            "window_mean_TS^4": cur["T4_mean"], "window_start_TS": cur["TS_start"] ** 4,
            "window_end_TS": cur["TS_end"] ** 4}


def openmask(cur, prev):
    ok = (cur["ice"] < 0.01) & (prev["ice"] < 0.01) & np.isfinite(cur["sst"]) & np.isfinite(prev["sst"])
    return ok


def alignment(blk):
    cf, pf = load(BLOCKS[blk]["fit"])
    cs, ps = load(BLOCKS[blk]["score"])
    of, os_ = openmask(cf, pf), openmask(cs, ps)
    out, maps = {}, {}
    for flus_name in ("FLUS", "FLUS_eam"):
        df, ds = sst_defs(cf, pf), sst_defs(cs, ps)
        for k in df:
            x, y = SIGMA * df[k][of].astype("f8"), cf[flus_name][of].astype("f8")
            eps = float((x * y).sum() / (x * x).sum())
            xs, ys = SIGMA * ds[k].astype("f8") * eps, cs[flus_name].astype("f8")
            r = np.where(os_, ys - xs, np.nan)
            ww = np.broadcast_to(W, r.shape)[os_]
            rms = float(np.sqrt((ww * r[os_] ** 2).sum() / ww.sum()))
            yv = ys[os_]
            ybar = (ww * yv).sum() / ww.sum()
            r2 = 1 - float((ww * r[os_] ** 2).sum() / (ww * (yv - ybar) ** 2).sum())
            # per-cell R^2 over time (cells open in >= 90% of windows)
            cnt = os_.sum(0)
            yy = np.where(os_, ys, np.nan)
            ssr = np.nansum(r ** 2, 0)
            sst_ = np.nansum((yy - np.nanmean(yy, 0)) ** 2, 0)
            cellr2 = np.where(cnt >= 0.9 * len(ys), 1 - ssr / np.maximum(sst_, 1e-9), np.nan)
            # anomaly R^2 per cell: remove each cell's score-period mean seasonal cycle by 73-window climatology
            out[f"{flus_name}|{k}"] = {"eps_fit": eps, "rms_Wm2": rms, "R2_pooled": r2,
                                       "cell_R2_median": float(np.nanmedian(cellr2)),
                                       "mean_residual_Wm2": float((ww * r[os_]).sum() / ww.sum())}
            maps[f"{blk}|{flus_name}|{k}|cellR2"] = cellr2.astype("f4")
    return out, maps


def harmonics(t):
    ph = 2 * np.pi * (np.asarray(t) % 365.0) / 365.0
    return np.stack([np.cos(ph), np.sin(ph), np.cos(2 * ph), np.sin(2 * ph)], 1)


def regress(years):
    cur, prev = load(years)
    n = len(cur["t"])
    ok = openmask(cur, prev)
    cell = ok.all(0)
    idx = np.where(cell.ravel())[0]
    y = (cur["sst"] - prev["sst"]).reshape(n, -1)[:, idx].astype("f8")
    Xf = [cur[f].reshape(n, -1)[:, idx].astype("f8") for f in FLUX] + [prev["sst"].reshape(n, -1)[:, idx].astype("f8")]
    H = harmonics(cur["t"])
    names = FLUX + ["sst_prev", "c1", "s1", "c2", "s2", "const"]
    ncell = len(idx)
    coef = np.full((len(names), ncell), np.nan)
    se = np.full_like(coef, np.nan)
    r2 = np.full(ncell, np.nan)
    fw = np.full(ncell, np.nan)
    L = 6
    for c in range(ncell):
        X = np.column_stack([x[:, c] for x in Xf] + [H, np.ones(n)])
        yc = y[:, c]
        XtX_inv = np.linalg.pinv(X.T @ X)
        b = XtX_inv @ X.T @ yc
        e = yc - X @ b
        # Newey-West
        Xe = X * e[:, None]
        S = Xe.T @ Xe
        for lag in range(1, L + 1):
            wl = 1 - lag / (L + 1)
            G = Xe[lag:].T @ Xe[:-lag]
            S += wl * (G + G.T)
        V = XtX_inv @ S @ XtX_inv
        coef[:, c], se[:, c] = b, np.sqrt(np.clip(np.diag(V), 0, None))
        r2[c] = 1 - (e @ e) / ((yc - yc.mean()) @ (yc - yc.mean()))
        if c % 997 == 0:  # Frisch-Waugh check on a subsample
            Xo = np.delete(X, 0, 1)
            P = Xo @ np.linalg.pinv(Xo)
            ry, rx = yc - P @ yc, X[:, 0] - P @ X[:, 0]
            fw[c] = (rx @ ry) / (rx @ rx)
    maps = {}
    for i, nm in enumerate(names[:6]):
        m = np.full(180 * 360, np.nan); m[idx] = coef[i]; maps[nm] = m.reshape(180, 360)
        s = np.full(180 * 360, np.nan); s[idx] = se[i]; maps[nm + "_se"] = s.reshape(180, 360)
    m = np.full(180 * 360, np.nan); m[idx] = r2; maps["R2"] = m.reshape(180, 360)
    wc = W.ravel()[idx]
    summ = {"n_windows": n, "open_cells": int(ncell), "open_area_frac": float(wc.sum() / W.sum())}
    for f in FLUX:
        b = coef[names.index(f)] * 100 * WARM[f]  # K per 100 W/m2 of warming flux
        z = coef[names.index(f)] / se[names.index(f)]
        summ[f] = {"K_per_100Wm2_warming_mean": float((wc * b).sum() / wc.sum()),
                   "median": float(np.median(b)), "p05": float(np.percentile(b, 5)), "p95": float(np.percentile(b, 95)),
                   "frac_area_negative(thermometer_sign_for_FLUS)": float((wc * (b < 0)).sum() / wc.sum()),
                   "frac_area_|z|>2": float((wc * (np.abs(z) > 2)).sum() / wc.sum())}
    summ["R2_median"] = float(np.median(r2))
    j = ~np.isnan(fw)
    summ["frisch_waugh_check_maxabs_diff"] = float(np.max(np.abs(fw[j] - coef[0][j])))
    # literal-thermometer reference: 1/(4 eps sigma T^3) at the cell's mean sst, K per 100 W/m2
    T = prev["sst"].reshape(n, -1)[:, idx].mean(0)
    lit = 100 / (4 * 0.9997 * SIGMA * T ** 3)
    summ["literal_thermometer_K_per_100Wm2_mean"] = float((wc * lit).sum() / wc.sum())
    return summ, maps


if __name__ == "__main__":
    result, allmaps = {"prereg": "PREREG.md 1.1", "blocks": BLOCKS}, {}
    for blk in BLOCKS:
        a, m = alignment(blk)
        result[f"alignment|{blk}"] = a
        allmaps.update(m)
        for part in ("score", "fit"):
            s, mp = regress(BLOCKS[blk][part])
            result[f"regression|{blk}|{part}"] = s
            for k, v in mp.items():
                allmaps[f"{blk}|{part}|{k}"] = v.astype("f4")
            print(blk, part, json.dumps({f: s[f]["K_per_100Wm2_warming_mean"] for f in FLUX}), flush=True)
    json.dump(result, open(os.path.join(RES, "f1.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(RES, "f1_maps.npz"), **allmaps)
    print("done")
