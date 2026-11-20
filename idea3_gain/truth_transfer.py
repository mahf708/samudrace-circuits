"""Idea 3 truth baselines from E3SM 6-hourly output (PREREG 3, baselines 2 and 3; rollout-link truth spectra).

stage A (per year):  python truth_transfer.py coeffs YEAR [YEAR...]
    SH coefficients (l, m <= 89) of the tracked variables at every 6-h record -> data/truth_sh/YYYY.npz
stage B:             python truth_transfer.py transfer
    climatology: fit-year mean coefficient per (day-of-year, hour), smoothed +-15 days (circular)
    per block and per l:
      AR(1)/transfer b_l = sum Re(a_{t+1} conj a_t) / sum |a_t|^2 over m and t, anomalies a about the climatology
        - 'damped_linear_fit'  : fitted on fit years (PREREG baseline 3)
        - 'truth_transfer_score': same statistic on score years (PREREG baseline 2; a restatement of 3 on other years)
      tendency power per l: <|x_{t+1}-x_t|^2>, and state power <|x_t|^2>
      daily 00Z power spectra for 1995, 2040, 2045 (rollout comparison)
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import xarray as xr

from hc import paths

VARS = ["T_1", "T_4", "T_7", "U_1", "U_4", "U_7", "V_4", "V_7", "PS", "TS"]
LM = 90
OUT = os.path.join(paths.OUT, "truth_sh")
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
BLOCKS = {"1990s": {"fit": [1995, 1996, 1997], "score": [1998, 1999]},
          "2040s": {"fit": list(range(2040, 2045)), "score": list(range(2045, 2050))}}


def coeffs_year(y):
    import torch

    torch.set_num_threads(4)
    from hc.spectra import _sht

    from hc.spectra import vector_power

    sht = _sht(torch.device("cpu"), "s")
    os.makedirs(OUT, exist_ok=True)
    C, T, UV4, UV7 = [], [], [], []
    for m in range(1, 13):
        d = xr.open_dataset(f"{paths.TRUTH_RUN}/{paths.TRUTH_PREFIX}.eam.h0.{y}-{m:02d}.nc", decode_times=False)
        arr = np.stack([d[v].values.astype("f4") for v in VARS], 1)  # (t, V, H, W)
        with torch.no_grad():
            c = sht(torch.from_numpy(arr))[..., :LM, :LM]
            d00 = torch.from_numpy(arr[::4])
            UV4.append(vector_power(d00[:, VARS.index("U_4")], d00[:, VARS.index("V_4")]).numpy())
            UV7.append(vector_power(d00[:, VARS.index("U_7")], d00[:, VARS.index("V_7")]).numpy())
        C.append(c.numpy().astype("c8"))
        T.append(d.time.values)
    np.savez(os.path.join(OUT, f"{y}.npz"), c=np.concatenate(C), t=np.concatenate(T), vars=np.array(VARS),
             uv4_vec=np.concatenate(UV4), uv7_vec=np.concatenate(UV7))
    return y


def power(c):
    p = np.abs(c) ** 2
    p[..., 1:] *= 2
    return p.sum(-1)


def transfer():
    import json

    os.makedirs(RES, exist_ok=True)
    out = {"vars": VARS}
    spectra = {}
    for blk, sp in BLOCKS.items():
        clim = None
        cnt = np.zeros((365, 4))
        for y in sp["fit"]:
            z = np.load(os.path.join(OUT, f"{y}.npz"))
            c = z["c"]
            if clim is None:
                clim = np.zeros((365, 4) + c.shape[1:], "c16")
            for i in range(c.shape[0]):
                doy, hr = (i // 4) % 365, i % 4
                clim[doy, hr] += c[i]
                cnt[doy, hr] += 1
        clim /= cnt[..., None, None, None]
        # +-15 day circular smoothing
        cs = np.zeros_like(clim)
        for s in range(-15, 16):
            cs += np.roll(clim, s, axis=0)
        clim = (cs / 31).astype("c8")
        for part in ("fit", "score"):
            num = 0.0
            den = 0.0
            tend = 0.0
            stp = 0.0
            n = 0
            for y in sp[part]:
                c = np.load(os.path.join(OUT, f"{y}.npz"))["c"]
                idx = np.arange(c.shape[0])
                a = c - clim[(idx // 4) % 365, idx % 4]
                num_m = np.real(a[1:] * np.conj(a[:-1]))
                num_m[..., 1:] *= 2
                num = num + num_m.sum(0).sum(-1)
                den = den + power(a[:-1]).sum(0)
                tend = tend + power(c[1:] - c[:-1]).sum(0)
                stp = stp + power(c).sum(0)
                n += c.shape[0] - 1
            key = "damped_linear_fit" if part == "fit" else "truth_transfer_score"
            out[f"{blk}|{key}"] = (num / den).tolist()  # (V, L)
            out[f"{blk}|{part}|tendency_power"] = (tend / n).tolist()
            out[f"{blk}|{part}|state_power"] = (stp / n).tolist()
    for y in (1995, 2040, 2045):
        z = np.load(os.path.join(OUT, f"{y}.npz"))
        c = z["c"][::4]  # daily 00Z
        spectra[f"truth_daily00Z_{y}"] = power(c).astype("f4")  # (365, V, L)
        spectra[f"truth_daily00Z_{y}_UV4vec"] = z["uv4_vec"].astype("f4")  # full-l vector-SHT power, as the rollout hook
        spectra[f"truth_daily00Z_{y}_UV7vec"] = z["uv7_vec"].astype("f4")
    json.dump(out, open(os.path.join(RES, "truth_transfer.json"), "w"))
    np.savez_compressed(os.path.join(RES, "truth_spectra.npz"), **spectra)
    print("done")


if __name__ == "__main__":
    if sys.argv[1] == "coeffs":
        ys = [int(a) for a in sys.argv[2:]]
        with ProcessPoolExecutor(min(8, len(ys))) as ex:
            for y in ex.map(coeffs_year, ys):
                print("coeffs", y, flush=True)
    else:
        transfer()
