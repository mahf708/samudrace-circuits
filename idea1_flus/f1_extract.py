"""F1 data extraction from E3SM truth: per 5-day ocean window, the fields the ocean was trained on plus
EAM 6-hourly surface temperature summaries for the same window.

Window w is the ocean record stamped t_w (end of window, noleap days since 1940-01-01); it averages (t_w-5d, t_w].
Saved per year (73 windows, float32, 180x360):
  sst        ocean 5-day mean SST (K)                      [training 'sst', +273.15]
  FLUS, FLDS, FSNS, LHFLX, SHFLX   MPAS 5-day mean fluxes with the training loader's signs (FLUS, LHFLX, SHFLX
                                   positive upward; FLDS, FSNS positive downward)
  ice        MPAS-SI 5-day mean ice area fraction
  TS_start   EAM TS of the 6-h record ending at t_w - 5d (window start)
  TS_end     EAM TS of the 6-h record ending at t_w (window end)
  T4_mean    mean over the window's 20 EAM records of TS^4
  FLUS_eam   mean over the window's 20 EAM records of EAM FLUS
usage: python f1_extract.py YEAR [YEAR ...]   -> data/f1/year_YYYY.npz
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import xarray as xr

from hc import paths

R, P = paths.TRUTH_RUN, paths.TRUTH_PREFIX
OUT = os.path.join(paths.OUT, "f1")
EAM0 = 52925 - 20075  # EAM days since 1850 minus ocean days since 1940 at 1995-01-01 (both noleap) = 32850


def ocean_year(y):
    ds = [xr.open_dataset(f"{R}/{P}.mpaso.hist.am.fmeDerivedFields5D.{y}-{m:02d}.remapped.nc", decode_times=False)
          for m in range(1, 13)]
    si = [xr.open_dataset(f"{R}/{P}.mpassi.hist.am.fmeSeaiceDerivedFields5D.{y}-{m:02d}.remapped.nc",
                          decode_times=False) for m in range(1, 13)]
    t = np.concatenate([d.time.values for d in ds])
    g = lambda v: np.concatenate([d[v].values for d in ds]).astype("f4")
    out = {"t": t, "sst": g("sst") + 273.15, "FLUS": -g("longWaveHeatFluxUp"), "FLDS": g("longWaveHeatFluxDown"),
           "FSNS": g("shortWaveHeatFlux"), "LHFLX": -g("latentHeatFlux"), "SHFLX": -g("sensibleHeatFlux"),
           "ice": np.concatenate([d["iceAreaTotal"].values for d in si]).astype("f4")}
    return out


def eam_months(y):
    months = [(y - 1, 12)] + [(y, m) for m in range(1, 13)] + [(y + 1, 1)]
    ts, fl, tt = [], [], []
    for yy, mm in months:
        f = f"{R}/{P}.eam.h0.{yy}-{mm:02d}.nc"
        if not os.path.exists(f):
            continue
        d = xr.open_dataset(f, decode_times=False)[["TS", "FLUS"]]
        tt.append(d.time.values - EAM0)
        ts.append(d.TS.values.astype("f4"))
        fl.append(d.FLUS.values.astype("f4"))
    return np.concatenate(tt), np.concatenate(ts), np.concatenate(fl)


def do_year(y):
    os.makedirs(OUT, exist_ok=True)
    o = ocean_year(y)
    te, ts, fl = eam_months(y)
    n = len(o["t"])
    S = np.empty((n, 180, 360), "f4"); E = np.empty_like(S); T4 = np.empty_like(S); FE = np.empty_like(S)
    for w, tw in enumerate(o["t"]):
        i_end = int(np.argmin(np.abs(te - tw)))
        i_start = int(np.argmin(np.abs(te - (tw - 5))))
        assert abs(te[i_end] - tw) < 1e-3 and abs(te[i_start] - (tw - 5)) < 1e-3, (y, w, tw)
        sl = slice(i_start + 1, i_end + 1)  # 20 records in (tw-5, tw]
        assert sl.stop - sl.start == 20
        S[w], E[w] = ts[i_start], ts[i_end]
        T4[w] = (ts[sl].astype("f8") ** 4).mean(0)
        FE[w] = fl[sl].mean(0)
    np.savez(os.path.join(OUT, f"year_{y}.npz"), TS_start=S, TS_end=E, T4_mean=T4, FLUS_eam=FE, **o)
    return y, n


if __name__ == "__main__":
    years = [int(a) for a in sys.argv[1:]]
    with ProcessPoolExecutor(min(len(years), 8)) as ex:
        for y, n in ex.map(do_year, years):
            print(y, n, flush=True)
