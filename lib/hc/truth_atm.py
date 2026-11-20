"""E3SM atmosphere truth (EAM 6-hourly) for any time, with the renames/scalings of C1A0's training loader.

`fields(t_days_since_1940, names)` -> {name: (180, 360) float64}. Times use the ocean files' units (noleap days
since 1940-01-01); EAM's own axis is days since 1850 (offset 32850). An EAM record stamped t averages (t-6h, t].
"""
import functools

import numpy as np
import xarray as xr

from hc import paths

EAM_OFFSET = 32850.0  # days 1850-01-01 -> 1940-01-01 (noleap)
RENAME = {"surface_precipitation_rate": ("PRECT", 1000.0), "frozen_precipitation_rate": ("PRECST", 1000.0),
          "global_mean_co2": ("co2vmr", 1.0)}
CUM = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


@functools.lru_cache(maxsize=4)
def _month(y, m):
    return xr.open_dataset(f"{paths.TRUTH_RUN}/{paths.TRUTH_PREFIX}.eam.h0.{y}-{m:02d}.nc", decode_times=False)


def _locate(t):
    te = t + EAM_OFFSET
    y = 1850 + int(te // 365)
    doy = te - 365 * (y - 1850)
    m = min(max(int(np.searchsorted(np.cumsum(CUM), doy, side="right")), 1), 12)
    for yy, mm in ((y, m), (y, m - 1 if m > 1 else 12), (y, m + 1 if m < 12 else 1)):
        try:
            d = _month(yy, mm)
        except FileNotFoundError:
            continue
        i = np.where(np.abs(d.time.values - te) < 1e-4)[0]
        if len(i):
            return d, int(i[0])
    raise KeyError(f"no EAM record at {t} (EAM {te})")


def fields(t, names):
    d, i = _locate(t)
    out = {}
    for n in names:
        src, scale = RENAME.get(n, (n, 1.0))
        out[n] = d[src].isel(time=i).values.astype("f8") * scale
    return out
