"""Truth-forced ocean network inputs from E3SM output, as in ocean-only (E11) training.

For the ocean step from stamp t_j (window ending t_j) to t_{j+1}:
  state channels (sst, ssh, T/S/u/v x 19, ocean_sea_ice_fraction, iceVolumeTotal) and LANDFRAC,
  sea_surface_fraction at t_j; forcing channels (next_step_forcing_names: TAUX, TAUY, precipitation, FLUS, FLDS,
  FSNS, LHFLX, SHFLX) at t_{j+1}. Renames/signs exactly as E35's loader. Normalized with C1A0's ocean normalizer;
  NaN (land / below sea floor) filled with 0 after normalization (input_masking fill_value 0), as the captured
  fme inputs. Also returns the truth target (80 outputs at t_{j+1}, physical units, NaN on land).
Times are noleap days since 1940-01-01 (the ocean files' units); stamps are multiples of 5.
"""
import functools

import numpy as np
import xarray as xr
import yaml

from hc import paths

R, P = paths.TRUTH_RUN, paths.TRUTH_PREFIX
FORCING = ["TAUX", "TAUY", "surface_precipitation_rate", "frozen_precipitation_rate", "FLUS", "FLDS", "FSNS",
           "LHFLX", "SHFLX"]


@functools.lru_cache(maxsize=1)
def names_norm():
    ck = yaml.safe_load(open(paths.CKPT_CFG_C1A0))["ocean"]["stepper"]["step"]["config"]
    nrm = ck["normalization"]["network"]
    return list(ck["in_names"]), list(ck["out_names"]), nrm["means"], nrm["stds"]


def stamp_to_ym(t):
    y = 1940 + int(t // 365)
    doy = t - 365 * (y - 1940)  # stamp is window END; a stamp at doy 0 belongs to the file of month 1 of year y
    cum = np.cumsum([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    m = int(np.searchsorted(cum, doy, side="right"))
    m = min(max(m, 1), 12)
    return y, m


@functools.lru_cache(maxsize=64)
def _month(y, m):
    d = xr.open_dataset(f"{R}/{P}.mpaso.hist.am.fmeDerivedFields5D.{y}-{m:02d}.remapped.nc", decode_times=False)
    c = xr.open_dataset(f"{R}/{P}.mpaso.hist.am.fmeDepthCoarsening5D.{y}-{m:02d}.remapped.nc", decode_times=False)
    s = xr.open_dataset(f"{R}/{P}.mpassi.hist.am.fmeSeaiceDerivedFields5D.{y}-{m:02d}.remapped.nc", decode_times=False)
    return d, c, s


@functools.lru_cache(maxsize=4)
def _landfrac(y):
    return xr.open_dataset(f"{paths.LANDFRAC5D}/landfrac5d.{y}.nc", decode_times=False)


def fields_at(t):
    """All 91 input names (and the 80 outputs) in physical units at stamp t (no time shifting)."""
    y, m = stamp_to_ym(t)
    for yy, mm in ((y, m), (y, m - 1 if m > 1 else 12), (y, m + 1 if m < 12 else 1)):
        d, c, s = _month(yy, mm)
        i = np.where(np.abs(d.time.values - t) < 1e-3)[0]
        if len(i):
            break
    else:
        raise KeyError(t)
    i = int(i[0])
    g = lambda ds, v: ds[v].values[i].astype("f8")
    f = {"sst": g(d, "sst") + 273.15, "ssh": g(d, "ssh"),
         "TAUX": -g(d, "windStressZonal"), "TAUY": -g(d, "windStressMeridional"),
         "FSNS": g(d, "shortWaveHeatFlux"), "FLDS": g(d, "longWaveHeatFluxDown"),
         "FLUS": -g(d, "longWaveHeatFluxUp"), "LHFLX": -g(d, "latentHeatFlux"), "SHFLX": -g(d, "sensibleHeatFlux"),
         "frozen_precipitation_rate": g(d, "snowFlux")}
    f["surface_precipitation_rate"] = g(d, "rainFlux") + f["frozen_precipitation_rate"]
    for k in range(19):
        f[f"salinityCoarsened_{k}"] = g(c, f"salinityCoarsened_{k}")
        f[f"temperatureCoarsened_{k}"] = g(c, f"temperatureCoarsened_{k}")
        f[f"velocityZonalCoarsened_{k}"] = g(c, f"velocityZonalCoarsened_{k}")
        f[f"velocityMeridionalCoarsened_{k}"] = g(c, f"velocityMeridionalCoarsened_{k}")
    f["ocean_sea_ice_fraction"] = g(s, "iceAreaTotal")
    f["iceVolumeTotal"] = g(s, "iceVolumeTotal")
    lf = _landfrac(y)
    j = np.where(np.abs(lf.time.values - t) < 1e-3)[0]
    if not len(j):  # the landfrac files may use another epoch; fall back to day-of-year matching
        doy = t - 365 * (y - 1940)
        j = np.where(np.abs((lf.time.values % 365) - doy % 365) < 1e-3)[0]
    j = int(j[0])
    f["LANDFRAC"] = lf["LANDFRAC"].values[j].astype("f8")
    f["sea_surface_fraction"] = lf["sea_surface_fraction"].values[j].astype("f8")
    return f


def build(t_j, ins=None, outs=None, mu=None, sd=None):
    """Normalized (91,180,360) float32 input for the step t_j -> t_j + 5, and the physical truth target (80,...).
    Defaults to C1A0's names/normalizer; pass another checkpoint's to build its inputs."""
    if ins is None:
        ins, outs, mu, sd = names_norm()
    a, b = fields_at(t_j), fields_at(t_j + 5)
    x = np.zeros((len(ins), 180, 360), "f4")
    land = ~np.isfinite(a["sst"])  # fme input_masking: fill 0 (normalized) on land for all but LANDFRAC
    for i, n in enumerate(ins):
        v = b[n] if n in FORCING else a[n]
        z = np.nan_to_num((v - mu[n]) / sd[n], nan=0.0)
        if n != "LANDFRAC":
            z[land] = 0.0
        x[i] = z
    y = np.stack([b[n] for n in outs]).astype("f4")
    return x, y


def stamp(year, month, day):
    cum = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    doy = sum(cum[:month]) + day - 1
    return 365 * (year - 1940) + doy
