"""Shared bits: the bridged observable, flux-shift edits, and the mixed-layer law."""
import numpy as np, torch

RHO_CP = 1025.0 * 3990.0          # J m-3 K-1
DT = 5 * 86400.0                  # s, one ocean step
SIGMA, EPS = 5.670e-8, 0.97
FLUX = ("FLUS", "FLDS", "FSNS", "LHFLX", "SHFLX")
SIGN = {"FLUS": -1, "FLDS": +1, "FSNS": +1, "LHFLX": -1, "SHFLX": -1}   # sign in Q (net downward into ocean)


class Bridge:
    """Ice-free, area-weighted ocean-mean of a field (the LAWFUL output bridge D_f)."""
    def __init__(self, ad, sst_ctrl, ice_ctrl):
        w = np.cos(np.deg2rad(ad.lat))[:, None] * np.ones((1, ad.W))
        self.ok = np.isfinite(sst_ctrl) & (np.nan_to_num(ice_ctrl, nan=1.0) < 0.01)
        self.w = np.where(self.ok, w, 0.0); self.w /= self.w.sum()
    def __call__(self, field):
        return float(np.sum(np.where(self.ok, field, 0.0) * self.w))


def shift_fn(ad, shifts):
    """Normalized-space patch for the network input: add shifts[name] W/m2 at ocean points
    (where the normalized sst input is non-zero, as in lawful_ideas/ocean_flux_arms.py)."""
    i_sst = ad.in_names.index("sst")
    items = [(ad.in_names.index(n), float(a) / float(ad.in_std[ad.in_names.index(n)])) for n, a in shifts.items() if a]
    def fn(x):
        x = x.clone(); ocean = (x[i_sst] != 0).to(x.dtype)
        for i, d in items:
            x[i] = x[i] + ocean * d
        return x
    return fn


def q_net(shifts):
    return sum(SIGN[n] * a for n, a in shifts.items())


def law(q, h):
    """Mixed-layer budget: dSST = Q dt / (rho cp h)."""
    return q * DT / (RHO_CP * h)
