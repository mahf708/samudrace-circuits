"""Spherical-harmonic power spectra on the 180x360 legendre-gauss grid (the model's grid).

scalar_power(f): power per degree l of a scalar field (..., 180, 360), summed over m (m>0 doubled).
vector_power(u, v): kinetic-energy-like power per degree from the vector SHT of (u, v).
"""
import torch
import torch_harmonics as th

_cache = {}


def _sht(dev, kind):
    key = (str(dev), kind)
    if key not in _cache:
        if kind == "s":
            _cache[key] = th.RealSHT(180, 360, grid="legendre-gauss").to(dev)
        elif kind == "is":
            _cache[key] = th.InverseRealSHT(180, 360, grid="legendre-gauss").to(dev)
        elif kind == "v":
            _cache[key] = th.RealVectorSHT(180, 360, grid="legendre-gauss").to(dev)
    return _cache[key]


def _fold(c):
    p = c.abs() ** 2
    p[..., 1:] = 2 * p[..., 1:]
    return p.sum(-1)


def scalar_power(f):
    f = torch.nan_to_num(f.float())
    return _fold(_sht(f.device, "s")(f).to(torch.complex128))


def vector_power(u, v):
    uv = torch.stack([torch.nan_to_num(u.float()), torch.nan_to_num(v.float())], dim=-3)
    c = _sht(u.device, "v")(uv)  # (..., 2, l, m)
    return _fold(c).sum(-2)


def sht(f):
    return _sht(f.device, "s")(f.float())


def isht(c):
    return _sht(c.device, "is")(c)
