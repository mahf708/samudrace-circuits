"""Idea 3 step 2: one-step (5-day) gain per SH degree band of the C1A0 ocean network, by JVP. PREREG 3.

Truth-forced inputs (hc.truth_ocean) at the same 12 dates as the atmosphere (Jan 16, Apr 16, Jul 15, Oct 13 of
1996, 2042, 2047). The ocean is deterministic (no seeds). Probes: band-limited random fields with flat power over
the band, MULTIPLIED BY THE OCEAN MASK (land stays 0), unit area-weighted RMS over ocean, physical units; 8 per
band. Gain(band) = sqrt(in-band output power / in-band input power), both spectra of ocean-masked fields (the
mask leaks power across degrees; the probe's own leakage is in the denominator, so persistence-like maps give ~1).
Pairs: diagonal sst, temperatureCoarsened_0, temperatureCoarsened_5, ssh; all pairs among them kept.
usage: python gain_ocn.py OUT_TAG  -> idea3_gain/results/gain_ocn_TAG.npz/.json
"""
import json
import os
import sys
import time

import numpy as np
import torch

from hc import graph, truth_ocean
from hc.spectra import _sht

TAG = sys.argv[1]
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
ocn, ins, outs, means, stds = graph.load_ocean()
ocn.requires_grad_(False)
DEV = graph.DEV
CH = ["sst", "temperatureCoarsened_0", "temperatureCoarsened_5", "ssh"]
BANDS = [(0, 2), (3, 5)] + [(l, l + 3) for l in range(6, 90, 4)]
NPROBE = 8
DATES = [truth_ocean.stamp(y, m, d) for y in (1996, 2042, 2047) for m, d in ((1, 16), (4, 16), (7, 15), (10, 13))]
sht, isht = _sht(DEV, "s"), _sht(DEV, "is")
LMAX, MMAX = sht.lmax, sht.mmax
latw = torch.tensor(np.cos(np.deg2rad(np.linspace(-89.5, 89.5, 180))), device=DEV, dtype=torch.float32)[:, None]
o_idx = torch.tensor([outs.index(c) for c in CH], device=DEV)
o_sd = torch.tensor([stds[c] for c in CH], device=DEV)[None, :, None, None]


def power(f):
    c = sht(f.float())
    p = c.abs() ** 2
    p[..., 1:] *= 2
    return p.sum(-1)


def probe(lo, hi, n, gen, mask):
    c = torch.zeros(n, LMAX, MMAX, dtype=torch.complex64, device=DEV)
    for l in range(lo, hi + 1):
        m = min(l, MMAX - 1) + 1
        z = torch.randn(n, m, 2, generator=gen, device=DEV)
        z = torch.complex(z[..., 0], z[..., 1]) / np.sqrt(2.0)
        z[:, 0] = z[:, 0].real.to(torch.complex64) * np.sqrt(2.0)
        c[:, l, :m] = z / np.sqrt(2 * l + 1)
    f = isht(c) * mask
    rms = torch.sqrt((f ** 2 * latw).sum((-2, -1)) / (latw * mask).sum())
    return f / rms[:, None, None]


P_in = np.zeros((len(DATES), len(CH), len(BANDS), LMAX))
P_r = np.zeros((len(DATES), len(CH), len(BANDS), len(CH), LMAX))
P_state = np.zeros((len(DATES), len(CH), LMAX))
t0 = time.time()
for di, t in enumerate(DATES):
    x, _ = truth_ocean.build(t)
    x0 = torch.tensor(x[None], device=DEV)
    mask = (x0[0, ins.index("sst")] != 0).float()
    gen = torch.Generator(device=DEV).manual_seed(500 + di)
    with torch.no_grad():
        y0 = ocn(x0)[:, o_idx] * o_sd
        P_state[di] = power(y0[0] * mask).double().cpu().numpy()
    for ii, cin in enumerate(CH):
        ci = ins.index(cin)
        for bi, (lo, hi) in enumerate(BANDS):
            d = probe(lo, hi, NPROBE, gen, mask)
            for s in range(0, NPROBE, 2):
                dd = d[s:s + 2]
                v = torch.zeros(dd.shape[0], *x0.shape[1:], device=DEV)
                v[:, ci] = dd / stds[cin]
                _, jv = torch.func.jvp(lambda xx: ocn(xx)[:, o_idx] * o_sd, (x0.repeat(dd.shape[0], 1, 1, 1),), (v,))
                P_in[di, ii, bi] += power(dd).sum(0).double().cpu().numpy()
                P_r[di, ii, bi] += power(jv * mask).sum(0).double().cpu().numpy()
    print(f"date {di} {time.time() - t0:.0f}s", flush=True)
np.savez_compressed(os.path.join(RES, f"gain_ocn_{TAG}.npz"), P_in=P_in, P_resp=P_r, P_state=P_state, bands=np.array(BANDS))
g = {}
for ii, c in enumerate(CH):
    rows = []
    for bi, (lo, hi) in enumerate(BANDS):
        num = P_r[:, ii, bi, ii, lo:hi + 1].sum(-1)
        den = P_in[:, ii, bi, lo:hi + 1].sum(-1)
        rows.append(np.sqrt(num / den))
    a = np.array(rows).T  # dates x bands
    g[c] = {"mean": a.mean(0).tolist(), "min": a.min(0).tolist(), "max": a.max(0).tolist()}
json.dump({"channels": CH, "bands": BANDS, "dates": DATES, "diag_gain": g, "seconds": time.time() - t0,
           "note": "ocean-masked probes and spectra; truth-forced inputs; 5-day step"},
          open(os.path.join(RES, f"gain_ocn_{TAG}.json"), "w"), indent=1)
print(json.dumps({c: [round(v, 2) for v in g[c]["mean"][::4]] for c in CH}))
