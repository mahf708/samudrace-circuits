"""Idea 1 step 3 (F2): localise the FLUS response below "skip.0 + layer 0". PREREG 1.3.

Bare C1A0 ocean network, fp32. Layer 0 (ConvNeXtBlock, the only reader of the input) is
    L0(x) = skip_module(x)  [1x1 conv, 91->280, no norm]
          + convblock(x)    [3x3 conv -> InstanceNorm -> GELU -> 3x3 conv -> InstanceNorm -> GELU -> 1x1 conv]
and its output feeds the U-Net twice: the deep path (down through D0) and the full-resolution skip (S0).
Exact patching (clean -> corrupted sender values) over the factorial
    layer-0 edge in {skip_module, convblock, convblock with clean InstanceNorm statistics frozen}
    x U-Net path  in {skip.0, deep}
Combos (each a single forward of the real network with some sender values from the corrupted run):
    full      : everything corrupted (= the model's response)
    s0only    : skip.0 reads corrupted L0, deep path clean
    deeponly  : deep path corrupted L0, skip.0 clean
    skipmod   : L0 = skip_module(x1) + convblock(x0) on both paths
    convblk   : L0 = skip_module(x0) + convblock(x1) on both paths
    convfrz   : L0 = skip_module(x0) + convblock(x1 with InstanceNorm mean/var from x0) on both paths
Perturbations ("input channels"): FLUS, FLDS, FSNS, LHFLX, SHFLX (dQ of warming flux), sst and
temperatureCoarsened_0 together ("sstT0", dT), and sst alone. Areas: uniform open ocean + 10x10 deg boxes on the
PREREG lattice (kept if >= 95% open ocean that date). dQ: +-25 (all combos), +-100 (full only); dT: +-0.5, +-2 K.
Metrics: perturbed-area mean dSST and projection on the full response over open ocean.
usage: python localize.py STATES_DIR TAG [stride=4] [first=0]  -> idea1_flus/results/localize_TAG.npz/.json
"""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from hc import graph

S, TAG = os.path.abspath(sys.argv[1]), sys.argv[2]
stride = int(sys.argv[3]) if len(sys.argv) > 3 else 4
first = int(sys.argv[4]) if len(sys.argv) > 4 else 0
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RES, exist_ok=True)
ocn, ins, outs, means, stds = graph.load_ocean()
G = graph.Graph(ocn)
L0 = ocn.layers[0]
DEV = graph.DEV
isst = outs.index("sst")
W = graph.Grid.w
FIXED = graph.fixed_ocean_mask(S)
PERTS = {"FLUS": [("FLUS", -1)], "FLDS": [("FLDS", 1)], "FSNS": [("FSNS", 1)], "LHFLX": [("LHFLX", -1)],
         "SHFLX": [("SHFLX", -1)], "sstT0": [("sst", 1), ("temperatureCoarsened_0", 1)], "sst": [("sst", 1)]}
AMP = {"flux": [-100.0, -25.0, 25.0, 100.0], "temp": [-2.0, -0.5, 0.5, 2.0]}
COMBOS = ["full", "s0only", "deeponly", "skipmod", "convblk", "convfrz"]
LAT_C, LON_C = [-45, -25, -5, 15, 35], [30, 110, 190, 270]


def boxes(openw):
    lat, lon = graph.Grid.lat, graph.Grid.lon
    out = {}
    for la in LAT_C:
        for lo in LON_C:
            m = (np.abs(lat - la)[:, None] <= 5) & (np.abs(((lon - lo + 180) % 360) - 180)[None, :] <= 5)
            if (W * (m & openw)).sum() >= 0.95 * (W * m).sum():
                out[f"box{la:+d}_{lo:03d}"] = m & openw
    return out


def convblock(x, stats=None, want_stats=False):
    """Replicates ConvNeXtBlock.convblock forward; stats = list of (mean, var) per InstanceNorm to freeze."""
    st, i_in = [], 0
    for layer in L0.convblock:
        if isinstance(layer, torch.nn.Conv2d) and layer.kernel_size[0] != 1:
            x = F.pad(x, (L0.N_pad, L0.N_pad, 0, 0), mode=L0.pad)
            x = F.pad(x, (0, 0, L0.N_pad, L0.N_pad), mode="constant")
            x = layer(x)
        elif isinstance(layer, torch.nn.InstanceNorm2d):
            mu = x.mean(dim=(2, 3), keepdim=True)
            var = x.var(dim=(2, 3), keepdim=True, unbiased=False)
            st.append((mu, var))
            if stats is not None:
                mu, var = stats[i_in]
            x = (x - mu) / torch.sqrt(var + layer.eps)
            i_in += 1
        else:
            x = layer(x)
    return (x, st) if want_stats else x


@torch.no_grad()
def check_exact(x):
    y_ref = ocn(x)
    sk = L0.skip_module(x)
    l0 = sk + convblock(x)
    y = G.run(x, src={"D0": l0, "S0": l0})
    return float((y - y_ref).abs().max()), float((l0 - L0(x)).abs().max())


@torch.no_grad()
def responses(x0, x1s):
    """x0: (1,91,H,W) clean; x1s: (B,91,H,W) corrupted. Returns dict combo -> (B,H,W) dSST (K)."""
    B = x1s.shape[0]
    sk0, (cb0, st0) = L0.skip_module(x0), convblock(x0, want_stats=True)
    l00 = sk0 + cb0
    y0 = G.run(x0, src={"D0": l00, "S0": l00})[:, isst]
    sk1, cb1 = L0.skip_module(x1s), convblock(x1s)
    cbf = convblock(x1s, stats=[(m.expand(B, -1, -1, -1), v.expand(B, -1, -1, -1)) for m, v in st0])
    l01 = sk1 + cb1
    L = {"full": (l01, l01), "s0only": (l00.expand(B, -1, -1, -1), l01), "deeponly": (l01, l00.expand(B, -1, -1, -1)),
         "skipmod": (sk1 + cb0,) * 2, "convblk": (sk0 + cb1,) * 2, "convfrz": (sk0 + cbf,) * 2}
    xx = x0.expand(B, -1, -1, -1)
    out = {}
    for c, (ld, ls) in L.items():
        out[c] = ((G.run(xx, src={"D0": ld, "S0": ls})[:, isst] - y0) * stds["sst"]).float().cpu().numpy()
    return out


ks = list(range(first, len(graph.state_files(S)), stride))
x_chk = torch.tensor(graph.load_state(ks[0], S)[None], device=DEV)
exact = check_exact(x_chk)
print("exactness (max|y-ref|, max|L0-ref|):", exact, flush=True)
rows, t0 = [], time.time()
for k in ks:
    xn = graph.load_state(k, S)
    ice = xn[ins.index("ocean_sea_ice_fraction")] * stds["ocean_sea_ice_fraction"] + means["ocean_sea_ice_fraction"]
    openw = FIXED & (ice < 0.01)
    areas = {"uniform": openw} | boxes(openw)
    x0 = torch.tensor(xn[None], device=DEV)
    hrules = {"dT0.2_floor20": graph.mixed_layer_depth(xn, ins, means, stds, dT=0.2),
              "dT0.5_floor20": graph.mixed_layer_depth(xn, ins, means, stds, dT=0.5),
              "fixed33.1": np.full((180, 360), 33.1), "fixed50": np.full((180, 360), 50.0)}
    for pname, comps in PERTS.items():
        amps = AMP["temp" if pname.startswith("sst") else "flux"]
        for aname, am in areas.items():
            amt = torch.tensor(am.astype("f4"), device=DEV)
            x1 = []
            for a in amps:
                x = x0[0].clone()
                for ch, sgn in comps:
                    x[ins.index(ch)] += sgn * a * amt / stds[ch]
                x1.append(x)
            R = responses(x0, torch.stack(x1))
            full = R["full"]
            wa = W * am
            wo = W * openw
            rec = {"k": k, "pert": pname, "area": aname, "area_frac": float(wa.sum() / W.sum()),
                   "amps": amps, "full_areamean": [float((r * wa).sum() / wa.sum()) for r in full]}
            if not pname.startswith("sst"):
                rec["law_K_per_100"] = {hn: float((graph.slab_law(100.0, h, pname) * wa).sum() / wa.sum())
                                        for hn, h in hrules.items()}
            for c in COMBOS:
                rec[c] = {}
                for i in range(4):
                    r = R[c][i]
                    rec[c].setdefault("areamean", []).append(float((r * wa).sum() / wa.sum()))
                    rec[c].setdefault("proj", []).append(float((r * full[i] * wo).sum() / (full[i] ** 2 * wo).sum()))
            rows.append(rec)
    print(f"state {k} {time.time() - t0:.0f}s areas={len(areas)}", flush=True)

json.dump({"states_dir": S, "stride": stride, "states": ks, "exactness": exact, "combos": COMBOS,
           "perts": list(PERTS), "rows": rows}, open(os.path.join(RES, f"localize_{TAG}.json"), "w"))
print("done", time.time() - t0)
