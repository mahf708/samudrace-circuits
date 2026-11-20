"""Skeptic check (independent implementation): FLUS skip.0 and deep-path shares through exege's samudra adapter
(EdgePatch semantics) on real C1A0 states, vs localize.py. base = perturbed (FLUS -25 W/m2 of warming flux,
i.e. FLUS +25, uniform open ocean), source = clean.
  skip.0 share = response with only input->l0.* and l0.*->out open (others read the clean source)
  deep share   = full - response with l0.*->enc1 closed
usage: python skeptic_adapter_s0.py STATES_DIR -> results/skeptic_adapter_s0.json"""
import json, os, sys
import numpy as np
from hc import graph, paths, set_numerics
from exege.circuits.contract import EdgePatch
from exege.circuits.contract import open_patchable

S = os.path.abspath(sys.argv[1]); set_numerics()
names = json.load(open(os.path.join(S, "names.json")))
_, ins, outs, mu, sd = graph.load_ocean()
json.dump({"in_names": ins, "out_names": outs, "means": mu, "stds": sd}, open(os.path.join(S, "meta.json"), "w"))
ad = open_patchable(S, "samudra", checkpoint=paths.CKPT["C1A0"], device="cuda")
fixed = graph.fixed_ocean_mask(S); W = graph.Grid.w
g = ad.graph(); rows = []
for k in range(0, 73, 12):
    clean = ad.states([k])
    x = graph.load_state(k, S)
    ice = x[ins.index("ocean_sea_ice_fraction")] * sd["ocean_sea_ice_fraction"] + mu["ocean_sea_ice_fraction"]
    ow = fixed & (ice < 0.01)
    pert = ad.perturb(clean, {"FLUS": (25.0 * ow).ravel()})
    y0 = ad.step(clean)["sst"][0].reshape(180, 360)
    am = lambda y: float(((y.reshape(180, 360) - y0) * W * ow).sum() / (W * ow).sum())
    full = am(ad.step(pert)["sst"][0])
    m = g.mask(0.0)
    for s_, r_ in [("input", "l0.skip_module"), ("input", "l0.convblock"), ("l0.skip_module", "out"), ("l0.convblock", "out")]:
        m[g.sender(s_), g.receiver(r_)] = 1
    s0 = am(ad.step(pert, EdgePatch(clean, m))["sst"][0])
    m2 = g.mask(1.0); m2[g.sender("l0.skip_module"), g.receiver("enc1")] = 0; m2[g.sender("l0.convblock"), g.receiver("enc1")] = 0
    deep = full - am(ad.step(pert, EdgePatch(clean, m2))["sst"][0])
    rows.append({"k": k, "full_K_per_25": full, "skip0_share": s0 / full, "deep_share": deep / full})
    print(rows[-1], flush=True)
out = {"rows": rows, "skip0_share_mean": float(np.mean([r["skip0_share"] for r in rows])),
       "deep_share_mean": float(np.mean([r["deep_share"] for r in rows])),
       "slope_K_per_100": float(np.mean([r["full_K_per_25"] for r in rows]) * -4)}
json.dump(out, open("idea1_flus/results/skeptic_adapter_s0.json", "w"), indent=1); print(out)
