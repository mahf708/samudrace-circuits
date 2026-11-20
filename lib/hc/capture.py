"""Run SamudrACE through the official fme.coupled.inference path and capture every ocean-network call.

Port of the laptop capture scripts (cross-flux-witness-circuit/capture_states.py and
lawful-ocean-flus-thermometer/capture.py), CUDA + config-driven paths. At every ocean call it saves the
exact fp32 tensor entering Samudra (x_XXX.npy: normalized, packed in in_names order, land = 0) and the raw
output (y_XXX.npy), plus meta.npz (lat, lon, per-output finite mask) for SamudraReplay.

usage:
  python -m hc.capture OUT --start 1985 --steps 73 [--ckpt C1A0] [--seed 0] [--deleak] [--atm-pred]

--start 1985|2015 picks the release IC (index 0|1). Forcing for start..start+N years is linked into OUT/forcing.
--deleak appends CombinedFluxConfig(name="FLUS", terms={"FLUS": 0.0}, sst_emission_coefficient=5.67e-8) to the
loaded coupled stepper (no checkpoint edit).
"""
import argparse
import json
import os
import time

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
import numpy as np
import yaml

from hc import paths

DELEAK = dict(name="FLUS", terms={"FLUS": 0.0}, sst_emission_coefficient=5.67e-8)


def link_forcing(dst, y0, y1):
    os.makedirs(dst, exist_ok=True)
    for y in range(y0, y1 + 1):
        for kind in ("atmosphere", "ocean"):
            f = f"{kind}-forcing-{y}.nc"
            src = os.path.join(paths.FORCING, f)
            if not os.path.exists(src):
                raise FileNotFoundError(src)
            d = os.path.join(dst, f)
            if not os.path.lexists(d):
                os.symlink(src, d)
    return dst


def build_config(out, start, steps, ckpt, seed, save_atm=False, save_ocn=True, monthly=False):
    c = yaml.safe_load(open(paths.INFER_CFG["C1A0"]))
    years = steps * 5 // 365 + 1
    fdir = link_forcing(os.path.join(out, "forcing"), start, start + years)
    c.update(n_coupled_steps=steps, coupled_steps_in_memory=1, experiment_dir=os.path.join(out, "fme"),
             checkpoint_path=paths.CKPT[ckpt], seed=seed)
    for r in ("atmosphere", "ocean"):
        c["forcing_loader"][r]["dataset"]["data_path"] = fdir
        c["initial_condition"][r]["path"] = paths.IC[r]
        c["data_writer"][r]["save_monthly_files"] = monthly
    c["data_writer"]["ocean"]["save_prediction_files"] = save_ocn
    c["data_writer"]["atmosphere"]["save_prediction_files"] = save_atm
    c["initial_condition"]["start_indices"] = {"first": {1985: 0, 2015: 1}[start], "n_initial_conditions": 1}
    c["forcing_loader"]["num_data_workers"] = 0
    c["logging"]["log_to_file"] = False
    os.makedirs(c["experiment_dir"], exist_ok=True)
    return c


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--start", type=int, default=1985)
    ap.add_argument("--steps", type=int, default=73)
    ap.add_argument("--ckpt", default="C1A0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--deleak", action="store_true")
    ap.add_argument("--atm-pred", action="store_true", help="also write atmosphere prediction files")
    ap.add_argument("--no-capture", action="store_true", help="do not save ocean x/y tensors")
    ap.add_argument("--monthly", action="store_true")
    a = ap.parse_args(argv)
    import fme.coupled.inference.inference as inf
    from fme.coupled.stepper import CombinedFluxConfig

    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    c = build_config(out, a.start, a.steps, a.ckpt, a.seed, save_atm=a.atm_pred, monthly=a.monthly)
    cfg = os.path.join(out, "config.yaml")
    yaml.safe_dump(c, open(cfg, "w"), sort_keys=False)
    open(os.path.join(out, "start_date.txt"), "w").write(f"{a.start}-01-01")

    calls = []
    orig = inf.InferenceConfig.load_stepper

    def load_stepper(self):
        st = orig(self)
        if a.deleak:
            st._config.combined_fluxes.append(CombinedFluxConfig(**DELEAK))
        ocn = st.ocean.modules[0].module
        so = st.ocean._step_obj
        json.dump({"in_names": list(so.in_names), "out_names": list(so.out_names),
                   "combined_fluxes": [str(x) for x in st._config.combined_fluxes]},
                  open(os.path.join(out, "names.json"), "w"), indent=1)
        if not a.no_capture:
            def pre(mod, args):
                k = len(calls)
                calls.append({"k": k, "t": time.time()})
                np.save(os.path.join(out, f"x_{k:03d}.npy"), args[0].detach().float().cpu().numpy()[0])

            def post(mod, args, output):
                np.save(os.path.join(out, f"y_{len(calls) - 1:03d}.npy"), output.detach().float().cpu().numpy()[0])

            ocn.register_forward_pre_hook(pre)
            ocn.register_forward_hook(post)
        return st

    inf.InferenceConfig.load_stepper = load_stepper
    t0 = time.time()
    inf.main(cfg)
    wall = time.time() - t0
    pred = os.path.join(out, "fme", "ocean", "autoregressive_predictions.nc")
    if os.path.exists(pred) and not a.no_capture:
        import xarray as xr

        ds = xr.open_dataset(pred).isel(sample=0, time=-1)
        names = json.load(open(os.path.join(out, "names.json")))["out_names"]
        mask = np.stack([np.isfinite(ds[n].values) for n in names])
        np.savez(os.path.join(out, "meta.npz"), lat=ds.lat.values, lon=ds.lon.values, out_mask=mask)
    json.dump({"n_calls": len(calls), "wall_s": wall, "start": f"{a.start}-01-01", "steps": a.steps,
               "seed": a.seed, "ckpt": a.ckpt, "ckpt_path": paths.CKPT[a.ckpt], "deleak": a.deleak,
               "note": "x_k enters the ocean at start + 5k days; y_k is its raw (normalized) output"},
              open(os.path.join(out, "capture.json"), "w"), indent=1)
    print("calls", len(calls), "wall", round(wall, 1), flush=True)


if __name__ == "__main__":
    main()
