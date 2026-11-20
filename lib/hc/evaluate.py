"""Coupled C1A0 runs started from E3SM TRUTH at any date (held-out years), through fme.coupled.evaluator.

The data loader is E35-CFT's own `heldout_1990s` inference loader (the exact dataset spec the model was
trained/validated with), with start times replaced. Writes predictions and the matching truth (target) files.
Optional --deleak (CombinedFluxConfig on the loaded stepper) and ocean-input capture like hc.capture.

usage: python -m hc.evaluate OUT --start 1995-01-01 [--start ...] --steps N [--seed 0] [--deleak] [--capture]
       [--save-atm] [--no-ocn-pred] [--monthly]
"""
import argparse
import copy
import json
import os
import time

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
import numpy as np
import yaml

from hc import paths
from hc.capture import DELEAK


def build_config(out, starts, steps, ckpt, seed, save_atm, save_ocn, monthly, mem):
    train = yaml.safe_load(open(paths.TRAIN_CFG))
    block = next(b for b in train["inference"] if b["name"] == "heldout_1990s")
    loader = copy.deepcopy(block["loader"])
    loader["start_indices"] = {"times": [f"{s}T00:00:00" for s in starts]}
    loader["num_data_workers"] = 2
    c = {
        "experiment_dir": os.path.join(out, "fme"),
        "n_coupled_steps": steps,
        "coupled_steps_in_memory": mem,
        "checkpoint_path": paths.CKPT[ckpt],
        "seed": seed,
        "logging": {"log_to_screen": True, "log_to_file": False, "log_to_wandb": False},
        "loader": loader,
        "data_writer": {
            "ocean": {"save_prediction_files": save_ocn, "save_monthly_files": monthly},
            "atmosphere": {"save_prediction_files": save_atm, "save_monthly_files": monthly},
        },
    }
    os.makedirs(c["experiment_dir"], exist_ok=True)
    return c


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--start", action="append", required=True)
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--ckpt", default="C1A0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--deleak", action="store_true")
    ap.add_argument("--capture", action="store_true")
    ap.add_argument("--save-atm", action="store_true")
    ap.add_argument("--no-ocn-pred", action="store_true")
    ap.add_argument("--monthly", action="store_true")
    ap.add_argument("--mem", type=int, default=1)
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--capture-atm", type=int, default=0,
                    help="save the first N atmosphere network calls: atm_x_KK.npy (input) and atm_y_KK.npy (output)")
    ap.add_argument("--atm-spectra", action="store_true",
                    help="record daily (every 4th call) pre-corrector SH spectra of T_4, T_7, (U,V)_4, (U,V)_7")
    a = ap.parse_args(argv)
    import fme.coupled.inference.evaluator as ev
    from fme.coupled.stepper import CombinedFluxConfig

    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    c = build_config(out, a.start, a.steps, a.ckpt, a.seed, a.save_atm, not a.no_ocn_pred, a.monthly, a.mem)
    cfg = os.path.join(out, "config.yaml")
    yaml.safe_dump(c, open(cfg, "w"), sort_keys=False)
    open(os.path.join(out, "start_date.txt"), "w").write(a.start[0])
    calls, spec_calls, spec_store = [], [0], []
    orig = ev.InferenceEvaluatorConfig.load_stepper

    def load_stepper(self):
        st = orig(self)
        if a.deleak:
            st._config.combined_fluxes.append(CombinedFluxConfig(**DELEAK))
        so = st.ocean._step_obj
        json.dump({"in_names": list(so.in_names), "out_names": list(so.out_names),
                   "combined_fluxes": [str(x) for x in st._config.combined_fluxes]},
                  open(os.path.join(out, "names.json"), "w"), indent=1)
        sa0 = st.atmosphere._step_obj
        json.dump({"in_names": list(sa0.in_names), "out_names": list(sa0.out_names)},
                  open(os.path.join(out, "atm_names.json"), "w"), indent=1)
        if a.capture_atm:
            atm_calls = [0]

            def atm_cap(mod, args, output):
                k = atm_calls[0]
                atm_calls[0] += 1
                if k < a.capture_atm:
                    np.save(os.path.join(out, f"atm_x_{k:02d}.npy"), args[0].detach().float().cpu().numpy())
                    np.save(os.path.join(out, f"atm_y_{k:02d}.npy"), output.detach().float().cpu().numpy())

            st.atmosphere.modules[0].module.register_forward_hook(atm_cap)
        if a.atm_spectra:
            from hc.spectra import scalar_power, vector_power

            sa = st.atmosphere._step_obj
            onames = list(sa.out_names)
            mu = {k: float(v) for k, v in sa.normalizer.means.items()}
            sd = {k: float(v) for k, v in sa.normalizer.stds.items()}
            ph = lambda y, n: y[:, onames.index(n)] * sd[n] + mu[n]

            def atm_hook(mod, args, output):
                k = spec_calls[0]
                spec_calls[0] += 1
                if k % 4:
                    return
                y = output.detach()
                rec = {"T_4": scalar_power(ph(y, "T_4")), "T_7": scalar_power(ph(y, "T_7")),
                       "UV_4": vector_power(ph(y, "U_4"), ph(y, "V_4")),
                       "UV_7": vector_power(ph(y, "U_7"), ph(y, "V_7"))}
                spec_store.append({n: r.cpu().numpy() for n, r in rec.items()} | {"call": k})

            st.atmosphere.modules[0].module.register_forward_hook(atm_hook)
        if a.capture:
            ocn = st.ocean.modules[0].module

            def pre(mod, args):
                x = args[0].detach().float().cpu().numpy()
                k = len(calls)
                calls.append(k)
                for b in range(x.shape[0]):
                    np.save(os.path.join(out, f"x_{k:03d}" + (f"_ic{b}" if x.shape[0] > 1 else "") + ".npy"), x[b])

            ocn.register_forward_pre_hook(pre)
        return st

    ev.InferenceEvaluatorConfig.load_stepper = load_stepper
    t0 = time.time()
    ev.main(cfg, override_dotlist=a.override or None)
    if spec_store:
        np.savez(os.path.join(out, "atm_spectra.npz"), call=np.array([r["call"] for r in spec_store]),
                 **{n: np.stack([r[n] for r in spec_store]) for n in ("T_4", "T_7", "UV_4", "UV_7")})
    json.dump({"starts": a.start, "steps": a.steps, "seed": a.seed, "ckpt": a.ckpt, "deleak": a.deleak,
               "wall_s": time.time() - t0, "n_ocean_calls": len(calls)},
              open(os.path.join(out, "evaluate.json"), "w"), indent=1)
    print("wall", round(time.time() - t0, 1), flush=True)


if __name__ == "__main__":
    main()
