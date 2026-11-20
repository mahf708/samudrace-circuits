"""Config-driven paths for every script in hackathon-circuits (no laptop paths).

Override any entry with an environment variable of the same name (e.g. HC_OUT=/elsewhere).
"""
import os

# Data SOURCES live in mahf708's scratch on Perlmutter (world-readable); this does not depend on who runs the
# code. Off Perlmutter, set HC_DATA_ROOT (or the individual HC_* variables) after downloading what you need.
_PS = os.environ.get("HC_DATA_ROOT", "/pscratch/sd/m/mahf708")


def _env(name, default):
    return os.environ.get(name, default)


ROOT = _env("HC_ROOT", os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..")))
# release folder (HF mahf708/e3smv3-lr-hist-samudrace, files renamed: CO2-noaero.tar == rc1 C1A0.tar)
RELEASE = _env("HC_RELEASE", f"{_PS}/samudrace-repro/e3smv3-lr-hist-samudrace")
CKPT = {
    "C1A0": _env("HC_CKPT_C1A0", f"{RELEASE}/checkpoints/CO2-noaero.tar"),  # md5 a5082a9aec1243e37e0820a99e77d059
    "C0A0": _env("HC_CKPT_C0A0", f"{RELEASE}/checkpoints/noCO2-noaero.tar"),  # md5 44585e626f9322cd6f6965d2462c498e
    "C1A1": _env("HC_CKPT_C1A1", f"{RELEASE}/checkpoints/CO2-aero.tar"),  # md5 e6f9d00fccd0b95ac7cdae1e3b27edf6
}
CKPT_MD5 = {
    "C1A0": "a5082a9aec1243e37e0820a99e77d059",
    "C0A0": "44585e626f9322cd6f6965d2462c498e",
    "C1A1": "e6f9d00fccd0b95ac7cdae1e3b27edf6",
}
INFER_CFG = {"C1A0": f"{RELEASE}/configs/inference-CO2-noaero.yaml"}
FORCING = f"{RELEASE}/forcing_data"
IC = {
    "atmosphere": f"{RELEASE}/initial_conditions/restart-ic-1985-2015_atmosphere.nc",
    "ocean": f"{RELEASE}/initial_conditions/restart-ic-1985-2015_ocean.nc",
}
# E3SM truth: the coupled dataset C1A0 was trained on (training run E35-CFT S01)
TRUTH_RUN = _env("HC_TRUTH", f"{_PS}/v3.LR.historical_0101.aigo/run")
TRUTH_PREFIX = "v3.LR.historical_0101.aigo"
LANDFRAC5D = f"{_PS}/e3sm-hist-inputs/landfrac5d"
MOMFRAC5D = f"{_PS}/e3sm-hist-inputs/momfrac5d"
TRAIN_CFG = f"{_PS}/aug26-cft/E35-CFT.aug26.cpl.A0_B16_C1_L0_O5_V0_W0_X0.S01/config.yaml"
# Train: 1940-01-06..1990-01-01 and 2000-01-06..2040-01-01. Validation: 1990-1994. Never seen: 1995-1999, 2040+.
SPLITS = {"train": [(1940, 1989), (2000, 2039)], "valid": [(1990, 1994)], "heldout": [(1995, 1999), (2040, 2064)]}
# outputs: big things on scratch, under this project
OUT = _env("HC_OUT", f"{ROOT}/data")
CKPT_CFG_C1A0 = os.path.join(os.path.dirname(__file__), "ckpt_config_C1A0.yaml")


def split_of(year: int) -> str:
    for name, ranges in SPLITS.items():
        if any(a <= year <= b for a, b in ranges):
            return name
    return "none"
