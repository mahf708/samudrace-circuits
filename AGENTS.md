# AGENTS.md: working with samudrace-circuits

This repository finishes three interpretability studies of **SamudrACE C1A0** (E3SMv3 coupled emulator: SFNO
atmosphere + Samudra U-Net ocean) against **E3SM truth** on Perlmutter. It is written to be picked up by a person or
an agent. Read this file, then `PREREG.md` (the fixed rules), then `results/manifest.json` (every claim, with the
JSON it comes from and the command that made it).

## What is where

| Path | What |
|---|---|
| `setup_env.sh` | rebuilds `.venv` (Python 3.12, torch 2.14.1+cu130, fme from ace tag `e3smv3-lr-hist-samudrace-rc1` = 1c29e2d9b, exege `mahf708/circuits`, this repo's `lib/` as package `hc`) |
| `lib/hc/paths.py` | **every path** (checkpoints, E3SM truth, splits); override with env vars `HC_*` |
| `lib/hc/graph.py` | `Graph`: exact re-implementation of `Samudra.forward` with edge places (skip.0..3, deep path, bottleneck); `load_ocean`, `load_ocean_any` (coupled or ocean-only checkpoints), masks, mixed-layer depth, slab law |
| `lib/hc/samudra_adapter.py` | `SamudraReplay`: exege `Intervenable` for the ocean (entry point `samudra-replay`) |
| `lib/hc/capture.py` | official `fme.coupled.inference` run from the release ICs (1985, 2015) capturing every ocean input; `--deleak` |
| `lib/hc/evaluate.py` | coupled run **started from E3SM truth at any date** (`fme.coupled.evaluator` with C1A0's own training loader); writes predictions + truth targets; `--capture`, `--capture-atm N`, `--atm-spectra`, `--deleak` |
| `lib/hc/truth_ocean.py` | truth-forced ocean inputs (state at t_j + truth fluxes of window j+1, exactly as ocean-only training), validated equal to fme's packing on all state channels |
| `lib/hc/spectra.py` | SH power spectra on the model's legendre-gauss grid |
| `lib/runner.sh` | tiny job queue (drop scripts in `data/queue/{gpu,cpu}`) |
| `idea1_flus/` | FLUS thermometer: F1 truth test, localisation, ablation, replication |
| `idea2_errors/` | error agreement with E3SM truth (Phase 2) |
| `idea3_gain/` | one-step gain per SH degree (JVP), truth baselines |
| `phase0/` | smoke tests, numerics check, laptop-number reproduction |
| `web/build.py` | builds the results page and `results/manifest.json` from the JSONs |

## Model, data, splits (do not get these wrong)

- C1A0 = HF `mahf708/e3smv3-lr-hist-samudrace@rc1 checkpoints/C1A0.tar`, md5 `a5082a9aec1243e37e0820a99e77d059`
  (older local copies call it `CO2-noaero.tar`). Training run E35-CFT S01: atmosphere from E02-FT S01, ocean from
  **E11-FT S01** (the hackathon "E11" ocean is C1A0's ocean before coupled fine-tuning).
- Truth: `v3.LR.historical_0101.aigo` (6-hourly EAM, 5-day-mean MPAS), 1940–2065.
- **Train 1940–1989 and 2000–2039; validation 1990–1994; never seen 1995–1999 and 2040+.** Score only on held-out
  years. 1985 and 2015 (the release ICs, the laptop states) are training years.
- Ocean records are stamped at the END of their 5-day window. The ocean step j→j+1 reads the state at t_j and the
  fluxes of window (t_j, t_j+5d], and predicts the state at t_j+5d.

## Pitfalls we hit (each cost time)

1. **TF32.** PyTorch enables TF32 for cuDNN convolutions by default and fme runs that way: the U-Net is then not
   batch-invariant (0.02 K). Perturbation/patching/JVP work calls `hc.set_numerics()` (fp32). Rollouts run as deployed.
2. **Land fill.** fme fills every input channel but LANDFRAC with 0 (normalized) on land. `sea_surface_fraction > 0.5`
   on such inputs marks 96% of the grid as ocean; use `graph.fixed_ocean_mask` (official sst output finite).
3. **Ice-shelf cavities** (Ross, Weddell, ~80S) are "open ocean" by the ice < 0.01 test; exclude them (lat > -70) for
   sensitivity.
4. **ICs must sit on 5-day stamps** (multiples of 5 noleap days since 1940-01-01): Jan 16, Apr 16, Jul 15, Oct 13...
5. **Forward-mode AD** (`torch.func.jvp`) needs contiguous primals/tangents (`repeat`, not `expand`) and parameters
   with `requires_grad=False` (else memory doubles). Set `net.checkpointing = 0` and MLP checkpointing off.
6. **Atmosphere noise.** `NoiseConditionedModel.forward` draws its own noise; to hold it fixed call
   `conditional_model(x, Context(noise=..., embedding_pos=pos_embed))` with your own draw.
7. **Samudra's in-place `fts += skip`**: hooks must `.clone()`. Upsample outputs are padded before the skip add.
8. **Perlmutter shared GPU nodes**: `srun` steps can wait forever on "Required ports are in use". Run one long-lived
   step (or a batch job) with `lib/runner.sh` and queue work as scripts.
9. **The leak test needs per-cell intercepts.** MPAS FLUS and σT⁴ differ by a spatially varying offset (pooled RMS
   22 W/m²), which hides the time relation in pooled statistics.

## Reproduce a number

Every claim in `results/manifest.json` has `script`, `command`, `inputs` and `json`. Typical pattern:

```bash
bash setup_env.sh
.venv/bin/python -m hc.evaluate data/states/t2045 --start 2045-01-01 --steps 73 --capture   # held-out ocean states
.venv/bin/python -m hc.meta data/states/t2045
.venv/bin/python idea1_flus/f1_extract.py 1995 1996 1997 1998 1999 2040 ... 2049            # truth windows (CPU)
.venv/bin/python idea1_flus/f1_analyze.py
.venv/bin/python idea1_flus/f1_model_compare.py data/states/t2045 2040s t2045
```
