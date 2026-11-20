# Inventory (Phase 0, 2026-11-19)

All paths are on Perlmutter. `$PSCRATCH=/pscratch/sd/m/mahf708`. Work directory: `$PSCRATCH/hackathon-circuits`
(git repo; large outputs in `data/`, git-ignored).

## 1. C1A0

| | |
|---|---|
| Path | `$PSCRATCH/samudrace-repro/e3smv3-lr-hist-samudrace/checkpoints/CO2-noaero.tar` |
| md5 | `a5082a9aec1243e37e0820a99e77d059`, **matches**. The local copy predates the rc1 renaming: HF rc1 `checkpoints/C1A0.tar` is 2176590511 bytes, the same as `CO2-noaero.tar`, and the local MANIFEST lists that md5 for `CO2-noaero.tar`. |
| Training run | `$PSCRATCH/aug26-cft/E35-CFT.aug26.cpl.A0_B16_C1_L0_O5_V0_W0_X0.S01` (wandb `tmnl25ga`, read from the checkpoint's `training_history`). 25 epochs of coupled fine-tuning (`n_coupled_steps: 4`, ensemble CRPS loss), EMA weights. |
| Atmosphere lineage | E02 (`extfo6xr`) → E02-FT S01 (`88qegkqv`) → E35-CFT |
| Ocean lineage | E11 S01 (`35g3pgfl`) → **E11-FT S01** (`fyxxulce`) → E35-CFT. `aigs-hack-sep26/hf-models/E11-FT-S01` has the same md5 (`a88f6a2c…`) as this E11-FT S01, so the hackathon "E11" ocean **is C1A0's ocean before coupled fine-tuning**. |
| Atmosphere | NoiseConditionedSFNO: encoder + 8 blocks, embed 384, noise embed 32 (isotropic), big skip, `dhconv`, linear filter, MLP ratio 2. 6 h step. In: LANDFRAC, OCNFRAC, ICEFRAC, PHIS, SOLIN, PS, TS, T_0..7, STW_0..7, U_0..7, V_0..7, Qat2m, Uat10m, Vat10m, Tat2m, global_mean_co2. Corrector: dry-air conservation, moisture budget (advection and precipitation), positivity for STW, precipitation, FLUS, FLUT, FLDS, FSNS, FSUTOA. |
| Ocean | Samudra U-Net, widths 280/380/480/520, dilations 1/2/4/8, InstanceNorm, circular padding. 5-day step. 91 in / 80 out. **FLUS is an ocean input** (one of the `next_step_forcing_names`: TAUX, TAUY, surface and frozen precipitation, FLUS, FLDS, FSNS, LHFLX, SHFLX). Corrector: positivity (salinity, ice volume) and sea-ice fraction; **nothing touches sst**. |
| Coupling | Ocean fluxes = atmosphere outputs averaged over the 20 inner steps, scaled by the open-water fraction (heat fluxes, precipitation) or by `momentum_flux_fraction` (TAUX, TAUY). `combined_fluxes: []`. The atmosphere sees the ocean's current SST for all 20 inner steps. |
| Release forcing | CO2 (`global_mean_co2`), SOLIN, PHIS, LANDFRAC; no aerosol |

**De-leak switch (F3), verified.** Appending `CombinedFluxConfig(name="FLUS", terms={"FLUS": 0.0},
sst_emission_coefficient=5.67e-8)` to `stepper._config.combined_fluxes` after `load_stepper` works without
editing the checkpoint (`hc/capture.py --deleak`). At step 0 the ocean's FLUS input equals 5.67e-8·sst⁴ (median
ratio 0.9997, the open-water scaling being the only difference), and every other channel is bit-identical to the
control. Control vs de-leaked FLUS: 5.6 W/m² RMS at step 0, 7.3 at step 1; ocean means 323.8 vs 324.5 W/m².

## 2. C1A0's data (E3SM truth)

| | |
|---|---|
| Run | E3SMv3 `v3.LR.historical_0101.aigo`: the historical member 0101 rerun with online emulator output. Historical to 2014, then the scenario extension. CO2 normalizer 366 ± 51 ppm. |
| Path | `$PSCRATCH/v3.LR.historical_0101.aigo/run` (7.0 TB), plus `$PSCRATCH/e3sm-hist-inputs/{landfrac5d,momfrac5d}` |
| Years on disk | 1940-01 to 2065-01, contiguous |
| Atmosphere | `*.eam.h0.YYYY-MM.nc`: monthly files of **6-hourly** instantaneous fields, 180×360 Gaussian grid, 80 model levels plus the coarsened T_0..7, U_0..7, V_0..7, STW_0..7 |
| Ocean | `*.mpaso.hist.am.fmeDerivedFields5D.*`, `fmeDepthCoarsening5D.*` and `*.mpassi.hist.am.fmeSeaiceDerivedFields5D.*`: **5-day means**, 19 layers to 6380 m |
| Training split (E35 `train_loader`) | **1940-01-06 to 1990-01-01 and 2000-01-06 to 2040-01-01** |
| Validation (E35 `validation`) | 1990-01-06 to 1995-01-01; also selected the best inference checkpoint (`heldout_1990s`, weight 1, from 1990–1994) |
| Never used for training or selection | **1995–1999 and 2040–2064.** (`future_2040` ran as an inference monitor with weight 0, so it was not used for selection.) |
| Variables checked present | SST (`sst`, 5-day mean; EAM `TS` 6-hourly), FLUS, FLDS, FSNS, LHFLX, SHFLX (MPAS `longWaveHeatFluxUp/Down`, `shortWaveHeatFlux`, `latentHeatFlux`, `sensibleHeatFlux`, sign-flipped in the loader as in training; EAM 6-hourly too), upper-ocean T (`temperatureCoarsened_0..18`), TAUX/TAUY, PRECT, FLUT, FSUTOA, SOLIN, PS, T_0..7, U_0..7, V_0..7. **All present.** Same grid as C1A0, so no regridding. |

**Consequences**
- The laptop states (1985 and 2015) and the two release ICs are in **training** years. Every scored result
  below uses 1995–1999 and 2040+ (held out). 1985 and 2015 serve only to reproduce the laptop numbers and as a
  "training-year" contrast.
- Ocean SST in the data is a 5-day **mean**. At ocean step j → j+1 the network reads sst[j] (the previous
  window's mean) and the next window's flux means FLUS[j+1] etc., and predicts sst[j+1]. F1 compares FLUS[j+1]
  with εσT⁴ for the previous-window mean, the target-window mean and the instantaneous window-start and
  window-end SST (EAM `TS` over open ocean, 6-hourly).
- In coupled mode FLUS is the atmosphere's own (driven by the ocean's current SST), so it is already close to
  start-of-window emission. C1A0's ocean was fine-tuned in coupled mode for 25 epochs after E11's truth-forced
  training and still shows the −2.04 slope.

## 3. Other SamudrACE checkpoints (used only in Idea 1 step 4 and Idea 3 step 4)

| Name | Path | md5 | How it differs from C1A0 |
|---|---|---|---|
| C0A0 (`noCO2-noaero.tar`) | release `checkpoints/` | `44585e626f9322cd6f6965d2462c498e` | E30-CFT S03: no CO2 input; atmosphere E01-FT S03; ocean **E11-FT S03 (another seed)** |
| C1A1 (`CO2-aero.tar`) | release `checkpoints/` | `e6f9d00fccd0b95ac7cdae1e3b27edf6` | E31-CFT S01: CO2 + aerosol inputs; atmosphere **E05-FT S01** (= hackathon E05); ocean **E11-FT S01, same as C1A0's** |
| Coupled seeds | `$PSCRATCH/aug26-cft/E30-CFT…S01..S03`, `E31-CFT…S01..S03`, `E35-CFT`, `E40-CFT S01/S02`, E19/E21/E24/E34/E37/E38 | best_inference_ckpt | seed and option variants |
| Ocean-only | `$PSCRATCH/aug26/E11…S01..S03`, `aug26-ft/E11-FT…S01..S03` | | truth-forced ocean, three seeds (pre- and post-fine-tune) |
| Ocean variants | `aug26/E12, E13, E14, E16` (loss weights W1–W4), `E17` (**1-day ocean**, O1) | | independent ocean trainings, so the best test of whether the shortcut generalises |
| Atmosphere-only | `aug26-ft/E01-FT…, E02-FT…, E05-FT…` etc.; `aigs-hack-sep26/hf-models/E05-FT-S01` | | for Idea 3 step 4 |

**Independence note.** C1A0 and C1A1 share an ocean init (E11-FT S01), and the hackathon E11 archive is that
same ocean. Only E11/E11-FT S02 and S03, C0A0 (S03) and E12–E17 are independent ocean trainings.

## 4. Latent archives

`$PSCRATCH/aigs-hack-sep26/archives` (also the HF cache `datasets--E3SM-Project--aigs-hack-sep26-latents`,
snapshot `99f32225…`). Each run folder has `step_LL.npy` per layer, `reference.nc` (physical fields),
`grid.npz`, `manifest.json` and `bases/` (PCA and SAE).

| Runs | Model | Start | Times | Layers |
|---|---|---|---|---|
| atm-E05S01-ctrl, -ctrl-seed1..4, co2x0.8, co2x1.25, sst+4K, aerfrz, steer-*, skip-* | E05-FT S01 atmosphere (SST prescribed) | 2015-01-03 12:00 from the **emulator's own century run** (not truth) | 20 (5 days) | 0 = encoder, 1–8 = blocks |
| atm-E05S01-ctrl-30d | same | same | 120 | 0, 2, 4, 6, 8 |
| ocn-E11S01-ctrl, ocn-E11S01-flusfrz | E11-FT S01 ocean, fluxes prescribed from E3SM | 2040-01-01 (held out for C1A0) | 20 (100 days) | 0 = first encoder block, 1 = last decoder block |

## 5. Environment and compute

- `setup_env.sh` rebuilds `.venv`: Python 3.12, torch 2.14.1+cu130 (the release's uv.lock pin), `git archive` of
  the ace tag (commit 1c29e2d9b) into `ace-tag/`, installed editable; exege `mahf708/circuits` (7fe6dc9) editable;
  `lib/` (package `hc`) editable, which registers the `samudra-replay` exege adapter entry point.
- Shared library `lib/hc`: `paths.py` (all paths, config-driven), `graph.py` (= cross-flux `common.Graph`),
  `samudra_adapter.py` (= `SamudraReplay`), `lawful.py`, `capture.py` (official fme path plus ocean capture,
  `--deleak`).
- Slurm accounts available: e3sm_g, m5451_g (GPU). Interactive work runs in `salloc --no-shell` allocations
  driven with `srun --jobid`.
