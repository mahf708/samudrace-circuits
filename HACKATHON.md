# Hackathon quickstart: SamudrACE C1A0 circuits, laws and errors

You (or your agent) can be running real C1A0 experiments against E3SM truth within ~20 minutes. Read this page,
then `AGENTS.md` (layout, libraries, pitfalls), `PREREG.md` (the rules) and `REPORT.md` (what is known). Results
page: https://portal.nersc.gov/project/m5451/mahf708/samudrace-circuits/

## 1. Set up (Perlmutter)

```bash
git clone https://github.com/mahf708/samudrace-circuits.git && cd samudrace-circuits
bash setup_env.sh          # ~10 min: .venv (py3.12, torch 2.14.1+cu130), fme @ ace tag e3smv3-lr-hist-samudrace-rc1,
                           # exege (circuits + laws + samudra adapter), this repo's lib as package `hc`
bash fetch_data.sh         # on Perlmutter: read-only symlinks to the shared captures; elsewhere: HF download
source .venv/bin/activate
```

Everything heavy reads from `/pscratch/sd/m/mahf708` (world-readable), and needs no copying:
- C1A0 and the other release checkpoints;
- the 7 TB E3SM truth run;
- C1A0's training configs.

Paths live in `lib/hc/paths.py`; override them with `HC_*` environment variables.

**GPU.** One A100 is plenty for everything here.

```bash
salloc -N 1 -C gpu -q shared_interactive -A <your_account> --gpus-per-node=1 -c 32 -t 02:00:00 --no-shell
srun --jobid <JOBID> --overlap -n1 --gpus=1 python ...
```

On shared nodes `srun` can hang on "Required ports are in use". If it does, run your work through one long-lived
step, or a batch job, using `lib/runner.sh`.

**Off Perlmutter.** C1A0 is at HF `mahf708/e3smv3-lr-hist-samudrace@rc1`. The captured states and truth extracts
are at HF dataset `mahf708/samudrace-circuits` (`fetch_data.sh` downloads them). Set `HC_RELEASE` and
`HC_CKPT_C1A0`. The full E3SM truth is not on HF, so experiments that need new truth states must run on Perlmutter.

## 2. Check it works (5 minutes)

```bash
python phase0/flus_slope.py data/states/ic1985 4   # C1A0 FLUS slope on 1985: ≈ −2.04 K per 100 W/m2
python -c "from exege.circuits.contract import open_patchable; from hc import paths; \
a = open_patchable('data/states/t2045', 'samudra', checkpoint=paths.CKPT['C1A0'], device='cuda'); print(a.graph().senders)"
```

### A 15-minute walkthrough: truth-started run to a scored number (tested, timings measured on one A100)

```bash
python -m hc.evaluate runs/demo --start 2045-01-01 --steps 4 --capture   # ~6.5 min: ~6 min is dataset indexing,
                                                                          # each 5-day coupled step ~3.5 s after that
python -m hc.meta runs/demo                                               # official ocean mask for the captures
python examples/walkthrough.py runs/demo 4                                # ~30 s
# -> runs/demo/walkthrough.json: FLUS slope ≈ -1.82 K per 100 W/m2, skip.0 share ≈ 0.85
```

`examples/walkthrough.py` is ~80 lines and shows the whole pattern:
- `hc.set_numerics()` (fp32);
- the fixed ocean mask;
- a physical-unit perturbation through `hc.graph`;
- an edge patch through the exege `samudra` adapter;
- a JSON with the result.

Copy it for your own experiment. A 73-step (1-year) truth-started run took 7 min (431 s), or 11 min (649 s) with `--capture`, which writes every ocean input.

## 3. What is ready to use

| Want | Use |
|---|---|
| Ocean inputs C1A0 actually saw, held-out years | `data/states/t1995`, `data/states/t2045`: 73 × 5-day steps of coupled runs started from truth (`x_###.npy`, normalized, land = 0), plus fme predictions **and truth targets** in `fme/ocean/` |
| Truth-forced ocean inputs, any date | `hc.truth_ocean.build(stamp)` → (input, truth target); stamps = multiples of 5 noleap days since 1940-01-01 |
| A coupled run from truth on any date (held-out years) | `python -m hc.evaluate OUT --start 2045-01-01 --steps N [--capture] [--capture-atm K] [--deleak] [--seed S]` |
| The de-leak switch | `--deleak` = `CombinedFluxConfig(name="FLUS", terms={"FLUS": 0.0}, sst_emission_coefficient=5.67e-8)` appended at load time |
| Atmosphere truth states, 12 dates (+ the same dates −5 d) | `data/states/atm12`, `data/states/atm12m5` (`atm_x_00.npy`: batch of 12, normalized, exact fme packing) |
| Exact U-Net patching | `hc.graph.Graph` (edges D0–D3, B, U1–U3, S0–S3), `idea1_flus/localize.py` (layer-0 sub-edges), exege `samudra` adapter (`Patchable`, `EdgeGradients` for EAP) |
| Truth per-window fields for the leak (SST, fluxes, EAM TS start/end/mean) | `data/f1/year_YYYY.npz` (1985–89, 1995–99, 2040–49) |
| Per-degree truth persistence | `idea3_gain/results/truth_transfer.json` |
| JVP on the atmosphere with fixed noise | `idea3_gain/gain_atm.py` (`f_out`, `make_noise`): copy the pattern |

## 4. Tracks: open questions with a head start

Each track lists what is known, the first command, and a rough cost.

**T1. Does de-leaking FLUS change the coupled climate? (Idea 1 step 2; PREREG 1.2)**
- One-step: input replacement shifts the next SST by +0.17 K, but the coupled FLUS is already close to
  start-of-window emission (difference ~6 W/m² RMS).
- Start: `python -m hc.evaluate runs/ctl_1995_s0 --start 1995-01-01 --steps 365 --seed 0 --mem 5 --monthly`
  and the same with `--deleak`. Do this for ICs 1995-01-01, 2040-01-01 and 2045-01-01 and seeds 0 and 1.
  Do not pass `--no-ocn-pred`: the ocean predictions and truth targets are written every step (about 1.5 GB per
  year), and scoring needs them.
- Cost (estimated, not measured): about 25 min per 5-year run on one A100, based on 1-year runs of ~4 min plus
  ~5 min of dataset indexing. Test a 73-step run first.
- Score against `autoregressive_target` (or the truth files): SST bias and drift, seasonal cycle, Niño3.4
  variance and spectrum, sea ice extent, 0–300 m heat content. Use the preregistered "same/better/worse" rule.

**T2. Do explanations keep the emulator's real errors? (Idea 2; PREREG 2)**
- **Starter (tested, ~1 min on one GPU):** `python idea2_errors/strata.py data/states/atm12 --layer 7`.
  - It runs one-step forecasts from 12 truth states with 4 seeds and scores them against E3SM truth at t + 6 h
    with the preregistered τ (K = 2).
  - Explanations included: seed ceiling, PCA-k at an SFNO block (k = 0 is the empty explanation), and the
    scalar-offset and affine-persistence controls.
  - Writes `results/strata_L7_K2.{json,md}`. Add explanations to `EXPLANATIONS`.
- **First results** (2047 score dates, pre-corrector):
  - Seed ceiling A_err 0.77–0.82.
  - PCA-128 at block 7 reaches A_err 0.88 (T_7), 0.87 (FLUT), 0.56 (FSUTOA).
  - The output-level controls stay ≤ 0.28.
  - **PS is "error" on 100% of nodes and precipitation on 68%: the corrector column is missing.** Add it first.
- Still missing: the corrector in/out column (use fme's corrector on the network output), more dates, the block
  bootstrap, the ocean (`hc.truth_ocean.build` gives truth-forced inputs and targets), and the pre-corrector strata.
- Error = model − truth at the same time.
- τ = max(2·σ_seed, 1% of the spatial SD).
- Explanations: ocean skip.0-only (patch the deep path to its fit-date mean), atmosphere blocks-only / big-skip-only,
  PCA-k at SFNO blocks.
- Report A_ok, A_err, the three-way split and the scalar-offset control.
- Start: `hc.evaluate --steps 1 --capture-atm 1 --seed {0..3}` on many dates (batch them as several `--start`).

**T3. Does the one-step gain predict where rollouts lose small scales? (Idea 3, rollout link)**
- **Starter (tested, CPU, seconds):** `python idea3_gain/rollout_spectra.py` writes
  `idea3_gain/results/rollout_spectra.{json,png}`, the rollout/truth power ratio per l over months 7–12.
  First result:
  - T_4 and T_7 power matches truth at all l (±6%).
  - Winds lose small-scale power: UV_4 ×0.83–0.85 and UV_7 ×0.89–0.90 at l = 80–89.
  - This fits the balanced-probe result: the one-step amount of each scale is right while the pattern
    decorrelates.
- 1-year rollouts already recorded daily pre-corrector spectra: `data/states/t1995/atm_spectra.npz`,
  `data/states/t2045/atm_spectra.npz`, `data/rollout1y/t2040/atm_spectra.npz` (T_4, T_7 scalar; (U,V)_4, (U,V)_7
  vector-SHT power per l).
- Truth spectra: `idea3_gain/truth_transfer.py transfer` writes `truth_spectra.npz`.
- Compare months 7–12 against the balanced-probe projection curves.

**T4. Why does coupled fine-tuning shrink the thermometer 3.5× and fix LHFLX's sign?**
- E11-FT S01 (−6.97) and C1A0 (−1.84) share an initialisation, so diff them with `idea1_flus/localize.py` and the
  adapter's `EdgeGradients`: which edges and channels changed?
- Checkpoints are listed in `idea1_flus/README.md` and `INVENTORY.md`.

**T5. Small, sharp questions**
- Why does SHFLX have the wrong sign under 10° boxes (−0.14) but not uniformly (+0.13)?
- Why do T_1 and PS small scales persist more than in truth?
- Does the balanced-probe result hold for other perturbation directions (seasonal anomaly, analog dates)?

**T6. exege features**
- Use the `samudra` adapter with `exege.circuits` (EAP, greedy pruning) and `exege.laws` on the ocean.
- PR: https://github.com/mahf708/exege/pull/12.

## 5. Rules that will save you a day (from PREREG and our own failures)

- Score on **1995–1999 / 2040–2049** only. 1985 and 2015 are training years.
- Fit on one set of dates, score on another; use size-matched random controls; use at least two perturbation
  families (uniform and box move InstanceNorm differently).
- Quote ratios between fluxes, not absolute "lawfulness". The mixed-layer depth rule moves absolute gains.
- Linearity collapses evidence: sufficiency = necessity and skip + deep ≈ 1 are one number.
- Call `hc.set_numerics()` (fp32) before any perturbation or gradient work.
- Every claim gets a control and a caveat; add yours to `results/manifest.json` via `web/build.py`.

## 6. A prompt to hand your agent

> You are working in the samudrace-circuits repo on Perlmutter. Read HACKATHON.md, AGENTS.md, PREREG.md and
> REPORT.md first. Use `hc` (lib/hc) for paths, captures, truth-forced inputs and exact patching; use the exege
> `samudra` adapter for edge patching and gradients. Score only on 1995–1999 and 2040–2049, run network experiments
> after `hc.set_numerics()`, and report every number with its control, caveat, JSON path and the command that made it.
> Task: <track and question>.
