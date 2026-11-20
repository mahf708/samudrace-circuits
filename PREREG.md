# Preregistration (written 2026-11-19, before any Phase 1 result)

Applies to C1A0 unless stated. Do not change any of this after seeing results. If a change is unavoidable,
append it to the change log at the bottom with the date and the reason, and say in REPORT.md which results it
touches.

## 0. Rules common to all three ideas

**Splits**
- Every **scored** result uses only the never-trained, never-validated years: **1995–1999 and 2040–2049**.
- Anything fitted (channel sets, PCA/SAE bases, ε, regressions, damped-linear coefficients, baselines) is fitted
  on **fit years** and scored on **score years**:

  | Block | Fit years | Score years |
  |---|---|---|
  | 1990s | 1995–1997 | 1998–1999 |
  | 2040s | 2040–2044 | 2045–2049 |

  The validation years (1990–1994) and training years (1985, 2015) are reported only as labelled contrasts
  ("in-distribution") or to reproduce laptop numbers.

**Numerics and corrector**
- Perturbation, patching and JVP experiments run in fp32 with TF32 off (`hc.set_numerics`).
- Rollouts run as deployed (fme defaults, TF32 on).
- The ocean is scored as the bare network; its corrector never touches sst or temperature.
- Atmosphere law and error tests use pre-corrector output (`RecordingCorrector`), with the post-corrector
  number beside it ("corrector in/out").

**Seeds**
- Every contrast is same-seed paired, and its noise yardstick is a same-seed floor.
- Seeds 0–3 are used wherever a seed spread is needed.

**Reporting**
- Area weights are cos(lat) on the Gaussian grid.
- The ocean mask is the fixed one: official sst output finite (`hc.graph.fixed_ocean_mask`). Never
  `sea_surface_fraction > 0.5` on land-filled inputs.
- "Open ocean" = fixed mask and input sea-ice fraction < 0.01.
- Every number is reported with its spread over states/dates (mean, min, max, or 5–95%), and every bound says
  whether it is a mean or a maximum.
- Uncertainty is a block bootstrap: blocks are contiguous runs of dates within one (year, season), 1000
  resamples, 95% percentile intervals.

**Controls**
- At least two corruption families everywhere.
- Size-matched random controls (same layer, same edge type, same number of channels or same area).
- Circuit and channel-set inclusion frequencies over resampled dates.

**Skeptic pass.** Before any headline number goes in REPORT.md, an independent implementation or a fresh
subagent working from code and JSON alone tries to break it. Claims that fail are downgraded and listed.

## 1. Idea 1: FLUS

### 1.1 F1: the leak in truth (CPU, E3SM data only)

**Data.** Ocean 5-day means plus EAM 6-hourly `TS` and FLUS, 1995–1999 and 2040–2049 (score). Fit years as in §0.

**(a) Alignment.** Fit ε by least squares on the fit years, separately for each SST definition, over open
ocean. On the score years, report the RMS and the per-cell R² of FLUS[j+1] − εσT⁴ for:
- T = sst[j], the previous-window mean (what the ocean reads);
- T = sst[j+1], the target-window mean;
- the window-mean T⁴ from 6-hourly TS;
- instantaneous TS at window start;
- instantaneous TS at window end.

Report global numbers and maps, both for the ocean dataset's FLUS (MPAS, what training fed) and EAM's FLUS.

**(b) Conditional slope in truth.** Per open-ocean cell, regress ΔSST = sst[j+1] − sst[j] on
[FLUS, FLDS, FSNS, LHFLX, SHFLX][j+1], sst[j] and annual + semiannual harmonics, by OLS on the score years.
- Sign convention: K per 100 W/m² of **warming** flux (FLUS enters with −1).
- Also fit (b′): the same regression without FLUS, then regress its residual on the FLUS residual (Frisch–Waugh).
  This is a restatement of (b) and will be labelled as such.
- HAC (Newey–West, lag 6) standard errors.
- Report the area-weighted global mean slope, the fraction of open-ocean area with the "thermometer" sign
  (raising FLUS warms), and maps.

**(c) Comparison with the model.** Use the model's per-node input-response slope maps from §1.3 (uniform shift)
on the score-year states.

**Verdicts, fixed now**
- **"Leak supported"** if all three hold:
  - the truth conditional slope has the thermometer sign over > 75% of open-ocean area;
  - the model/truth global slope ratio is in [0.33, 3];
  - the area-weighted pattern correlation of the model and truth slope maps is > 0.3.
- **"Leak refuted"** if the truth conditional slope has the lawful sign (raising FLUS cools) over > 50% of
  open-ocean area.
- Otherwise **"something else to explain"**, reported with the numbers.
- The alignment test (a) is supporting evidence only: end-of-window better than start-of-window by > 20% in RMS.

### 1.2 F3: coupled de-leak

**Arms.**
- Control (C1A0 as released).
- De-leaked: `CombinedFluxConfig(name="FLUS", terms={"FLUS": 0.0}, sst_emission_coefficient=5.67e-8)`.

**ICs.** Truth states at **1995-01-01, 2040-01-01 and 2045-01-01**, through the coupled evaluator from the
E3SM dataset, plus the two release restart ICs (1985 and 2015, labelled training-year). **2 seeds (0, 1) per IC
per arm**, seed-paired across arms. Each run is **5 years** (365 coupled steps). If cheap, add 2050-01-01 and
seeds 2–3.

**Scoring against truth, years 1–5.** Seasonal means and anomalies use the truth climatology of the same years.
- Global and regional SST bias: tropics 20S–20N, Niño3.4, N Atlantic 40–60N, Southern Ocean 40–60S.
- Drift: linear trend of the global-mean SST error.
- Seasonal-cycle RMSE.
- Niño3.4 anomaly variance and its spectrum (Welch, years 1–5).
- NH and SH sea-ice extent (area with fraction > 0.15).
- Upper-ocean heat content, 0–300 m: layers 0–7 thickness-weighted temperature.

**Verdict per metric.** "Same" if |de-leak − control| < 2 × the paired-seed standard error. Otherwise "better"
or "worse" by distance to truth. The headline is the count of metrics in each class. No metric is promoted
after seeing results.

### 1.3 F2/F3: network localisation and ablation (bare ocean network, fp32)

**States**
- Primary (held out): ocean inputs in the coupled run from truth ICs at 1995-01-01 and 2045-01-01. Captured
  every step for 73 steps (1 year), stride 4 for patching (19 states per year).
- Secondary: truth-forced inputs (truth ocean state j plus truth fluxes j+1, exactly what E11 trained on) at the
  same dates, built from the E3SM dataset.
- Laptop reproduction: 1985 captured states.

**Perturbation families (both always)**
- **Uniform**: open-ocean shift.
- **Box**: 10×10° open-ocean box. 16 preregistered box centres on a fixed lat/lon lattice, keeping only those
  ≥ 95% open ocean on that date: lat ∈ {−45, −25, −5, 15, 35}, lon ∈ {30, 110, 190, 270}.
- dQ ∈ {−100, −25, 25, 100} W/m² of warming flux, for FLUS, FLDS, FSNS, LHFLX and SHFLX. Slopes are through the
  origin.
- **Third corruption family (patching only):** analog-date replacement of the FLUS channel by its value
  6 months later in the same year.

**Metrics (both always)**
1. Perturbed-area mean dSST: area-weighted, over the perturbed area.
2. Projection: ⟨response, full response⟩ / ⟨full, full⟩ over open ocean.

Quote every gain with perturbed area, metric, year and MLD rule.

**MLD rules (sweep, all reported):**
- 0.2 K criterion, 20 m floor (laptop default);
- 0.5 K criterion, 20 m floor;
- fixed 33.1 m (the laptop's FLDS fit);
- fixed 50 m.

Lawfulness is quoted only as flux ratios (X/FLDS) and signs, never as absolute gain.

**Localisation**
- Layer-0 ConvNeXt split into edges: depthwise conv, MLP (pointwise convs + norm + activation), residual /
  1×1 `skip_module`.
- Per input channel: patch the perturbed input to the clean input for one channel group at a time (sst, FLUS,
  temperatureCoarsened_0, rest) at the layer-0 input, through each sub-edge.
- Report each edge's share of the full response under both metrics and both perturbation families.
- Shares come from exact patching (clean → corrupted). Because the network is nearly linear in dQ,
  sufficiency = necessity and skip + deep ≈ 1; these are reported once, not as separate evidence.

**Ablation**
- (i) Remove the FLUS input direction: replace FLUS by 5.67e-8·sst[j]⁴ times the open-water fraction, i.e. the
  de-leak value, at the input.
- (ii) Zero-ablate (mean-ablate) the greedy channel set at the best localised place. The set is selected on fit
  states (1985 captured + 1995 fit-year states) by greedy FLUS-response share, sized k ∈ {10, 32}, and scored
  on score-year states. Controls: 20 random size-matched channel sets at the same place.
- Re-score every flux family after each ablation. Success = the FLUS slope's sign flips to lawful or its
  magnitude falls below 25% of the original, **and** the FLDS and FSNS slopes stay within ±25% of their
  unablated values, **and** both random-set controls do not achieve the same.

### 1.4 Replication (only after 1.1–1.3)

FLUS, FLDS, FSNS, LHFLX and SHFLX uniform and box slopes, perturbed-area mean metric, on 1995 truth-forced
inputs, for:
- E11 S01–S03 and E11-FT S01–S03;
- E12, E13, E14, E16, E17;
- C0A0, C1A1, E30-CFT S01–S03, E31-CFT S01–S03.

Report as a table. "Has the shortcut" = FLUS slope with the thermometer sign over > 75% of open-ocean area.

## 2. Idea 2: errors

### 2.1 Error strata

**Atmosphere**
- One-step 6 h forecasts from truth states.
- Dates: score blocks of §0. In each year, the 15th of Jan, Apr, Jul and Oct at 00 and 12 UTC, 8 ICs per year;
  20 score-years-equivalent split into fit and score as §0.
- 4 seeds per IC.

**Ocean**
- One-step 5-day forecasts from truth-forced inputs, deterministic, on every 6th window of each year.

**Error and tolerance**
- Error e = model − truth at the same time, per node and variable.
- Tolerance per node and variable: τ_k = max(k · σ_seed, ε_abs).
  - Atmosphere: σ_seed is the SD over the 4 seeds of the model output at that node (same IC), so the floor
    matches the seed pairing.
  - Ocean: σ_seed is the SD of the 5-day ocean output across 4 atmosphere seeds in coupled mode from the same
    truth IC.
  - ε_abs is 1% of the variable's global spatial SD.
- **Primary k = 2; sweep k ∈ {1, 2, 3}** reported in full.
- Night FSUTOA uses the measured night seed SD (laptop 2.6 W/m², remeasured here).
- Node = "error" if |e| > τ, else "ok". Everything area-weighted, ocean variables masked.

### 2.2 Explanations

- **(a) Ocean skip.0 only.** The decoder reads skip.0 from the clean run; the deep path (upsampled stream into
  level 0) is patched to a baseline.
  - Primary baseline: mean ablation, the deep-path value on the time-mean input over fit dates.
  - Secondary: analog date (same season, other fit year).
- **(b) Atmosphere blocks only / big skip only.** The other path is patched to the same two baseline families.
  Corrector in and out.
- **(c) Latent reconstruction.** PCA-k at SFNO block 4 and block 7 outputs, k ∈ {8, 32, 128}, fit on fit-year
  latents and scored on score years. Plus an SAE at block 7 if one is fitted on C1A0 fit-year latents (TopK,
  1536 features, k = 32).

### 2.3 Scores per explanation (per variable, region, season)

- A_ok = P(|y_expl − y_model| ≤ τ | ok)
- A_err = P(|y_expl − y_model| ≤ τ | error)
- Three-way split on error nodes:
  - reproduces: |y_expl − y_model| ≤ τ;
  - corrects: not reproduces and |y_expl − truth| ≤ τ;
  - other.
- Controls:
  - scalar offset: y_base + c, with c fitted per variable on fit dates;
  - per-component gain: y_base + g·(y_model − y_base), g fitted per variable on fit dates;
  - empty explanation: everything patched to baseline;
  - seed ceiling: another seed's output as the "explanation".
- Bootstrap as §0.
- Headline variables: TS, T_7, U_7, PS, PRECT, FLUT, FSUTOA for the atmosphere; sst and temperatureCoarsened_0
  for the ocean.

### 2.4 Pre-corrector strata

Same ICs, pre-corrector atmosphere output:
- raw PRECT < 0;
- raw FSNS < 0;
- raw FSUTOA < 0;
- night FSUTOA < −τ_night, with night = SOLIN = 0.

Report A_ok and A_err of the same explanations on these strata.

## 3. Idea 3: one-step gain per SH degree

**States**
- Atmosphere: truth states, 4 seasons (15 Jan/Apr/Jul/Oct, 00 UTC) × 3 years × 3 seeds.
- Years: 1996, 2042, 2047, all score years.
- Ocean: truth-forced inputs, same 12 dates.

**Operator**
- JVP of the network map (input → output, pre-corrector), linearised at each state with the noise draw fixed
  (seeded), in physical units: input perturbation in K, m/s, Pa; output in the same.
- Probes: band-limited random fields, 8 per degree band, generated in spectral space with flat amplitude in
  degree band l. Bands: l = 0–2, 3–5, then width 4 to l = 89.
- Gain(l) = ‖P_l J δ_l‖ / ‖δ_l‖, area-weighted L2.

**Pairs**
- Diagonal: T_1, T_4, T_7, U_1, U_4, U_7, V_4, V_7, PS, TS.
- Cross pairs: T_7 → U_7, U_7 → T_7, PS → U_7, TS → T_7.
- Ocean: sst, temperatureCoarsened_0 and _5, ssh.
- Noise gain: the JVP with respect to the noise input, same norms.

**Linearity check.** JVP vs finite difference at ε ∈ {0.01, 0.1} of the per-band SD; must agree within 10%, or
the band is flagged.

**Baselines**
- (1) Persistence: gain 1.
- (2) Truth one-step spectral transfer: per l, the regression slope of truth anomaly coefficients at t + Δt on t
  (anomaly about the fit-year daily climatology), fitted on fit years and evaluated on score years. Plus the
  ratio of truth to model tendency spectra.
- (3) Damped linear: per-l AR(1) coefficient fitted on fit-year truth.
- Rollout link: 1-year (atmosphere-in-coupled) rollouts from the 3 truth ICs. Compare KE and T_7 spectra
  averaged over months 7–12 with truth. Test whether the l where gain falls below the truth transfer predicts
  the l where rollout spectra fall below truth (Spearman over pairs and checkpoints).

## Change log

- **2026-11-19 (before any Phase 1 result), §1.3 layer-0 edges.** Samudra's layer-0 block has no depthwise conv.
  It is `skip_module` (1×1 conv, no norm) + convblock (3×3 conv → InstanceNorm → GELU → 3×3 conv →
  InstanceNorm → GELU → 1×1 conv). The edges are therefore `skip_module` and `convblock`, crossed with the U-Net
  paths skip.0 and deep. **Added** (not replacing anything): a `convblock` edge with the InstanceNorm mean/var
  frozen at their clean values, to test the InstanceNorm-statistics explanation of box vs uniform. Layer 0 is the
  network's only reader of the input, so "per input channel" means per perturbed input (FLUS, FLDS, FSNS, LHFLX,
  SHFLX, sst, sst + temperatureCoarsened_0).
- **2026-11-19, §1.3 box lattice.** The text said 16 centres, but the listed lattice is 5 lat × 4 lon = 20. The
  listed lattice is used: all 20 centres, kept if ≥ 95% open ocean that date.
- **2026-11-19, §2.1 and §3 dates.** Coupled-evaluator ICs must sit on the ocean's 5-day stamps. "15th of
  Jan/Apr/Jul/Oct" becomes the nearest stamps: Jan 16, Apr 16, Jul 15 and Oct 13 (00 UTC).
- **2026-11-19, §3 ocean gain.** The ocean is deterministic, so "3 seeds" applies to the atmosphere only.
- **2026-11-20, §3 (exploratory addition, made AFTER seeing the single-variable gains).** Single-variable,
  single-level probes are unbalanced, and the model rejects part of them. Both the atmosphere (T_7 at l ≤ 2:
  0.49 vs truth persistence 1.00) and the ocean (sst alone 0.22 vs sst + 0–20 m 0.50) show this. **Added** a
  balanced probe: the truth 5-day state difference x(t) − x(t − 5 d), over all prognostic input channels together
  (forcings and fractions held fixed), band-passed per SH band, as the JVP tangent. Gain per variable =
  √(in-band output power / in-band input power). It is directly comparable to the truth per-degree AR(1)
  transfer. Labelled exploratory everywhere; the preregistered single-variable curves are still reported.
- **2026-11-20, §3 rollout link: not done in this phase.** It is deferred to Phase 2, where the multi-year
  control rollouts provide the spectra. The 1-year runs from 1995, 2040 and 2045 recorded daily spectra and are
  kept.
