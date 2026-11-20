# SamudrACE C1A0: three ideas finished at scale (state at 2026-11-20, before the hackathon)

Model: SamudrACE **C1A0** (HF `mahf708/e3smv3-lr-hist-samudrace@rc1`, md5 `a5082a9aec1243e37e0820a99e77d059`).
Truth: the E3SMv3 run it was trained on (`v3.LR.historical_0101.aigo`). **Scored only on years the model never
saw: 1995–1999 and 2040–2049.** Rules were fixed before results in `PREREG.md`; every deviation is in its change
log. Every number below is in `results/manifest.json` with its JSON, script and command. Results page:
https://portal.nersc.gov/project/m5451/mahf708/samudrace-circuits/

## One page: headline findings

| # | Finding | Control | Caveat |
|---|---|---|---|
| 1 | **In E3SM truth, the ocean's FLUS input is a near-exact thermometer of the SST it must predict.** Per open-ocean cell, ΔSST regressed on the window's fluxes + previous SST: conditional FLUS slope **−17.0 K per 100 W/m²** of warming flux (2045–49; −17.2 in 1998–99, −17.2 in 1988–89), thermometer sign on 99.5% of open ocean, per-cell R² median 0.998. The literal value 1/(4εσT³) is −17.3. | Given FLUS, the other four fluxes have conditional slopes ≈ 0 (medians < 0.003 in magnitude). Frisch–Waugh restatement agrees to 5e-9. | The preregistered *alignment* check (end-of-window beats start-of-window by > 20% pooled RMS) is **not met** (22.21 vs 22.28 W/m²). A spatial offset between MPAS FLUS and σT⁴ dominates pooled statistics; per-cell R² favours the target window (0.770 vs 0.744). |
| 2 | **C1A0 does not copy that relation.** Model FLUS slope **−1.84** (2045 states) / −1.86 (1995), **0.11×** truth, pattern correlation **0.03** / 0.07 (−0.02 without ice-shelf cavities). Its FLDS/FSNS responses keep the law's sign (+0.32/+0.29) where truth's conditional slopes are ≈ 0. Preregistered verdict: **"something else to explain"**. | Verdict rule fixed in PREREG 1.1 (thermometer area > 75%, ratio in [0.33, 3], r > 0.3). | Model states are coupled-run inputs from truth ICs; uniform shifts for this comparison. |
| 3 | **Training decides the shortcut's strength.** All 18 ocean checkpoints have it (thermometer area ≥ 0.988). Ocean-only training on leaky FLUS: E11 seeds −4.0…−4.2; more ocean-only fine-tuning (E11-FT): **−6.9…−7.1**; after coupled fine-tuning (FLUS from the atmosphere, not leaky): C1A0/C1A1/C0A0 and E30/E31 seeds **−1.7…−2.5** K per 100 W/m² (1995 truth-forced inputs). Coupled fine-tuning also turns LHFLX from the wrong sign (−0.05…−0.56) to the lawful one (+0.03…+0.08) in every case. | Loss-weight variants E12–E16: −4.1…−5.5, in line with ocean-only. | Not all independent: C1A1 shares C1A0's ocean init; E11-FT S01 is C1A0's ocean before coupled fine-tuning. E17 (1-day ocean) excluded. |
| 4 | **Where it lives, below "skip.0 + layer 0".** skip.0 carries **0.86** of the uniform FLUS response (box 0.91) on 2045 and on 1995 states. Inside layer 0, FLUS splits **0.53 / 0.47** between the linear 1×1 `skip_module` and the InstanceNorm-ed convblock (box 0.35/0.65), whereas an SST perturbation goes **1.00** through the 1×1 `skip_module`. | **Skeptic pass:** exege's new `samudra` Patchable adapter (independent implementation, EdgePatch semantics) gives skip.0 share **0.854**. Projection metric agrees within 0.05. | Near-linear response: sufficiency = necessity and skip.0 + deep ≈ 1 are one result. |
| 5 | **InstanceNorm statistics account for box vs uniform, for FLUS.** Freezing the convblock's IN statistics raises its uniform share 0.47 → 0.94 and leaves the box share unchanged (0.65); predicted un-normalised uniform slope −2.65 vs measured box −2.58. | 1995 replicates to ±0.01. | Does not hold for FLDS (prediction 1.34×, measured 1.07×). Box/uniform is 1.43×, not the laptop's ~2×. Under boxes SHFLX has the wrong sign (−0.14). |
| 6 | **Removing the shortcut:** not at layer-0 channel granularity (the greedy 32 channels that carry 92% of FLUS also carry FLDS: FLUS ×0.08, FLDS ×−0.06); at the input yes (FLUS := 5.67e-8·sst⁴ keeps FLDS ×1.00, FSNS ×1.02) at a cost of **+0.17 K** in the next SST (open-ocean mean, RMS 0.49 K). | 20 random size-matched sets per size: FLUS ×0.97 (10) / ×0.87 (32), 0 successes. | FLUS = 0 under input replacement is by construction; the coupled multi-year de-leak test (PREREG 1.2) was not run before the hackathon (see Open). |
| 7 | **One-step scale gain (atmosphere).** With balanced perturbations (truth 5-day differences, all prognostic channels; *exploratory*), C1A0 keeps the *amount* of each scale about as E3SM does (U_7 magnitude 0.43 vs truth AR(1) 0.45 at l ≈ 55) but **decorrelates the pattern faster**: projection 0.85× truth at l ≈ 8, 0.6× at l ≈ 55. Exceptions: upper-level T_1 small scales persist longer than in truth; TS amplifies mid scales. | Truth per-degree AR(1) from E3SM 6-h output (fit and score years agree within 0.02). | The **preregistered single-variable** gains are confounded by balance: an isolated one-level perturbation is partly rejected (T_7 at l ≤ 2: 0.49 vs persistence 1), and 30–50% of response power leaves the probed band for T_4…V_7. Balanced probe = one 5-day difference direction per state. |
| 8 | **Ocean 5-day gain is flat in degree to l = 89** (sst 0.21–0.22, 0–20 m T 0.25–0.28, layer-5 T 0.61–0.73, ssh 0.10–0.13; 12 dates). | | Single-channel gains understate persistence: sst and the 0–20 m layer describe the same water (sst alone 0.22 K/K, sst + 0–20 m together 0.50 K/K). |

## Idea 1: FLUS thermometer

**Verified:** findings 1–6 above. The laptop number reproduces on Perlmutter: −2.042 [−2.076, −2.012] K per 100
W/m² over 73 states of 1985 (laptop −2.04 [−2.08, −2.01]); wrong sign on 99.85% of open ocean. LHFLX/FLDS 0.26 and
SHFLX/FLDS 0.27 reproduce.

**Refuted or narrowed:**
- **"Target leak" as the full explanation is narrowed.** The leak exists in the data (finding 1), but the model's
  response is 0.11× the truth relation with an unrelated pattern. The preregistered verdict is "something else to
  explain". Training history (finding 3) explains most of the strength.
- **"E11 shows the shortcut too, so it is not unique to C1A0" is not independent evidence.** The hackathon E11
  ocean *is* C1A0's ocean before coupled fine-tuning. Independent ocean trainings (E11 S02/S03, E12–E16) do show
  it.
- **"A box gives ~2× the uniform response"** narrows to 1.43× on held-out states.

**Open:**
- The coupled de-leak scorecard (PREREG 1.2) was not run: 5 ICs × 2 arms × 2 seeds × 5 years, about 10
  GPU-hours. The switch is verified (`hc.capture --deleak`, `hc.evaluate --deleak`).
- Why coupled fine-tuning fixes LHFLX's sign.
- The SHFLX sign under boxes.

## Idea 2: errors (hackathon track T2; starter only)

**Starter result** (`idea2_errors/strata.py`; 4 score dates in 2047, 4 seeds, pre-corrector; no bootstrap):
- Seed ceiling A_err 0.77–0.82.
- PCA-128 at block 7: A_err 0.88 (T_7), 0.87 (FLUT), 0.56 (FSUTOA).
- Scalar-offset and affine-persistence controls ≤ 0.28.
- PS and precipitation errors are dominated by the missing corrector.

**Ready:**
- One-step forecasts from truth with matching targets: `hc.evaluate` writes `autoregressive_predictions.nc` and
  `autoregressive_target.nc`.
- Truth-forced ocean inputs, validated equal to fme's packing on all state channels (`hc.truth_ocean`).
- Atmosphere first-call capture (`--capture-atm`).
- The tolerance rule τ = max(k·σ_seed, 1% of the spatial SD), k ∈ {1, 2, 3}, is preregistered.

## Idea 3: scale gain

**Verified:**
- Truth per-degree AR(1) baselines.
- Preregistered single-variable JVP gains for 10 variables × 12 states × 3 noise draws, linear within 0.5% (FD at
  ε = 0.01).
- Noise gain.
- Ocean gain.

**Narrowed:** the preregistered single-variable gain is not a scale-selectivity measure on its own (balance).

**Open:**
- The rollout-spectra link (starter `idea3_gain/rollout_spectra.py`), months 7–12 of 1-year runs: T_4/T_7 power
  matches truth at all l (±6%), while winds lose 11–17% at l = 80–89. Linking this quantitatively to the one-step
  curves is open.
- Other checkpoints.
- The ViT comparison.

## Numbers that are restatements of one another

- F1 conditional FLUS coefficient and its Frisch–Waugh version.
- Truth AR(1) transfer on score years and the damped-linear fit on fit years (one statistic, two year sets).
- Edge-patching sufficiency and necessity, and skip.0 + deep ≈ 1, under a near-linear response. Likewise the
  skip_module + convblock shares.
- The laptop's zero-net violation, Eq. 10 M and joint M all follow from the FLUS/FLDS ratio.
- The model's flux ratios X/FLDS and its slopes are the same numbers in different units (ratios are what to quote:
  the MLD rule moves absolute gains, see `localize_summary_*.json` `gain_vs_slab_law`).
