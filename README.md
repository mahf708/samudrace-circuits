# samudrace-circuits

Interpretability of the **SamudrACE C1A0** coupled emulator of E3SMv3 (SFNO atmosphere + Samudra U-Net ocean),
finished at scale on Perlmutter against **E3SM truth**, scored only on years the model never saw (1995–1999,
2040–2049). Built for a hackathon: participants and their agents should start from here.

- **Start here:** [`HACKATHON.md`](HACKATHON.md) (setup, what is ready, open tracks, an agent prompt)
- **Findings:** [`REPORT.md`](REPORT.md) and the results page https://portal.nersc.gov/project/m5451/mahf708/samudrace-circuits/
- **For agents:** [`AGENTS.md`](AGENTS.md) (layout, libraries, pitfalls) and `results/manifest.json` (every claim
  with its control, caveat, JSON, script and command)
- **Rules:** [`PREREG.md`](PREREG.md) (fixed before results, with a change log) · [`INVENTORY.md`](INVENTORY.md)
- **Data products:** HF dataset [`mahf708/samudrace-circuits`](https://huggingface.co/datasets/mahf708/samudrace-circuits)
  (`bash fetch_data.sh`)
- **exege feature:** the `samudra` Patchable adapter, https://github.com/mahf708/exege/pull/12

| Dir | Idea |
|---|---|
| `idea1_flus/` | The ocean reads upward longwave (FLUS) as an SST thermometer: the leak in truth, localisation, ablation, replication across 18 checkpoints |
| `idea2_errors/` | Do explanations keep the emulator's real errors? (a hackathon track) |
| `idea3_gain/` | How much of each spatial scale survives one step (JVP, balanced probes, truth per-degree persistence) |
| `lib/hc/` | Shared library: paths, exact Samudra graph/replay, truth-start runs, truth-forced inputs, spectra |
| `phase0/` | Smoke tests, numerics (TF32) check, reproduction of the laptop number |
| `web/` | Builds the results page and manifest from the JSONs |

C1A0 weights: HF [`mahf708/e3smv3-lr-hist-samudrace@rc1`](https://huggingface.co/mahf708/e3smv3-lr-hist-samudrace)
(`checkpoints/C1A0.tar`). Code: `fme` from E3SM-Project/ace tag `e3smv3-lr-hist-samudrace-rc1`.
