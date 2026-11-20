# Idea 3 — how much of each spatial scale survives one step?

| Step | Script | Output | Command |
|---|---|---|---|
| truth states for 12 dates (first atmosphere call of truth-started runs) | `hc.evaluate --capture-atm 1` | `data/states/atm12` | `python -m hc.evaluate data/states/atm12 --steps 1 --capture-atm 1 --no-ocn-pred --capture --start 1996-01-16 …` (Jan 16, Apr 16, Jul 15, Oct 13 × 1996, 2042, 2047) |
| atmosphere single-variable JVP gain (preregistered) | `gain_atm.py` (checkpoints per state/seed in `results/gain_parts/`) | `results/gain_atm_{a,b,c,d}.npz` | `python gain_atm.py data/states/atm12 a 0 1 2` … |
| summary + figure | `gain_summary.py` | `results/gain_atm_summary.json`, `gain_atm.png` | `python gain_summary.py a b c d` |
| truth per-degree AR(1) transfer, truth spectra | `truth_transfer.py` | `results/truth_transfer.json`, `truth_spectra.npz` | `python truth_transfer.py coeffs 1995 … 2049 && python truth_transfer.py transfer` |
| ocean 5-day gain (truth-forced inputs) | `gain_ocn.py` | `results/gain_ocn_v1.json` | `python gain_ocn.py v1` |
| balanced probes (EXPLORATORY; PREREG change log) | `gain_atm_bal.py` | `results/gain_atm_bal_v2.json` | `python -m hc.evaluate data/states/atm12m5 … (dates − 5 d)` then `python gain_atm_bal.py data/states/atm12 data/states/atm12m5 v2` |

Gains are in physical units; noise is held fixed by passing our own seeded draw to `conditional_model`. The
single-variable gain is unbalanced (the model rejects part of an isolated one-level perturbation); compare the
balanced projection with truth's AR(1).
