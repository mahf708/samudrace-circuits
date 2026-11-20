# Idea 1 — the ocean reads FLUS as an SST thermometer

All on C1A0 unless stated; scored on held-out years (1995–1999, 2040–2049). Rules: `../PREREG.md` §1.

| Step | Script | Output | Command |
|---|---|---|---|
| F1 truth extraction (CPU) | `f1_extract.py` | `data/f1/year_YYYY.npz` | `python f1_extract.py 1995 … 1999 2040 … 2049 1985 … 1989` |
| F1 alignment + conditional regression | `f1_analyze.py` | `results/f1.json`, `results/f1_maps.npz` | `python f1_analyze.py` |
| held-out ocean states (coupled run from truth) | `hc.evaluate`, `hc.meta` | `data/states/t1995`, `t2045` | `python -m hc.evaluate data/states/t2045 --start 2045-01-01 --steps 73 --capture --atm-spectra && python -m hc.meta data/states/t2045` |
| F1 model vs truth + verdict | `f1_model_compare.py` | `results/f1_compare_t{1995,2045}.json` | `python f1_model_compare.py data/states/t2045 2040s t2045` |
| F2 localisation (layer-0 sub-edges × skip.0/deep, IN-frozen) | `localize.py`, `localize_summary.py` | `results/localize_*.json`, `localize_summary_*.{json,md}` | `python localize.py data/states/t2045 t2045 4 && python localize_summary.py t2045` |
| F2 skeptic (independent: exege `samudra` adapter) | `skeptic_adapter_s0.py` | `results/skeptic_adapter_s0.json` | `python skeptic_adapter_s0.py data/states/t2045` |
| F3 ablation (greedy vs random channel sets; input de-leak) | `ablate.py`, `ablate_summary.py` | `results/ablate_l0_v1*.json` | `python ablate.py l0 data/states/ic1985,data/states/t1995 data/states/t2045 v1 && python ablate_summary.py ablate_l0_v1` |
| Replication across checkpoints | `replicate.py` | `results/replicate_*.json` | `python replicate.py NAME=CKPT …` (see `data/queue/done/08_replicate_*.sh`) |
| F3 coupled de-leak rollouts | Phase 2 | | `python -m hc.evaluate … --deleak` |

Slopes are K per 100 W/m² of warming flux (FLUS, LHFLX, SHFLX enter with −1): the thermometer sign is negative.
Every gain is quoted with perturbed area (uniform open ocean vs 10×10° boxes on the PREREG lattice), metric
(perturbed-area mean; projection) and year. fp32 (TF32 off) for all network experiments.
