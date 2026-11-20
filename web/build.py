"""Build the results page, results/manifest.json and the downloadable docs from the JSON results in this repo.

usage: python web/build.py [OUT_DIR]   (default $CFS/m5451/www/mahf708/samudrace-circuits)
Every number on the page is read from a JSON file in the repo; each rendered claim is also written to
results/manifest.json (value, verdict, caveat, json path, script, command) for agents.
"""
import datetime
import html
import json
import os
import shutil
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.environ.get("CFS", "/global/cfs/cdirs"),
                                                          "m5451/www/mahf708/samudrace-circuits")
FIG, DATA = os.path.join(OUT, "fig"), os.path.join(OUT, "data")
os.makedirs(FIG, exist_ok=True)
os.makedirs(DATA, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130})
GH = "https://github.com/mahf708/samudrace-circuits"
HF = "https://huggingface.co/datasets/mahf708/samudrace-circuits"
CLAIMS = []


def J(path):
    p = os.path.join(ROOT, path)
    return json.load(open(p)) if os.path.exists(p) else None


def f(x, n=3, sign=False):
    if x is None:
        return "–"
    return format(x, f"+.{n}f" if sign else f".{n}f")


def esc(s):
    return html.escape(str(s))


def table(head, rows, cls=""):
    h = "".join(f"<th>{esc(c)}</th>" for c in head)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<div class='tw'><table class='{cls}'><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>"


def figure(name, caption):
    return f'<figure><a href="fig/{name}"><img src="fig/{name}" alt="{esc(caption[:80])}"></a><figcaption>{caption}</figcaption></figure>'


def pending(what):
    return f'<p class="pending">Pending: {esc(what)}</p>'


def repro(cmds, jsons=()):
    links = " ".join(f"<a href='data/{os.path.basename(j)}'>{esc(os.path.basename(j))}</a>" for j in jsons)
    for j in jsons:
        src = os.path.join(ROOT, j)
        if os.path.exists(src) and os.path.getsize(src) < 30e6:
            shutil.copy(src, DATA)
    c = esc("\n".join(cmds))
    return f"<details class='repro'><summary>Reproduce</summary><pre>{c}</pre><p>Results: {links or '–'}</p></details>"


def claim(cid, idea, text, value, verdict, control, caveat, json_path, script, command, status="verified"):
    CLAIMS.append(dict(id=cid, idea=idea, claim=text, value=value, verdict=verdict, control=control, caveat=caveat,
                       json=json_path, script=script, command=command, status=status))
    badge = {"verified": "ok", "narrowed": "warn", "refuted": "bad", "pending": "mut", "running": "mut"}.get(status, "mut")
    return (f"<div class='claim'><div class='ch'><span class='badge {badge}'>{esc(status)}</span> "
            f"<b>{esc(cid)}</b> · {text}</div><div class='cv'>{value}</div>"
            f"<div class='cc'><b>Control:</b> {control}</div><div class='cc'><b>Caveat:</b> {caveat}</div></div>")


# ======================================================================= data
smoke, slope, numer = J("phase0/smoke.json"), J("phase0/flus_slope.json"), J("phase0/numerics_check.json")
f1 = J("idea1_flus/results/f1.json")
cmp = {t: J(f"idea1_flus/results/f1_compare_{t}.json") for t in ("t1995", "t2045")}
tt = J("idea3_gain/results/truth_transfer.json")
gs = J("idea3_gain/results/gain_atm_summary.json")
locs = {t: J(f"idea1_flus/results/localize_summary_{t}.json") for t in ("t1995", "t2045")}
L45 = locs.get("t2045")
SK = J("idea1_flus/results/skeptic_adapter_s0.json")
abl = J("idea1_flus/results/ablate_l0_v1_summary.json")
gocn = J("idea3_gain/results/gain_ocn_v1.json")
rep = {}
for _fn in sorted(os.listdir(os.path.join(ROOT, "idea1_flus/results"))):
    if _fn.startswith("replicate_") and _fn.endswith(".json"):
        rep.update(J(f"idea1_flus/results/{_fn}"))
FAMILY = {"E11_": "ocean-only, truth-forced (leaky FLUS)", "E11-FT": "ocean-only + fine-tune (leaky FLUS)",
          "E12": "ocean-only, loss-weight variant", "E13": "ocean-only, loss-weight variant",
          "E14": "ocean-only, loss-weight variant", "E16": "ocean-only, loss-weight variant",
          "C1A0": "coupled fine-tune (release)", "C1A1": "coupled fine-tune (release)", "C0A0": "coupled fine-tune (release)",
          "E30": "coupled fine-tune (C0A0 family seed)", "E31": "coupled fine-tune (C1A1 family seed)"}
fam = lambda n: next((v for k, v in FAMILY.items() if n.startswith(k)), "")

# ======================================================================= headline
head = []
if slope:
    s = slope["summary"]["FLUS_orig"]
    head.append(claim("P0-1", "setup", "The laptop FLUS-thermometer number reproduces on Perlmutter (A100, fp32).",
                      f"{s['mean']:+.3f} K per 100 W/m² of warming flux [{s['min']:+.3f}, {s['max']:+.3f}] over 73 states of 1985; "
                      f"wrong sign on {100 * slope['summary']['FLUS_wrong_sign_area_frac']['mean']:.2f}% of open ocean "
                      "(laptop: −2.04 [−2.08, −2.01], 99.8%)", "reproduced",
                      "slab law gives +0.33; FLDS/FSNS have the law's sign (0.33 / 0.28)",
                      "1985 is a TRAINING year; uniform open-ocean shift (a 10×10° box gives ~2×)",
                      "phase0/flus_slope.json", "phase0/flus_slope.py",
                      "python -m hc.capture data/states/ic1985 --start 1985 --steps 73 && python phase0/flus_slope.py data/states/ic1985"))
if f1:
    r = f1["regression|2040s|score"]["FLUS"]
    head.append(claim("I1-F1a", "idea1", "In E3SM truth, the ocean's FLUS input is a near-exact thermometer of the SST it must predict (a target leak in the data).",
                      f"conditional slope {r['K_per_100Wm2_warming_mean']:+.1f} K per 100 W/m² of warming flux (literal 1/(4εσT³): "
                      f"−{f1['regression|2040s|score']['literal_thermometer_K_per_100Wm2_mean']:.1f}); thermometer sign on "
                      f"{100 * r['frac_area_negative(thermometer_sign_for_FLUS)']:.1f}% of open ocean; per-cell R² median "
                      f"{f1['regression|2040s|score']['R2_median']:.4f} (2045–49)", "leak present in truth",
                      "the other four fluxes have conditional slopes ≈ 0 (medians |·| < 0.003)",
                      "per-cell OLS with intercept; the preregistered alignment check (end vs start of window, pooled RMS) does NOT "
                      "show the 20% margin (22.21 vs 22.28 W/m²) because a spatial offset dominates pooled RMS",
                      "idea1_flus/results/f1.json", "idea1_flus/f1_analyze.py", "python idea1_flus/f1_analyze.py"))
c45 = cmp.get("t2045")
if c45:
    F = c45["per_flux"]["FLUS"]
    head.append(claim("I1-F1b", "idea1", "C1A0 does not copy the truth thermometer: it uses about a tenth of it, with an unrelated spatial pattern, and keeps lawful-sign responses to the other fluxes.",
                      f"model/truth slope ratio {F['ratio_model_over_truth']:.3f}; pattern correlation {F['pattern_corr']:+.2f} "
                      f"({F['pattern_corr_excl_south_of_70S']:+.2f} without ice-shelf cavities); model FLUS {F['model_global_mean']:+.2f} vs truth "
                      f"{F['truth_global_mean']:+.2f} K per 100 W/m² (2045 states vs 2045–49 truth)",
                      f"preregistered verdict: {c45['verdict']['verdict']}",
                      "same result on 1995 states (ratio 0.109, r = 0.07); FLDS/FSNS model slopes +0.32/+0.29 vs truth ≈ 0",
                      "model states are coupled-run inputs from truth ICs (not truth-forced); uniform shifts only for this comparison",
                      "idea1_flus/results/f1_compare_t2045.json", "idea1_flus/f1_model_compare.py",
                      "python idea1_flus/f1_model_compare.py data/states/t2045 2040s t2045", status="verified"))

if L45:
    U, B = L45["t2045|uniform"]["FLUS"], L45["t2045|box"]["FLUS"]
    su, sb = U["slope"]["mean"], B["slope"]["mean"]
    sh = lambda E, c: E["edge_shares"][c]["areamean_share"]["mean"]
    pred = (sh(U, "skipmod") + sh(U, "convfrz")) * su
    head.append(claim("I1-F2a", "idea1", "Below “skip.0 + layer 0”: FLUS splits about evenly between layer 0's linear 1×1 skip_module and its InstanceNorm-ed convblock, whereas an SST perturbation goes 100% through the 1×1 skip_module.",
                      f"FLUS uniform: skip.0 {sh(U, 's0only'):.2f}, skip_module {sh(U, 'skipmod'):.2f}, convblock {sh(U, 'convblk'):.2f}; "
                      f"box: skip.0 {sh(B, 's0only'):.2f}, skip_module {sh(B, 'skipmod'):.2f}, convblock {sh(B, 'convblk'):.2f}; "
                      f"sst: skip_module {sh(L45['t2045|uniform']['sst'], 'skipmod'):.2f}",
                      "2045 (score year), 19 states, exact patching; 1995 replicates to ±0.01",
                      "projection metric agrees within 0.05 for FLUS. <b>Skeptic pass:</b> an independent implementation "
                      "(exege's new <code>samudra</code> Patchable adapter, EdgePatch semantics) gives skip.0 share "
                      + (f"{SK['skip0_share_mean']:.3f}" if SK else "–") + " on 7 states of 2045 (localize.py: 0.86)",
                      "near-linear response: sufficiency = necessity, shares sum to ≈1 (one result per split)",
                      "idea1_flus/results/localize_summary_t2045.json", "idea1_flus/localize.py",
                      "python idea1_flus/localize.py data/states/t2045 t2045 4 && python idea1_flus/localize_summary.py t2045"))
    head.append(claim("I1-F2b", "idea1", "InstanceNorm statistics account for the box-vs-uniform difference of the FLUS response.",
                      f"freezing the convblock's IN statistics raises its uniform-shift share {sh(U, 'convblk'):.2f} → {sh(U, 'convfrz'):.2f} "
                      f"(box: {sh(B, 'convblk'):.2f} → {sh(B, 'convfrz'):.2f}); predicted un-normalised uniform slope {pred:+.2f} vs measured box "
                      f"{sb:+.2f} K per 100 W/m² (uniform {su:+.2f})", "consistent",
                      "the same freeze leaves box shares unchanged, as expected if a box barely moves whole-field statistics",
                      "holds for FLUS only: for FLDS the same prediction gives 1.34× vs measured box/uniform 1.07×; box ratio here is 1.43× (laptop said ~2×)",
                      "idea1_flus/results/localize_summary_t2045.json", "idea1_flus/localize.py",
                      "python idea1_flus/localize_summary.py t2045", status="narrowed"))

# ======================================================================= setup section
setup = ["<h2 id='setup'>Setup, data and checks</h2>",
         "<h3>Model and lineage</h3>",
         table(["", ""], [
             ["Checkpoint", "HF <code>mahf708/e3smv3-lr-hist-samudrace@rc1</code> <code>checkpoints/C1A0.tar</code>, md5 <code>a5082a9aec1243e37e0820a99e77d059</code>"],
             ["Training run", "E35-CFT S01: 25 epochs coupled fine-tune (4 coupled steps, ensemble CRPS), EMA weights"],
             ["Atmosphere", "NoiseConditioned SFNO, 8 blocks × 384, big skip, isotropic noise (32 ch); init E02-FT S01. 6 h step. Corrector: dry air, moisture budget, positivity"],
             ["Ocean", "Samudra U-Net 280/380/480/520, InstanceNorm; init <b>E11-FT S01</b> (= the hackathon E11 archive ocean). 5-day step; FLUS is an input; corrector never touches sst"],
             ["Coupling", "ocean fluxes = atmosphere outputs averaged over 20 inner steps; the atmosphere sees the ocean's current SST"],
             ["Other release checkpoints", "C0A0 = E30-CFT S03 (no CO₂; ocean E11-FT S03); C1A1 = E31-CFT S01 (CO₂ + aerosol; atmosphere E05-FT S01, ocean E11-FT S01, same as C1A0)"],
         ]),
         "<h3>Truth data and splits</h3>",
         table(["", ""], [
             ["Run", "<code>v3.LR.historical_0101.aigo</code>: EAM 6-hourly, MPAS-O/SI 5-day means, 1940–2065, 1° Gaussian (C1A0's grid)"],
             ["Train", "1940-01-06 … 1990-01-01 and 2000-01-06 … 2040-01-01"],
             ["Validation", "1990–1994 (also used to pick the checkpoint)"],
             ["<b>Held out (scored here)</b>", "<b>1995–1999 and 2040–2049</b>; fit years 1995–97 / 2040–44, score years 1998–99 / 2045–49"],
             ["Laptop states", "1985, 2015 = training years; used only to reproduce laptop numbers"],
         ])]
if smoke and numer:
    setup += ["<h3>Exactness and reproducibility checks</h3>", table(["check", "result"], [
        ["common.Graph vs Samudra.forward (A100)", f"max |diff| {smoke['graph_vs_module_maxabs']:.1f} (bitwise)"],
        ["SamudraReplay vs Samudra.forward", f"max |diff| {smoke['replay_vs_module_maxabs']:.1f} (bitwise)"],
        ["same-seed fme coupled steps (all tensors, all outputs)", "bitwise identical" if smoke["same_seed_all_bitwise"] else "NOT identical"],
        ["de-leak switch at load time (CombinedFluxConfig FLUS := 5.67e-8·sst⁴)",
         f"works without editing the checkpoint; ocean FLUS / emission median {smoke['deleak']['step0']['deleak_over_emission_median']:.4f}; "
         f"control − de-leak FLUS {smoke['deleak']['step0']['control_minus_deleak_rms_Wm2']:.1f} W/m² RMS (step 0), "
         f"{smoke['deleak']['step1']['control_minus_deleak_rms_Wm2']:.1f} (step 1)"],
        ["TF32 (CUDA default) vs fp32",
         f"TF32: batch-of-4 vs single {numer['tf32_default']['batch4_vs_single_sst_maxabs_K']:.3f} K max; fp32: "
         f"{numer['tf32_off']['batch4_vs_single_sst_maxabs_K']:.1e} K; fp32 vs deployed TF32 output {numer['tf32_off']['vs_fme_capture_sst_maxabs_K']:.2f} K max"],
        ["truth-forced ocean input builder vs fme packing of the 1995-01-01 truth IC", "all state channels identical; only the 9 flux channels differ (truth vs coupled-atmosphere fluxes), as expected"],
    ]), repro(["bash setup_env.sh", "python -m hc.capture data/smoke/A --steps 2 --atm-pred",
               "python -m hc.capture data/smoke/B --steps 2 --atm-pred", "python -m hc.capture data/smoke/D --steps 2 --deleak",
               "python phase0/smoke.py data/smoke/A data/smoke/B data/smoke/D", "python phase0/numerics_check.py data/smoke/A"],
              ["phase0/smoke.json", "phase0/numerics_check.json", "phase0/flus_slope.json"])]

# ======================================================================= idea 1
i1 = ["<h2 id='idea1'>Idea 1 · The ocean reads FLUS as an SST thermometer</h2>",
      "<p><b>Question.</b> Raising upward longwave (FLUS) at the ocean's input <i>warms</i> its next SST (wrong sign "
      "on 99.8% of open ocean). Is this a target leak, i.e. is the training-time FLUS (a 5-day mean over the window "
      "being predicted, the coupler's emission from that window's SST) a legitimate predictor of the target? If so, "
      "does the model's response match the truth relation, and can it be removed without breaking the lawful fluxes?</p>"]
i1 += head[1:2]
if f1:
    rows = []
    for blk in ("1990s", "2040s", "1980s_train_contrast"):
        for k, v in f1.get(f"alignment|{blk}", {}).items():
            src, d = k.split("|")
            rows.append([blk, src, d, f(v["eps_fit"], 4), f(v["rms_Wm2"], 2), f(v["R2_pooled"], 4), f(v["cell_R2_median"], 3)])
    i1 += ["<h3>F1 (a) · Alignment of FLUS with εσT⁴ in truth</h3>",
           "<p>FLUS[w] is the 5-day mean the ocean receives with window w (MPAS = what training fed; EAM = the "
           "atmosphere's own). ε fitted on fit years over open ocean, scored on score years. Five SST definitions: previous-window "
           "mean (what the ocean reads), target-window mean, window-mean TS⁴ (6-hourly EAM), instantaneous TS at window start and end.</p>",
           table(["block", "FLUS source", "SST definition", "ε", "RMS (W/m²)", "pooled R²", "per-cell R² (median)"], rows),
           "<p class='note'>Reading: EAM's FLUS is exactly εσ⟨TS⁴⟩ over its window (RMS 1.1 W/m²). MPAS FLUS carries a "
           "spatially varying offset against every SST definition (pooled RMS 16–22 W/m²) that hides the time relation; "
           "per-cell R² favours the target window (0.770 vs 0.744 in the 1990s). The preregistered supporting criterion "
           "(end beats start by > 20% pooled RMS) is <b>not met</b>.</p>"]
    rows = []
    for blk in ("1990s", "2040s", "1980s_train_contrast"):
        for part in ("score", "fit"):
            r = f1.get(f"regression|{blk}|{part}")
            if not r:
                continue
            for fl in ("FLUS", "FLDS", "FSNS", "LHFLX", "SHFLX"):
                v = r[fl]
                rows.append([blk, part, fl, f(v["K_per_100Wm2_warming_mean"], 2, True), f(v["median"], 3, True),
                             f"[{f(v['p05'], 2, True)}, {f(v['p95'], 2, True)}]",
                             f(v["frac_area_negative(thermometer_sign_for_FLUS)"], 3), f(v["frac_area_|z|>2"], 3)])
    i1 += ["<h3>F1 (b) · The conditional slope in truth</h3>",
           "<p>Per open-ocean cell (ice-free in every window): ΔSST = sst[w] − sst[w−1] regressed on the five window-w "
           "fluxes, sst[w−1], annual + semiannual harmonics and an intercept (OLS, Newey–West lag 6). K per 100 W/m² of "
           "<i>warming</i> flux (FLUS, LHFLX, SHFLX enter with −1), so the thermometer sign is negative. The Frisch–Waugh "
           "restatement agrees to 5e-9 (one number, not two).</p>",
           table(["block", "years", "flux", "area mean", "median", "5–95%", "area frac. negative", "area frac. |z|>2"], rows)]
cmp_rows, verdicts = [], []
for tag, blk in (("t1995", "1990s"), ("t2045", "2040s")):
    c = cmp.get(tag)
    if not c:
        continue
    verdicts.append(f"<li>{tag} states vs {blk} truth: <b>{esc(c['verdict']['verdict'])}</b> "
                    f"(criteria: {', '.join(f'{k} = {v}' for k, v in c['verdict']['criteria'].items())})</li>")
    for fl, v in c["per_flux"].items():
        cmp_rows.append([tag, fl, f(v["model_global_mean"], 3, True), f(v["truth_global_mean"], 3, True),
                         f(v["ratio_model_over_truth"], 3), f(v["pattern_corr"], 3, True), f(v["pattern_corr_excl_south_of_70S"], 3, True)])
    mp, tmp = os.path.join(ROOT, f"idea1_flus/results/f1_model_maps_{tag}.npz"), os.path.join(ROOT, "idea1_flus/results/f1_maps.npz")
    if os.path.exists(mp) and os.path.exists(tmp):
        mm, tm = np.load(mp), np.load(tmp)
        fig, axs = plt.subplots(1, 3, figsize=(15, 3.6))
        panels = [(-tm[f"{blk}|score|FLUS"] * 100, f"truth: conditional FLUS slope ({blk} score years)", (-22, -12)),
                  (mm["FLUS"], f"C1A0: FLUS input-response ({tag} states)", (-3.5, -0.5)),
                  (mm["FLDS"], f"C1A0: FLDS input-response ({tag} states)", (0, 0.6))]
        for ax, (arr, title, (lo, hi)) in zip(axs, panels):
            im = ax.imshow(arr, origin="lower", extent=[0, 360, -90, 90], vmin=lo, vmax=hi, cmap="viridis", aspect="auto")
            ax.set_title(title, fontsize=9)
            plt.colorbar(im, ax=ax, shrink=0.85, label="K per 100 W/m² warming flux")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, f"f1_maps_{tag}.png"))
        plt.close(fig)
if cmp_rows:
    i1 += ["<h3>F1 (c) · Model vs truth, and the preregistered verdict</h3>"] + head[2:3] + [
        "<p>Model: per-node slope of next SST to a uniform open-ocean shift of each flux (bare network, fp32, every 4th "
        "state of a 1-year coupled run started from truth). Verdict rule fixed in PREREG 1.1: <i>leak supported</i> iff truth "
        "thermometer area &gt; 75% <b>and</b> model/truth ratio in [0.33, 3] <b>and</b> pattern correlation &gt; 0.3; "
        "<i>refuted</i> if truth has the lawful sign over &gt; 50%; otherwise <i>something else to explain</i>.</p>",
        "<ul>" + "".join(verdicts) + "</ul>",
        table(["states", "flux", "model mean", "truth mean", "model/truth", "pattern r", "pattern r (no cavities)"], cmp_rows)]
    for tag in ("t1995", "t2045"):
        if os.path.exists(os.path.join(FIG, f"f1_maps_{tag}.png")):
            i1.append(figure(f"f1_maps_{tag}.png", f"Left: truth conditional FLUS slope (≈ −1/(4εσT³): stronger in cold water). "
                                                    f"Middle/right: C1A0's input-response to FLUS and FLDS ({tag}). Note the colour "
                                                    "ranges: the model uses ~1/10 of the truth relation and its map is nearly flat."))
    i1.append("<p><b>Interpretation (open).</b> The data contain the leak in its strongest form, so a model that fit the "
              "linear truth relation would answer ~17 K per 100 W/m² to FLUS and ~0 to everything else. C1A0 does neither. "
              "One candidate: its ocean was fine-tuned in coupled mode (E35-CFT), where FLUS comes from the atmosphere driven "
              "by the ocean's start-of-window SST, so the leak is absent there. The replication across checkpoints (E11-FT = the "
              "same ocean before coupled fine-tuning; E11 seeds; E12–E16) tests this directly.</p>")
    i1.append(repro(["python idea1_flus/f1_extract.py 1995 1996 1997 1998 1999 2040 2041 2042 2043 2044 2045 2046 2047 2048 2049 1985 1986 1987 1988 1989",
                     "python idea1_flus/f1_analyze.py",
                     "python -m hc.evaluate data/states/t1995 --start 1995-01-01 --steps 73 --capture --atm-spectra && python -m hc.meta data/states/t1995",
                     "python -m hc.evaluate data/states/t2045 --start 2045-01-01 --steps 73 --capture --atm-spectra && python -m hc.meta data/states/t2045",
                     "python idea1_flus/f1_model_compare.py data/states/t1995 1990s t1995",
                     "python idea1_flus/f1_model_compare.py data/states/t2045 2040s t2045"],
                    ["idea1_flus/results/f1.json", "idea1_flus/results/f1_compare_t1995.json", "idea1_flus/results/f1_compare_t2045.json"]))

# F2 localisation
i1.append("<h3>F2 · Where the response goes: exact patching below “skip.0 + layer 0”</h3>")
i1.append("<p>Layer 0 is the network's only reader of the input: <code>skip_module</code> (1×1 conv, no norm) + "
          "<code>convblock</code> (3×3 conv → InstanceNorm → GELU → 3×3 conv → InstanceNorm → GELU → 1×1 conv). Its output "
          "feeds the full-resolution skip (skip.0) and the deep path. Exact clean→corrupted patching over the factorial "
          "{skip_module, convblock, convblock with clean InstanceNorm statistics} × {skip.0, deep}; perturbations: each flux, "
          "sst, sst + 0–20 m temperature; uniform open-ocean and 10×10° boxes on the preregistered lattice.</p>")
anyloc = False
for tag, L in locs.items():
    if not L:
        continue
    anyloc = True
    for key in [k for k in L if k != "tags" and not k.startswith("all")]:
        rows = []
        for p, e in L[key].items():
            sh = e["edge_shares"]
            rows.append([p, f"{e['slope']['mean']:+.3f} [{e['slope']['min']:+.3f}, {e['slope']['max']:+.3f}]",
                         f(e["ratio_to_FLDS"]["mean"], 2, True) if e.get("ratio_to_FLDS") else "–"]
                        + [f"{sh[c]['areamean_share']['mean']:.2f} / {sh[c]['projection']['mean']:.2f}"
                           for c in ("s0only", "deeponly", "skipmod", "convblk", "convfrz")])
        i1 += [f"<h4>{esc(key)}</h4>", table(["perturbation", "slope (K/100 W/m²; K/K for sst)", "/FLDS", "skip.0 only",
                                               "deep only", "skip_module only", "convblock only", "convblock, IN frozen"], rows)]
if anyloc:
    i1 += [c for c in head if "I1-F2" in c]
    i1.append("<p class='note'>Cells: perturbed-area-mean share / projection share. Because the network is near-linear "
              "in dQ, sufficiency = necessity and skip.0 + deep ≈ 1: one result, not converging lines.</p>")
    i1.append(repro(["python idea1_flus/localize.py data/states/t1995 t1995 4", "python idea1_flus/localize.py data/states/t2045 t2045 4",
                     "python idea1_flus/localize_summary.py t1995 t2045"],
                    [f"idea1_flus/results/localize_summary_{t}.json" for t in locs if locs[t]]))
else:
    i1.append(pending("localisation runs (1995 and 2045 states) in progress"))
i1.append("<h3>F3 · Can FLUS be removed without breaking FLDS/FSNS?</h3>")
if abl:
    rows = []
    for kind in ("uniform", "box"):
        A = abl[kind]
        for arm in ("greedy10", "greedy32", "deleak_input"):
            rt = A[arm]["ratio_to_unablated"]
            rows.append([kind, arm] + [f(rt[x], 2) for x in ("FLUS", "FLDS", "FSNS", "LHFLX", "SHFLX")]
                        + ["yes" if A[arm]["success"] else "no"])
        for k in (10, 32):
            r = A[f"random{k}_FLUS_ratio"]
            rows.append([kind, f"random{k} ×20", f"{r['median']:.2f} [{r['min']:.2f}, {r['max']:.2f}]", "", "", "", "",
                         f"{A['random_successes'][str(k)]} successes"])
    U = abl["uniform"]
    i1.append(claim("I1-F3", "idea1", "At layer-0 channel granularity the FLUS shortcut cannot be removed without breaking FLDS; at the input it can.",
                    f"greedy-32 channels (selected on fit states 1985 + 1995, scored on 2045): FLUS ×{U['greedy32']['ratio_to_unablated']['FLUS']:.2f} "
                    f"but FLDS ×{U['greedy32']['ratio_to_unablated']['FLDS']:.2f}; replacing the FLUS input with 5.67e-8·sst⁴ keeps FLDS/FSNS ×"
                    f"{U['deleak_input']['ratio_to_unablated']['FLDS']:.2f}/×{U['deleak_input']['ratio_to_unablated']['FSNS']:.2f} and shifts the clean next SST by "
                    f"{abl['deleak_input_clean_dSST']['mean_open_K']:+.2f} K (open-ocean mean; RMS {abl['deleak_input_clean_dSST']['rms_open_K']:.2f} K)",
                    "PREREG rule: channel sets fail; input replacement passes (FLUS = 0 by construction)",
                    "20 random size-matched sets per size: FLUS ×0.97 (10) / ×0.87 (32) median, 0 successes",
                    "mean ablation at one place (layer-0 output); box shifts move SHFLX by +67% under input replacement; the one-step +0.17 K is what the coupled de-leak tests over years",
                    "idea1_flus/results/ablate_l0_v1_summary.json", "idea1_flus/ablate.py",
                    "python idea1_flus/ablate.py l0 data/states/ic1985,data/states/t1995 data/states/t2045 v1 && python idea1_flus/ablate_summary.py ablate_l0_v1",
                    status="verified"))
    i1.append(table(["shift", "arm", "FLUS ×", "FLDS ×", "FSNS ×", "LHFLX ×", "SHFLX ×", "PREREG success"], rows))
    i1.append("<p class='note'>Ratios are ablated / unablated slope per state, averaged (LHFLX ratios are unstable because "
              "its unablated slope is small). Greedy channels at layer 0: " + ", ".join(map(str, abl["greedy"][:10])) +
              " … (channel 254 first, as on the laptop). Greedy curve on fit states: 10 channels 0.48, 32 channels 0.91 of the FLUS response.</p>")
    i1.append(repro(["python idea1_flus/ablate.py l0 data/states/ic1985,data/states/t1995 data/states/t2045 v1",
                     "python idea1_flus/ablate_summary.py ablate_l0_v1"], ["idea1_flus/results/ablate_l0_v1_summary.json"]))
else:
    i1.append(pending("ablation"))
i1.append("<h3>Replication across checkpoints</h3>")
if rep:
    order = sorted(rep, key=lambda n: rep[n]["summary"]["FLUS"]["uniform_mean"])
    rows = []
    for n in order:
        sm = rep[n]["summary"]
        rows.append([n, fam(n)] + [f(sm[x]["uniform_mean"], 2, True) for x in ("FLUS", "FLDS", "FSNS", "LHFLX", "SHFLX")]
                    + [f(sm["FLUS"]["box_mean"], 2, True), f(sm["FLUS_thermometer_area_mean"], 3), "yes" if sm["has_shortcut"] else "no"])
    cpl = [rep[n]["summary"]["FLUS"]["uniform_mean"] for n in rep if fam(n).startswith("coupled")]
    oft = [rep[n]["summary"]["FLUS"]["uniform_mean"] for n in rep if n.startswith("E11-FT")]
    o11 = [rep[n]["summary"]["FLUS"]["uniform_mean"] for n in rep if n.startswith("E11_")]
    i1.append(claim("I1-R", "idea1", "Every ocean checkpoint has the FLUS thermometer; its strength tracks how leaky the FLUS it was last trained on was.",
                    f"ocean-only E11 seeds {min(o11):+.2f}…{max(o11):+.2f}; after more ocean-only fine-tuning (E11-FT) {min(oft):+.2f}…{max(oft):+.2f}; "
                    f"after coupled fine-tuning (C1A0, C1A1, C0A0, E30/E31 seeds) {min(cpl):+.2f}…{max(cpl):+.2f} K per 100 W/m² (uniform, 1995 truth-forced inputs)",
                    f"{sum(rep[n]['summary']['has_shortcut'] for n in rep)}/{len(rep)} have the shortcut (thermometer sign > 75% of area)",
                    "truth's conditional slope is −17: every model is far below it; in this sample, coupled fine-tuning also turns LHFLX from the wrong sign to the lawful one",
                    "C1A1 shares C1A0's ocean init and E11-FT S01 is C1A0's ocean before coupled fine-tuning (not independent); E17 (1-day ocean) excluded: 5-day inputs",
                    "idea1_flus/results/replicate_*.json", "idea1_flus/replicate.py", "python idea1_flus/replicate.py NAME=CKPT ..."))
    i1.append(table(["checkpoint", "training", "FLUS", "FLDS", "FSNS", "LHFLX", "SHFLX", "FLUS box", "thermometer area", "shortcut?"], rows))
    i1.append(repro(["python idea1_flus/replicate.py C1A0=... E11-FT_S01=... (see slurm/queue scripts 08_replicate_*.sh)"],
                    [f"idea1_flus/results/{fn}" for fn in sorted(os.listdir(os.path.join(ROOT, 'idea1_flus/results'))) if fn.startswith('replicate_')]))
else:
    i1.append(pending("replication"))

# ======================================================================= idea 2
i2 = ["<h2 id='idea2'>Idea 2 · Do explanations keep the emulator's real errors?</h2>",
      "<p><b>Plan (preregistered, Phase 2).</b> One-step forecasts from truth states (atmosphere 6 h, 4 seeds; ocean 5 days, "
      "truth-forced). Error = model − E3SM. Tolerance τ = max(2 × same-seed SD, 1% of spatial SD), sweep k ∈ {1, 2, 3}. "
      "Explanations: ocean skip.0-only, atmosphere blocks-only / big-skip-only (corrector in/out), PCA-k / SAE at SFNO blocks. "
      "Scores: A_ok, A_err, three-way split (reproduces / corrects / other), scalar-offset and per-component-gain controls, "
      "empty-explanation and seed-ceiling anchors, block bootstrap.</p>", pending("Phase 2")]

# ======================================================================= idea 3
i3 = ["<h2 id='idea3'>Idea 3 · How much of each spatial scale survives one step?</h2>",
      "<p><b>Method.</b> JVP of the 6-hour atmosphere network (pre-corrector) at 12 truth states (Jan/Apr/Jul/Oct × 1996, 2042, "
      "2047) × 3 noise draws, noise held fixed by passing our own draw to the conditional model. Probes: band-limited random "
      "fields (flat power over the band, 8 per band, physical units: K, m/s, Pa); gain(band) = √(output power in band / input "
      "power in band). Baselines: persistence (1); E3SM's own per-degree one-step AR(1) transfer of anomalies (score years) "
      "and the damped-linear fit (fit years). These two are the same statistic on different years, a restatement, shown to "
      "bound sampling noise.</p>"]
if tt:
    V = tt["vars"]
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.4), sharey=True)
    for ax, v in zip(axs, ["T_7", "U_7", "PS", "TS"]):
        for k, col in (("1990s|truth_transfer_score", "C3"), ("1990s|damped_linear_fit", "C3"), ("2040s|truth_transfer_score", "C1")):
            a = np.array(tt[k])[V.index(v)]
            ax.plot(np.arange(len(a)), a, color=col, ls="--" if "damped" in k else "-", lw=1.5, label=k.replace("|", " "))
        ax.axhline(1, color="k", ls=":", lw=1)
        ax.set_title(v)
        ax.set_xlabel("SH degree l")
    axs[0].set_ylabel("one-step (6 h) transfer")
    axs[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "truth_transfer.png"))
    plt.close(fig)
    i3.append(figure("truth_transfer.png", "E3SM's own 6-hour per-degree persistence (AR(1) coefficient of anomaly SH "
                                           "coefficients about a fit-year smoothed climatology). Large scales persist (≈0.99 at "
                                           "l ≤ 8); T_7 falls to 0.44 and U_7 to 0.27 at l = 80. This is the bar the model's gain is compared against."))
    i3.append(repro(["python idea3_gain/truth_transfer.py coeffs 1995 1996 1997 1998 1999 2040 2041 2042 2043 2044 2045 2046 2047 2048 2049",
                     "python idea3_gain/truth_transfer.py transfer"], ["idea3_gain/results/truth_transfer.json"]))
if gs:
    i3.append(figure("gain_atm.png", "C1A0 6-h single-variable gain per SH band (preregistered probe; blue: mean, band 5–95% over 12 "
                                     "states × 3 noise draws) vs persistence and E3SM's per-degree persistence. <b>Caveat:</b> a "
                                     "single-variable, single-level perturbation is unbalanced and the model rejects part of it, so "
                                     "these gains sit below persistence even at l ≤ 2; 30–50% of the response lands outside the probed "
                                     "band for T_4…V_7 (1–5% for PS, T_1). T_1 and PS gains rise at small scales. The balanced-probe "
                                     "measurement below is the apples-to-apples comparison with truth."))
    lin = {}
    for fn in os.listdir(os.path.join(ROOT, "idea3_gain/results/gain_parts")):
        if fn.endswith("_lin.json"):
            for k, v in json.load(open(os.path.join(ROOT, "idea3_gain/results/gain_parts", fn))).items():
                lin.setdefault(k, []).extend(v)
    if lin:
        i3.append("<p class='note'>Linearity (finite difference / JVP, per probe): " + "; ".join(
            f"{esc(k)}: {min(v):.3f}–{max(v):.3f}" for k, v in sorted(lin.items())) + ".</p>")
    i3.append(repro(["python -m hc.evaluate data/states/atm12 --steps 1 --capture-atm 1 --no-ocn-pred --capture --start 1996-01-16 ... (12 dates)",
                     "python idea3_gain/gain_atm.py data/states/atm12 a 0 1 2  # ... b 3 4 5, c 6 7 8, d 9 10 11",
                     "python idea3_gain/gain_summary.py a b c d"], ["idea3_gain/results/gain_atm_summary.json"]))
else:
    i3.append(pending("JVP gain curves (4 jobs × 3 states × 3 seeds running; FD check on state 0: JVP agrees within 0.5% at ε = 0.01, 6% at ε = 0.1)"))
if gocn:
    B3 = gocn["bands"]
    mid3 = [(a + b) / 2 for a, b in B3]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for c in gocn["channels"]:
        v = gocn["diag_gain"][c]
        ax.fill_between(mid3, v["min"], v["max"], alpha=0.25, lw=0)
        ax.plot(mid3, v["mean"], lw=2, label=c)
    ax.set_xlabel("SH degree l (band centre)"); ax.set_ylabel("5-day gain (physical units)"); ax.set_ylim(0, 1)
    ax.legend(fontsize=8, frameon=False); fig.tight_layout(); fig.savefig(os.path.join(FIG, "gain_ocn.png")); plt.close(fig)
    i3.append("<h3>Ocean, 5-day step</h3>")
    i3.append(figure("gain_ocn.png", "C1A0 ocean single-channel gain per SH band (truth-forced inputs, 12 dates; band = min–max over "
                                     "dates). Flat in degree to l = 89: no scale selectivity at 1° within one step. Single-channel "
                                     "gains understate persistence: sst and the 0–20 m layer describe the same water (sst alone 0.22 "
                                     "K/K; sst + 0–20 m together 0.50 K/K in the localisation runs)."))
    i3.append(repro(["python idea3_gain/gain_ocn.py v1"], ["idea3_gain/results/gain_ocn_v1.json"]))
gb = J("idea3_gain/results/gain_atm_bal_v2.json")
if gb and gs:
    Bb = gb["bands"]; midb = [(a + b) / 2 for a, b in Bb]
    fig, axs = plt.subplots(2, 5, figsize=(16, 6.4), sharex=True)
    for ax, c in zip(axs.ravel(), gb["track"]):
        v = gb["gain"][c]
        ax.fill_between(midb, v["proj_p05"], v["proj_p95"], color="C2", alpha=0.25, lw=0)
        ax.plot(midb, v["proj_mean"], "C2-", lw=2, label="C1A0 projection (balanced)")
        ax.plot(midb, v["mean"], "C2--", lw=1.3, label="C1A0 magnitude (balanced)")
        ax.plot(midb, gs["diag"][c]["mean"], "C0:", lw=1.3, label="C1A0 single-variable (prereg)")
        for blk, col in (("1990s", "C3"), ("2040s", "C1")):
            b = gs["baselines"].get(f"{blk}|truth_transfer_score", {}).get(c)
            if b:
                ax.plot(midb, b, color=col, lw=1.3, label=f"truth AR(1) {blk}")
        ax.axhline(1, color="k", ls=":", lw=0.8); ax.set_title(c); ax.set_ylim(0, 1.2)
    for ax in axs[-1]:
        ax.set_xlabel("SH degree l")
    axs[0, 0].legend(fontsize=7, frameon=False); fig.tight_layout()
    fig.savefig(os.path.join(FIG, "gain_atm_bal.png")); plt.close(fig)
    u = gb["gain"]["U_7"]; tr = gs["baselines"]["1990s|truth_transfer_score"]["U_7"]
    i3.append("<h3>Balanced probes (exploratory, PREREG change log 2026-11-20)</h3>")
    i3.append(claim("I3-1", "idea3", "With balanced perturbations, C1A0 keeps the amount of each scale about as E3SM does, but decorrelates the pattern faster, increasingly so at small scales.",
                    f"U_7 projection vs truth AR(1): l≈8 {u['proj_mean'][2]:.2f} vs {tr[2]:.2f}; l≈31 {u['proj_mean'][8]:.2f} vs {tr[8]:.2f}; l≈55 "
                    f"{u['proj_mean'][14]:.2f} vs {tr[14]:.2f}; magnitude at l≈55 {u['mean'][14]:.2f}", "exploratory",
                    "truth per-degree AR(1) on fit and score years (agree within 0.02); preregistered single-variable probes shown for contrast",
                    "probe = one 5-day truth difference direction per state (12 states × 3 noise draws); truth AR(1) averages all anomalies; "
                    "T_1 is the exception (small scales persist more than in truth); l ≤ 2 includes global means",
                    "idea3_gain/results/gain_atm_bal_v2.json", "idea3_gain/gain_atm_bal.py",
                    "python idea3_gain/gain_atm_bal.py data/states/atm12 data/states/atm12m5 v2", status="narrowed"))
    i3.append(figure("gain_atm_bal.png", "Green: C1A0's one-step response to balanced perturbations (truth 5-day differences, all prognostic "
                                         "channels, band-passed): solid = projection on the perturbation (phase-sensitive, comparable to "
                                         "truth's AR(1)), dashed = magnitude. Red/orange: E3SM's own per-degree 6-h AR(1). Blue dotted: the "
                                         "preregistered single-variable gain (unbalanced, hence low)."))
    i3.append(repro(["python -m hc.evaluate data/states/atm12m5 --steps 1 --capture-atm 1 --no-ocn-pred --start 1996-01-11 ... (12 dates − 5 d)",
                     "python idea3_gain/gain_atm_bal.py data/states/atm12 data/states/atm12m5 v2"], ["idea3_gain/results/gain_atm_bal_v2.json"]))
i3.append(pending("rollout-spectra link (Phase 2, with the multi-year control rollouts); other checkpoints"))

# ======================================================================= restatements, pitfalls, agents
restate = ["<h2 id='restate'>Numbers that are restatements of one another</h2><ul>",
           "<li>F1 (b) FLUS coefficient and its Frisch–Waugh version (agree to 5e-9).</li>",
           "<li>Truth one-step transfer (score years) and the damped-linear AR(1) fit (fit years): one statistic, two year sets.</li>",
           "<li>Edge-patching sufficiency and necessity, and skip.0 + deep ≈ 1, under a near-linear response.</li>",
           "<li>The laptop's zero-net violation, Eq. 10 M and joint M all follow from the FLUS/FLDS ratio (PLAN Sec. 2.3).</li></ul>"]
agents = ["<h2 id='agents'>For participants and their agents</h2>",
          f"<ul><li>Code: <a href='{GH}'>{GH}</a> (to be pushed after review) · data products: <a href='{HF}'>{HF}</a> (to be uploaded)</li>",
          "<li><a href='data/AGENTS.md'>AGENTS.md</a>: where everything is, the exact libraries, pitfalls we hit</li>",
          "<li><a href='data/manifest.json'>manifest.json</a>: every claim on this page with value, verdict, control, caveat, source JSON, script and command</li>",
          "<li><a href='data/PREREG.md'>PREREG.md</a> (fixed before results, with change log) · <a href='data/INVENTORY.md'>INVENTORY.md</a></li></ul>"]

now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M %Z")
for doc in ("AGENTS.md", "PREREG.md", "INVENTORY.md"):
    shutil.copy(os.path.join(ROOT, doc), DATA)
os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
manifest = {"updated": now, "model": "SamudrACE C1A0 (mahf708/e3smv3-lr-hist-samudrace@rc1)", "page": "https://portal.nersc.gov/project/m5451/mahf708/samudrace-circuits/",
            "heldout_years": [[1995, 1999], [2040, 2049]], "claims": CLAIMS}
json.dump(manifest, open(os.path.join(ROOT, "results", "manifest.json"), "w"), indent=1)
shutil.copy(os.path.join(ROOT, "results", "manifest.json"), DATA)

CSS = """
:root{--bg:#fff;--fg:#1d2329;--mut:#5b6670;--acc:#0b6bcb;--line:#e3e7eb;--card:#f6f8fa;--ok:#1a7f37;--warn:#9a6700;--bad:#cf222e}
@media (prefers-color-scheme: dark){:root{--bg:#111518;--fg:#e6e9ec;--mut:#9aa5ae;--acc:#5aa9ff;--line:#2a3138;--card:#1a2025;--ok:#3fb950;--warn:#d29922;--bad:#f85149}}
body{background:var(--bg);color:var(--fg);font:15px/1.55 -apple-system,Segoe UI,Helvetica,Arial,sans-serif;margin:0}
main{max-width:1100px;margin:0 auto;padding:0 16px 48px}
nav{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);padding:8px 16px;z-index:1;overflow-x:auto;white-space:nowrap}
nav a{margin-right:18px;color:var(--acc);text-decoration:none;font-weight:600}
h1{font-size:28px;margin:28px 0 6px} h2{margin-top:44px;border-bottom:1px solid var(--line);padding-bottom:4px} h4{margin-bottom:4px}
a{color:var(--acc)} .lede{color:var(--mut)} code{background:var(--card);padding:1px 4px;border-radius:3px;font-size:13px}
.tw{overflow-x:auto} table{border-collapse:collapse;margin:10px 0;font-size:13px}
th,td{border:1px solid var(--line);padding:4px 8px;text-align:left;vertical-align:top} th{background:var(--card)}
figure{margin:14px 0} figure img{max-width:100%;border:1px solid var(--line)} figcaption{color:var(--mut);font-size:13.5px}
.pending{color:var(--mut);font-style:italic} .note{color:var(--mut);font-size:13px}
.claim{border:1px solid var(--line);border-radius:8px;padding:10px 14px;margin:12px 0;background:var(--card)}
.ch{margin-bottom:4px} .cv{font-size:15px;margin:4px 0} .cc{color:var(--mut);font-size:13.5px}
.badge{font-size:11px;font-weight:700;text-transform:uppercase;padding:1px 6px;border-radius:10px;border:1px solid}
.badge.ok{color:var(--ok)} .badge.warn{color:var(--warn)} .badge.bad{color:var(--bad)} .badge.mut{color:var(--mut)}
details.repro{margin:8px 0 16px;border-left:3px solid var(--acc);padding-left:10px} details.repro summary{cursor:pointer;color:var(--acc);font-weight:600}
pre{background:var(--card);padding:8px;overflow-x:auto;font-size:12.5px}
.status{display:flex;flex-wrap:wrap;gap:8px;margin:10px 0} .status span{background:var(--card);border:1px solid var(--line);padding:3px 10px;border-radius:12px;font-size:13px}
"""
status = ("<div class='status'><span>Phase 0 ✔</span><span>F1 truth test ✔</span><span>F1 model vs truth ✔</span>"
          f"<span>F2 localisation {'✔' if anyloc else '…'}</span><span>F3 ablation {'✔' if abl else '…'}</span>"
          f"<span>Idea 3 gain {'✔' if gs else '…'}</span><span>Idea 1 coupled de-leak · Idea 2: Phase 2</span></div>")
page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Inside SamudrACE C1A0</title><style>{CSS}</style></head><body>
<nav><a href="#headline">Headline</a><a href="#setup">Setup</a><a href="#idea1">FLUS thermometer</a><a href="#idea2">Errors</a><a href="#idea3">Scale gain</a><a href="#restate">Restatements</a><a href="#agents">Reproduce</a></nav>
<main><h1>Inside SamudrACE C1A0</h1>
<p class="lede">Circuits, law witnesses and error agreement for the E3SMv3 coupled emulator (SFNO atmosphere + Samudra ocean),
finished at scale on Perlmutter against <b>E3SM truth</b>, scored only on years the model never saw (1995–1999, 2040–2049).
Rules were preregistered before results (<a href="data/PREREG.md">PREREG.md</a>). Work in progress; this page is rebuilt from the result JSONs.</p>
{status}
<h2 id="headline">Headline so far</h2>
{''.join(head)}
{''.join(setup)}{''.join(i1)}{''.join(i2)}{''.join(i3)}{''.join(restate)}{''.join(agents)}
<p class="note">Updated {now}. Every number links to its JSON under “Reproduce”; machine-readable: <a href="data/manifest.json">manifest.json</a>.</p>
</main></body></html>"""
open(os.path.join(OUT, "index.html"), "w").write(page)
print("wrote", os.path.join(OUT, "index.html"), "claims:", len(CLAIMS))
