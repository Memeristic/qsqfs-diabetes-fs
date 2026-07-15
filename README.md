# QSQ-FS · Multimodal Diabetes Feature Selection

A **Quorum Sensing & Quorum Quenching** swarm heuristic for feature selection on
high-dimensional, multimodal clinical data (MIMIC-IV), with a two-stage pipeline
(per-modality exploration → multimodal fusion), a neural classifier, and an
interactive Streamlit dashboard.

Performance is estimated with **nested cross-validation**: feature selection
runs inside each outer training fold only, so reported AUC/accuracy are unbiased
estimates of generalisation rather than resubstitution figures. This implements
the methodology of the thesis (Chapter 3).

---

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate      # optional
pip install -r requirements.txt
python verify_setup.py          # ~1–2 min end-to-end smoke test on synthetic data
streamlit run app.py            # launch the dashboard
```

`verify_setup.py` prints the nested-CV numbers, then runs the comparative
and ablation studies at tiny budgets so you can confirm everything works before
scaling up.

---

## What the pipeline does

1. **Load** any supported source through one loader (`src/datasets.py`): real
   MIMIC-IV (`hosp/` + `icu/`, `.csv` or `.csv.gz`), the synthetic demo, or any
   tidy one-row-per-patient CSV (Pima, UCI, your own export). Modalities are
   partitioned automatically for non-MIMIC data, so the pipeline is not
   MIMIC-only.
2. **Build modalities** — labs, vitals, medications, diagnoses (MIMIC), or
   keyword-grouped modalities (generic CSV). Diabetes-defining ICD codes and
   diabetes drugs are stripped (label leakage) in every path.
3. **Stage 1** runs QSQ-FS *per modality* to shortlist features, with an
   AUC-aware (`balanced`) fitness suited to imbalanced cohorts.
4. **Stage 2** fuses the shortlists and runs QSQ-FS again with a
   frequency-weighted initialisation.
5. **Fuse & classify** with multimodal fusion (`src/fusion.py`): a feature-level
   shared latent space, a decision-level per-modality ensemble, or the default
   **hybrid** of the two, with minority-class balancing and a tuned decision
   threshold to protect sensitivity. An MLP or KNN can be selected instead.
6. **Evaluate** with nested CV: steps 3–5 run **inside each outer training fold
   only**; the held-out fold is scored once. Reported metrics are the mean ± 95%
   CI across folds, plus feature-selection stability.
7. **Compare & ablate** — QSQ-FS vs the proposal's named metaheuristics
   (**RIME, PLO, HGS**) and GA/PSO plus SVM/RandomForest/XGBoost (Wilcoxon
   significance), and a four-way ablation of QSQ-FS's mechanisms.

> **Performance vs interpretation.** The Dashboard tab's numbers come from nested
> CV. The Feature Selection tab fits on the full dataset to *show which features
> the method favours* — that view is explicitly labelled interpretation, not a
> performance estimate.

---

## Repository layout

```
qsqfs-diabetes-fs/
├── app.py                     # Streamlit dashboard (8 tabs)
├── config.yaml                # single canonical config (keys map to QSQFS args)
├── verify_setup.py            # end-to-end smoke test
├── requirements.txt           # core deps; torch/xgboost/shap optional (commented)
├── packages.txt, runtime.txt  # Streamlit Community Cloud hints
├── .streamlit/config.toml     # theme + server settings
├── .github/workflows/ci-deploy.yml   # lint + smoke tests on push
├── push_to_github.sh          # one-time push helper
├── DEPLOYMENT.md              # GitHub + Streamlit Cloud, click-by-click
├── matlab/
│   ├── QSQ_FS.m               # reference optimiser (EMA, guided refinement, elitism, seed)
│   └── demo_QSQ_FS.m          # minimal MATLAB example
├── src/
│   ├── qsfs.py                # QSQ-FS engine
│   ├── evaluation.py          # nested-CV + leakage-free select-then-score
│   ├── neural_model.py        # MLP (best-weight restore, L2, seeded)
│   ├── data_loader.py         # synthetic + real long-format loaders
│   ├── modality_builder.py    # modality pivots, leakage filters, modality map
│   ├── stage1_runner.py       # full-data Stage-1 (interpretation)
│   ├── stage2_fuser.py        # full-data Stage-2 (frequency-weighted)
│   ├── comparative_analysis.py# GA/PSO/RIME/PLO/HGS + SVM/RF/XGB, equal budget
│   ├── ablation.py            # use_qs/use_qq/use_cache/elitism switches
│   ├── stats_analysis.py      # permutation importance, clinical alignment, LaTeX
│   └── plotting.py            # convergence/ROC/confusion/comparison figures
└── tests/test_smoke.py        # pytest CI checks
```

---

## Configuration

Everything lives in `config.yaml`. The keys under `stage1:` and `stage2:` map
**directly** onto `QSQFS.__init__`. Notable knobs:

- `evaluation.n_outer_folds` — nested-CV outer folds (5 default; 10 for final runs).
- `preprocessing.combine_how` — `inner` (complete cases) or `outer` (mean-impute;
  recommended for real MIMIC-IV, where most patients lack ICU vitals).
- `comparative.n_agents` / `comparative.max_iter` — shared, equal budget for every
  metaheuristic baseline.
- `neural_model.l2_alpha` — the L2 weight-decay penalty applied by the MLP
  classifier.

---

## Running on real MIMIC-IV

1. Download the MIMIC-IV demo (or full set, with credentialing) and point
   `paths.data_root` at the folder that contains `hosp/` and `icu/`.
2. Set `demo_mode: false` (or toggle it off in the sidebar).
3. Consider `combine_how: outer` so patients without ICU stays are retained.

The loaders read both `.csv` and `.csv.gz`; `labevents` and `chartevents` are
pivoted from long format. When you pick **Real MIMIC-IV folder** in the app it
first *validates* the folder (checks the required files/columns exist) and shows
a live progress bar during the slow chunked reads.

---

## Three data sources (app) — and using your own data

The sidebar **Data source** selector supports:

- **Demo (synthetic)** — an overlapping-distribution synthetic cohort;
  always works, no download.
- **Real MIMIC-IV folder** — validated before the (slow) pipeline runs.
- **Upload my own CSV** — a generic **column-mapping** layer lets you use *any*
  tidy one-row-per-patient CSV (a different EHR export, a Kaggle dataset, ...).
  You tell the app which column is the patient id, which is the label (must be
  binary), and which features are numeric vs categorical; it builds the matrix
  (numeric mean-imputed, categoricals one-hot encoded) and runs the same
  leak-free pipeline. MIMIC-IV stays a first-class preset — this is additive.

The **MATLAB** port has the same choices without a web UI: `run_pipeline('ui')`
pops up a chooser (Demo / MIMIC-IV folder / single CSV) with native
file/folder pickers, `run_pipeline('real')` (no path) pops the folder picker,
and `run_pipeline('csv', '/path/data.csv')` runs on an arbitrary CSV. See
[`matlab/README.md`](matlab/README.md).

---

## How the MATLAB port and the Streamlit app relate

They are **two independent implementations of the same algorithm**, not a
client/server pair — they do not call each other at runtime.

- **`app.py` + `src/` (Python)** is the **user-facing product**: the interactive
  dashboard, the thing you deploy to Streamlit Community Cloud.
- **`matlab/` (MATLAB/Octave)** is a **standalone reference/validation
  implementation** of the identical pipeline (same two-stage QSQ-FS, same
  leak-free nested-CV protocol, same modality construction and leakage filters).

Why keep both? A second, independent implementation in a different language is
strong evidence the method is correctly specified and not an artefact of one
codebase — and on the real MIMIC-IV demo the two agree (e.g. both report ~35%
diabetes prevalence and comparable nested-CV AUC). Use the Streamlit app to
*explore and present*; use the MATLAB code to *reproduce/validate* or to run
where MATLAB is the required environment. There is deliberately **no MATLAB
GUI** beyond the dataset pickers: the Streamlit app already fills the
interactive-front-end role and has a free hosting path, whereas a MATLAB GUI
would need a paid license/Runtime to run.

---

## Reproducing the deep analysis and figures

Two scripts regenerate the full analytical outputs (the numbers and figures behind
the reports) on any dataset. Run them from the project root so `src` is importable:

```bash
# Dataset-agnostic driver: selection + multimodal fusion + comparative
# (incl. RIME/PLO/HGS) + ablation -> analysis_out/ . Works on any source:
python run_analysis.py                             # synthetic demo
python run_analysis.py mimic /path/to/mimic-iv     # real MIMIC-IV folder
python run_analysis.py csv  /path/to/data.csv      # any tidy CSV (label auto-detected)
python make_thesis_figures.py analysis_out         # fusion + comparative figures

# SPSS-style statistics (group descriptives, t-test/Mann-Whitney, Cohen's d,
# chi-square, FDR correction) are written by run_analysis.py to
# analysis_out/spss_*.csv plus an effect_sizes.png figure.

# Stability check: repeat the fusion nested-CV across several random seeds.
python run_robustness.py mimic /path/to/mimic-iv   # AUC/F1/sensitivity mean +/- SD

# Narrative analysis: nested-CV metrics, confusion matrix, comparative + ablation,
# clinical alignment, permutation importance, and 6 core figures -> analysis_out/
python run_full_analysis.py /path/to/mimic-iv     # or: python run_full_analysis.py demo

# The 20-figure deep-analysis suite (classical stats + ML diagnostics) -> analysis_out/fig20/
python run_20_figures.py /path/to/mimic-iv         # or: python run_20_figures.py demo
```

Both write only aggregate results (metrics, counts, feature names) — never
per-patient rows — and default to the `config.yaml` `data_root` if no path is given.
The figures are plain PNGs suitable for dropping straight into a thesis.

---

## Optional accelerators

The default classifier is a regularised **sklearn** MLP so the app runs anywhere,
including Streamlit Community Cloud, with no heavy downloads. To enable extras,
uncomment the relevant lines in `requirements.txt`:

- `torch` — GPU/BatchNorm MLP (`src/neural_model.py` auto-detects it).
- `xgboost` — adds an XGBoost comparative baseline.
- `shap` — SHAP feature importance (otherwise permutation importance is used).

---

## Deployment

See [`DEPLOYMENT.md`](DEPLOYMENT.md). In short: push with `./push_to_github.sh
<your-repo-url>`, then link the repo once on Streamlit Community Cloud with main
file `app.py`. After that, every push redeploys automatically and the CI workflow
keeps `main` green.

---

## Results note

Headline result on the **MIMIC-IV clinical database demo** (100 patients,
135 features, 35% prevalence), under nested cross-validation with feature
selection performed inside each outer training fold:

| Metric | Value |
|---|---|
| AUC | **0.581** (95% CI 0.480 – 0.682) |
| Accuracy | 0.52 |
| F1 | 0.497 |
| Features selected | ~19 of 135 |
| Majority-class baseline | **0.65 accuracy** |

The AUC confidence interval includes 0.50. On a cohort of this size, and with the
diagnostic criterion excluded (see `docs/leakage_control.md`), the model is **not
distinguishable from chance**, and no comparison against any baseline reaches
significance under Bonferroni correction.

This is reported as-is. Two things drive it, and both are properties of the
setting rather than defects in the search:

1. **The exclusion policy is consequential.** Serum glucose alone achieves a
   univariate AUC of ~0.87 on this cohort. Because glucose *defines* the
   diagnosis, admitting it as a predictor would reduce the task to label
   recovery; excluding it removes most of the separable signal, which is the
   intended behaviour of the protocol.
2. **n = 100 is the binding constraint.** Ten outer folds give ~10 test patients
   each; the resulting confidence intervals are ±0.10 wide, and a 10-sample
   Wilcoxon cannot reach a Bonferroni-corrected threshold. The demo subset cannot
   resolve an effect of this size. Credentialed access to the full MIMIC-IV
   (~300k patients) is required for a powered test.

Strong full-feature learners (SVM, RF) outperform every feature-selection method
paired with KNN here. That is an expected outcome under an honest protocol.

---

## Citation / academic use

This is research code accompanying a Master's thesis (Wenzhou University). If you
build on it, please cite the thesis.
