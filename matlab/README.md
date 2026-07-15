# QSQ-FS — MATLAB implementation

A full MATLAB port of the Python pipeline (`src/`, `app.py`), not just the
core optimiser. Portable to plain MATLAB (no toolboxes required) **and**
GNU Octave 8.x, except for `QSQ_FS.m` itself, which uses MATLAB's
`RandStream`/`rand(stream, ...)`/`randperm(stream, ...)` idiom for
independent, reproducible random streams — this is MATLAB-only. An
Octave-compatible variant of that one idiom can be swapped in if you need to
run the whole thing under Octave (see "Running under Octave" below); every
other file here is Octave-clean as shipped.

## Files

| File | Mirrors (Python) | Purpose |
|---|---|---|
| `QSQ_FS.m` | `src/qsfs.py` | Core two-stage QS/QQ optimiser, with ablation switches. |
| `generate_synthetic_data.m` | `src/data_loader.py` (demo path) | Overlapping-distribution synthetic cohort. |
| `load_mimic_data.m` | `src/data_loader.py` (real path) | Loads a real MIMIC-IV directory tree. |
| `load_large_event_table.m` | `_load_large_event_table` | Scalable, chunked, frequency-filtered reader for `labevents`/`chartevents` (100M–330M+ rows). |
| `build_modalities.m` | `src/modality_builder.py` | Labs/vitals/meds/dx modality construction, drug-leakage filtering, ICD-leakage stripping, sparse-coverage handling, combined matrix. |
| `mlp_train_evaluate.m` | `src/neural_model.py` | Toolbox-free MLP (manual backprop); early stopping restores best-validation weights. |
| `nested_cv_pipeline.m` | `src/evaluation.py` | Leakage-free nested CV: Stage 1 per modality → Stage 2 fusion → final classifier, all inside the outer train fold only. |
| `ablation_study.m` | `src/ablation.py` | QS/QQ/cache/elitism ablation using `QSQ_FS.m`'s switches. |
| `feature_selectors.m` | `src/optimizers.py` | RIME / PLO / HGS / GA / PSO as first-class binary feature selectors (equal-budget, shared leak-free fitness). |
| `multimodal_fusion.m` | `src/fusion.py` | Feature-, decision-, and hybrid-level multimodal fusion (toolbox-free manual-MLP encoders and heads). |
| `comparative_baselines.m` | `src/comparative_analysis.py` | QSQ-FS vs RIME / PLO / HGS / GA / PSO (equal budget) vs a full-feature logistic-regression baseline. |
| `run_pipeline.m` | `verify_setup.py` | End-to-end driver, demo or real data. |

## Quick start

```matlab
run_pipeline               % synthetic demo data, ~35 features, prints AUC/ACC
run_pipeline('ui')         % pops up a dataset chooser: Demo / MIMIC-IV folder / CSV
run_pipeline('real')       % pops up a folder picker for a MIMIC-IV directory
run_pipeline('real', '/path/to/mimic-iv')   % real MIMIC-IV, explicit path
run_pipeline('csv', '/path/to/data.csv')    % any tidy CSV with a 'label' column
```

## Named optimisers and multimodal fusion

Two additions bring the MATLAB reference in line with the proposal:

```matlab
% The proposal's named metaheuristics as first-class selectors (equal budget):
sel = feature_selectors('RIME', X, y, 30, 20, 42, 3, 3, 'balanced');
sel = feature_selectors('PLO',  X, y, 30, 20, 42, 3, 3, 'balanced');
sel = feature_selectors('HGS',  X, y, 30, 20, 42, 3, 3, 'balanced');

% Multimodal fusion (feature-level shared latent / decision-level / hybrid):
m = multimodal_fusion(Xtr, ytr, Xte, yte, modalityMap, 'hybrid', cfg, 42);

demo_fusion_and_optimizers   % runs both on synthetic data, no MIMIC needed
```

The `ui` / path-less `real` modes open native `uigetdir`/`uigetfile` dialogs so
you can point-and-click your dataset (the MATLAB counterpart to the Streamlit
data-source selector). They need a MATLAB/Octave desktop; under `-nodisplay`,
pass the path explicitly. `load_generic_csv.m` handles arbitrary CSVs (numeric
columns used as-is, categoricals one-hot encoded, binary label) and is portable
to Octave.

## Ablation switches

`QSQ_FS.m` exposes four switches — `useQS`, `useQQ`, `useCache`, `elitism`
(all default `true` = full algorithm) — so `ablation_study.m` can disable each
mechanism independently. Quorum Quenching uses the effective fitness of thesis
Sec 3.5.3 / Eq 3.8 (effective fitness = raw fitness − suppression), matching
`src/qsfs.py`'s `_evaluate_population`.

## Real MIMIC-IV setup

1. Obtain MIMIC-IV access via PhysioNet (requires CITI training
   credentialing) and download it — this cannot be done from within this
   pipeline; MIMIC-IV is not publicly downloadable.
2. Lay the files out exactly as they ship from PhysioNet:
   ```
   dataRoot/hosp/{patients,diagnoses_icd,d_icd_diagnoses,labevents,
                  d_labitems,pharmacy}.csv[.gz]
   dataRoot/icu/{chartevents,d_items}.csv[.gz]
   ```
3. **Decompress `.csv.gz` first** (`gunzip *.csv.gz`) — `load_mimic_data.m`
   uses plain `fopen`/`textscan` for portability (no toolbox / `gunzip`
   dependency), which cannot stream gzip directly. (MATLAB users with the
   Statistics/Datastore toolboxes can swap in `tabularTextDatastore`, which
   does support `.gz`, if preferred.)
4. `run_pipeline('real', '/path/to/mimic-iv')`.

`labevents` (~120M rows) and `chartevents` (~330M rows) are read in
2M-row chunks by `load_large_event_table.m`: pass 1 streams only `itemid` to
get true frequency counts across the whole file, pass 2 keeps only rows
whose itemid is in the most-frequent set. Peak memory is bounded by the
*filtered* result, not the source table.

### Real-data robustness (quote-aware parsing + input checks)

Two things that arise on real MIMIC-IV (but not the synthetic demo) are handled:

- **Quoted free-text with embedded commas.** Real event tables have free-text
  columns (e.g. `labevents.comments`) whose values contain commas and doubled
  quotes inside quotation marks. The readers parse with `textscan`'s `%q`
  (RFC-4180 quote-aware) instead of `%s`, so those rows do not mis-split and
  desync the rest of the file. `%q` behaves identically on MATLAB and Octave.
- **Clear, early errors.** `load_mimic_data`, `load_large_event_table`, and
  `build_modalities` validate their inputs up front and fail with an
  actionable message instead of a deep stack trace: missing folder /
  missing `hosp/`, a `.csv.gz` handed in un-decompressed, a file with no header,
  `top_k` ≤ 0, missing `diagnoses`, an empty cohort after an `inner` join, zero
  feature columns, or a label with only one class (which would otherwise fail
  deep inside cross-validation).

### Role of this MATLAB code / front-end

This MATLAB port is the research and validation reference implementation. The
user-facing application is the Streamlit app (`app.py`); there is deliberately
no MATLAB App Designer GUI. Rationale: the Streamlit app already provides the
interactive front-end and has a real deployment path (Streamlit Community
Cloud), whereas a MATLAB GUI would need a MATLAB license or MATLAB Runtime to
run and has no comparable free hosting story. Use this MATLAB code to (a)
independently reproduce and validate the Python results in a second language,
and (b) run the algorithm where MATLAB is the required environment. Keep new
user-facing features in the Streamlit app.

## Parity with the Python engine

- **Stage-2 frequency-weighted pool.** `QSQ_FS.m` accepts `featurePool` and
  `poolWeights` parameters: Stage 2 seeds its colonies from that pool, weighted
  by each feature's cross-modality Stage-1 selection count, while the
  mutation/guided-refinement operators can still flip any of the full `d` bits
  — exactly like `src/qsfs.py`. `nested_cv_pipeline.m` builds the weight vector
  from its Stage-1 `poolWeight` map and calls `QSQ_FS` on the *full* training
  matrix, so returned indices are already global (no remap). This matches thesis
  Sec 3.5.1.
- **Classifiers.** `mlp_train_evaluate.m` and `comparative_baselines.m`'s
  logistic-regression baseline are toolbox-free (manual backprop / IRLS) so
  everything here runs on base MATLAB or Octave with zero toolbox
  dependencies. With the Statistics and Machine Learning Toolbox,
  `fitcnet`/`fitcsvm`/`TreeBagger` are drop-in replacements at the marked call
  sites in `comparative_baselines.m`.

## Validation

Every file here was validated under **GNU Octave 8.4** (MATLAB-language
compatible), with a small compatibility shim standing in only for the
`RandStream` object idiom that Octave does not implement; the shim is not
shipped, and `QSQ_FS.m` uses real `RandStream` throughout. Results:

- `QSQ_FS.m` recovers the 3 informative features out of 20 on a synthetic
  separation task (Stage 1 → Stage 2 fitness lift 0.77 → 0.87).
- All four ablation switches produce measurably different behaviour (No-Cache is
  ~2× slower with identical selected subsets; No-QQ / No-QS select worse subsets).
- **Real MIMIC-IV demo (v2.2, 100 patients), end-to-end.** The full real-data
  path — `load_mimic_data.m` → `build_modalities.m` → `nested_cv_pipeline.m` —
  was run against the PhysioNet MIMIC-IV clinical database demo.
  `load_large_event_table.m` streamed the real `labevents` (~108k rows) and
  `chartevents` (~669k rows), frequency-filtered itemids, and pivoted them;
  `build_modalities.m` produced a clean 100 × 116 matrix (35% diabetes
  prevalence, no NaN/Inf), with the diabetes-drug leakage filter firing on
  ~1000 real free-text drug rows and diabetes ICD codes stripped. Leakage-free
  nested CV gave AUC ≈ 0.78, and the most stable selected feature across folds
  was itemid 50931 (serum glucose).
- The Stage-2 pool parameters (`featurePool`/`poolWeights`) were unit-tested:
  given a pool, `QSQ_FS` recovers the informative in-pool features and also
  selects informative features outside the pool, confirming the seed biases but
  does not constrain the search.
- All input-validation paths were tested to confirm they raise the intended
  clear errors (missing folder, `.gz` not decompressed, one-class label, empty
  cohort, bad `top_k`, and so on).
- `nested_cv_pipeline.m` end-to-end on synthetic data gave AUC ≈ 0.83,
  consistent with the Python nested-CV numbers on the equivalent demo.
- `ablation_study.m` and `comparative_baselines.m` reproduce the same
  qualitative pattern as the Python results (a strong full-feature classical
  baseline can beat any single feature-selection method on a small/dense
  problem — expected behaviour).

If you have access to real MATLAB, re-running `run_pipeline` there is a
recommended sanity check before any real-MIMIC-IV run; Octave's JIT is slower on
the manual per-colony loops, so real MATLAB will also be substantially faster
for large population/iteration budgets.

## Running under Octave

Every file except `QSQ_FS.m` runs unmodified on Octave 8.x. `QSQ_FS.m` uses the
`RandStream` object idiom for independent reproducible streams; to run it under
Octave, replace the `localRng`/`RandStream` calls with Octave's `rand`/`randperm`
seeding. All other files (data loading, modality building, nested CV, ablation,
baselines) are Octave-clean as shipped.
