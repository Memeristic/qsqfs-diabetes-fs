"""
app.py — QSQ-FS Multimodal Diabetes Feature Selection dashboard
================================================================
Streamlit front-end for the QSQ-FS multimodal feature-selection framework.

How results are reported in the UI:
  * Performance metrics (AUC/Acc/F1) come from src.evaluation.nested_cv_evaluate
    -- feature selection runs inside each outer TRAIN fold only, so no test-fold
    information leaks into selection.
  * The feature lists in the "Feature Selection" tab come from a full-data fit
    and are labelled INTERPRETATION (which features the method favours), not a
    performance estimate.
  * ROC / confusion use concatenated out-of-fold predictions.

Three data sources are supported:
  1. Demo (synthetic)         — always works, no download.
  2. Real MIMIC-IV folder     — validated before the (slow) pipeline runs.
  3. Upload my own CSV        — a generic column-mapping layer (src/schema.py)
                                lets ANY tidy one-row-per-patient CSV be used,
                                not just MIMIC-IV.

Run locally:   streamlit run app.py
"""

from __future__ import annotations

import io
import json
import os
import tempfile
import time
import zipfile

import numpy as np
import pandas as pd
import streamlit as st
import yaml

from src.data_loader import MIMICDataLoader
from src.figures20 import build_figures
from src.leakage import assert_no_leakage, leaky_columns, resolve_leaky_itemids
from src.modality_builder import (
    extract_diabetes_label, build_labs, build_vitals, build_medications,
    build_diagnoses, build_combined_matrix, build_modality_map,
)
from src.stage1_runner import run_all_modalities
from src.stage2_fuser import run_stage2_fusion
from src.evaluation import nested_cv_evaluate
from src.comparative_analysis import run_comparative_analysis
from src.ablation import run_ablation_study
from src.stats_analysis import (
    check_clinical_alignment, build_item_label_map,
    generate_latex_results_table, generate_ablation_latex_table,
)
from src import plotting_interactive as pi   # interactive (on-screen)
from src import schema                       # folder validation + CSV mapping
from ui.theme import inject_theme
from ui import captions as cap

st.set_page_config(page_title="QSQ-FS Diabetes FS", page_icon="🧬", layout="wide")

APP_VERSION = "v2.4.0"


# --------------------------------------------------------------------------- #
# Config + caching                                                            #
# --------------------------------------------------------------------------- #
@st.cache_data
def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


@st.cache_data(show_spinner=False)
def load_and_build(demo_mode: bool, combine_how: str, data_root: str,
                   _cfg: dict, _progress_cb=None):
    """Load + build modalities.

    ``data_root`` is passed explicitly (and is *hashable*) so it forms part of
    Streamlit's cache key. Two things depend on this:

      1. Correctness — it is written into the loader config below, so the loader
         reads the folder the user selected rather than the default path in
         ``config.yaml``.
      2. Cache invalidation — pointing the app at a different MIMIC-IV folder
         changes the cache key, so the results are recomputed for that folder.

    ``_cfg`` / ``_progress_cb`` stay underscore-prefixed (unhashed): the config
    dict isn't hashable and the progress callback shouldn't affect caching. On a
    cache HIT nothing runs (instant)."""
    cfg = dict(_cfg)
    cfg["demo_mode"] = demo_mode
    # Copy the nested ``paths`` dict before mutating so we never clobber the
    # shared global config object, then point the loader at the chosen folder.
    cfg["paths"] = dict(cfg.get("paths", {}))
    cfg["paths"]["data_root"] = data_root
    data = MIMICDataLoader(cfg).load_all(progress_cb=_progress_cb)
    label_df = extract_diabetes_label(
        data.get("patients"), data.get("diagnoses_icd"), data.get("d_icd_diagnoses"))

    # The exclusion policy applies on every path that builds a feature matrix.
    # Criterion analytes are keyed by opaque itemids, so they are resolved from
    # the database's own dictionaries and withheld by identifier before the
    # top-k ranking that fills each modality.
    leaky_ids = resolve_leaky_itemids(data.get("d_labitems"), data.get("d_items"))

    mod = {}
    for key, fn, arg in [
        ("labs", build_labs, data.get("labevents")),
        ("vitals", build_vitals, data.get("chartevents")),
        ("meds", build_medications, data.get("pharmacy")),
        ("dx", build_diagnoses, data.get("diagnoses_icd")),
    ]:
        df, _ = (fn(arg, label_df, leaky_itemids=leaky_ids)
                 if key in ("labs", "vitals") else fn(arg, label_df))
        if df is not None:
            mod[key] = df
    combined, names = build_combined_matrix(mod, how=combine_how)

    # The synthetic simulator emits named analytes, so the policy is also applied
    # to the column names; on the clinical path this is a no-op because the
    # itemid-level exclusion has already run. The assertion is the hard
    # post-condition: no excluded feature may reach the matrix on any source.
    withheld = set(leaky_columns(names, leaky_ids))
    names = [n for n in names if n not in withheld]
    assert_no_leakage(names, leaky_ids)

    mmap = build_modality_map(mod, names)
    ref_tables = {"d_labitems": data.get("d_labitems"), "d_items": data.get("d_items")}
    return mod, combined, names, mmap, ref_tables


def ss(key, default=None):
    return st.session_state.get(key, default)


cfg = load_config()

# --------------------------------------------------------------------------- #
# Sidebar                                                                      #
# --------------------------------------------------------------------------- #
st.sidebar.title("🧬 QSQ-FS Controls")

dark_mode = st.sidebar.toggle("🌙 Dark mode", value=False)
inject_theme(dark_mode)

st.sidebar.divider()
st.sidebar.markdown("#### 📁 Data source")
data_source = st.sidebar.radio(
    "Choose data source",
    ["Demo (synthetic)", "Real MIMIC-IV folder", "Upload my own CSV"],
    index=0 if cfg.get("demo_mode", True) else 1, label_visibility="collapsed")

demo_mode = data_source == "Demo (synthetic)"
uploaded_df = None
data_root_input = None

if data_source == "Real MIMIC-IV folder":
    mimic_uploads = st.sidebar.file_uploader(
        "Upload MIMIC-IV table(s)", type=["csv", "gz"], accept_multiple_files=True,
        help="Select any of: diagnoses_icd, labevents, patients, d_labitems, "
             "pharmacy, chartevents, d_items (.csv or .csv.gz). Files are "
             "matched by name and sorted into hosp/ or icu/ automatically.")
    if mimic_uploads:
        up_root, recognised, unrecognised = schema.save_uploaded_mimic_files(mimic_uploads)
        data_root_input = str(up_root)
        if recognised:
            st.sidebar.success("Recognised: " + ", ".join(recognised))
        if unrecognised:
            st.sidebar.warning(
                "Not a recognised MIMIC-IV table name, skipped: "
                + ", ".join(unrecognised))
    else:
        data_root_input = None

    st.sidebar.caption("Never commit real patient data or use it on a public deployment.")

elif data_source == "Upload my own CSV":
    up = st.sidebar.file_uploader(
        "Upload a CSV (one row per patient)", type=["csv"])
    if up is not None:
        try:
            uploaded_df = pd.read_csv(up)
            st.sidebar.success(f"Loaded {uploaded_df.shape[0]:,} rows × {uploaded_df.shape[1]} cols")
        except Exception as exc:
            st.sidebar.error(f"Could not read that CSV: {exc}")

st.sidebar.divider()
st.sidebar.markdown("#### ⚙️ Model settings")
combine_how = st.sidebar.selectbox(
    "Modality join", ["inner", "outer"],
    index=0 if cfg["preprocessing"]["combine_how"] == "inner" else 1,
    help="How to combine modalities: 'inner' keeps patients present in all "
         "modalities; 'outer' keeps everyone and mean-imputes the gaps.")
n_outer = st.sidebar.slider("Nested-CV outer folds", 3, 10, cfg["evaluation"]["n_outer_folds"])
classifier = st.sidebar.selectbox("Final classifier", ["fusion", "auto", "mlp", "knn"], index=0)
if classifier == "fusion":
    fusion_strategy = st.sidebar.selectbox(
        "Fusion strategy", ["hybrid", "feature", "decision"], index=0,
        help="feature = shared latent space; decision = per-modality ensemble; "
             "hybrid = average of both.")
    cfg.setdefault("neural_model", {})["fusion_strategy"] = fusion_strategy

st.sidebar.divider()
st.sidebar.markdown("#### 🎚️ Search budget")
cfg["stage1"]["max_iter_stage1"] = st.sidebar.slider(
    "Stage-1 iterations", 5, 40, cfg["stage1"]["max_iter_stage1"])
cfg["stage2"]["max_iter_stage2"] = st.sidebar.slider(
    "Stage-2 iterations", 5, 80, cfg["stage2"]["max_iter_stage2"])

with st.sidebar.expander("ℹ️ Methodology note"):
    st.caption(cap.leakage_note())


# --------------------------------------------------------------------------- #
# First-run welcome (shown before any data is loaded / on empty states)       #
# --------------------------------------------------------------------------- #
def render_welcome():
    st.info(
        "**Welcome 👋** This app finds the small set of clinical measurements that "
        "best predict diabetes, using a bio-inspired search (QSQ-FS) with "
        "leakage-free, nested cross-validation scoring.\n\n"
        "**To start:** pick a **Data source** in the sidebar (leave it on *Demo* to "
        "try it instantly), then open the **🏠 Dashboard** tab and click "
        "**▶️ Run nested-CV pipeline**. Everything else (method comparison, "
        "ablation, clinical checks, export) unlocks from there.")


# --------------------------------------------------------------------------- #
# Generic uploaded-CSV column mapping                                         #
# --------------------------------------------------------------------------- #
def resolve_uploaded_csv(df: pd.DataFrame):
    """Interactive column-mapping → (X, y, names, mmap). Blocks with st.stop()
    until the user confirms a valid mapping."""
    st.subheader("Map your columns")
    st.caption("Your file isn't assumed to be MIMIC-IV — tell the app what its "
               "columns mean and it will build a feature matrix. Guesses are "
               "pre-filled; adjust as needed.")
    with st.expander("Preview uploaded data", expanded=False):
        st.dataframe(df.head(8), width="stretch")

    guess = schema.infer_column_kinds(df)
    cols = list(df.columns)

    # On an arbitrary CSV the app cannot know what the label means, so the
    # exclusion policy is advisory rather than enforced: surface any column that
    # looks like a diagnostic criterion and let the analyst decide.
    suspect = leaky_columns(cols)
    if suspect:
        st.warning(
            "**Possible label leakage.** These columns look like diagnostic "
            "criteria, treatment-indicating codes, or consequences of treatment: "
            + ", ".join(f"`{c}`" for c in suspect)
            + ". If your label is diabetes, including them lets the model recover "
              "the diagnosis instead of predicting it. Deselect them below unless "
              "you have a specific reason to keep them.")

    c1, c2 = st.columns(2)
    label_col = c1.selectbox(
        "Label column (what to predict — must have exactly 2 values)", cols,
        index=cols.index(guess["label_col"]) if guess["label_col"] in cols else 0)
    id_options = ["(none)"] + cols
    id_default = guess["id_col"] if guess["id_col"] in cols else "(none)"
    id_col = c2.selectbox("Patient ID column (optional, ignored as a feature)",
                          id_options, index=id_options.index(id_default))
    id_col = None if id_col == "(none)" else id_col

    feature_pool = [c for c in cols if c not in (label_col, id_col)]
    num_default = [c for c in guess["numeric"] if c in feature_pool]
    numeric_cols = st.multiselect(
        "Numeric feature columns", feature_pool, default=num_default,
        help="Continuous measurements (age, BMI, lab values, ...).")
    categorical_cols = st.multiselect(
        "Categorical feature columns (one-hot encoded)", feature_pool,
        default=[c for c in feature_pool if c not in numeric_cols],
        help="Discrete categories (sex, smoking status, ...).")

    if st.button("✅ Build feature matrix from this mapping", type="primary"):
        try:
            X, y, names, mmap = schema.build_matrix_from_mapping(
                df, label_col, numeric_cols, categorical_cols, id_col)
            st.session_state.update(
                up_X=X, up_y=y, up_names=names, up_mmap=mmap,
                up_sig=(df.shape, label_col, tuple(numeric_cols), tuple(categorical_cols)))
            st.success(f"Built {X.shape[0]} patients × {X.shape[1]} features. "
                       f"Open the Dashboard tab to run the pipeline.")
        except ValueError as exc:
            st.error(str(exc))

    sig = (df.shape, label_col, tuple(numeric_cols), tuple(categorical_cols))
    if ss("up_X") is not None and ss("up_sig") == sig:
        return ss("up_X"), ss("up_y"), ss("up_names"), ss("up_mmap")
    st.stop()


# --------------------------------------------------------------------------- #
# Resolve X, y, names, mmap from the chosen source                            #
# --------------------------------------------------------------------------- #
st.title("QSQ-FS · Multimodal Diabetes Feature Selection")
st.caption(f"Quorum Sensing & Quorum Quenching heuristic for multimodal feature selection · {APP_VERSION}")

combined = None  # only set for the MIMIC/demo path (used by Data Explorer)
mod, ref_tables = {}, {}

if data_source == "Upload my own CSV":
    if uploaded_df is None:
        render_welcome()
        st.info("👈 Upload a CSV in the sidebar to begin.")
        st.stop()
    X, y, names, mmap = resolve_uploaded_csv(uploaded_df)
    st.session_state.update(X=X, y=y, names=names, mmap=mmap, mod={}, ref_tables={})

elif data_source == "Real MIMIC-IV folder":
    if data_root_input is None:
        render_welcome()
        st.info("👈 Upload your MIMIC-IV files in the sidebar to begin.")
        st.stop()
    report = schema.validate_mimic_folder(data_root_input or "")
    if report["errors"]:
        st.error("This data isn't a complete MIMIC-IV export yet:")
        st.markdown("**Missing / required:**")
        for e in report["errors"]:
            st.markdown(f"- {e}")
        if report["warnings"]:
            st.markdown("**Recommended, not required:**")
            for w in report["warnings"]:
                st.markdown(f"- {w}")
        st.caption("Fix the above, or switch to **Demo** in the sidebar to keep exploring.")
        st.stop()
    for w in report["warnings"]:
        st.warning(w)
    try:
        prog = st.progress(0.0, text="Preparing to load real MIMIC-IV...")

        def _load_cb(frac, msg):
            # frac is a 0..1 float for per-table steps, or None for the
            # indeterminate chunked-read sub-steps → hold the bar, update text.
            if isinstance(frac, (int, float)):
                prog.progress(min(float(frac), 1.0), text=msg)
            else:
                prog.progress(0.5, text=msg)

        mod, combined, names, mmap, ref_tables = load_and_build(
            False, combine_how, data_root_input, cfg, _progress_cb=_load_cb)
        prog.empty()
        if combined is None or len(combined) == 0:
            st.error("No data could be built from that folder. Check it contains "
                     "hosp/ and icu/ subfolders, or switch to Demo mode.")
            st.stop()
        X = combined[names].values.astype(float)
        y = combined["label"].values.astype(int)
        st.session_state.update(X=X, y=y, names=names, mmap=mmap, mod=mod, ref_tables=ref_tables)
    except Exception as exc:
        st.error(f"Data loading failed: {exc}\n\nWith real MIMIC-IV this usually means "
                 "a column name doesn't match what the loader expects, or a required "
                 "file is missing. Demo mode always works if you want to keep going.")
        st.stop()

else:  # Demo (synthetic)
    try:
        mod, combined, names, mmap, ref_tables = load_and_build(
            True, combine_how, "__demo__", cfg)
        X = combined[names].values.astype(float)
        y = combined["label"].values.astype(int)
        st.session_state.update(X=X, y=y, names=names, mmap=mmap, mod=mod, ref_tables=ref_tables)
    except Exception as exc:
        st.error(f"Demo data failed to build: {exc}")
        st.stop()


# --------------------------------------------------------------------------- #
# Tabs — Comparative and Ablation are grouped under one "Analysis" tab          #
# --------------------------------------------------------------------------- #
tabs = st.tabs(["🏠 Dashboard", "🔬 Data Explorer", "🧩 Feature Selection",
                "📊 Analysis", "🩺 Deep Analysis", "🖼️ Figures",
                "📐 Code & Equations", "💾 Export"])

# =========================================================================== #
# TAB 1 — Dashboard                                                           #
# =========================================================================== #
with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Samples", f"{X.shape[0]:,}")
    c2.metric("Features", X.shape[1])
    c3.metric("Modalities", len(mmap))
    c4.metric("Prevalence", f"{100*y.mean():.1f}%")
    st.caption(cap.prevalence_meaning(float(y.mean())))

    st.subheader("Run the nested-CV pipeline")
    st.caption(cap.leakage_note())
    if st.button("▶️ Run nested-CV pipeline", type="primary"):
        try:
            prog = st.progress(0.0, text="Starting...")
            cb = lambda frac, msg: prog.progress(min(frac, 1.0), text=msg)
            t0 = time.time()
            res = nested_cv_evaluate(X, y, mmap, cfg, n_outer=n_outer,
                                     classifier=classifier, progress_cb=cb)
            res["elapsed"] = time.time() - t0
            st.session_state["nested"] = res
            prog.empty()
        except Exception as exc:
            st.error(f"The pipeline hit an error: {exc}\n\nTry fewer folds or the "
                     "Demo dataset to confirm your setup, then re-run.")

    res = ss("nested")
    if res:
        a, b, c, d = st.columns(4)
        a.metric("AUC", f"{res['auc_mean']:.3f}", f"±{res['auc_ci95']:.3f} CI")
        b.metric("Accuracy", f"{res['accuracy_mean']:.3f}", f"±{res['accuracy_std']:.3f}")
        c.metric("F1", f"{res['f1_mean']:.3f}")
        d.metric("Mean # features", f"{res['n_features_mean']:.1f}")
        st.caption(f"{res['n_outer']}-fold nested CV · {res.get('elapsed', 0):.1f}s · classifier={classifier}")

        with st.expander("What do these numbers mean? (plain language)", expanded=True):
            st.markdown(f"- {cap.auc_meaning(res['auc_mean'])}")
            st.markdown(f"- {cap.accuracy_meaning(res['accuracy_mean'], float(y.mean()))}")
            st.markdown(f"- {cap.f1_meaning(res['f1_mean'])}")
            st.markdown(f"- {cap.n_features_meaning(res['n_features_mean'], X.shape[1])}")

        pf = pd.DataFrame(res["per_fold"])[["accuracy", "auc", "f1", "n_features"]]
        pf.index = [f"Fold {i+1}" for i in range(len(pf))]
        st.dataframe(pf.style.format("{:.4f}"), width="stretch")

        st.plotly_chart(pi.plot_selection_frequency_interactive(
            res["selection_frequency"], names), width="stretch")
        st.caption(cap.selection_frequency_meaning())
    else:
        render_welcome()

# =========================================================================== #
# TAB 2 — Data Explorer                                                       #
# =========================================================================== #
with tabs[1]:
    st.subheader("Class balance & modalities")
    c1, c2 = st.columns([1, 2])
    with c1:
        st.write(pd.Series(y).value_counts().rename({0: "No DM", 1: "DM"}).to_frame("count"))
        st.write({k: len(v) for k, v in mmap.items()})
    with c2:
        st.bar_chart({k: len(v) for k, v in mmap.items()})
    st.caption("Each modality is a group of related measurements (labs, vitals, "
               "medications, diagnoses). More features isn't always better — the "
               "whole point of QSQ-FS is to find the few that matter.")
    if combined is not None:
        st.subheader("Feature preview")
        st.dataframe(combined[names].describe().T.style.format("{:.2f}"),
                     width="stretch", height=300)
        if st.checkbox("Show feature correlation heatmap"):
            import matplotlib.pyplot as plt
            corr = combined[names].corr()
            fig, ax = plt.subplots(figsize=(9, 7))
            im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
            ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
            ax.set_xticklabels(names, rotation=90, fontsize=6)
            ax.set_yticklabels(names, fontsize=6); fig.colorbar(im, fraction=0.046)
            st.pyplot(fig)
            st.caption("Red = two features move together, blue = they move oppositely. "
                       "Highly correlated features are partly redundant.")
    else:
        st.info("Feature preview/correlation is available for the demo and MIMIC-IV "
                "sources. Your uploaded matrix is ready — head to the Dashboard to run it.")

# =========================================================================== #
# TAB 3 — Feature Selection (interpretation)                                  #
# =========================================================================== #
with tabs[2]:
    st.subheader("Two-stage selection (full-data fit — interpretation only)")
    st.caption("Shows WHICH features QSQ-FS favours when it sees all the data. "
               "This is for understanding the method, not for scoring it — for an "
               "unbiased performance estimate use the Dashboard's nested-CV run.")
    if st.button("🧩 Run two-stage selection"):
        try:
            with st.spinner("Stage 1 (per modality)..."):
                s1_sel, s1_models, _ = run_all_modalities(mod, cfg) if mod else ({}, {}, None)
            if not mod:
                # uploaded single-modality path: run stage-2 style search on all features
                with st.spinner("Running selection on uploaded features..."):
                    s2 = run_stage2_fusion(X, y, names, {"uploaded": list(range(len(names)))}, cfg)
                st.session_state.update(s1_sel={"uploaded": list(range(len(names)))},
                                        s1_models={}, s2=s2)
            else:
                with st.spinner("Stage 2 (fusion, frequency-weighted init)..."):
                    s2 = run_stage2_fusion(X, y, names, s1_sel, cfg)
                st.session_state.update(s1_sel=s1_sel, s1_models=s1_models, s2=s2)
        except Exception as exc:
            st.error(f"Selection failed: {exc}")

    if ss("s1_sel"):
        st.markdown("**Stage 1 — selected per modality**")
        for m, sel in ss("s1_sel").items():
            with st.expander(f"{m} — {len(sel)} features"):
                st.write(sel)
        s2 = ss("s2")
        final = [names[i] for i in s2.get_selected_features()]
        st.markdown(f"**Stage 2 — final fused subset ({len(final)} features)**")
        st.write(final)
        st.metric("Best fitness (full-data)", f"{s2.get_best_fitness():.4f}")
        st.plotly_chart(pi.plot_convergence_interactive(
            s2.get_convergence(), s2.get_feature_counts(),
            title="Stage-2 convergence (full-data fit)"), width="stretch")
        st.caption(cap.convergence_meaning())

# =========================================================================== #
# TAB 4 — Analysis (Comparative + Ablation, sub-navigated)                    #
# =========================================================================== #
with tabs[3]:
    sub = st.tabs(["📊 Comparative", "🧪 Ablation"])

    # ---- Comparative ----
    with sub[0]:
        st.subheader("QSQ-FS vs baselines — leak-free, equal budget")
        st.caption("Every method selects on the train fold and is scored on the "
                   "held-out test fold; metaheuristics share the same population "
                   "and iteration budget, so it's a fair race.")
        cc = cfg["comparative"]
        n_agents = st.slider("Population (all metaheuristics)", 10, 50, cc["n_agents"])
        max_iter = st.slider("Iterations (all metaheuristics)", 5, 40, cc["max_iter"])
        if st.button("📊 Run comparative analysis"):
            try:
                with st.spinner("Running all methods under identical protocol..."):
                    comp = run_comparative_analysis(X, y, n_trials=n_outer,
                                                    n_agents=n_agents, max_iter=max_iter)
                st.session_state["comp"] = comp
            except Exception as exc:
                st.error(f"Comparative analysis failed: {exc}")
        comp = ss("comp")
        if comp:
            rows = []
            for name, r in comp.items():
                if name.startswith("_"):
                    continue
                rows.append({"Method": name, "AUC": np.nanmean(r["auc"]),
                             "Acc": r["accuracy"].mean(), "F1": np.nanmean(r["f1"]),
                             "Features": r["n_features"].mean()})
            df = pd.DataFrame(rows).sort_values("AUC", ascending=False)
            st.dataframe(df.style.format({"AUC": "{:.4f}", "Acc": "{:.4f}",
                                          "F1": "{:.4f}", "Features": "{:.1f}"}),
                         width="stretch")
            st.plotly_chart(pi.plot_comparison_interactive(comp, "auc"),
                            width="stretch")
            st.caption(cap.comparison_meaning())
            st.markdown("**Wilcoxon signed-rank vs QSQ-FS (Bonferroni-corrected)**")
            st.dataframe(pd.DataFrame(comp["_stats"]), width="stretch")
            st.caption("Note: strong full-feature learners (SVM/RF) may beat any "
                       "feature-selection method on KNN — this is expected under a "
                       "leakage-free protocol.")

    # ---- Ablation ----
    with sub[1]:
        st.subheader("Mechanism ablation — explicit switches")
        st.caption("Turn each mechanism off one at a time (Quorum Sensing, Quorum "
                   "Quenching, caching, elitism) to see how much each contributes. "
                   "Caching affects RUNTIME, not accuracy — as it should.")
        if st.button("🧪 Run ablation study"):
            try:
                with st.spinner("Ablating mechanisms under leak-free protocol..."):
                    abl = run_ablation_study(X, y, n_trials=n_outer)
                st.session_state["abl"] = abl
            except Exception as exc:
                st.error(f"Ablation failed: {exc}")
        abl = ss("abl")
        if abl:
            rows = []
            for name, r in abl.items():
                if name.startswith("_"):
                    continue
                rows.append({"Variant": name, "Acc": r["accuracy"].mean(),
                             "Std": r["accuracy"].std(), "Features": r["n_features"].mean(),
                             "KNN evals": r["n_evals"], "Runtime (s)": r["runtime"][0]})
            st.dataframe(pd.DataFrame(rows).style.format(
                {"Acc": "{:.4f}", "Std": "{:.4f}", "Features": "{:.1f}", "Runtime (s)": "{:.2f}"}),
                width="stretch")
            st.markdown("**Contribution & significance vs Full QSQ-FS**")
            st.dataframe(pd.DataFrame(abl["_stats"]), width="stretch")

# =========================================================================== #
# TAB 5 — Deep Analysis                                                       #
# =========================================================================== #
with tabs[4]:
    st.subheader("Out-of-fold diagnostics & clinical alignment")
    res = ss("nested")
    if not res:
        st.info("Run the nested-CV pipeline (Dashboard) first — this tab visualises "
                "its out-of-fold predictions.")
    else:
        if len(res.get("oof_proba", [])) > 0:
            proba, yt = res["oof_proba"], res["oof_y"]
            pred = (proba >= 0.5).astype(int)
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(pi.plot_roc_interactive(yt, proba), width="stretch")
                st.caption(cap.roc_meaning())
            with c2:
                st.plotly_chart(pi.plot_confusion_interactive(yt, pred), width="stretch")
                st.caption(cap.confusion_meaning())
        freq = res["selection_frequency"]
        stable = [names[i] for i in np.argsort(freq)[::-1] if freq[i] >= 0.5]
        ref_tables = ss("ref_tables", {})
        item_map = build_item_label_map(stable, ref_tables.get("d_labitems"),
                                        ref_tables.get("d_items"))
        align = check_clinical_alignment(stable, names, item_label_map=item_map)
        st.markdown(f"**Clinical alignment** — {align['n_matched']}/{align['total_selected']} "
                    f"stable features are known biomarkers "
                    f"(overlap = {align['biomarker_overlap']:.0%})")
        st.caption("A sanity check: are the features the search trusts the ones "
                   "clinicians already associate with diabetes? Overlap here is "
                   "reassuring; novel features may be genuine comorbidity signals.")
        if align["matched"]:
            st.dataframe(pd.DataFrame(align["matched"]), width="stretch")
        if align["unmatched"]:
            st.caption("Other stable features (candidate / comorbidity signals): "
                       + ", ".join(align["unmatched"]))

# =========================================================================== #
# TAB 6 — Code & Equations                                                    #
# =========================================================================== #
# =========================================================================== #
# TAB 6 — Figures (F01-F20)                                                   #
# =========================================================================== #
with tabs[5]:
    st.subheader("Publication figures (F01–F20)")
    st.caption("The twenty analysis figures, rendered at 300 dpi from the cohort "
               "currently loaded. Download individually or as a single archive.")

    X_f, y_f, names_f, mmap_f = ss("X"), ss("y"), ss("names"), ss("mmap")
    comp_f = ss("comp")

    if X_f is None:
        st.info("Load a dataset in the sidebar first.")
    elif not comp_f:
        st.info("Run the **Comparative** study on the Analysis tab first — the "
                "method-comparison figures (F19, F20) are built from its table.")
    else:
        if st.button("🖼️  Generate the 20 figures", type="primary"):
            outdir = tempfile.mkdtemp(prefix="qsqfs_fig_")
            bar = st.progress(0.0, text="Starting…")
            done = {"n": 0}

            def _tick(msg):
                done["n"] += 1
                bar.progress(min(done["n"] / 20.0, 1.0), text=msg)

            try:
                comp_table = (comp_f.get("table", []) if isinstance(comp_f, dict)
                              else list(comp_f))
                manifest = build_figures(
                    np.asarray(X_f), np.asarray(y_f), list(names_f), mmap_f, cfg,
                    comparative_table=comp_table, out_dir=outdir,
                    feature_labels=ss("feature_labels") or {}, progress=_tick)
                bar.progress(1.0, text=f"{len(manifest)} figures ready")
                st.session_state["fig_dir"] = outdir
                st.session_state["fig_manifest"] = manifest
            except Exception as exc:
                bar.empty()
                st.error(f"Figure generation failed: {exc}")

        manifest = ss("fig_manifest")
        figdir = ss("fig_dir")

        if manifest and figdir:
            figroot = os.path.join(figdir, "fig20")

            # one archive containing every figure plus the manifest
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                for item in manifest:
                    fp = os.path.join(figroot, item["file"])
                    if os.path.exists(fp):
                        z.write(fp, item["file"])
                z.writestr("manifest.json", json.dumps(manifest, indent=2))
            st.download_button(
                "⬇️  Download all 20 figures (.zip)", buf.getvalue(),
                file_name="qsqfs_figures.zip", mime="application/zip",
                type="primary")

            st.divider()

            for item in manifest:
                fp = os.path.join(figroot, item["file"])
                if not os.path.exists(fp):
                    continue
                st.markdown(f"**{item['id']} · {item['title']}**")
                st.image(fp, width="stretch")
                st.caption(item["desc"])
                with open(fp, "rb") as fh:
                    st.download_button(
                        f"⬇️  {item['file']}", fh.read(), file_name=item["file"],
                        mime="image/png", key=f"dl_{item['id']}")
                st.divider()


with tabs[6]:
    st.subheader("Methodology (thesis Ch.3) and implementation")
    st.latex(r"J(c) = \alpha\,\mathrm{Acc}(c) + (1-\alpha)\left(1 - \frac{\lVert c\rVert_1}{d}\right)")
    st.latex(r"n_{\min}=\max(1,\lceil 0.05d\rceil),\quad n_{\max}=\lceil 0.50d\rceil")
    st.latex(r"\mathrm{AI}_j = \max_{s\in S}\; c_{s,j}\big(\hat F_s\,w_{AI} + (1-w_{AI})U\big)")
    st.latex(r"\mathrm{AI}^{(t)} = \beta\,\mathrm{AI}^{(t)}_{\text{new}} + (1-\beta)\,\mathrm{AI}^{(t-1)}\quad(\beta=0.70)")
    st.latex(r"\mathrm{supp}(c)=\max\big(0,\ \theta_s - F_{\text{raw}}(c)\big)\;\text{(QQ uses RAW fitness)}")
    st.markdown("""
**Implementation notes:**
- **Leakage-free evaluation** — feature selection is nested inside each outer
  cross-validation training fold (`src/evaluation.py`).
- **Elitism** — the global best solution is re-injected each generation (`src/qsfs.py`).
- **Fitness caching** — evaluated subsets are memoised; caching changes runtime,
  not the selected subset.
- **Quorum Quenching on raw fitness** — the suppression term uses the raw
  objective, so penalties do not compound (Eq 3.8).
- **Deterministic seeding** — each evaluation is seeded by
  `(random_state, stage, iter, colony)` for reproducibility.
- **Frequency-weighted Stage-2 initialisation** — the Stage-1 selection pool
  weights the Stage-2 starting population.
- **Regularised MLP classifier** — L2 weight decay via `l2_alpha`.
- **Long-format handling** — `chartevents` and `pharmacy` are pivoted from
  MIMIC-IV long format; diabetes-defining codes and drugs are removed to
  prevent label leakage.
- **Equal search budgets across baselines** and permutation-based feature importance.
""")

# =========================================================================== #
# TAB 7 — Export                                                              #
# =========================================================================== #
with tabs[7]:
    st.subheader("Export results")
    res = ss("nested"); comp = ss("comp"); abl = ss("abl")
    payload = {}
    if res:
        payload["nested_cv"] = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                for k, v in res.items()
                                if k not in ("per_fold", "oof_proba", "oof_y")}
    if comp:
        st.code(generate_latex_results_table(comp), language="latex")
        st.download_button("⬇️ comparative_table.tex",
                           generate_latex_results_table(comp), "comparative_table.tex")
    if abl:
        st.code(generate_ablation_latex_table(abl, abl.get("_stats", [])), language="latex")
    if payload:
        st.download_button("⬇️ results.json", json.dumps(payload, indent=2, default=str),
                           "qsqfs_results.json", "application/json")
    sel_df = pd.DataFrame({"feature": names})
    if res is not None:
        sel_df["selection_frequency"] = res["selection_frequency"]
    buf = io.StringIO(); sel_df.to_csv(buf, index=False)
    st.download_button("⬇️ feature_stability.csv", buf.getvalue(), "feature_stability.csv")
    if not (res or comp or abl):
        st.info("Run something first (Dashboard / Analysis) — results will appear here to export.")

st.markdown("---")
st.caption(f"QSQ-FS {APP_VERSION} · multimodal diabetes feature selection · "
           "performance estimated via nested cross-validation.")
