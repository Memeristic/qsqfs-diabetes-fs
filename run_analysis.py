"""
run_analysis.py
===============
End-to-end, dataset-agnostic analysis driver. Produces every metric, table and
figure the thesis needs under an output directory, from any supported source.

Usage
-----
    python run_analysis.py                          # synthetic demo
    python run_analysis.py mimic /path/to/mimic-iv  # real MIMIC-IV folder
    python run_analysis.py csv  /path/to/data.csv   # any tidy CSV (label auto-detected)
    python run_analysis.py csv  data.csv Outcome    # explicit label column

Only AGGREGATE outputs are written (metrics, counts, feature names) -- never any
per-patient row -- consistent with the never-expose-patient-data rule for
clinical data.

What it runs, in order:
  1. Load + partition into modalities (any dataset).
  2. Two-stage QSQ-FS feature selection on the full data (for interpretation).
  3. Leak-free nested CV with the multimodal FUSION classifier + tuned threshold
     (the primary performance estimate), plus a KNN reference.
  4. Fusion-strategy comparison: feature- vs decision- vs hybrid-level.
  5. Comparative study: QSQ-FS vs RIME, PLO, HGS, GA, PSO and classical
     SVM/RF/XGBoost baselines, with Wilcoxon significance.
  6. Ablation of QSQ-FS's four mechanisms.
  7. Confusion matrix / ROC at the tuned operating point.
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

from sklearn.metrics import confusion_matrix, roc_auc_score

from src.datasets import load_dataset
from src.evaluation import nested_cv_evaluate
from src.comparative_analysis import run_comparative_analysis
from src.ablation import run_ablation_study
from src.fusion import MultimodalFusion, tune_threshold
from src import plotting

RS = 42


def _parse_args(argv):
    if len(argv) <= 1:
        return "synthetic", {}
    src = argv[1].lower()
    if src in ("demo", "synthetic"):
        return "synthetic", {}
    if src == "mimic":
        root = argv[2] if len(argv) > 2 else None
        return "mimic", {"data_root": root}
    if src == "csv":
        path = argv[2]
        opts = {"path": path}
        if len(argv) > 3:
            opts["label_col"] = argv[3]
        return "csv", opts
    # a bare path
    if argv[1].lower().endswith(".csv"):
        return argv[1], {}
    return src, {}


def main():
    source, opts = _parse_args(sys.argv)
    out = os.environ.get("OUT_DIR", "analysis_out")
    fig = os.path.join(out, "figures")
    os.makedirs(fig, exist_ok=True)
    cfg = yaml.safe_load(open("config.yaml"))
    cfg.setdefault("neural_model", {})["fusion_strategy"] = \
        cfg.get("neural_model", {}).get("fusion_strategy", "hybrid")
    summary = {}

    # ---------- 1. LOAD ----------
    print(f">> loading source={source} opts={opts}", flush=True)
    if source == "mimic":
        cfg.setdefault("preprocessing", {})["combine_how"] = "outer"
    X, y, names, mmap, meta = load_dataset(source, cfg, **opts)
    summary["dataset"] = meta
    print("   dataset:", meta, flush=True)

    # ---------- 2. FULL-DATA TWO-STAGE SELECTION (interpretation) ----------
    print(">> full-data two-stage QSQ-FS selection ...", flush=True)
    from src.qsfs import QSQFS
    pool, pool_w = [], {}
    for mod, cols in mmap.items():
        cols = list(cols)
        if not cols:
            continue
        m1 = QSQFS(n_colonies=cfg["stage1"]["n_colonies"],
                   max_iter_stage1=cfg["stage1"]["max_iter_stage1"], max_iter_stage2=0,
                   alpha=cfg["stage1"].get("alpha", 0.85),
                   max_frac=cfg["stage1"].get("max_frac", 0.50),
                   min_frac=cfg["stage1"].get("min_frac", 0.05),
                   fitness_metric="balanced", k_nn=cfg["stage1"]["k_nn"],
                   cv_folds=cfg["stage1"]["cv_folds"], random_state=RS, verbose=False)
        m1.fit(X[:, cols], y)
        for li in m1.get_selected_features():
            g = cols[int(li)]
            pool.append(g); pool_w[g] = pool_w.get(g, 0) + 1
    pool = sorted(set(pool)) or list(range(len(names)))
    weights = [pool_w.get(g, 1) for g in pool]
    m2 = QSQFS(n_colonies=cfg["stage2"]["n_colonies"], max_iter_stage1=0,
               max_iter_stage2=cfg["stage2"]["max_iter_stage2"], fitness_metric="balanced",
               alpha=cfg["stage2"].get("alpha", 0.85),
               max_frac=cfg["stage2"].get("max_frac", 0.50),
               min_frac=cfg["stage2"].get("min_frac", 0.05),
               rho=cfg["stage2"]["rho"], k_nn=cfg["stage2"]["k_nn"],
               cv_folds=cfg["stage2"]["cv_folds"], random_state=RS, verbose=False)
    m2.fit(X, y, feature_pool=pool, pool_weights=weights)
    final_idx = m2.get_selected_features()
    final_feats = [names[i] for i in final_idx]
    if m2.saturated():
        print("   WARNING: selected subset sits on the cardinality ceiling "
              f"(n_max={m2.n_max}); the parsimony term is too weak to bind.", flush=True)
    summary["full_data_selection"] = {
        "stage1_pool_size": len(pool), "stage2_n_selected": len(final_feats),
        "n_max_ceiling": int(m2.n_max),
        "saturated_at_ceiling": bool(m2.saturated()),
        "stage2_best_fitness": round(float(m2.get_best_fitness()), 4),
        "stage2_selected_features": final_feats}
    f = plotting.plot_convergence(m2.get_convergence(), m2.get_feature_counts(),
                                  title="Stage-2 convergence (full-data fit)")
    f.savefig(os.path.join(fig, "convergence.png"), dpi=150, bbox_inches="tight"); plt.close(f)

    # ---------- 3. NESTED CV: fusion (primary) + knn (reference) ----------
    print(">> nested CV -- multimodal fusion (primary) ...", flush=True)
    t0 = time.time()
    res = nested_cv_evaluate(X, y, mmap, cfg, n_outer=cfg["evaluation"]["n_outer_folds"],
                             classifier="fusion", random_state=RS)
    res["elapsed"] = time.time() - t0
    print(">> nested CV -- knn reference ...", flush=True)
    res_knn = nested_cv_evaluate(X, y, mmap, cfg, n_outer=cfg["evaluation"]["n_outer_folds"],
                                 classifier="knn", random_state=RS)
    summary["nested_cv_fusion"] = _cv_summary(res)
    summary["nested_cv_knn"] = _cv_summary(res_knn)
    pf = pd.DataFrame(res["per_fold"])[["accuracy", "auc", "f1", "n_features"]]
    pf.index = [f"Fold {i+1}" for i in range(len(pf))]
    pf.to_csv(os.path.join(out, "per_fold.csv"))
    summary["per_fold_fusion"] = pf.round(4).reset_index().rename(
        columns={"index": "fold"}).to_dict("records")

    # selection stability
    freq = res["selection_frequency"]
    stab = pd.DataFrame({"feature": names, "selection_frequency": freq}).sort_values(
        "selection_frequency", ascending=False)
    stab.to_csv(os.path.join(out, "feature_stability.csv"), index=False)
    f = plotting.plot_selection_frequency(freq, names, top_n=20)
    f.savefig(os.path.join(fig, "selection_frequency.png"), dpi=150, bbox_inches="tight"); plt.close(f)
    stable = [names[i] for i in np.argsort(freq)[::-1] if freq[i] >= 0.5]
    summary["stable_features"] = {"threshold": 0.5, "n_stable": len(stable),
                                  "features": stable,
                                  "top10": stab.head(10).round(3).to_dict("records")}

    # ---------- 4. FUSION-STRATEGY COMPARISON (reuses fold selections) ----------
    print(">> fusion-strategy comparison (feature / decision / hybrid) ...", flush=True)
    from src.fusion import score_fusion_strategies
    strat_res = score_fusion_strategies(
        X, y, mmap, res["fold_selected"], n_outer=cfg["evaluation"]["n_outer_folds"],
        random_state=RS)
    fusion_rows = [{"Strategy": s, "AUC": round(v["auc"], 4),
                    "Accuracy": round(v["accuracy"], 4), "F1": round(v["f1"], 4),
                    "Sensitivity": round(v["sensitivity"], 4)}
                   for s, v in strat_res.items()]
    pd.DataFrame(fusion_rows).to_csv(os.path.join(out, "fusion_strategies.csv"), index=False)
    summary["fusion_strategies"] = fusion_rows

    # ---------- 5. CONFUSION / ROC at tuned operating point ----------
    print(">> confusion + ROC (tuned threshold, out-of-fold) ...", flush=True)
    proba, yt = res["oof_proba"], res["oof_y"]
    # Decisions are taken at the threshold each fold tuned on its own training
    # partition, so the operating point is never chosen using the labels it is
    # scored against.
    pred = res["oof_pred"]
    thr = float(np.mean([m.get("threshold", 0.5) for m in res["per_fold"]]))
    cm = confusion_matrix(yt, pred)
    tn, fp, fn_, tp = cm.ravel()
    summary["confusion"] = {
        "threshold_mean_across_folds": round(float(thr), 3),
        "threshold_tuned_on": "training partition of each outer fold",
        "tn": int(tn), "fp": int(fp),
        "fn": int(fn_), "tp": int(tp),
        "sensitivity_recall": round(tp / (tp + fn_) if (tp + fn_) else 0.0, 4),
        "specificity": round(tn / (tn + fp) if (tn + fp) else 0.0, 4),
        "ppv_precision": round(tp / (tp + fp) if (tp + fp) else 0.0, 4),
        "npv": round(tn / (tn + fn_) if (tn + fn_) else 0.0, 4),
        "oof_auc": round(float(roc_auc_score(yt, proba)) if len(np.unique(yt)) > 1 else float("nan"), 4),
        "n_oof": int(len(yt))}
    pd.DataFrame(cm, index=["Actual: No DM", "Actual: DM"],
                 columns=["Pred: No DM", "Pred: DM"]).to_csv(
        os.path.join(out, "confusion_matrix.csv"))
    f = plotting.plot_confusion(yt, pred); f.savefig(os.path.join(fig, "confusion.png"), dpi=150, bbox_inches="tight"); plt.close(f)
    f = plotting.plot_roc(yt, proba); f.savefig(os.path.join(fig, "roc.png"), dpi=150, bbox_inches="tight"); plt.close(f)

    # ---------- 6. COMPARATIVE (incl. RIME / PLO / HGS) ----------
    print(">> comparative analysis (QSQ-FS vs RIME/PLO/HGS/GA/PSO + SVM/RF/XGB) ...", flush=True)
    comp = run_comparative_analysis(X, y, n_trials=cfg["comparative"]["n_trials"],
                                    n_agents=cfg["comparative"]["n_agents"],
                                    max_iter=cfg["comparative"]["max_iter"], random_state=RS)
    rows = []
    for name, r in comp.items():
        if name.startswith("_"):
            continue
        rows.append({"Method": name, "AUC": round(float(np.nanmean(r["auc"])), 4),
                     "AUC_std": round(float(np.nanstd(r["auc"])), 4),
                     "Accuracy": round(float(r["accuracy"].mean()), 4),
                     "F1": round(float(np.nanmean(r["f1"])), 4),
                     "Features": round(float(r["n_features"].mean()), 1)})
    comp_df = pd.DataFrame(rows).sort_values("AUC", ascending=False)
    comp_df.to_csv(os.path.join(out, "comparative.csv"), index=False)
    summary["comparative"] = {"table": comp_df.to_dict("records"),
                              "stats": comp.get("_stats", {})}
    f = plotting.plot_comparison(comp, "auc"); f.savefig(os.path.join(fig, "comparison_auc.png"), dpi=150, bbox_inches="tight"); plt.close(f)

    # ---------- 7. ABLATION ----------
    print(">> ablation study ...", flush=True)
    abl = run_ablation_study(X, y, n_trials=cfg["ablation"]["n_trials"], random_state=RS)
    arows = []
    for name, r in abl.items():
        if name.startswith("_"):
            continue
        arows.append({"Variant": name, "Accuracy": round(float(r["accuracy"].mean()), 4),
                      "Acc_std": round(float(r["accuracy"].std()), 4),
                      "Features": round(float(r["n_features"].mean()), 1),
                      "Runtime_s": round(float(r["runtime"][0]), 2)})
    pd.DataFrame(arows).to_csv(os.path.join(out, "ablation.csv"), index=False)
    summary["ablation"] = {"table": arows, "stats": abl.get("_stats", {})}

    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=2, default=str)

    # ---------- 8. SPSS-STYLE STATISTICAL ANALYSIS ----------
    print(">> SPSS-style statistical analysis (descriptives, t-tests, effect sizes) ...", flush=True)
    from src.spss_analysis import run_spss_style_analysis
    spss = run_spss_style_analysis(X, y, names, top_k=25)
    for key, df in spss.items():
        df.to_csv(os.path.join(out, f"spss_{key}.csv"), index=False)
    ind = spss["independent_samples"]
    summary["spss_analysis"] = {
        "n_features_tested": int(len(ind)),
        "n_significant_fdr": int((ind["Sig"] != "ns").sum()) if not ind.empty else 0,
        "top_discriminators": ind.head(10).to_dict("records") if not ind.empty else []}
    # SPSS-style figure: effect size (Cohen's d) of the strongest predictors
    if not ind.empty:
        top = ind.head(15).iloc[::-1]
        efig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(top))))
        colors = ["#dc2626" if s != "ns" else "#94a3b8" for s in top["Sig"]]
        ax.barh(range(len(top)), top["Cohens_d"], color=colors)
        ax.set_yticks(range(len(top))); ax.set_yticklabels(top["Feature"], fontsize=8)
        ax.axvline(0, color="#334155", lw=0.8)
        ax.axvline(0.8, color="#16a34a", ls="--", lw=0.8, label="large effect (d=0.8)")
        ax.axvline(-0.8, color="#16a34a", ls="--", lw=0.8)
        ax.set_xlabel("Cohen's d (diabetic - non-diabetic)")
        ax.set_title("Standardised group difference by feature (SPSS-style)")
        ax.legend(fontsize=8)
        efig.savefig(os.path.join(fig, "effect_sizes.png"), dpi=150, bbox_inches="tight")
        plt.close(efig)

    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=2, default=str)
    print("\n>> DONE.")
    print(">> Fusion nested-CV:", summary["nested_cv_fusion"])
    print(">> Figures:", sorted(os.listdir(fig)))


def _cv_summary(res):
    return {"n_outer": int(res["n_outer"]),
            "auc_mean": round(float(res["auc_mean"]), 4),
            "auc_ci95": round(float(res["auc_ci95"]), 4),
            "accuracy_mean": round(float(res["accuracy_mean"]), 4),
            "accuracy_std": round(float(res["accuracy_std"]), 4),
            "f1_mean": round(float(res["f1_mean"]), 4),
            "f1_std": round(float(res["f1_std"]), 4),
            "n_features_mean": round(float(res["n_features_mean"]), 2),
            "elapsed_s": round(res.get("elapsed", 0.0), 1)}


def _sensitivity(y_true, pred):
    tp = int(np.sum((pred == 1) & (y_true == 1)))
    fn = int(np.sum((pred == 0) & (y_true == 1)))
    return tp / (tp + fn) if (tp + fn) else 0.0


if __name__ == "__main__":
    main()
