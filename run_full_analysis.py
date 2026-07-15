"""
run_full_analysis.py — one-shot deep analysis on the REAL MIMIC-IV demo.
Produces every result + figure the report needs, saved under analysis_out/.
Only AGGREGATE outputs are written (metrics, counts, feature names) — no
per-patient rows — consistent with the never-expose-patient-data rule.
"""
import os, json, warnings, time
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import yaml
from src.data_loader import MIMICDataLoader
from src.modality_builder import (extract_diabetes_label, build_labs, build_vitals,
    build_medications, build_diagnoses, build_combined_matrix, build_modality_map)
from src.stage1_runner import run_all_modalities
from src.stage2_fuser import run_stage2_fusion
from src.evaluation import nested_cv_evaluate
from src.comparative_analysis import run_comparative_analysis
from src.ablation import run_ablation_study
from src.stats_analysis import (check_clinical_alignment, build_item_label_map)
from src.leakage import resolve_leaky_itemids, assert_no_leakage
from src import plotting
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, confusion_matrix

OUT = "analysis_out"; FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)
RS = 42
summary = {}

# ---------- 1. LOAD REAL DATA ----------
# Usage:
#   python run_full_analysis.py                    # uses config.yaml data_root
#   python run_full_analysis.py /path/to/mimic-iv  # explicit MIMIC-IV folder
#   python run_full_analysis.py demo               # synthetic demo data
import sys
cfg = yaml.safe_load(open("config.yaml"))
arg = sys.argv[1] if len(sys.argv) > 1 else None
if arg == "demo":
    cfg["demo_mode"] = True
    print(">> loading DEMO (synthetic) data ...", flush=True)
else:
    cfg["demo_mode"] = False
    if arg:
        cfg["paths"]["data_root"] = arg
    cfg["preprocessing"]["combine_how"] = "outer"
    print(f">> loading real MIMIC-IV from {cfg['paths']['data_root']} ...", flush=True)
data = MIMICDataLoader(cfg).load_all()
label_df = extract_diabetes_label(data.get("patients"), data.get("diagnoses_icd"), data.get("d_icd_diagnoses"))
leaky_ids = resolve_leaky_itemids(data.get("d_labitems"), data.get("d_items"))
mod = {}
for key, fn, arg in [("labs",build_labs,data.get("labevents")),("vitals",build_vitals,data.get("chartevents")),
                     ("meds",build_medications,data.get("pharmacy")),("dx",build_diagnoses,data.get("diagnoses_icd"))]:
    df,_ = (fn(arg, label_df, leaky_itemids=leaky_ids) if key in ("labs","vitals") else fn(arg, label_df))
    if df is not None: mod[key] = df
combined, names = build_combined_matrix(mod, how="outer")
assert_no_leakage(names, leaky_ids)
mmap = build_modality_map(mod, names)
ref_tables = {"d_labitems": data.get("d_labitems"), "d_items": data.get("d_items")}
X = combined[names].values.astype(float); y = combined["label"].values.astype(int)
summary["dataset"] = {"n_patients": int(X.shape[0]), "n_features": int(X.shape[1]),
                      "prevalence_pct": round(100*float(y.mean()),1),
                      "n_positive": int(y.sum()), "n_negative": int((1-y).sum()),
                      "modalities": {k: len(v) for k,v in mmap.items()}}
print("   dataset:", summary["dataset"], flush=True)

# ---------- 2. FULL-DATA TWO-STAGE SELECTION (interpretation) ----------
print(">> full-data two-stage selection (interpretation) ...", flush=True)
s1_sel, s1_models, _ = run_all_modalities(mod, cfg)
s2 = run_stage2_fusion(X, y, names, s1_sel, cfg)
final_idx = s2.get_selected_features()
final_feats = [names[i] for i in final_idx]
summary["full_data_selection"] = {
    "stage1_per_modality": {m: len(sel) for m, sel in s1_sel.items()},
    "stage2_n_selected": len(final_feats),
    "stage2_best_fitness": round(float(s2.get_best_fitness()),4),
    "stage2_selected_features": final_feats}
fig = plotting.plot_convergence(s2.get_convergence(), s2.get_feature_counts(),
                                title="Stage-2 convergence (full-data fit)")
fig.savefig(os.path.join(FIG,"convergence.png"), dpi=150, bbox_inches="tight"); plt.close(fig)

# ---------- 3. NESTED-CV (performance estimate) ----------
print(">> nested-CV (leak-free performance) ...", flush=True)
t0 = time.time()
res = nested_cv_evaluate(X, y, mmap, cfg, n_outer=cfg["evaluation"]["n_outer_folds"],
                         classifier="knn")
res["elapsed"] = time.time()-t0
summary["nested_cv"] = {
    "n_outer": int(res["n_outer"]),
    "auc_mean": round(float(res["auc_mean"]),4), "auc_ci95": round(float(res["auc_ci95"]),4),
    "accuracy_mean": round(float(res["accuracy_mean"]),4), "accuracy_std": round(float(res["accuracy_std"]),4),
    "f1_mean": round(float(res["f1_mean"]),4), "f1_std": round(float(res["f1_std"]),4),
    "n_features_mean": round(float(res["n_features_mean"]),2),
    "elapsed_s": round(res["elapsed"],1)}
pf = pd.DataFrame(res["per_fold"])[["accuracy","auc","f1","n_features"]]
pf.index = [f"Fold {i+1}" for i in range(len(pf))]
pf.to_csv(os.path.join(OUT,"per_fold.csv"))
summary["per_fold"] = pf.round(4).reset_index().rename(columns={"index":"fold"}).to_dict("records")

# selection frequency table + figure
freq = res["selection_frequency"]
stab_df = pd.DataFrame({"feature": names, "selection_frequency": freq}).sort_values(
    "selection_frequency", ascending=False)
stab_df.to_csv(os.path.join(OUT,"feature_stability.csv"), index=False)
fig = plotting.plot_selection_frequency(freq, names, top_n=20); 
fig.savefig(os.path.join(FIG,"selection_frequency.png"), dpi=150, bbox_inches="tight"); plt.close(fig)
stable = [names[i] for i in np.argsort(freq)[::-1] if freq[i] >= 0.5]
summary["stable_features"] = {"threshold": 0.5, "n_stable": len(stable), "features": stable,
    "top10": stab_df.head(10).round(3).to_dict("records")}

# ---------- 4. CONFUSION MATRIX + ROC (out-of-fold) ----------
print(">> confusion matrix + ROC (out-of-fold) ...", flush=True)
proba = res["oof_proba"]; yt = res["oof_y"]; pred = (proba >= 0.5).astype(int)
cm = confusion_matrix(yt, pred)
tn, fp, fn_, tp = cm.ravel()
sens = tp/(tp+fn_) if (tp+fn_) else 0.0
spec = tn/(tn+fp) if (tn+fp) else 0.0
ppv  = tp/(tp+fp) if (tp+fp) else 0.0
npv  = tn/(tn+fn_) if (tn+fn_) else 0.0
oof_auc = roc_auc_score(yt, proba) if len(np.unique(yt))>1 else float("nan")
summary["confusion"] = {"tn":int(tn),"fp":int(fp),"fn":int(fn_),"tp":int(tp),
    "sensitivity_recall": round(sens,4),"specificity": round(spec,4),
    "ppv_precision": round(ppv,4),"npv": round(npv,4),
    "oof_auc": round(float(oof_auc),4),"n_oof": int(len(yt))}
pd.DataFrame(cm, index=["Actual: No DM","Actual: DM"],
             columns=["Pred: No DM","Pred: DM"]).to_csv(os.path.join(OUT,"confusion_matrix.csv"))
fig = plotting.plot_confusion(yt, pred); fig.savefig(os.path.join(FIG,"confusion.png"), dpi=150, bbox_inches="tight"); plt.close(fig)
fig = plotting.plot_roc(yt, proba); fig.savefig(os.path.join(FIG,"roc.png"), dpi=150, bbox_inches="tight"); plt.close(fig)

# ---------- 5. COMPARATIVE ANALYSIS ----------
print(">> comparative analysis vs baselines ...", flush=True)
comp = run_comparative_analysis(X, y, n_trials=5, n_agents=20, max_iter=15, random_state=RS)
rows = []
for name, r in comp.items():
    if name.startswith("_"): continue
    rows.append({"Method": name, "AUC": round(float(np.nanmean(r["auc"])),4),
                 "AUC_std": round(float(np.nanstd(r["auc"])),4),
                 "Accuracy": round(float(r["accuracy"].mean()),4),
                 "F1": round(float(np.nanmean(r["f1"])),4),
                 "Features": round(float(r["n_features"].mean()),1)})
comp_df = pd.DataFrame(rows).sort_values("AUC", ascending=False)
comp_df.to_csv(os.path.join(OUT,"comparative.csv"), index=False)
summary["comparative"] = {"table": comp_df.to_dict("records"), "stats": comp["_stats"]}
fig = plotting.plot_comparison(comp, "auc"); fig.savefig(os.path.join(FIG,"comparison_auc.png"), dpi=150, bbox_inches="tight"); plt.close(fig)

# ---------- 6. ABLATION ----------
print(">> ablation study ...", flush=True)
abl = run_ablation_study(X, y, n_trials=5, random_state=RS)
arows = []
for name, r in abl.items():
    if name.startswith("_"): continue
    arows.append({"Variant": name, "Accuracy": round(float(r["accuracy"].mean()),4),
                  "Acc_std": round(float(r["accuracy"].std()),4),
                  "Features": round(float(r["n_features"].mean()),1),
                  "KNN_evals": int(r["n_evals"]), "Runtime_s": round(float(r["runtime"][0]),2)})
abl_df = pd.DataFrame(arows); abl_df.to_csv(os.path.join(OUT,"ablation.csv"), index=False)
summary["ablation"] = {"table": abl_df.to_dict("records"), "stats": abl["_stats"]}

# ---------- 7. CLINICAL ALIGNMENT ----------
print(">> clinical alignment ...", flush=True)
item_map = build_item_label_map(stable, ref_tables.get("d_labitems"), ref_tables.get("d_items"))
align = check_clinical_alignment(stable, names, item_label_map=item_map)
summary["clinical_alignment"] = {
    "n_matched": align["n_matched"], "total_selected": align["total_selected"],
    "biomarker_overlap_pct": round(100*align["biomarker_overlap"],1),
    "matched": align["matched"], "unmatched": align["unmatched"]}
if align["matched"]:
    pd.DataFrame(align["matched"]).to_csv(os.path.join(OUT,"clinical_alignment.csv"), index=False)

# ---------- 8. FEATURE IMPORTANCE (permutation, interpretation) ----------
print(">> permutation importance on stable subset ...", flush=True)
imp_summary = []
if stable:
    idx = [names.index(f) for f in stable]
    Xs = X[:, idx]
    rf = RandomForestClassifier(n_estimators=300, random_state=RS).fit(Xs, y)
    pi_res = permutation_importance(rf, Xs, y, n_repeats=20, random_state=RS, scoring="roc_auc")
    imp_df = pd.DataFrame({"feature": stable, "importance_mean": pi_res.importances_mean,
                           "importance_std": pi_res.importances_std}).sort_values(
                           "importance_mean", ascending=False)
    imp_df.to_csv(os.path.join(OUT,"feature_importance.csv"), index=False)
    imp_summary = imp_df.round(4).to_dict("records")
    fig, ax = plt.subplots(figsize=(8, max(3, 0.4*len(stable))))
    d = imp_df.iloc[::-1]
    ax.barh(range(len(d)), d["importance_mean"], xerr=d["importance_std"], color="#2563eb")
    ax.set_yticks(range(len(d))); ax.set_yticklabels(d["feature"], fontsize=8)
    ax.set_xlabel("Permutation importance (drop in AUC when shuffled)")
    ax.set_title("Feature importance on stable subset (interpretation)")
    fig.savefig(os.path.join(FIG,"feature_importance.png"), dpi=150, bbox_inches="tight"); plt.close(fig)
summary["feature_importance"] = imp_summary

json.dump(summary, open(os.path.join(OUT,"summary.json"),"w"), indent=2, default=str)
print("\n>> DONE. Figures:", sorted(os.listdir(FIG)))
print(">> Summary keys:", list(summary.keys()))
