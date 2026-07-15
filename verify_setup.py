#!/usr/bin/env python3
"""
verify_setup.py
===============
End-to-end smoke test for the QSQ-FS framework. Runs:
  data load -> modalities -> nested-CV pipeline -> comparative -> ablation
on synthetic demo data with tiny settings (well under a minute). No Torch /
Streamlit required (the MLP falls back to a regularised sklearn model).

    pip install -r requirements.txt
    python verify_setup.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import yaml

from src.data_loader import MIMICDataLoader
from src.leakage import resolve_leaky_itemids
from src.modality_builder import (
    extract_diabetes_label, build_labs, build_vitals, build_medications,
    build_diagnoses, build_combined_matrix, build_modality_map,
)
from src.evaluation import nested_cv_evaluate
from src.comparative_analysis import run_comparative_analysis
from src.ablation import run_ablation_study


def main():
    t0 = time.time()
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    # tiny settings for a fast smoke test
    config["stage1"].update({"n_colonies": 10, "max_iter_stage1": 5, "cv_folds": 3})
    config["stage2"].update({"n_colonies": 10, "max_iter_stage2": 5, "cv_folds": 3})
    config["neural_model"].update({"max_epochs": 60, "early_stopping_patience": 8})

    print("[1/5] Loading synthetic demo data...")
    data = MIMICDataLoader(config).load_all()
    label_df = extract_diabetes_label(
        data.get("patients"), data.get("diagnoses_icd"), data.get("d_icd_diagnoses"))

    print("[2/5] Building modalities...")
    mod = {}
    leaky_ids = resolve_leaky_itemids(data.get("d_labitems"), data.get("d_items"))
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
    assert mod, "No modalities built."
    combined, names = build_combined_matrix(mod, how="inner")
    y = combined["label"].values.astype(int)
    X = combined[names].values.astype(float)
    mmap = build_modality_map(mod, names)
    print(f"    Combined: {X.shape[0]} samples x {X.shape[1]} features; "
          f"modalities={ {k: len(v) for k, v in mmap.items()} }")
    print(f"    Prevalence: {100*y.mean():.1f}% positive")

    print("[3/5] Nested-CV pipeline (unbiased performance estimate)...")
    res = nested_cv_evaluate(X, y, mmap, config, n_outer=3, random_state=42)
    print(f"    AUC      = {res['auc_mean']:.4f} +/- {res['auc_std']:.4f} "
          f"(95% CI +/-{res['auc_ci95']:.4f})")
    print(f"    Accuracy = {res['accuracy_mean']:.4f} +/- {res['accuracy_std']:.4f}")
    print(f"    F1       = {res['f1_mean']:.4f}")
    print(f"    Features = {res['n_features_mean']:.1f} (mean over folds)")
    top = np.argsort(res["selection_frequency"])[::-1][:8]
    print("    Most stable features:",
          [(names[i], round(float(res["selection_frequency"][i]), 2)) for i in top])

    print("[4/5] Comparative analysis (leak-free, equal budget)...")
    comp = run_comparative_analysis(X, y, n_trials=3, n_agents=12, max_iter=6)
    for name in [k for k in comp if not k.startswith("_")]:
        a = comp[name]["accuracy"]
        print(f"    {name:14s} acc={a.mean():.4f}  auc={np.nanmean(comp[name]['auc']):.4f}  "
              f"feat={comp[name]['n_features'].mean():.1f}")

    print("[5/5] Ablation study (per-mechanism switches)...")
    abl = run_ablation_study(X, y, n_trials=3)
    for name in [k for k in abl if not k.startswith("_")]:
        r = abl[name]
        print(f"    {name:18s} acc={r['accuracy'].mean():.4f}  "
              f"evals={r['n_evals']}  rt={r['runtime'][0]:.2f}s")

    print(f"\nAll steps completed in {time.time()-t0:.1f}s.  Setup OK.")
    print("Run the dashboard with:  streamlit run app.py")


if __name__ == "__main__":
    main()
