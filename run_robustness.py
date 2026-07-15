"""
run_robustness.py
=================
Repeat the fusion nested-CV across several random seeds and report the mean and
spread of AUC / F1 / sensitivity. A small spread indicates the headline result
is stable rather than an artefact of one data split.

    python run_robustness.py                          # synthetic demo
    python run_robustness.py mimic /path/to/mimic-iv  # real MIMIC-IV
    python run_robustness.py csv  data.csv Outcome    # any tidy CSV

Seeds default to 42, 7, 13, 21, 99 (override with SEEDS="1,2,3").
"""

from __future__ import annotations

import os
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import yaml

from src.datasets import load_dataset
from src.evaluation import nested_cv_evaluate


def _parse(argv):
    if len(argv) <= 1:
        return "synthetic", {}
    if argv[1].lower() == "mimic":
        return "mimic", {"data_root": argv[2] if len(argv) > 2 else None}
    if argv[1].lower() == "csv":
        opts = {"path": argv[2]}
        if len(argv) > 3:
            opts["label_col"] = argv[3]
        return "csv", opts
    return argv[1], {}


def main():
    source, opts = _parse(sys.argv)
    cfg = yaml.safe_load(open("config.yaml"))
    if source == "mimic":
        cfg.setdefault("preprocessing", {})["combine_how"] = "outer"
    X, y, names, mmap, meta = load_dataset(source, cfg, **opts)
    print(f"dataset: {meta['n_patients']} x {meta['n_features']} "
          f"prevalence {meta['prevalence_pct']}%")

    seeds = [int(s) for s in os.environ.get("SEEDS", "42,7,13,21,99").split(",")]
    n_outer = cfg.get("evaluation", {}).get("n_outer_folds", 5)
    aucs, f1s, senss = [], [], []
    for seed in seeds:
        r = nested_cv_evaluate(X, y, mmap, cfg, n_outer=n_outer,
                               classifier="fusion", random_state=seed)
        p, t = r["oof_proba"], r["oof_y"]
        pred = (p >= 0.5).astype(int)
        tp = int(np.sum((pred == 1) & (t == 1)))
        fn = int(np.sum((pred == 0) & (t == 1)))
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        aucs.append(r["auc_mean"]); f1s.append(r["f1_mean"]); senss.append(sens)
        print(f"seed {seed:3d}: AUC={r['auc_mean']:.3f} "
              f"F1={r['f1_mean']:.3f} sensitivity={sens:.3f}")

    print(f"\nRobustness over {len(seeds)} seeds: "
          f"AUC {np.mean(aucs):.3f}+/-{np.std(aucs):.3f}  "
          f"F1 {np.mean(f1s):.3f}+/-{np.std(f1s):.3f}  "
          f"sensitivity {np.mean(senss):.3f}+/-{np.std(senss):.3f}")


if __name__ == "__main__":
    main()
