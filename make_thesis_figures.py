"""
make_thesis_figures.py
======================
Render the publication figures for the additions (fusion-strategy comparison
and the full metaheuristic comparison) from a completed analysis_out/summary.json.

    python make_thesis_figures.py [analysis_dir]

Reads <analysis_dir>/summary.json and writes PNGs into <analysis_dir>/figures/.
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def fusion_strategy_figure(summary, fig_dir):
    fs = summary.get("fusion_strategies")
    if not fs:
        return
    strategies = [r["Strategy"].capitalize() for r in fs]
    metrics = ["AUC", "F1", "Sensitivity"]
    x = np.arange(len(strategies))
    width = 0.25
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for i, m in enumerate(metrics):
        ax.bar(x + i * width, [r[m] for r in fs], width, label=m)
    ax.set_xticks(x + width)
    ax.set_xticklabels(strategies)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Multimodal fusion strategies")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(os.path.join(fig_dir, "fusion_strategies.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def comparative_figure(summary, fig_dir):
    tbl = summary.get("comparative", {}).get("table")
    if not tbl:
        return
    tbl = sorted(tbl, key=lambda r: r["AUC"])
    methods = [r["Method"] for r in tbl]
    aucs = [r["AUC"] for r in tbl]
    errs = [r.get("AUC_std", 0) for r in tbl]
    # highlight the metaheuristics vs classical baselines
    classical = {"SVM", "RandomForest", "XGBoost", "LogReg_AllFeatures"}
    colors = ["#94a3b8" if m in classical else "#2563eb" for m in methods]
    colors = ["#f59e0b" if m == "QSQ-FS" else c for m, c in zip(methods, colors)]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.barh(range(len(methods)), aucs, xerr=errs, color=colors)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods)
    ax.set_xlabel("AUC (nested CV, leak-free)")
    ax.set_title("QSQ-FS vs named metaheuristics and classical baselines")
    ax.set_xlim(0.5, 1.0)
    ax.grid(axis="x", alpha=0.3)
    fig.savefig(os.path.join(fig_dir, "comparative_all_methods.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "analysis_out"
    fig_dir = os.path.join(out, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    summary = json.load(open(os.path.join(out, "summary.json")))
    fusion_strategy_figure(summary, fig_dir)
    comparative_figure(summary, fig_dir)
    print("Wrote:", sorted(f for f in os.listdir(fig_dir)
                           if f in ("fusion_strategies.png", "comparative_all_methods.png")))


if __name__ == "__main__":
    main()
