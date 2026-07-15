"""
src/plotting.py
===============
Matplotlib helpers (Agg-safe) returning Figure objects for Streamlit / export.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve

plt.rcParams.update({"figure.dpi": 110, "font.size": 10})


def plot_convergence(history: List[float], feature_counts: Optional[List[int]] = None,
                     stage_boundary: Optional[int] = None, title="QSQ-FS Convergence"):
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(history, color="#2563eb", lw=2, label="Best fitness")
    ax1.set_xlabel("Iteration"); ax1.set_ylabel("Fitness", color="#2563eb")
    ax1.tick_params(axis="y", labelcolor="#2563eb")
    if stage_boundary:
        ax1.axvline(stage_boundary, ls="--", color="#9333ea", alpha=0.7)
        ax1.text(stage_boundary, ax1.get_ylim()[0], " Stage 1|2", color="#9333ea", fontsize=8)
    if feature_counts:
        ax2 = ax1.twinx()
        ax2.plot(feature_counts, color="#16a34a", lw=1.2, alpha=0.6, label="# features")
        ax2.set_ylabel("# features", color="#16a34a")
        ax2.tick_params(axis="y", labelcolor="#16a34a")
    ax1.set_title(title); fig.tight_layout()
    return fig


def plot_roc(y_true, proba, title="ROC (out-of-fold)"):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    if len(np.unique(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, proba)
        from sklearn.metrics import auc as _auc
        ax.plot(fpr, tpr, color="#2563eb", lw=2, label=f"AUC = {_auc(fpr, tpr):.3f}")
    ax.plot([0, 1], [0, 1], ls="--", color="gray", alpha=0.6)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(title); ax.legend(loc="lower right"); fig.tight_layout()
    return fig


def plot_confusion(y_true, y_pred, title="Confusion (out-of-fold)"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["No DM", "DM"]); ax.set_yticklabels(["No DM", "DM"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title)
    fig.colorbar(im, fraction=0.046); fig.tight_layout()
    return fig


def plot_comparison(results: Dict[str, dict], metric="auc", title=None):
    names, means, errs = [], [], []
    for name, res in results.items():
        if name.startswith("_") or metric not in res:
            continue
        vals = np.asarray(res[metric], dtype=float)
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            continue
        names.append(name); means.append(vals.mean())
        errs.append(vals.std() / np.sqrt(len(vals)) if len(vals) > 1 else 0)
    order = np.argsort(means)[::-1]
    names = [names[i] for i in order]; means = [means[i] for i in order]; errs = [errs[i] for i in order]
    colors = ["#dc2626" if n == "QSQ-FS" else "#64748b" for n in names]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(names, means, yerr=errs, color=colors, capsize=4)
    ax.set_ylabel(metric.upper()); ax.set_title(title or f"{metric.upper()} by method")
    ax.set_ylim(min(means) - 0.05, 1.0 if max(means) < 1 else max(means) + 0.05)
    plt.xticks(rotation=30, ha="right"); fig.tight_layout()
    return fig


def plot_selection_frequency(freq: np.ndarray, names: List[str], top_n=20,
                             title="Feature selection stability (across outer folds)"):
    idx = np.argsort(freq)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(8, max(4, len(idx) * 0.3)))
    ax.barh([names[i] for i in idx][::-1], [freq[i] for i in idx][::-1], color="#2563eb")
    ax.set_xlabel("Fraction of folds selected"); ax.set_xlim(0, 1); ax.set_title(title)
    fig.tight_layout()
    return fig
