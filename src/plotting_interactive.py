"""
src/plotting_interactive.py — interactive Plotly versions of the charts
=======================================================================
These mirror the matplotlib helpers in ``src/plotting.py`` one-for-one, but
return Plotly figures for ``st.plotly_chart`` so the dashboard charts are
zoomable / hoverable. ``src/plotting.py`` is kept for static export (PNG for
print / LaTeX) — this module is purely for on-screen use.

Every function returns a ``plotly.graph_objects.Figure``. They intentionally
take the SAME arguments as their matplotlib twins so the app can swap between
static and interactive with a one-line change.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc as sk_auc, confusion_matrix

# A small, colour-blind-friendly palette reused across all charts.
_ACCENT = "#2563eb"
_ACCENT2 = "#f59e0b"
_GREY = "#94a3b8"


def _base_layout(fig: go.Figure, title: str, height: int = 380) -> go.Figure:
    """Apply one consistent, theme-neutral layout to every figure."""
    fig.update_layout(
        title=dict(text=title, x=0.02, xanchor="left", font=dict(size=15)),
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="closest",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(148,163,184,0.2)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.2)", zeroline=False)
    return fig


def plot_roc_interactive(y_true, proba, title: str = "ROC (out-of-fold)") -> go.Figure:
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    fpr, tpr, _ = roc_curve(y_true, proba)
    roc_auc = sk_auc(fpr, tpr)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={roc_auc:.3f})",
                             line=dict(color=_ACCENT, width=3),
                             hovertemplate="FPR=%{x:.3f}<br>TPR=%{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance",
                             line=dict(color=_GREY, width=1, dash="dash"),
                             hoverinfo="skip"))
    fig.update_xaxes(title="False positive rate", range=[-0.02, 1.02])
    fig.update_yaxes(title="True positive rate", range=[-0.02, 1.02])
    return _base_layout(fig, title)


def plot_confusion_interactive(y_true, y_pred, title: str = "Confusion (out-of-fold)") -> go.Figure:
    cm = confusion_matrix(np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int))
    labels = ["No DM", "DM"]
    z = cm.astype(int)
    text = [[str(v) for v in row] for row in z]
    fig = go.Figure(data=go.Heatmap(
        z=z, x=labels, y=labels, colorscale="Blues", showscale=True,
        text=text, texttemplate="%{text}", textfont=dict(size=18),
        hovertemplate="Predicted %{x}<br>Actual %{y}<br>Count=%{z}<extra></extra>"))
    fig.update_xaxes(title="Predicted")
    fig.update_yaxes(title="Actual", autorange="reversed")
    return _base_layout(fig, title, height=360)


def plot_convergence_interactive(history: List[float],
                                 feature_counts: Optional[List[int]] = None,
                                 title: str = "Convergence") -> go.Figure:
    history = list(history)
    it = list(range(1, len(history) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=it, y=history, mode="lines+markers", name="Best fitness",
                             line=dict(color=_ACCENT, width=3), marker=dict(size=5),
                             hovertemplate="iter %{x}<br>fitness=%{y:.4f}<extra></extra>"))
    fig.update_xaxes(title="Iteration")
    fig.update_yaxes(title="Best fitness")
    if feature_counts is not None and len(feature_counts) == len(history):
        fig.add_trace(go.Scatter(x=it, y=feature_counts, mode="lines", name="# features",
                                 line=dict(color=_ACCENT2, width=2, dash="dot"), yaxis="y2",
                                 hovertemplate="iter %{x}<br>%{y} features<extra></extra>"))
        fig.update_layout(yaxis2=dict(title="# features", overlaying="y", side="right",
                                      showgrid=False))
    return _base_layout(fig, title)


def plot_comparison_interactive(results: Dict[str, dict], metric: str = "auc",
                                title: Optional[str] = None) -> go.Figure:
    names, means, errs = [], [], []
    for name, r in results.items():
        if name.startswith("_") or metric not in r:
            continue
        vals = np.asarray(r[metric], dtype=float)
        names.append(name)
        means.append(np.nanmean(vals))
        errs.append(np.nanstd(vals))
    order = np.argsort(means)[::-1]
    names = [names[i] for i in order]
    means = [means[i] for i in order]
    errs = [errs[i] for i in order]
    colors = [_ACCENT if n.upper().startswith("QSQ") else _GREY for n in names]
    fig = go.Figure(go.Bar(
        x=names, y=means, marker_color=colors,
        error_y=dict(type="data", array=errs, visible=True, color="rgba(100,100,100,0.6)"),
        hovertemplate="%{x}<br>" + metric.upper() + "=%{y:.4f}<extra></extra>"))
    fig.update_yaxes(title=metric.upper(), range=[0, 1])
    return _base_layout(fig, title or f"Method comparison ({metric.upper()})")


def plot_selection_frequency_interactive(freq: np.ndarray, names: List[str],
                                         top_n: int = 20,
                                         title: str = "Feature selection stability") -> go.Figure:
    freq = np.asarray(freq, dtype=float)
    order = np.argsort(freq)[::-1][:top_n]
    yn = [names[i] for i in order][::-1]
    xv = [freq[i] for i in order][::-1]
    colors = [_ACCENT if v >= 0.5 else _GREY for v in xv]
    fig = go.Figure(go.Bar(
        x=xv, y=yn, orientation="h", marker_color=colors,
        hovertemplate="%{y}<br>selected in %{x:.0%} of folds<extra></extra>"))
    fig.update_xaxes(title="Selection frequency (fraction of folds)", range=[0, 1])
    fig.add_vline(x=0.5, line_dash="dash", line_color=_ACCENT2,
                  annotation_text="stable ≥ 50%", annotation_position="top")
    return _base_layout(fig, title, height=max(360, 22 * len(yn)))
