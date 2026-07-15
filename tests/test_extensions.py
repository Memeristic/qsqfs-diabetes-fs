"""Smoke tests for the fusion, optimizer, and dataset-loader additions.

Fast, dependency-light (no torch), synthetic data only. Run: pytest -q
"""
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from src.optimizers import get_selector, REGISTRY
from src.fusion import MultimodalFusion, tune_threshold, score_fusion_strategies
from src.datasets import _infer_modalities_from_names


def _toy(weights=(0.6, 0.4), seed=0):
    X, y = make_classification(n_samples=200, n_features=16, n_informative=4,
                              n_redundant=2, weights=list(weights), random_state=seed)
    return X, y


def test_every_optimizer_selects_features():
    X, y = _toy()
    for name in REGISTRY:
        sel = get_selector(name, n_agents=10, max_iter=5, random_state=1).fit(X, y).selected()
        assert len(sel) >= 1, f"{name} selected nothing"


def test_optimizer_auc_metric_runs():
    X, y = _toy()
    s = get_selector("RIME", n_agents=10, max_iter=5, metric="auc", random_state=1).fit(X, y)
    assert 0.0 <= s.best_fit <= 1.0


def test_fusion_strategies_produce_probabilities():
    X, y = _toy()
    mm = {"a": list(range(8)), "b": list(range(8, 16))}
    for strat in ("feature", "decision", "hybrid"):
        f = MultimodalFusion(modality_map=mm, strategy=strat, random_state=1).fit(X, y)
        p = f.predict_proba(X)[:, 1]
        assert p.shape == (X.shape[0],)
        assert np.all((p >= 0) & (p <= 1))


def test_threshold_tuning_lifts_recall_on_imbalanced():
    X, y = _toy(weights=(0.8, 0.2))
    f = MultimodalFusion(strategy="hybrid", random_state=1).fit(X, y)
    p = f.predict_proba(X)[:, 1]
    thr = tune_threshold(y, p, target="f1")
    assert 0.0 < thr < 1.0


def test_score_fusion_strategies_reuses_selections():
    X, y = _toy()
    mm = {"a": list(range(8)), "b": list(range(8, 16))}
    fold_selected = [list(range(16))] * 5
    res = score_fusion_strategies(X, y, mm, fold_selected, n_outer=5, random_state=1)
    assert set(res) == {"feature", "decision", "hybrid"}
    for v in res.values():
        assert 0.0 <= v["auc"] <= 1.0


def test_modality_inference_partitions_all_columns():
    names = ["Glucose", "BMI", "Age", "heart_rate", "random_col", "Insulin"]
    mm = _infer_modalities_from_names(names)
    covered = sorted(c for cols in mm.values() for c in cols)
    assert covered == list(range(len(names)))       # every column assigned exactly once


def test_spss_style_analysis_tables():
    """SPSS-style battery returns the expected tables with significance flags."""
    from src.spss_analysis import run_spss_style_analysis
    import numpy as np
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=200, n_features=10, n_informative=4,
                              weights=[0.6, 0.4], random_state=0)
    names = [f"feat_{i}" for i in range(X.shape[1])]
    res = run_spss_style_analysis(X, y, names, top_k=5)
    assert set(res) == {"independent_samples", "descriptives", "normality"}
    ind = res["independent_samples"]
    for col in ("Feature", "t", "MannWhitney_U", "Cohens_d", "AUC", "Sig", "t_p_FDR"):
        assert col in ind.columns
    # descriptives must have two rows (groups) per kept feature
    assert set(res["descriptives"]["Group"]) == {"Diabetic", "Non-diabetic"}
    # at least one informative feature should be significant on separable data
    assert (ind["Sig"] != "ns").any()
