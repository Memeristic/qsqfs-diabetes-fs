"""
src/evaluation.py
=================
Leakage-free evaluation protocols.

Feature selection is performed *inside* the training partition of an outer
cross-validation loop and never sees the held-out test fold. Reported metrics
are therefore unbiased estimates of generalisation, with proper across-fold
variance.

Two entry points:

  nested_cv_evaluate(...)   Full pipeline (Stage-1 per modality -> Stage-2 fusion
                            -> final classifier) wrapped in outer CV. Returns
                            per-fold AUC/Accuracy/F1 + selection-stability counts.

  select_then_score(...)    Generic "run a selector on the train fold, score the
                            chosen subset on the test fold" used by the
                            comparative and ablation studies so every method is
                            judged under the identical leak-free protocol with an
                            equal evaluation budget.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

try:
    from loguru import logger
except ImportError:                       # pragma: no cover
    import logging
    logger = logging.getLogger(__name__)

from src.qsfs import QSQFS


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def _restrict_modality_map(modality_map: Dict[str, Sequence[int]],
                           selected: Sequence[int]) -> Dict[str, List[int]]:
    """Re-express a global modality map in terms of a selected feature subset.

    `selected` are global column indices; the fusion model is fed X[:, selected],
    so its modality map must use positions 0..len(selected)-1. Preserves which
    modality each surviving feature belonged to.
    """
    selected = list(selected)
    pos = {g: i for i, g in enumerate(selected)}
    out: Dict[str, List[int]] = {}
    for name, cols in (modality_map or {}).items():
        local = [pos[g] for g in cols if g in pos]
        if local:
            out[name] = local
    if not out:
        out = {"all": list(range(len(selected)))}
    return out
def _ci95(values: Sequence[float]) -> Tuple[float, float, float]:
    """Return (mean, std, half-width of 95% CI) for a small sample."""
    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    half = 1.96 * std / np.sqrt(len(arr)) if len(arr) > 1 else 0.0
    return mean, std, half


def _knn_score_on_holdout(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    selected: np.ndarray, k: int = 3,
) -> Dict[str, float]:
    """Train a KNN on the selected columns of train, score on test (no leakage)."""
    if selected is None or len(selected) == 0:
        return {"accuracy": 0.5, "auc": 0.5, "f1": 0.0, "n_features": 0}
    scaler = StandardScaler().fit(X_train[:, selected])
    Xtr = scaler.transform(X_train[:, selected])
    Xte = scaler.transform(X_test[:, selected])
    knn = KNeighborsClassifier(n_neighbors=k).fit(Xtr, y_train)
    proba = knn.predict_proba(Xte)[:, 1] if len(np.unique(y_train)) > 1 else np.zeros(len(y_test))
    pred = (proba >= 0.5).astype(int)
    auc = roc_auc_score(y_test, proba) if len(np.unique(y_test)) > 1 else float("nan")
    return {
        "accuracy": accuracy_score(y_test, pred),
        "auc": auc,
        "f1": f1_score(y_test, pred, zero_division=0),
        "n_features": int(len(selected)),
        "proba": proba,
        "y_test": np.asarray(y_test),
    }


# --------------------------------------------------------------------------- #
# Generic leak-free selector evaluation (comparative / ablation)              #
# --------------------------------------------------------------------------- #
def select_then_score(
    selector: Callable[[np.ndarray, np.ndarray], np.ndarray],
    X: np.ndarray,
    y: np.ndarray,
    n_outer: int = 5,
    k_eval: int = 3,
    random_state: int = 42,
) -> Dict[str, np.ndarray]:
    """
    `selector(X_train, y_train) -> selected_indices` is fitted on each outer
    train fold; the subset is scored on the matching test fold. Returns arrays
    of per-fold accuracy / auc / f1 / n_features.
    """
    skf = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=random_state)
    accs, aucs, f1s, nfs = [], [], [], []
    for tr, te in skf.split(X, y):
        sel = selector(X[tr], y[tr])
        m = _knn_score_on_holdout(X[tr], y[tr], X[te], y[te], np.asarray(sel), k=k_eval)
        accs.append(m["accuracy"]); aucs.append(m["auc"]); f1s.append(m["f1"]); nfs.append(m["n_features"])
    return {
        "accuracy": np.array(accs), "auc": np.array(aucs),
        "f1": np.array(f1s), "n_features": np.array(nfs),
    }


# --------------------------------------------------------------------------- #
# Full nested-CV pipeline evaluation                                          #
# --------------------------------------------------------------------------- #
def nested_cv_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    modality_map: Dict[str, List[int]],
    config: dict,
    n_outer: int = 5,
    random_state: int = 42,
    classifier: str = "auto",
    progress_cb: Optional[Callable[[float, str], None]] = None,
) -> dict:
    """
    Two-stage QSQ-FS pipeline under outer CV.

    Parameters
    ----------
    X            : combined (unscaled) feature matrix, shape (n_samples, n_features)
    y            : binary labels
    modality_map : {modality_name: [global column indices]}
    config       : full config dict (uses config['stage1'] / config['stage2'] /
                   config['neural_model'])
    classifier   : 'mlp', 'knn', or 'auto' (mlp if torch/sklearn-MLP available)

    Returns a dict with per-fold metrics, aggregates with 95% CIs, and
    selection-frequency stability across folds.
    """
    from src.neural_model import train_mlp, evaluate_model  # local import (optional torch)

    skf = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=random_state)
    n_features = X.shape[1]
    fold_metrics: List[Dict[str, float]] = []
    selection_counts = np.zeros(n_features)
    fold_selected: List[List[int]] = []
    oof_proba: List[np.ndarray] = []
    oof_y: List[np.ndarray] = []
    oof_pred: List[np.ndarray] = []      # decisions at each fold's OWN train-tuned threshold
    stage1_cfg = config.get("stage1", {})
    stage2_cfg = config.get("stage2", {})
    nm_cfg = config.get("neural_model", {})

    for fold, (tr, te) in enumerate(skf.split(X, y), start=1):
        if progress_cb:
            progress_cb(fold / n_outer, f"Outer fold {fold}/{n_outer}")
        X_tr, X_te, y_tr, y_te = X[tr], X[te], y[tr], y[te]

        # ---- Stage 1: per modality, TRAIN ONLY ----
        pool: List[int] = []
        pool_weight: Dict[int, float] = {}
        for mod, cols in modality_map.items():
            cols = list(cols)
            if len(cols) == 0:
                continue
            m1 = QSQFS(
                n_colonies=stage1_cfg.get("n_colonies", 25),
                max_iter_stage1=stage1_cfg.get("max_iter_stage1", 15),
                max_iter_stage2=0,
                alpha=stage1_cfg.get("alpha", 0.85),
                max_frac=stage1_cfg.get("max_frac", 0.50),
                min_frac=stage1_cfg.get("min_frac", 0.05),
                w_AI=stage1_cfg.get("w_AI", 0.50),
                delta1=stage1_cfg.get("delta1", 0.97),
                weak_thresh1=stage1_cfg.get("weak_thresh1", 0.30),
                stagnation_window=stage1_cfg.get("stagnation_window", 15),
                diversity_thresh=stage1_cfg.get("diversity_thresh", 0.05),
                k_nn=stage1_cfg.get("k_nn", 3),
                cv_folds=stage1_cfg.get("cv_folds", 5),
                random_state=random_state + fold,
                verbose=False,
            )
            m1.fit(X_tr[:, cols], y_tr)
            for local_idx in m1.get_selected_features():
                g = cols[int(local_idx)]
                pool.append(g)
                pool_weight[g] = pool_weight.get(g, 0.0) + 1.0

        pool = sorted(set(pool)) or list(range(n_features))
        weights = [pool_weight.get(g, 1.0) for g in pool]

        # ---- Stage 2: fusion, TRAIN ONLY ----
        m2 = QSQFS(
            n_colonies=stage2_cfg.get("n_colonies", 40),
            max_iter_stage1=0,
            max_iter_stage2=stage2_cfg.get("max_iter_stage2", 40),
            alpha=stage2_cfg.get("alpha", 0.85),
            max_frac=stage2_cfg.get("max_frac", 0.50),
            min_frac=stage2_cfg.get("min_frac", 0.05),
            w_AI=stage2_cfg.get("w_AI", 0.50),
            delta2=stage2_cfg.get("delta2", 0.95),
            rho=stage2_cfg.get("rho", 0.80),
            stagnation_window=stage2_cfg.get("stagnation_window", 15),
            diversity_thresh=stage2_cfg.get("diversity_thresh", 0.05),
            k_nn=stage2_cfg.get("k_nn", 3),
            cv_folds=stage2_cfg.get("cv_folds", 5),
            random_state=random_state + fold,
            verbose=False,
        )
        m2.fit(X_tr, y_tr, feature_pool=pool, pool_weights=weights)
        selected = m2.get_selected_features()
        if len(selected) == 0:
            selected = np.array(pool)
        fold_selected.append([int(s) for s in selected])
        selection_counts[selected] += 1

        # ---- Final classifier on TRAIN, evaluate on TEST ----
        if classifier == "fusion":
            # Multimodal fusion over the SELECTED features, keeping each
            # feature's modality membership so the fusion head sees genuine
            # per-modality views. Threshold is tuned on the train fold only
            # (F1-optimal) to lift sensitivity on the imbalanced cohort.
            from src.fusion import MultimodalFusion, tune_threshold
            from sklearn.model_selection import train_test_split
            sel_map = _restrict_modality_map(modality_map, selected)
            fmodel = MultimodalFusion(
                modality_map=sel_map,
                strategy=nm_cfg.get("fusion_strategy", "hybrid"),
                latent_dim=nm_cfg.get("latent_dim", 8),
                hidden_layers=nm_cfg.get("hidden_layers", [32]),
                l2_alpha=nm_cfg.get("l2_alpha", 1e-3),
                class_weight="balanced",
                random_state=random_state + fold,
            ).fit(X_tr[:, selected], y_tr)
            # tune threshold on an internal split of the training fold
            if np.min(np.bincount(y_tr)) >= 4:
                Xa, Xb, ya, yb = train_test_split(
                    X_tr[:, selected], y_tr, test_size=0.3,
                    stratify=y_tr, random_state=random_state + fold)
                tmodel = MultimodalFusion(
                    modality_map=sel_map,
                    strategy=nm_cfg.get("fusion_strategy", "hybrid"),
                    latent_dim=nm_cfg.get("latent_dim", 8),
                    hidden_layers=nm_cfg.get("hidden_layers", [32]),
                    l2_alpha=nm_cfg.get("l2_alpha", 1e-3),
                    class_weight="balanced",
                    random_state=random_state + fold,
                ).fit(Xa, ya)
                thr = tune_threshold(yb, tmodel.predict_proba(Xb)[:, 1], target="f1")
            else:
                thr = 0.5
            proba = fmodel.predict_proba(X_te[:, selected])[:, 1]
            pred = (proba >= thr).astype(int)
            auc = roc_auc_score(y_te, proba) if len(np.unique(y_te)) > 1 else float("nan")
            metrics = {"accuracy": float(np.mean(pred == y_te)),
                       "auc": float(auc),
                       "f1": float(f1_score(y_te, pred, zero_division=0)),
                       "threshold": float(thr),
                       "proba": proba, "y_test": np.asarray(y_te)}
        elif classifier in ("mlp", "auto"):
            scaler = StandardScaler().fit(X_tr[:, selected])
            Xtr_s = scaler.transform(X_tr[:, selected])
            Xte_s = scaler.transform(X_te[:, selected])
            model = train_mlp(Xtr_s, y_tr, nm_cfg, random_state=random_state + fold)
            metrics = evaluate_model(model, Xte_s, y_te)
        else:
            metrics = _knn_score_on_holdout(
                X_tr, y_tr, X_te, y_te, selected, k=stage2_cfg.get("k_nn", 3))
        metrics["n_features"] = int(len(selected))
        fold_metrics.append(metrics)
        if "proba" in metrics and "y_test" in metrics:
            oof_proba.append(np.asarray(metrics["proba"]))
            oof_y.append(np.asarray(metrics["y_test"]))
            # The operating point is fixed on the training partition of this
            # fold and applied unchanged to its held-out test fold. Pooling these
            # decisions yields an unbiased confusion matrix, since no test label
            # participates in choosing the threshold applied to it.
            fold_thr = float(metrics.get("threshold", 0.5))
            oof_pred.append((np.asarray(metrics["proba"]) >= fold_thr).astype(int))
        logger.info(f"[nested-CV] fold {fold}: AUC={metrics.get('auc', float('nan')):.4f} "
                    f"ACC={metrics['accuracy']:.4f} F1={metrics['f1']:.4f} "
                    f"({metrics['n_features']} feats)")

    # ---- Aggregate ----
    def agg(key):
        vals = [m[key] for m in fold_metrics if not np.isnan(m.get(key, np.nan))]
        return _ci95(vals) if vals else (float("nan"), 0.0, 0.0)

    acc_m, acc_s, acc_ci = agg("accuracy")
    auc_m, auc_s, auc_ci = agg("auc")
    f1_m, f1_s, f1_ci = agg("f1")
    nf_m, nf_s, _ = agg("n_features")

    stability = selection_counts / n_outer    # fraction of folds each feature was chosen
    oof_p = np.concatenate(oof_proba) if oof_proba else np.array([])
    oof_t = np.concatenate(oof_y) if oof_y else np.array([])
    oof_d = np.concatenate(oof_pred) if oof_pred else np.array([])
    return {
        "per_fold": fold_metrics,
        "fold_selected": fold_selected,
        "accuracy_mean": acc_m, "accuracy_std": acc_s, "accuracy_ci95": acc_ci,
        "auc_mean": auc_m, "auc_std": auc_s, "auc_ci95": auc_ci,
        "f1_mean": f1_m, "f1_std": f1_s, "f1_ci95": f1_ci,
        "n_features_mean": nf_m, "n_features_std": nf_s,
        "selection_frequency": stability,
        "oof_proba": oof_p, "oof_y": oof_t, "oof_pred": oof_d,
        "n_outer": n_outer,
    }
