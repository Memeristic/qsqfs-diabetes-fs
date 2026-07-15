"""
src/ablation.py
===============
Mechanism-attribution ablation.

Design notes:
  * Variants use explicit switches on QSQFS (use_qs / use_qq / use_cache) to
    turn each mechanism off independently. "No Cache" disables memoisation, so
    it affects runtime only.
  * Accuracy is measured leak-free: select on the train fold, score on the
    held-out test fold (src.evaluation.select_then_score). Because fitness is a
    deterministic function of the subset, the cache affects runtime only.
  * Significance uses the Wilcoxon signed-rank test with Bonferroni correction
    on paired per-fold accuracy vectors.
  * Runtime per variant is measured directly from QSQFS.runtime / n_evals.
"""

from __future__ import annotations

import time
from typing import Dict, List

import numpy as np
from scipy import stats

try:
    from loguru import logger
except ImportError:                       # pragma: no cover
    import logging
    logger = logging.getLogger(__name__)

from src.qsfs import QSQFS
from src.evaluation import select_then_score

# Mirrors the Stage-1/Stage-2 search budget in config.yaml so that each variant
# isolates a single mechanism of the reported configuration, holding the
# population size, iteration count and fitness weighting fixed.
BASE = dict(n_colonies=40, max_iter_stage1=15, max_iter_stage2=40,
            alpha=0.85, cv_folds=5, k_nn=3, verbose=False)

VARIANTS: Dict[str, dict] = {
    "Full QSQ-FS":      {},
    "No QS (w_AI off)": {"use_qs": False},
    "No QQ archive":    {"use_qq": False},
    "No Cache":         {"use_cache": False},   # genuine disable (was inverted before)
    "No Elitism":       {"elitism": False},
}


def _make_selector(overrides, random_state):
    def selector(Xtr, ytr):
        m = QSQFS(**BASE, **overrides, random_state=random_state)
        m.fit(Xtr, ytr)
        selector.last_runtime = m.runtime
        selector.last_evals = m.n_evals
        return m.get_selected_features()
    selector.last_runtime = 0.0
    selector.last_evals = 0
    return selector


def run_ablation_study(X, y, n_trials=5, random_state=42) -> dict:
    """n_trials -> number of outer CV folds (paired across variants)."""
    X = np.asarray(X, float)
    y = np.asarray(y).astype(int)
    n_outer = max(3, n_trials)
    results: Dict[str, dict] = {}

    for name, ov in VARIANTS.items():
        logger.info(f"Ablation: {name}")
        # accuracy under leak-free protocol
        sel = _make_selector(ov, random_state)
        m = select_then_score(sel, X, y, n_outer=n_outer, random_state=random_state)
        # runtime/evals on a single full-data fit (representative)
        t0 = time.time()
        probe = QSQFS(**BASE, **ov, random_state=random_state)
        probe.fit(X, y)
        rt = time.time() - t0
        results[name] = {
            "accuracy": m["accuracy"], "auc": m["auc"], "f1": m["f1"],
            "n_features": m["n_features"],
            "runtime": np.array([rt]),
            "n_evals": int(probe.n_evals),
        }
        logger.info(f"  acc={m['accuracy'].mean():.4f}+/-{m['accuracy'].std():.4f} "
                    f"feat={m['n_features'].mean():.1f} rt={rt:.2f}s evals={probe.n_evals}")

    # Wilcoxon vs Full
    base = results["Full QSQ-FS"]["accuracy"]
    alpha_bonf = 0.05 / max(1, len(VARIANTS) - 1)
    stat_rows: List[dict] = []
    for name, res in results.items():
        if name == "Full QSQ-FS":
            continue
        diff = base - res["accuracy"]
        if np.allclose(diff, 0) or len(diff) < 5:
            w, p = np.nan, 1.0
        else:
            w, p = stats.wilcoxon(diff, alternative="greater", zero_method="wilcox")
        stat_rows.append({
            "mechanism": name,
            "contribution_pct": round(float(np.mean(diff)) * 100, 3),
            "w_statistic": round(float(w), 3) if not np.isnan(w) else "n/a",
            "p_value": round(float(p), 4),
            "significant": bool(p < alpha_bonf),
            "alpha_bonf": round(alpha_bonf, 4),
        })
    results["_stats"] = stat_rows
    results["_n_outer"] = n_outer
    return results
