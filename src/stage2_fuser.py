"""
src/stage2_fuser.py
====================
Run Stage-2 fusion on the FULL dataset for interpretation / feature display.
Passes per-feature Stage-1 selection frequency as `pool_weights` so the
frequency-weighted Stage-2 initialisation (thesis Sec 3.5.1) is actually used.

Performance estimates come from src.evaluation.nested_cv_evaluate, NOT this
full-data fit.
"""
from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
from src.qsfs import QSQFS
try:
    from loguru import logger
except ImportError:  # pragma: no cover
    import logging; logger = logging.getLogger(__name__)


def run_stage2_fusion(X_combined: np.ndarray, y: np.ndarray,
                      all_global_col_names: List[str],
                      modality_selected: Dict[str, List[str]],
                      config: Optional[dict] = None, random_state: int = 42) -> QSQFS:
    name_to_idx = {n: i for i, n in enumerate(all_global_col_names)}
    weight: Dict[int, float] = {}
    for names in modality_selected.values():
        for n in names:
            if n in name_to_idx:
                weight[name_to_idx[n]] = weight.get(name_to_idx[n], 0.0) + 1.0
    pool = sorted(weight.keys())
    if not pool:
        logger.warning("Stage2: empty pool - using all features.")
        pool = list(range(X_combined.shape[1]))
        weights = None
    else:
        weights = [weight[g] for g in pool]
    cfg = (config or {}).get("stage2", {})
    model = QSQFS(
        n_colonies=cfg.get("n_colonies", 40),
        max_iter_stage1=0, max_iter_stage2=cfg.get("max_iter_stage2", 40),
        alpha=cfg.get("alpha", 0.95), w_AI=cfg.get("w_AI", 0.50),
        delta2=cfg.get("delta2", 0.95), rho=cfg.get("rho", 0.80),
        stagnation_window=cfg.get("stagnation_window", 15),
        diversity_thresh=cfg.get("diversity_thresh", 0.05),
        k_nn=cfg.get("k_nn", 3), cv_folds=cfg.get("cv_folds", 5),
        random_state=random_state, verbose=True,
    )
    model.fit(X_combined, y, feature_pool=pool, pool_weights=weights)
    return model
