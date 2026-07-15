"""
src/stage1_runner.py
====================
Run Stage-1 (per-modality) selection on the FULL dataset for interpretation /
feature-stability display. NOTE: performance numbers must come from
src.evaluation.nested_cv_evaluate (this path intentionally fits on all data to
show *which* features the method favours, not to estimate accuracy).

QSQFS scales internally per CV fold, so no pre-scaling is done here; a global
StandardScaler at this point would both double-scale and peek across all rows.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import pandas as pd
from src.qsfs import QSQFS
try:
    from loguru import logger
except ImportError:  # pragma: no cover
    import logging; logger = logging.getLogger(__name__)


def run_stage1_on_modality(df: pd.DataFrame, label_col="label", config=None,
                           random_state=42, modality_name="") -> Tuple[List[str], Optional[QSQFS], List[str]]:
    if df is None or len(df) < 10:
        return [], None, []
    feats = [c for c in df.columns if c not in ("subject_id", label_col)]
    if not feats:
        return [], None, []
    X = df[feats].values.astype(float)
    y = df[label_col].values.astype(int)
    cfg = (config or {}).get("stage1", {})
    model = QSQFS(
        n_colonies=cfg.get("n_colonies", 25),
        max_iter_stage1=cfg.get("max_iter_stage1", 15),
        max_iter_stage2=0,
        alpha=cfg.get("alpha", 0.95), w_AI=cfg.get("w_AI", 0.50),
        delta1=cfg.get("delta1", 0.97), weak_thresh1=cfg.get("weak_thresh1", 0.30),
        stagnation_window=cfg.get("stagnation_window", 15),
        diversity_thresh=cfg.get("diversity_thresh", 0.05),
        k_nn=cfg.get("k_nn", 3), cv_folds=cfg.get("cv_folds", 5),
        random_state=random_state, verbose=False,
    )
    model.fit(X, y)
    sel = [feats[i] for i in model.get_selected_features() if i < len(feats)]
    logger.info(f"Stage1 '{modality_name}': {len(sel)}/{len(feats)} "
                f"(fitness={model.get_best_fitness():.4f})")
    return sel, model, feats


def run_all_modalities(modality_dfs: Dict[str, pd.DataFrame], config=None, random_state=42):
    all_sel, all_models, all_feats = {}, {}, {}
    for i, (name, df) in enumerate(modality_dfs.items()):
        sel, mdl, feats = run_stage1_on_modality(
            df, config=config, random_state=random_state + i, modality_name=name)
        all_sel[name] = sel
        if mdl is not None:
            all_models[name] = mdl
        all_feats[name] = feats
    logger.info(f"Stage1 total: {sum(len(v) for v in all_sel.values())} features "
                f"across {len(modality_dfs)} modalities")
    return all_sel, all_models, all_feats
