"""
src/datasets.py
===============
One entry point that turns any supported source into the tuple the rest of the
pipeline consumes::

    X, y, feature_names, modality_map, meta = load_dataset(source, **opts)

Supported `source` values:
  * "mimic"     : a real MIMIC-IV directory (opts: data_root). Uses the full
                  labs/vitals/meds/dx modality builder.
  * "synthetic" : the built-in overlapping-distribution demo cohort.
  * "csv"       : any tidy one-row-per-patient CSV (opts: path, label_col,
                  id_col). Modalities are inferred from column-name prefixes so
                  multimodal fusion still has something to fuse; if nothing can
                  be grouped, a single "clinical" modality is used.
  * a *.csv path directly is treated as source="csv".

`modality_map` always partitions the columns, so downstream fusion and the
two-stage selector work on every dataset, not just MIMIC. This is what makes the
pipeline dataset-agnostic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from loguru import logger
except ImportError:                       # pragma: no cover
    import logging
    logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Column-name -> modality heuristics (for generic tabular data)               #
# --------------------------------------------------------------------------- #
_MODALITY_KEYWORDS = {
    "labs": ["glucose", "insulin", "hba1c", "cholesterol", "chol", "hdl", "ldl",
             "triglyceride", "creatinine", "sodium", "potassium", "chloride",
             "bun", "wbc", "hemoglobin", "platelet", "albumin", "bilirubin",
             "alt", "ast", "crp", "lab", "serum", "blood", "pedigree"],
    "vitals": ["heart", "pulse", "bp", "sbp", "dbp", "pressure", "resp",
               "temperature", "temp", "spo2", "oxygen", "vital", "bmi",
               "weight", "height", "skin", "thickness"],
    "demographics": ["age", "gender", "sex", "race", "ethnic", "pregnan"],
    "meds": ["med", "drug", "insulin_dose", "metformin", "prescri"],
    "dx": ["dx", "icd", "diagnos", "comorbid", "history"],
}


def _infer_modalities_from_names(names: List[str]) -> Dict[str, List[int]]:
    """Group feature columns into modalities by keyword matching on their names.

    Any column that matches no keyword goes into "clinical". Empty modalities
    are dropped. Guarantees a partition of range(len(names)).
    """
    assigned = {}
    for idx, raw in enumerate(names):
        low = str(raw).lower()
        placed = None
        for mod, keys in _MODALITY_KEYWORDS.items():
            if any(k in low for k in keys):
                placed = mod
                break
        assigned[idx] = placed or "clinical"
    mm: Dict[str, List[int]] = {}
    for idx, mod in assigned.items():
        mm.setdefault(mod, []).append(idx)
    # collapse singletons/tiny groups into "clinical" so encoders have >=2 cols
    small = [m for m, cols in mm.items() if len(cols) < 2 and m != "clinical"]
    for m in small:
        mm.setdefault("clinical", []).extend(mm.pop(m))
    if not mm:
        mm = {"clinical": list(range(len(names)))}
    return {k: sorted(v) for k, v in mm.items() if v}


# --------------------------------------------------------------------------- #
# Loaders                                                                      #
# --------------------------------------------------------------------------- #
def _load_mimic(config: dict, data_root: Optional[str]) -> Tuple:
    from src.data_loader import MIMICDataLoader
    from src.modality_builder import (
        extract_diabetes_label, build_labs, build_vitals, build_medications,
        build_diagnoses, build_combined_matrix, build_modality_map)
    from src.leakage import resolve_leaky_itemids, assert_no_leakage

    cfg = dict(config)
    cfg["demo_mode"] = False
    if data_root:
        cfg.setdefault("paths", {})["data_root"] = data_root
    cfg.setdefault("preprocessing", {})["combine_how"] = \
        cfg.get("preprocessing", {}).get("combine_how", "outer")

    data = MIMICDataLoader(cfg).load_all()
    label_df = extract_diabetes_label(
        data.get("patients"), data.get("diagnoses_icd"), data.get("d_icd_diagnoses"))

    # Criterion analytes (glucose / HbA1c) are keyed by opaque itemids, so they
    # are resolved against the database's own dictionaries and excluded by id.
    leaky_ids = resolve_leaky_itemids(data.get("d_labitems"), data.get("d_items"))

    mod = {}
    for key, fn, arg in [("labs", build_labs, data.get("labevents")),
                         ("vitals", build_vitals, data.get("chartevents")),
                         ("meds", build_medications, data.get("pharmacy")),
                         ("dx", build_diagnoses, data.get("diagnoses_icd"))]:
        df, _ = (fn(arg, label_df, leaky_itemids=leaky_ids)
                 if key in ("labs", "vitals") else fn(arg, label_df))
        if df is not None:
            mod[key] = df
    how = cfg["preprocessing"]["combine_how"]
    combined, names = build_combined_matrix(mod, how=how)
    mmap = build_modality_map(mod, names)
    X = combined[names].values.astype(float)
    y = combined["label"].values.astype(int)
    assert_no_leakage(names, leaky_ids)
    meta = {"source": "mimic", "modalities": {k: len(v) for k, v in mmap.items()},
            "leakage_policy": "tier1_criterion_analytes+tier2_treatment_codes+tier3_medications",
            "n_criterion_itemids_excluded": len(leaky_ids)}
    return X, y, names, mmap, meta


def _load_synthetic(config: dict) -> Tuple:
    from src.data_loader import MIMICDataLoader
    from src.modality_builder import (
        extract_diabetes_label, build_labs, build_vitals, build_medications,
        build_diagnoses, build_combined_matrix, build_modality_map)
    from src.leakage import assert_no_leakage, leaky_columns
    cfg = dict(config)
    cfg["demo_mode"] = True
    data = MIMICDataLoader(cfg).load_all()
    label_df = extract_diabetes_label(
        data.get("patients"), data.get("diagnoses_icd"), data.get("d_icd_diagnoses"))
    mod = {}
    for key, fn, arg in [("labs", build_labs, data.get("labevents")),
                         ("vitals", build_vitals, data.get("chartevents")),
                         ("meds", build_medications, data.get("pharmacy")),
                         ("dx", build_diagnoses, data.get("diagnoses_icd"))]:
        df, _ = fn(arg, label_df)
        if df is not None:
            mod[key] = df
    combined, names = build_combined_matrix(mod, how="inner")

    # The simulator emits named analytes directly, so the exclusion policy is
    # applied to the column names. Criterion analytes (glucose, HbA1c) are
    # withheld here exactly as they are on the clinical path; the retained
    # signal comes from admissible risk factors (adiposity, blood pressure,
    # lipids, inflammatory markers), keeping the fixture consistent with the
    # protocol under which real results are reported.
    excluded = set(leaky_columns(names))
    names = [n for n in names if n not in excluded]
    if excluded:
        logger.info(f"synthetic: withheld {len(excluded)} criterion-analyte "
                    f"column(s) under the exclusion policy: {sorted(excluded)}")

    mmap = build_modality_map(mod, names)
    X = combined[names].values.astype(float)
    y = combined["label"].values.astype(int)
    assert_no_leakage(names)
    meta = {"source": "synthetic", "modalities": {k: len(v) for k, v in mmap.items()},
            "n_criterion_columns_excluded": len(excluded)}
    return X, y, names, mmap, meta


def _load_csv(path: str, label_col: Optional[str] = None,
              id_col: Optional[str] = None) -> Tuple:
    from src.schema import infer_column_kinds, build_matrix_from_mapping
    df = pd.read_csv(path, low_memory=False)
    kinds = infer_column_kinds(df)
    label = label_col or kinds["label_col"]
    if label is None:
        raise ValueError(
            f"Could not find a label column in {Path(path).name}. Pass "
            f"label_col explicitly. Columns: {list(df.columns)}")
    idc = id_col if id_col is not None else kinds["id_col"]
    numeric = [c for c in kinds["numeric"] if c not in (label, idc)]
    categorical = [c for c in kinds["categorical"] if c not in (label, idc)]
    X, y, names, _ = build_matrix_from_mapping(
        df, label_col=label, numeric_cols=numeric,
        categorical_cols=categorical, id_col=idc)
    mmap = _infer_modalities_from_names(names)
    meta = {"source": "csv", "path": str(path), "label_col": label,
            "modalities": {k: len(v) for k, v in mmap.items()}}
    return X, y, names, mmap, meta


# --------------------------------------------------------------------------- #
# Public entry point                                                          #
# --------------------------------------------------------------------------- #
def load_dataset(source: str, config: Optional[dict] = None, **opts) -> Tuple[
        np.ndarray, np.ndarray, List[str], Dict[str, List[int]], dict]:
    """Load any supported source into (X, y, names, modality_map, meta).

    Examples
    --------
    load_dataset("mimic", config, data_root="/path/to/mimic-iv")
    load_dataset("synthetic", config)
    load_dataset("csv", path="diabetes.csv", label_col="Outcome")
    load_dataset("/data/pima.csv")                      # path == csv shortcut
    """
    config = config or {}
    src = str(source).strip()

    if src.lower().endswith(".csv") or (Path(src).exists() and Path(src).is_file()):
        X, y, names, mmap, meta = _load_csv(
            src, label_col=opts.get("label_col"), id_col=opts.get("id_col"))
    elif src.lower() == "mimic":
        X, y, names, mmap, meta = _load_mimic(config, opts.get("data_root"))
    elif src.lower() in ("synthetic", "demo"):
        X, y, names, mmap, meta = _load_synthetic(config)
    elif src.lower() == "csv":
        X, y, names, mmap, meta = _load_csv(
            opts["path"], label_col=opts.get("label_col"), id_col=opts.get("id_col"))
    else:
        raise ValueError(f"Unknown source '{source}'. Use 'mimic', 'synthetic', "
                         f"'csv', or a path to a .csv file.")

    # universal sanity checks (fail early, clear message)
    if X.shape[0] < 2 * len(np.unique(y)):
        logger.warning(f"Very small cohort: {X.shape[0]} rows.")
    if len(np.unique(y)) < 2:
        raise ValueError("The label has only one class; both positive and "
                         "negative cases are required.")
    meta.update({"n_patients": int(X.shape[0]), "n_features": int(X.shape[1]),
                 "prevalence_pct": round(100 * float(np.mean(y)), 1),
                 "n_positive": int(np.sum(y)), "n_negative": int(np.sum(1 - y))})
    logger.info(f"Loaded {meta['source']}: {meta['n_patients']} patients x "
                f"{meta['n_features']} features, prevalence {meta['prevalence_pct']}%")
    return X, y, names, mmap, meta
