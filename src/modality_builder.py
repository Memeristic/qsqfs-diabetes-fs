"""
src/modality_builder.py
========================
Builds per-modality feature matrices and the combined matrix.

Key behaviour:
  * Real MIMIC-IV `chartevents` and `labevents` are LONG format (itemid /
    valuenum); `build_vitals` and `build_labs` pivot them to one column per
    itemid, falling back to a wide numeric builder for the synthetic demo.
  * The diabetes-drug leakage filter is applied in both the wide (synthetic)
    and long (real) medication paths.
  * `build_modality_map` returns {modality: [global column indices]} for the
    nested-CV pipeline.
  * `build_combined_matrix` supports an inner join or an outer join with mean
    imputation, and reports the surviving sample count and prevalence.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from loguru import logger
except ImportError:                       # pragma: no cover
    import logging
    logger = logging.getLogger(__name__)

from src.leakage import (
    is_diabetes_code as _is_diabetes_defining_code,
    is_leaky_code,
    is_leaky_medication,
    resolve_leaky_itemids,
    leaky_columns,
    assert_no_leakage,
    leakage_report,
)

# --------------------------------------------------------------------------- #
# Diabetes-drug leakage filter                                                #
# --------------------------------------------------------------------------- #
# Real MIMIC-IV `pharmacy.medication` / `prescriptions.drug` values are free
# text with brand names, salts, routes and doses, e.g. "Insulin Glargine",
# "MetFORMIN (Glucophage) 500 MG Tab", "insulin lispro", "Jardiance",
# "Metformin-Sitagliptin". To reliably remove diabetes medications (which would
# otherwise leak the label) the filter uses a case-insensitive substring/regex
# match against the generic and common brand names of every diabetes-drug
# class, applied to the raw drug-name text before it ever becomes a column.
_DIABETES_DRUG_PATTERNS = [
    # Biguanides
    r"metformin", r"glucophage",
    # Insulins (all formulations/brands)
    r"insulin", r"lantus", r"humalog", r"novolog", r"novorapid", r"levemir",
    r"tresiba", r"apidra", r"toujeo", r"basaglar", r"humulin", r"fiasp",
    # Sulfonylureas
    r"glipizide", r"glucotrol", r"glyburide", r"glibenclamide", r"glimepiride",
    r"amaryl", r"gliclazide", r"tolbutamide", r"chlorpropamide",
    # Meglitinides
    r"repaglinide", r"nateglinide", r"prandin", r"starlix",
    # Thiazolidinediones
    r"pioglitazone", r"actos", r"rosiglitazone", r"avandia",
    # Alpha-glucosidase inhibitors
    r"acarbose", r"miglitol",
    # DPP-4 inhibitors
    r"sitagliptin", r"januvia", r"saxagliptin", r"onglyza",
    r"linagliptin", r"tradjenta", r"alogliptin", r"nesina",
    # GLP-1 receptor agonists
    r"liraglutide", r"victoza", r"exenatide", r"byetta", r"bydureon",
    r"dulaglutide", r"trulicity", r"semaglutide", r"ozempic", r"rybelsus",
    r"lixisenatide", r"tirzepatide", r"mounjaro",
    # SGLT2 inhibitors
    r"empagliflozin", r"jardiance", r"canagliflozin", r"invokana",
    r"dapagliflozin", r"farxiga", r"ertugliflozin", r"steglatro",
    # Amylin analogue
    r"pramlintide", r"symlin",
]
DIABETES_DRUG_RE = re.compile("(" + "|".join(_DIABETES_DRUG_PATTERNS) + ")",
                              re.IGNORECASE)


def is_diabetes_drug_name(text: str) -> bool:
    """Antidiabetic agent, or a hypoglycaemia-rescue agent administered because
    the patient is under glycaemic management. Policy lives in src.leakage."""
    return is_leaky_medication(text)


def is_diabetes_code(code: str) -> bool:
    """Diabetes-DEFINING code. Used to derive the label, so it must stay narrow:
    widening it here would change what is being predicted. Feature-side stripping
    uses the broader `is_leaky_code` (which also removes insulin-use codes)."""
    return _is_diabetes_defining_code(code)


def extract_diabetes_label(patients_df, diagnoses_df, d_icd_df) -> pd.DataFrame:
    if patients_df is None:
        raise ValueError("patients_df is required for label extraction.")
    if diagnoses_df is None:
        logger.warning("No diagnoses - random labels (demo fallback).")
        rng = np.random.default_rng(42)
        return pd.DataFrame({
            "subject_id": patients_df["subject_id"].unique(),
            "label": rng.choice([0, 1], len(patients_df), p=[0.75, 0.25]),
        })
    diab = set(diagnoses_df.loc[
        diagnoses_df["icd_code"].apply(is_diabetes_code), "subject_id"].unique())
    subs = patients_df["subject_id"].unique()
    df = pd.DataFrame({"subject_id": subs, "label": [int(s in diab) for s in subs]})
    pos = int(df["label"].sum())
    logger.info(f"Label: {pos}/{len(df)} positive ({100*pos/len(df):.1f}%)")
    return df


# --------------------------------------------------------------------------- #
# Generic builders                                                            #
# --------------------------------------------------------------------------- #
def _finalize(pivot: pd.DataFrame, label_df: pd.DataFrame, name: str,
              feature_cols: List[str], min_coverage: float = 0.05
              ) -> Tuple[Optional[pd.DataFrame], List[str]]:
    """Merge a pivoted modality with the label and finalise column names.

    Real MIMIC-IV lab/vital panels are sparse -- a given patient is rarely
    measured on all top-K items -- so requiring complete cases across every
    column would drop most or all patients. Instead of a hard `.dropna()` across
    all columns, this uses the same mean-imputation approach as the
    cross-modality outer join:
      1. Drop item columns with almost no coverage (< `min_coverage` of
         patients ever measured on them) -- these carry little signal.
      2. Keep any patient with at least one value among the surviving columns.
      3. Mean-impute the remaining per-column missingness.
    """
    merged = pivot.merge(label_df, on="subject_id", how="inner")
    if len(merged) == 0:
        logger.warning(f"Modality '{name}': no patients after label merge - skipped.")
        return None, []

    coverage = merged[feature_cols].notna().mean()
    keep_cols = coverage[coverage >= min_coverage].index.tolist()
    if len(keep_cols) < len(feature_cols):
        logger.info(f"Modality '{name}': dropped {len(feature_cols) - len(keep_cols)} "
                    f"item column(s) with <{min_coverage:.0%} patient coverage")
    feature_cols = keep_cols
    if not feature_cols:
        logger.warning(f"Modality '{name}': no columns met the coverage threshold - skipped.")
        return None, []

    merged = merged[["subject_id", "label"] + feature_cols]
    merged = merged.dropna(subset=feature_cols, how="all")   # need >=1 real value
    if len(merged) < 10:
        logger.warning(f"Modality '{name}': only {len(merged)} samples - skipped.")
        return None, []
    n_imputed = int(merged[feature_cols].isna().sum().sum())
    if n_imputed:
        merged[feature_cols] = merged[feature_cols].apply(lambda s: s.fillna(s.mean()))
        logger.info(f"Modality '{name}': mean-imputed {n_imputed} sparse cell(s) "
                    f"across {len(feature_cols)} columns")

    rename = {c: f"{name}_{c}" for c in feature_cols}
    merged = merged.rename(columns=rename)
    names = [f"{name}_{c}" for c in feature_cols]
    logger.info(f"Built '{name}': {len(merged)} samples x {len(names)} features")
    return merged, names


def build_numeric_modality(df, label_df, name, top_k=None):
    if df is None or len(df) == 0:
        return None, []
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    cols = [c for c in num if c != "subject_id"]
    if top_k:
        cols = cols[:top_k]
    if not cols:
        return None, []
    pivot = df[["subject_id"] + cols].groupby("subject_id").mean().reset_index()
    return _finalize(pivot, label_df, name, cols)


def _drop_leaky_items(df, leaky_itemids, name: str):
    """Remove criterion-analyte rows (glucose / HbA1c) BEFORE top-k selection.

    Excluded items are removed before ranking so that all K slots are filled by
    admissible items and the modality retains its intended width.
    """
    if df is None or not leaky_itemids or "itemid" not in df.columns:
        return df
    mask = df["itemid"].isin(leaky_itemids)
    n = int(mask.sum())
    if n:
        logger.info(f"{name}: dropped {n:,} rows for {df.loc[mask, 'itemid'].nunique()} "
                    f"criterion-analyte itemids (glucose/HbA1c) before top-k selection")
    return df.loc[~mask]


def _pivot_long(df, value_col, item_col, top_k, prefix):
    top = df[item_col].value_counts().head(top_k).index
    filt = df[df[item_col].isin(top)].copy()
    pivot = (filt.pivot_table(index="subject_id", columns=item_col,
                              values=value_col, aggfunc="mean").reset_index())
    pivot.columns = ["subject_id"] + [f"{prefix}{c}" for c in pivot.columns[1:]]
    return pivot


def build_labs(labevents, label_df, top_k=50, leaky_itemids=None):
    if labevents is None:
        return None, []
    if "itemid" not in labevents.columns:                       # wide / synthetic
        return build_numeric_modality(labevents, label_df, "labs", top_k)
    logger.info("Pivoting real labevents (long format)...")
    labevents = _drop_leaky_items(labevents, leaky_itemids, "labs")
    pivot = _pivot_long(labevents, "valuenum", "itemid", top_k, "item_")
    cols = [c for c in pivot.columns if c != "subject_id"]
    return _finalize(pivot, label_df, "labs", cols)


def build_vitals(chartevents, label_df, top_k=30, leaky_itemids=None):
    """Real chartevents is long format -> pivot by itemid (not a mean over itemid)."""
    if chartevents is None:
        return None, []
    if "itemid" not in chartevents.columns:                     # wide / synthetic
        return build_numeric_modality(chartevents, label_df, "vitals", top_k)
    logger.info("Pivoting real chartevents (long format)...")
    chartevents = _drop_leaky_items(chartevents, leaky_itemids, "vitals")
    pivot = _pivot_long(chartevents, "valuenum", "itemid", top_k, "item_")
    cols = [c for c in pivot.columns if c != "subject_id"]
    return _finalize(pivot, label_df, "vitals", cols)


def _strip_diabetes_drugs(feature_cols: List[str], merged: pd.DataFrame, name: str):
    leak = [c for c in feature_cols if is_diabetes_drug_name(c.replace(f"{name}_", "", 1))]
    if leak:
        merged = merged.drop(columns=leak, errors="ignore")
        logger.info(f"Modality '{name}': removed {len(leak)} diabetes-drug leakage cols "
                    f"({', '.join(leak[:5])}{'...' if len(leak) > 5 else ''})")
    return merged, [c for c in feature_cols if c not in leak]


def build_medications(pharmacy, label_df, top_k=50):
    if pharmacy is None:
        return None, []
    # Detect long format directly by the presence of a known drug-name column.
    # (Counting object columns is ambiguous: a real long-format `pharmacy` table
    # also has exactly one object column -- the drug-name text -- and would be
    # mis-routed into the wide/synthetic numeric branch.)
    med_col = next((c for c in ["medication", "drug", "ndc"] if c in pharmacy.columns), None)
    if med_col is None:
        # wide / synthetic binary columns
        merged, names = build_numeric_modality(pharmacy, label_df, "meds", top_k)
        if merged is None:
            return None, []
        merged, names = _strip_diabetes_drugs(names, merged, "meds")   # filter here too
        return merged, names
    # long / real
    # Drop diabetes-drug rows BEFORE top-k selection, using the substring matcher
    # on the raw free-text name (handles brand names, salts, dose strings and
    # combination products that an exact match on the column suffix would miss).
    is_leak = pharmacy[med_col].astype(str).apply(is_diabetes_drug_name)
    n_leak_rows = int(is_leak.sum())
    clean = pharmacy.loc[~is_leak]
    if n_leak_rows:
        logger.info(f"meds: dropped {n_leak_rows:,} rows naming a diabetes drug "
                    f"before top-k selection (real-data leakage guard)")
    top = clean[med_col].value_counts().head(top_k).index
    filt = clean[clean[med_col].isin(top)].copy()
    filt["present"] = 1
    pivot = (filt.pivot_table(index="subject_id", columns=med_col, values="present",
                              aggfunc="max", fill_value=0).reset_index())
    raw_cols = list(pivot.columns[1:])
    pivot.columns = ["subject_id"] + [f"meds_{str(c).replace(' ', '_')}" for c in raw_cols]
    merged = pivot.merge(label_df, on="subject_id", how="inner")
    feat = [c for c in merged.columns if c not in ("subject_id", "label")]
    # Also filter the sanitised column names, to catch any residual matches
    # introduced by string normalisation.
    merged, feat = _strip_diabetes_drugs(feat, merged, "meds")
    if len(merged) < 10 or not feat:
        return None, []
    logger.info(f"Built 'meds': {len(merged)} samples x {len(feat)} features")
    return merged, feat


def build_diagnoses(diagnoses_icd, label_df, top_k=30):
    if diagnoses_icd is None:
        return None, []
    if "icd_code" not in diagnoses_icd.columns:
        return build_numeric_modality(diagnoses_icd, label_df, "dx", top_k)
    top = diagnoses_icd["icd_code"].value_counts().head(top_k).index
    filt = diagnoses_icd[diagnoses_icd["icd_code"].isin(top)].copy()
    filt["present"] = 1
    pivot = (filt.pivot_table(index="subject_id", columns="icd_code", values="present",
                              aggfunc="max", fill_value=0).reset_index())
    pivot.columns = ["subject_id"] + [f"dx_{c}" for c in pivot.columns[1:]]
    merged = pivot.merge(label_df, on="subject_id", how="inner")
    feat = [c for c in merged.columns if c not in ("subject_id", "label")]
    # strip diabetes-defining codes (label leakage)
    leak = [c for c in feat if is_leaky_code(c.replace("dx_", ""))]
    merged = merged.drop(columns=leak, errors="ignore")
    feat = [c for c in feat if c not in leak]
    if len(merged) < 10 or not feat:
        return None, []
    logger.info(f"Built 'dx': {len(merged)} samples x {len(feat)} features "
                f"(removed {len(leak)} diabetes-code leakage cols)")
    return merged, feat


# --------------------------------------------------------------------------- #
# Combine                                                                     #
# --------------------------------------------------------------------------- #
def build_combined_matrix(modality_dfs: Dict[str, pd.DataFrame], how: str = "inner"):
    """
    Join all modalities on subject_id. `how='outer'` keeps every patient and
    mean-imputes missing modality features (recommended for real MIMIC-IV where
    most patients lack ICU vitals); `how='inner'` keeps only complete cases.
    """
    combined: Optional[pd.DataFrame] = None
    for name, df in modality_dfs.items():
        feat = [c for c in df.columns if c not in ("subject_id", "label")]
        sub = df[["subject_id"] + feat]
        if combined is None:
            combined = df[["subject_id", "label"]].copy()
        combined = combined.merge(sub, on="subject_id", how=how)
    if combined is None or len(combined) == 0:
        return None, []
    feat_cols = [c for c in combined.columns if c not in ("subject_id", "label")]
    if how == "outer":
        combined[feat_cols] = combined[feat_cols].apply(lambda s: s.fillna(s.mean()))
        combined = combined.dropna(subset=["label"])
    else:
        combined = combined.dropna()
    pos = int(combined["label"].sum())
    logger.info(f"Combined ({how}): {len(combined)} samples x {len(feat_cols)} features, "
                f"{100*pos/max(1,len(combined)):.1f}% positive")
    return combined, feat_cols


def build_modality_map(modality_dfs: Dict[str, pd.DataFrame],
                       global_names: List[str]) -> Dict[str, List[int]]:
    """Map each modality to the global column indices it contributed."""
    name_to_idx = {n: i for i, n in enumerate(global_names)}
    out: Dict[str, List[int]] = {}
    for mod, df in modality_dfs.items():
        feats = [c for c in df.columns if c not in ("subject_id", "label")]
        out[mod] = [name_to_idx[f] for f in feats if f in name_to_idx]
    return out
