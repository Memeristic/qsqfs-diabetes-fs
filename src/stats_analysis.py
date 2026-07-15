"""
src/stats_analysis.py
=====================
Post-selection analysis.

Contents:
  * `_permutation_importance` permutes each feature against the true labels and
    measures the resulting drop in performance.
  * The SHAP path is optional and used only if the package is present and a
    model object is supplied; otherwise permutation importance is used.
  * Clinical alignment is reported as `biomarker_overlap` -- the fraction of
    selected features that are known biomarkers.
  * An optional lab item-id -> label mapping via d_labitems lets alignment work
    on real MIMIC-IV, where labs are coded numerically.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    from loguru import logger
except ImportError:                       # pragma: no cover
    import logging
    logger = logging.getLogger(__name__)

KNOWN_DIABETES_FEATURES: Dict[str, str] = {
    "glucose": "Fasting plasma glucose (ADA >=126 mg/dL)",
    "hba1c": "Glycated haemoglobin HbA1c (ADA >=6.5%)",
    "triglycerides": "Hypertriglyceridaemia - metabolic syndrome",
    "crp": "C-reactive protein - chronic inflammation in T2DM",
    "ldl": "LDL cholesterol - dyslipidaemia",
    "hdl": "Low HDL cholesterol - metabolic syndrome",
    "creatinine": "Renal impairment - diabetic nephropathy",
    "sbp": "Systolic blood pressure - hypertension comorbidity",
    "dbp": "Diastolic blood pressure",
}

# Keyword variants used to recognise each canonical marker both in short
# synthetic-style names ("hba1c") and in the free-text item descriptions found
# in real MIMIC-IV's d_labitems/d_items ("Hemoglobin A1c", "% Hemoglobin A1c").
_CANON_KEYWORDS: Dict[str, List[str]] = {
    "glucose": ["glucose"],
    "hba1c": ["hemoglobin a1c", "haemoglobin a1c", "hba1c", "a1c", "glycated hemoglobin"],
    "triglycerides": ["triglyceride"],
    "crp": ["c-reactive protein", "c reactive protein", "crp"],
    "ldl": ["ldl cholesterol", "ldl,", "low-density lipoprotein", "ldl"],
    "hdl": ["hdl cholesterol", "high-density lipoprotein", "hdl"],
    "creatinine": ["creatinine"],
    "sbp": ["systolic"],
    "dbp": ["diastolic"],
}


def build_item_label_map(feature_names: List[str],
                         d_labitems: Optional[pd.DataFrame] = None,
                         d_items: Optional[pd.DataFrame] = None) -> Dict[str, str]:
    """On real MIMIC-IV, labs/vitals are coded as `labs_item_50931` /
    `vitals_item_220045` after the long-format pivot, so matching the bare
    feature name against KNOWN_DIABETES_FEATURES fails (name.split('_')[-1]
    yields a numeric item-id, not a clinical term).

    This resolves each `..._item_<itemid>` feature to its human-readable
    description via MIMIC-IV's `d_labitems` (labs) / `d_items` (chartevents)
    reference tables, so `check_clinical_alignment` can match on real content
    instead of an opaque code.
    """
    def _lookup_table(df: Optional[pd.DataFrame]) -> Dict[str, str]:
        if df is None or "itemid" not in df.columns:
            return {}
        label_col = next((c for c in ("label", "long_title", "abbreviation")
                          if c in df.columns), None)
        if label_col is None:
            return {}
        return dict(zip(df["itemid"].astype(str), df[label_col].astype(str)))

    lab_lookup = _lookup_table(d_labitems)
    item_lookup = _lookup_table(d_items)
    out: Dict[str, str] = {}
    for name in feature_names:
        if "_item_" not in name:
            continue
        itemid = name.rsplit("_item_", 1)[-1]
        desc = lab_lookup.get(itemid) or item_lookup.get(itemid)
        if desc:
            out[name] = desc
    return out


def check_clinical_alignment(selected_names: List[str],
                             all_feature_names: List[str],
                             item_label_map: Optional[Dict[str, str]] = None) -> dict:
    """In addition to matching short synthetic-style names directly, this also
    matches real MIMIC-IV `..._item_<id>` features via their resolved
    description text (see `build_item_label_map`), using keyword substrings
    rather than exact equality."""
    item_label_map = item_label_map or {}
    matched, unmatched = [], []
    for name in selected_names:
        stripped = name.split("_", 1)[-1].lower() if "_" in name else name.lower()
        desc_text = item_label_map.get(name, "").lower()
        canon = None
        for c, kws in _CANON_KEYWORDS.items():
            if stripped == c or any(kw in stripped for kw in kws):
                canon = c
                break
        if canon is None and desc_text:
            for c, kws in _CANON_KEYWORDS.items():
                if any(kw in desc_text for kw in kws):
                    canon = c
                    break
        if canon is not None:
            matched.append({
                "feature": name, "canonical": canon,
                "clinical_note": KNOWN_DIABETES_FEATURES.get(canon, canon),
                "resolved_via": "item_label_map" if desc_text and canon not in stripped else "name",
            })
        else:
            unmatched.append(name)
    total = len(selected_names)
    return {
        "matched": matched, "unmatched": unmatched,
        "n_matched": len(matched), "n_unmatched": len(unmatched),
        "biomarker_overlap": len(matched) / total if total else 0.0,
        "total_selected": total,
    }


def compute_feature_importance(model, X_selected: np.ndarray, y: np.ndarray,
                               feature_names: List[str],
                               max_samples: int = 300) -> Optional[pd.DataFrame]:
    """SHAP if available + model supports it; else working permutation importance."""
    if model is not None and SHAP_AVAILABLE:
        try:
            sample = X_selected[:max_samples]
            explainer = shap.KernelExplainer(model.predict_proba, shap.kmeans(sample, 10))
            sv = explainer.shap_values(sample, silent=True)
            if isinstance(sv, list):
                sv = sv[1]
            imp = np.abs(sv).mean(axis=0)
            return (pd.DataFrame({"feature": feature_names[:len(imp)],
                                  "importance": imp})
                    .sort_values("importance", ascending=False))
        except Exception as exc:
            logger.warning(f"SHAP failed ({exc}); using permutation importance.")
    return _permutation_importance(model, X_selected, y, feature_names)


def _permutation_importance(model, X, y, feature_names):
    """Drop in accuracy when each feature is shuffled (vs the TRUE labels)."""
    from sklearn.metrics import accuracy_score
    if model is None:
        return pd.DataFrame({"feature": feature_names, "importance": [0.0] * len(feature_names)})
    base = accuracy_score(y, (model.predict_proba(X)[:, 1] >= 0.5).astype(int))
    rng = np.random.default_rng(42)
    imps = []
    for j in range(X.shape[1]):
        Xp = X.copy()
        Xp[:, j] = rng.permutation(Xp[:, j])
        acc = accuracy_score(y, (model.predict_proba(Xp)[:, 1] >= 0.5).astype(int))
        imps.append(base - acc)                      # positive => feature matters
    return (pd.DataFrame({"feature": feature_names[:len(imps)], "importance": imps})
            .sort_values("importance", ascending=False))


def generate_latex_results_table(comparative_results: dict,
                                 caption="Comparative analysis (nested CV, mean over folds)",
                                 label="tab:comparative") -> str:
    rows = []
    for name, res in comparative_results.items():
        if name.startswith("_"):
            continue
        rows.append({
            "Algorithm": name,
            "AUC": f"{np.nanmean(res.get('auc', [np.nan])):.4f}",
            "Acc": f"{np.mean(res['accuracy']):.4f}",
            "F1": f"{np.mean(res.get('f1', [np.nan])):.4f}",
            "Features": f"{np.mean(res['n_features']):.1f}",
        })
    rows.sort(key=lambda r: float(r["AUC"]) if r["AUC"] != "nan" else -1, reverse=True)
    cols = ["Algorithm", "AUC", "Acc", "F1", "Features"]
    lines = [r"\begin{table}[ht]", r"\centering", rf"\caption{{{caption}}}",
             rf"\label{{{label}}}", r"\begin{tabular}{lrrrr}", r"\hline",
             " & ".join(cols) + r" \\", r"\hline"]
    for r in rows:
        bold = r["Algorithm"] == "QSQ-FS"
        cells = [(r"\textbf{" + str(r[c]) + "}") if bold else str(r[c]) for c in cols]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def generate_ablation_latex_table(ablation_results: dict, stats_rows: List[dict],
                                  label="tab:ablation") -> str:
    lines = [r"\begin{table}[ht]", r"\centering",
             r"\caption{Ablation study (nested CV): mechanism contribution}",
             rf"\label{{{label}}}", r"\begin{tabular}{lrrrr}", r"\hline",
             r"Variant & Acc & Std & Features & $\Delta$Acc (\%) \\", r"\hline"]
    full = np.mean(ablation_results.get("Full QSQ-FS", {}).get("accuracy", [0]))
    for name, res in ablation_results.items():
        if name.startswith("_"):
            continue
        acc = res["accuracy"]
        delta = (np.mean(acc) - full) * 100
        sign = "-" if delta < 0 else "+"
        lines.append(f"{name} & {np.mean(acc):.4f} & {np.std(acc):.4f} & "
                     f"{np.mean(res['n_features']):.1f} & {sign}{abs(delta):.2f} \\\\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)
