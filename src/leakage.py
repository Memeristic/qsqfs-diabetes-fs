"""
src/leakage.py
==============
Label-leakage exclusion policy for the diabetes prediction task.

The binary label is derived from ICD-coded diabetes diagnoses. Any feature that
encodes, defines, or is a direct clinical consequence of that diagnosis must be
removed before modelling, otherwise the task degenerates from prediction into
label recovery.

Three exclusion tiers are applied:

  TIER 1 -- Diagnostic criterion.
      Serum/plasma glucose and glycated haemoglobin are the analytes on which
      the diagnosis of diabetes is *defined* (ADA criteria: FPG >= 7.0 mmol/L,
      HbA1c >= 48 mmol/mol). Retaining them makes the prediction circular.
      Resolved dynamically against `d_labitems` / `d_items` by pattern, so the
      policy holds on the full database and not only on the demo subset.

  TIER 2 -- Treatment-indicating diagnosis codes.
      Codes that do not name diabetes but assert its management, principally
      ICD-9 V58.67 and ICD-10 Z79.4 ("long-term (current) use of insulin"),
      restate the label in another vocabulary.

  TIER 3 -- Consequence-of-treatment medications.
      Antidiabetic agents (glucose-lowering) and the hypoglycaemia rescue
      agents administered *because* a patient is under glycaemic management
      (glucagon, 50% dextrose, oral glucose gel). The latter are consequences
      of the label, not predictors of it.

The exclusion is enforced at feature-construction time; `assert_no_leakage()`
provides a hard post-condition that callers run against the final matrix.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Sequence, Set

import pandas as pd

try:
    from loguru import logger
except ImportError:                       # pragma: no cover
    import logging
    logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# TIER 1 -- diagnostic-criterion analytes (matched against dictionary labels)  #
# --------------------------------------------------------------------------- #
CRITERION_ANALYTE_PATTERN = re.compile(
    r"glucose"
    r"|glycated\s*h(a)?emoglobin"
    r"|glycosylated\s*h(a)?emoglobin"
    r"|h(a)?emoglobin\s*a1c"
    r"|hba1c"
    r"|\ba1c\b",
    re.IGNORECASE,
)


# --------------------------------------------------------------------------- #
# TIER 2 -- diagnosis codes                                                    #
# --------------------------------------------------------------------------- #
DIABETES_CODE_PREFIXES = ("250", "E08", "E09", "E10", "E11", "E12", "E13")

# Codes asserting glycaemic management without naming diabetes.
INSULIN_USE_CODES = {
    "V5867",   # ICD-9  Long-term (current) use of insulin
    "V586",    # ICD-9  Long-term (current) use of medicament (insulin sub-codes)
    "Z794",    # ICD-10 Long term (current) use of insulin
    "Z7984",   # ICD-10 Long term (current) use of oral hypoglycaemic drugs
}


# --------------------------------------------------------------------------- #
# TIER 3 -- medications                                                        #
# --------------------------------------------------------------------------- #
_ANTIDIABETIC_PATTERNS = [
    # Biguanides
    r"metformin", r"glucophage",
    # Insulins (all formulations / brands)
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

# Administered *because* the patient is under glycaemic management. These are
# consequences of the label; including them recovers the diagnosis rather than
# predicting it.
_HYPOGLYCAEMIA_RESCUE_PATTERNS = [
    r"glucagon",
    r"dextrose\s*50", r"\bd50\b", r"d50w", r"dextrose\s*25", r"\bd25\b",
    r"glucose\s*gel", r"oral\s*glucose", r"glucose\s*tab",
    r"glucose\s*control",          # e.g. "Boost Glucose Control" enteral feeds
]

MEDICATION_LEAK_RE = re.compile(
    "(" + "|".join(_ANTIDIABETIC_PATTERNS + _HYPOGLYCAEMIA_RESCUE_PATTERNS) + ")",
    re.IGNORECASE,
)


# --------------------------------------------------------------------------- #
# Predicates                                                                   #
# --------------------------------------------------------------------------- #
def is_diabetes_code(code: str) -> bool:
    """Diabetes-defining diagnosis code (equivalent to the label by construction)."""
    s = str(code).strip().upper().replace(".", "")
    return s.startswith(DIABETES_CODE_PREFIXES)


def is_insulin_use_code(code: str) -> bool:
    """Code asserting long-term glycaemic management (label in another vocabulary)."""
    s = str(code).strip().upper().replace(".", "")
    return s in INSULIN_USE_CODES


def is_leaky_code(code: str) -> bool:
    return is_diabetes_code(code) or is_insulin_use_code(code)


def is_leaky_medication(name: str) -> bool:
    """Antidiabetic agent, or a rescue agent given because of glycaemic management."""
    return bool(MEDICATION_LEAK_RE.search(str(name)))


def is_criterion_analyte(label: str) -> bool:
    """Glucose or HbA1c -- the analytes the diagnosis is defined on."""
    return bool(CRITERION_ANALYTE_PATTERN.search(str(label)))


# --------------------------------------------------------------------------- #
# Dictionary-driven itemid resolution                                          #
# --------------------------------------------------------------------------- #
def resolve_leaky_itemids(
    d_labitems: Optional[pd.DataFrame] = None,
    d_items: Optional[pd.DataFrame] = None,
) -> Set[int]:
    """Resolve every glucose/HbA1c itemid from the database's own dictionaries.

    Laboratory and chart features are keyed by opaque numeric identifiers, so a
    name-based filter applied to column names cannot see them. The identifiers
    are therefore resolved against `d_labitems` / `d_items` and excluded by id.
    """
    leaky: Set[int] = set()
    for table, name in ((d_labitems, "d_labitems"), (d_items, "d_items")):
        if table is None or "itemid" not in table.columns or "label" not in table.columns:
            continue
        hit = table[table["label"].astype(str).apply(is_criterion_analyte)]
        ids = {int(i) for i in hit["itemid"].tolist()}
        leaky |= ids
        if ids:
            logger.info(f"leakage: {len(ids)} criterion-analyte itemids resolved from {name}")
    return leaky


def leaky_columns(
    columns: Iterable[str],
    leaky_itemids: Optional[Set[int]] = None,
) -> List[str]:
    """Return the subset of built feature columns that violate the policy.

    Recognises the project's column conventions:
      labs_item_<itemid> / vitals_item_<itemid>   -> itemid match (Tier 1)
      meds_<drug name>                            -> medication match (Tier 3)
      dx_<icd code>                               -> code match (Tier 2)
    """
    leaky_itemids = leaky_itemids or set()
    out: List[str] = []
    for c in columns:
        s = str(c)
        m = re.match(r"^(labs|vitals)_item_(\d+)$", s)
        if m and int(m.group(2)) in leaky_itemids:
            out.append(c)
            continue
        if s.startswith("meds_") and is_leaky_medication(s[len("meds_"):]):
            out.append(c)
            continue
        if s.startswith("dx_") and is_leaky_code(s[len("dx_"):]):
            out.append(c)
            continue
        # dictionary-named columns (synthetic / wide-format sources)
        if is_criterion_analyte(s):
            out.append(c)
    return out


def assert_no_leakage(
    columns: Sequence[str],
    leaky_itemids: Optional[Set[int]] = None,
) -> None:
    """Hard post-condition: raise if any excluded feature reached the matrix.

    Called after the combined feature matrix is built. A failure here means the
    exclusion policy was bypassed somewhere upstream, and every downstream number
    would be invalid -- so this raises rather than warns.
    """
    bad = leaky_columns(columns, leaky_itemids)
    if bad:
        raise ValueError(
            "Label leakage: excluded features reached the feature matrix: "
            + ", ".join(map(str, bad[:20]))
            + (f" (+{len(bad) - 20} more)" if len(bad) > 20 else "")
        )
    logger.info(f"leakage check passed: {len(columns)} features, none excluded-listed")


def leakage_report(
    columns: Sequence[str],
    leaky_itemids: Optional[Set[int]] = None,
) -> Dict[str, List[str]]:
    """Itemised record of what the policy removed, for the methods chapter."""
    tier1, tier2, tier3 = [], [], []
    leaky_itemids = leaky_itemids or set()
    for c in columns:
        s = str(c)
        m = re.match(r"^(labs|vitals)_item_(\d+)$", s)
        if (m and int(m.group(2)) in leaky_itemids) or is_criterion_analyte(s):
            tier1.append(s)
        elif s.startswith("dx_") and is_leaky_code(s[len("dx_"):]):
            tier2.append(s)
        elif s.startswith("meds_") and is_leaky_medication(s[len("meds_"):]):
            tier3.append(s)
    return {
        "tier1_criterion_analytes": tier1,
        "tier2_treatment_codes": tier2,
        "tier3_medications": tier3,
    }
