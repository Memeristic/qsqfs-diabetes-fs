"""
src/schema.py — dataset validation & generic column mapping
===========================================================
Two jobs, both aimed at making the app work on data that isn't the built-in
MIMIC-IV demo:

1. ``validate_mimic_folder`` — before the (slow) real-MIMIC pipeline runs,
   check the folder actually looks like a MIMIC-IV export and return a clear,
   specific list of what's missing, instead of letting a cryptic exception
   surface deep in the loader.

2. A column-mapping layer (``infer_column_kinds`` +
   ``build_matrix_from_mapping``) so a user can upload any one-row-per-patient
   CSV — a different EHR export, a Kaggle dataset, and so on — tell the app
   which column is the patient id, which is the label, and which columns are
   categorical vs numeric, and get a usable feature matrix without any
   hardcoded MIMIC table/column names. The MIMIC path remains a first-class
   preset; this is additive.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# 1. MIMIC-IV folder validation                                               #
# --------------------------------------------------------------------------- #
# Files the pipeline actually reads, and the columns each must expose. Only the
# columns the loader/label-builder depend on are listed (not every column).
_REQUIRED = {
    "hosp/diagnoses_icd": ["subject_id", "icd_code"],
    "hosp/labevents": ["subject_id", "itemid", "valuenum"],
}
_RECOMMENDED = {
    "hosp/patients": ["subject_id"],
    "hosp/d_labitems": ["itemid"],
    "hosp/pharmacy": ["subject_id"],
    "icu/chartevents": ["subject_id", "itemid", "valuenum"],
    "icu/d_items": ["itemid"],
}


def _resolve(root: Path, rel: str):
    """Return the first existing path for `rel`, trying .csv then .csv.gz."""
    for cand in (root / f"{rel}.csv", root / f"{rel}.csv.gz"):
        if cand.exists():
            return cand
    return None


def _header_columns(path: Path) -> List[str]:
    """Read only the header row (handles .gz), return column names lowercased."""
    comp = "gzip" if path.suffix == ".gz" else None
    head = pd.read_csv(path, nrows=0, compression=comp)
    return [c.strip().lower() for c in head.columns]


def validate_mimic_folder(data_root: str | Path) -> Dict[str, object]:
    """Check a folder looks like a MIMIC-IV export.

    Returns a dict: ``ok`` (bool — required pieces all present), ``errors``
    (blocking problems, list[str]), ``warnings`` (non-blocking, list[str]),
    ``found`` (list[str] of tables located).
    """
    root = Path(data_root)
    errors: List[str] = []
    warnings: List[str] = []
    found: List[str] = []

    if not root.exists():
        return {"ok": False, "found": [],
                "errors": [f'Folder "{root}" does not exist.'], "warnings": []}
    if not (root / "hosp").is_dir():
        errors.append('No "hosp/" subfolder found. MIMIC-IV must be laid out as '
                      'data_root/hosp/... and data_root/icu/...')
    if not (root / "icu").is_dir():
        warnings.append('No "icu/" subfolder — the vitals (chartevents) modality '
                        'will be skipped. Labs / meds / diagnoses can still run.')

    def _check(rel: str, cols: Sequence[str], required: bool):
        path = _resolve(root, rel)
        if path is None:
            msg = f'Missing file: {rel}.csv (or .csv.gz).'
            (errors if required else warnings).append(msg)
            return
        found.append(rel)
        try:
            have = set(_header_columns(path))
            missing = [c for c in cols if c not in have]
            if missing:
                msg = (f'{path.name}: missing column(s) {missing}. '
                       f'Found columns: {sorted(have)[:8]}...')
                (errors if required else warnings).append(msg)
        except Exception as exc:  # unreadable / not a CSV
            (errors if required else warnings).append(
                f'Could not read {path.name}: {exc}')

    for rel, cols in _REQUIRED.items():
        _check(rel, cols, required=True)
    for rel, cols in _RECOMMENDED.items():
        _check(rel, cols, required=False)

    # Common case: the user points at the folder but the files are still gzip-compressed.
    if not errors:
        gz_only = all(_resolve(root, r) is not None
                      and _resolve(root, r).suffix == ".gz"
                      for r in _REQUIRED)
        if gz_only:
            warnings.append("Files are gzip-compressed (.csv.gz). The Python loader "
                            "reads .gz directly, but the MATLAB port needs them "
                            "decompressed first (gunzip *.csv.gz).")

    return {"ok": len(errors) == 0, "errors": errors,
            "warnings": warnings, "found": found}


# Which subfolder each recognised MIMIC-IV table belongs in, keyed by the
# table's own filename stem (lowercased, no extension) so uploads are matched
# regardless of what folder they came from on the user's machine.
_TABLE_LOCATION = {
    "diagnoses_icd": "hosp", "labevents": "hosp", "patients": "hosp",
    "d_labitems": "hosp", "pharmacy": "hosp",
    "chartevents": "icu", "d_items": "icu",
}


def save_uploaded_mimic_files(uploaded_files) -> Tuple[Path, List[str], List[str]]:
    """Write Streamlit ``UploadedFile`` objects into a temp hosp/icu layout.

    Matches each upload to a known MIMIC-IV table by filename (ignoring
    ``.csv``/``.csv.gz`` and any path the browser included), so files can be
    selected in any order or folder. Returns ``(root, recognised, unrecognised)``
    — ``root`` is ready to pass straight into ``validate_mimic_folder`` /
    the loader, exactly like a real on-disk MIMIC-IV export.
    """
    root = Path(tempfile.mkdtemp(prefix="qsqfs_mimic_upload_"))
    (root / "hosp").mkdir(parents=True, exist_ok=True)
    (root / "icu").mkdir(parents=True, exist_ok=True)

    recognised: List[str] = []
    unrecognised: List[str] = []

    for uf in uploaded_files:
        name = Path(uf.name).name  # strip any client-side path
        stem = name.lower()
        for suffix in (".csv.gz", ".csv"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        subfolder = _TABLE_LOCATION.get(stem)
        if subfolder is None:
            unrecognised.append(name)
            continue
        ext = ".csv.gz" if name.lower().endswith(".gz") else ".csv"
        dest = root / subfolder / f"{stem}{ext}"
        with open(dest, "wb") as f:
            shutil.copyfileobj(uf, f)
        recognised.append(f"{subfolder}/{stem}{ext}")

    return root, recognised, unrecognised


# --------------------------------------------------------------------------- #
# 2. Generic CSV column-mapping                                               #
# --------------------------------------------------------------------------- #
def infer_column_kinds(df: pd.DataFrame) -> Dict[str, object]:
    """Best-effort guesses to pre-fill the mapping widgets.

    Returns suggested ``id_col``, ``label_col``, ``numeric`` list and
    ``categorical`` list. Heuristics only — the user can override everything.
    """
    cols = list(df.columns)
    lower = {c: str(c).strip().lower() for c in cols}

    # label: a column literally named label/target/outcome/y/diabetes, ideally
    # binary; otherwise the last binary column; otherwise None.
    label_names = {"label", "target", "outcome", "class", "y", "diabetes", "dm"}
    label_col = next((c for c in cols if lower[c] in label_names), None)
    if label_col is None:
        for c in cols:
            vals = pd.unique(df[c].dropna())
            if len(vals) == 2:
                label_col = c
                break

    # id: a column named *id / subject_id / patient*, or one that's unique per
    # row AND integer/string-like. A continuous float feature (e.g. Glucose with
    # all-distinct values) must NOT be mistaken for an id, or a real predictor
    # would be silently dropped.
    id_col = next((c for c in cols
                   if lower[c] in {"subject_id", "patient_id", "id", "patientid", "hadm_id"}
                   or lower[c].endswith("_id")), None)
    if id_col is None:
        for c in cols:
            if c == label_col or not df[c].is_unique:
                continue
            s = df[c]
            is_floaty = pd.api.types.is_float_dtype(s) or (
                pd.to_numeric(s, errors="coerce").notna().all()
                and not (pd.to_numeric(s, errors="coerce") % 1 == 0).all())
            if not is_floaty:                 # integer- or string-typed unique col
                id_col = c
                break

    numeric, categorical = [], []
    for c in cols:
        if c in (id_col, label_col):
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        # mostly-numeric with a decent number of distinct values -> numeric
        if s.notna().mean() >= 0.8 and df[c].nunique(dropna=True) > 10:
            numeric.append(c)
        elif s.notna().mean() >= 0.8 and pd.api.types.is_numeric_dtype(df[c]):
            numeric.append(c)
        else:
            categorical.append(c)
    return {"id_col": id_col, "label_col": label_col,
            "numeric": numeric, "categorical": categorical}


def build_matrix_from_mapping(df: pd.DataFrame, label_col: str,
                              numeric_cols: Sequence[str],
                              categorical_cols: Sequence[str],
                              id_col: str | None = None,
                              max_categorical_levels: int = 20
                              ) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, List[int]]]:
    """Turn an arbitrary tidy CSV into (X, y, names, modality_map).

    - numeric columns are coerced to float and mean-imputed
    - categorical columns are one-hot encoded (rare levels beyond
      ``max_categorical_levels`` are grouped into "__other__")
    - the label is coerced to 0/1 (the *larger* string/þvalue → 1 if non-numeric)

    Raises ``ValueError`` with a plain-language message on the common mistakes
    (label missing, label not binary, no usable feature columns).
    """
    if label_col not in df.columns:
        raise ValueError(f'Label column "{label_col}" is not in the CSV.')

    # ---- label -> 0/1 ----
    y_raw = df[label_col]
    y_num = pd.to_numeric(y_raw, errors="coerce")
    if y_num.notna().all():
        uy = sorted(pd.unique(y_num.dropna()))
        if len(uy) != 2:
            raise ValueError(
                f'The label column "{label_col}" has {len(uy)} distinct values '
                f'({uy[:5]}). This app does binary classification — the label must '
                f'have exactly two values (e.g. 0 and 1).')
        y = (y_num == uy[1]).astype(int).values
    else:
        uy = sorted(map(str, pd.unique(y_raw.dropna())))
        if len(uy) != 2:
            raise ValueError(
                f'The label column "{label_col}" has {len(uy)} distinct values. '
                f'It must have exactly two (e.g. "yes"/"no").')
        y = (y_raw.astype(str) == uy[1]).astype(int).values

    blocks: List[np.ndarray] = []
    names: List[str] = []

    # ---- numeric ----
    num = [c for c in numeric_cols if c in df.columns and c != label_col]
    if num:
        M = df[num].apply(pd.to_numeric, errors="coerce")
        M = M.fillna(M.mean())
        # any column still all-NaN (no numeric values at all) -> drop
        good = [c for c in num if M[c].notna().any()]
        M = M[good].fillna(0.0)
        if good:
            blocks.append(M.values.astype(float))
            names.extend(good)

    # ---- categorical (one-hot) ----
    cat = [c for c in categorical_cols if c in df.columns and c != label_col]
    for c in cat:
        s = df[c].astype("string").fillna("__missing__")
        top = s.value_counts().head(max_categorical_levels).index
        s = s.where(s.isin(top), other="__other__")
        dummies = pd.get_dummies(s, prefix=str(c))
        if dummies.shape[1] > 0:
            blocks.append(dummies.values.astype(float))
            names.extend(list(dummies.columns))

    if not blocks:
        raise ValueError("No usable feature columns were selected. Pick at least "
                         "one numeric or categorical column (not just the id/label).")

    X = np.hstack(blocks).astype(float)
    # single modality "uploaded" (this generic path has no clinical modalities)
    mmap = {"uploaded": list(range(X.shape[1]))}
    return X, y, names, mmap
