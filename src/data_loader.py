"""
src/data_loader.py
==================
Loads MIMIC-IV CSV/CSV.GZ, MATLAB .mat, CSV uploads, or generates synthetic
demo data.

The synthetic generator uses heavily OVERLAPPING class distributions so the
demo presents a realistic, non-trivial signal: diabetics have higher
glucose/HbA1c on average, but the distributions overlap, so no single feature
is a giveaway. The label is derived from ICD codes; the diabetes-defining codes
and diabetes drugs are stripped downstream by modality_builder in both the wide
and long paths, so they cannot leak into the feature matrix.

Real MIMIC-IV paths read both `.csv` and `.csv.gz`, and `chartevents` from the
`icu/` sub-folder (long format, handled in modality_builder).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

try:
    from loguru import logger
except ImportError:                       # pragma: no cover
    import logging
    logger = logging.getLogger(__name__)

DIABETES_ICD9_PREFIX = "250"
DIABETES_ICD10_PREFIXES = ("E08", "E09", "E10", "E11", "E12", "E13")


class MIMICDataLoader:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.demo_mode: bool = config.get("demo_mode", True)
        self.data_root = Path(config["paths"]["data_root"])

    def load_all(self, progress_cb=None) -> Dict[str, pd.DataFrame]:
        if self.demo_mode:
            logger.warning("DEMO MODE - generating non-trivial synthetic data.")
            return self._generate_synthetic_data()
        return self._load_real_data(progress_cb=progress_cb)

    # ------------------------------------------------------------------ #
    # Real MIMIC-IV                                                       #
    # ------------------------------------------------------------------ #
    # Real MIMIC-IV `labevents` (~120M rows) and `chartevents` (~330M rows)
    # cannot be `pd.read_csv(..., low_memory=False)`'d whole into memory
    # (this would exhaust memory on anything beyond the demo). These two tables
    # are read in chunks, restricted to the columns actually used downstream,
    # and pre-filtered to the top-N most frequent itemids -- everything else
    # is discarded before it is ever materialised as a full DataFrame.
    LARGE_LONG_TABLES = {"labevents", "chartevents"}

    def _load_real_data(self, progress_cb=None) -> Dict[str, pd.DataFrame]:
        hosp = ["admissions", "patients", "diagnoses_icd", "d_icd_diagnoses",
                "labevents", "d_labitems", "prescriptions", "pharmacy"]
        icu = ["chartevents", "d_items", "icustays"]
        data: Dict[str, pd.DataFrame] = {}
        # progress_cb(fraction_0_to_1, human_message) — optional; used by the
        # Streamlit UI to show a live progress bar during the slow chunked reads
        # (the big labevents/chartevents tables give no feedback otherwise).
        total = len(hosp) + len(icu)
        done = 0

        def _step(name):
            nonlocal done
            done += 1
            if progress_cb:
                progress_cb(done / total, f"Loading {name} ({done}/{total})...")

        for t in hosp:
            df = self._try_load(self.data_root / "hosp" / f"{t}.csv", t,
                                progress_cb=progress_cb)
            if df is not None:
                data[t] = df
            _step(t)
        for t in icu:
            df = self._try_load(self.data_root / "icu" / f"{t}.csv", t,
                                progress_cb=progress_cb)
            if df is not None:
                data[t] = df
            _step(t)
        if progress_cb:
            progress_cb(1.0, "All tables loaded.")
        return data

    def _try_load(self, path: Path, name: str, progress_cb=None):
        for p in (path, path.with_suffix(".csv.gz")):
            if p.exists():
                try:
                    if name in self.LARGE_LONG_TABLES:
                        df = self._load_large_event_table(p, name, progress_cb=progress_cb)
                    else:
                        df = pd.read_csv(p, low_memory=False)
                    logger.info(f"Loaded {name}: {len(df):,} rows x {df.shape[1]} cols")
                    return df
                except Exception as exc:
                    logger.error(f"Failed to load {name}: {exc}")
        logger.warning(f"Table not found: {path.name} - skipping.")
        return None

    def _load_large_event_table(self, path: Path, name: str,
                                chunksize: int = 2_000_000,
                                progress_cb=None) -> pd.DataFrame:
        """Scalable read for labevents/chartevents.

        Pass 1 streams only the `itemid` column (int32, near-zero memory) to
        get true frequency counts across the WHOLE table. Pass 2 streams
        `subject_id, itemid, valuenum` and keeps only rows whose itemid is in
        the frequent set, so peak memory is bounded by the filtered result,
        not by the source table (2-330M+ rows on real MIMIC-IV).
        """
        top_k = self.config.get("preprocessing", {}).get(
            "top_labs" if name == "labevents" else "top_vitals", 50)
        # keep a generous buffer above top_k: downstream code still applies
        # its own top-k after merging with the (smaller) surviving cohort.
        keep_n = max(top_k * 3, top_k + 10)

        logger.info(f"{name}: pass 1/2 - streaming itemid frequency "
                    f"(chunksize={chunksize:,})...")
        if progress_cb:
            progress_cb(None, f"{name}: scanning item frequencies (pass 1 of 2)...")
        counts: "pd.Series" = pd.Series(dtype="int64")
        for chunk in pd.read_csv(path, usecols=["itemid"],
                                 dtype={"itemid": "int32"}, chunksize=chunksize):
            counts = counts.add(chunk["itemid"].value_counts(), fill_value=0)
        top_items = set(counts.sort_values(ascending=False).head(keep_n).index.astype(int))
        logger.info(f"{name}: keeping top {len(top_items)} itemids by frequency "
                    f"(out of {len(counts):,} distinct)")

        logger.info(f"{name}: pass 2/2 - streaming filtered rows...")
        if progress_cb:
            progress_cb(None, f"{name}: reading {len(top_items)} kept items (pass 2 of 2)...")
        usecols = ["subject_id", "itemid", "valuenum"]
        dtypes = {"subject_id": "int32", "itemid": "int32", "valuenum": "float32"}
        parts = []
        for chunk in pd.read_csv(path, usecols=usecols, dtype=dtypes, chunksize=chunksize):
            parts.append(chunk[chunk["itemid"].isin(top_items)])
        df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=usecols)
        logger.info(f"{name}: {len(df):,} rows retained after itemid filtering "
                    f"(scalable path)")
        return df

    @staticmethod
    def load_matlab(mat_path) -> Dict[str, pd.DataFrame]:
        """Load a .mat file into named DataFrames.

        Each 2-D numeric variable is kept under its own name so the caller can
        decide which table it corresponds to, rather than assuming a fixed
        variable is `labevents`.
        """
        from scipy.io import loadmat
        mat = loadmat(mat_path)
        out = {}
        for k, v in mat.items():
            if k.startswith("_"):
                continue
            if isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[0] > 0 and v.shape[1] > 0:
                try:
                    out[k] = pd.DataFrame(v)
                    logger.info(f"MATLAB var '{k}': {v.shape[0]}x{v.shape[1]} -> DataFrame")
                except Exception as exc:
                    logger.warning(f"Could not convert MATLAB var '{k}': {exc}")
            else:
                logger.warning(f"Skipping MATLAB var '{k}' (not a non-empty 2-D array)")
        if not out:
            logger.error("No usable 2-D numeric variables found in .mat file.")
        return out

    @staticmethod
    def load_csv_upload(file_obj) -> pd.DataFrame:
        return pd.read_csv(file_obj, low_memory=False)

    # ------------------------------------------------------------------ #
    # Synthetic demo data                                                #
    # ------------------------------------------------------------------ #
    def _generate_synthetic_data(self) -> Dict[str, pd.DataFrame]:
        rng = np.random.default_rng(42)
        n = 1000
        patients = pd.DataFrame({
            "subject_id": np.arange(1, n + 1),
            "gender": rng.choice(["M", "F"], n),
            "anchor_age": rng.integers(20, 90, n),
        })

        # Latent risk -> ~25% prevalence. Features overlap heavily (non-trivial).
        risk = (
            0.018 * (patients["anchor_age"].values - 55)
            + rng.normal(0, 1.0, n)
        )
        prob = 1.0 / (1.0 + np.exp(-risk))
        flag = (rng.random(n) < np.clip(prob, 0.05, 0.6)).astype(int)

        def overlap(mean0, mean1, sd, effect=0.6):
            # small mean shift relative to sd => overlapping, realistic signal
            base = np.where(flag == 1, mean1, mean0)
            return base + rng.normal(0, sd, n) + effect * sd * rng.normal(0, 1, n) * 0

        labs = pd.DataFrame({
            "subject_id": patients["subject_id"],
            "glucose":       np.where(flag == 1, 128, 104) + rng.normal(0, 28, n),
            "hbA1c":         np.where(flag == 1, 6.6, 5.6) + rng.normal(0, 0.9, n),
            "creatinine":    rng.normal(1.0, 0.3, n),
            "sodium":        rng.normal(140, 3, n),
            "potassium":     rng.normal(4.0, 0.5, n),
            "chloride":      rng.normal(102, 3, n),
            "bun":           rng.normal(15, 5, n),
            "wbc":           rng.normal(7.5, 2, n),
            "hemoglobin":    rng.normal(13.5, 1.5, n),
            "platelets":     rng.normal(250, 50, n),
            "alt":           rng.normal(35, 10, n),
            "ast":           rng.normal(30, 8, n),
            "alk_phos":      rng.normal(80, 20, n),
            "bilirubin":     rng.normal(0.8, 0.2, n),
            "albumin":       rng.normal(4.0, 0.4, n),
            "triglycerides": np.where(flag == 1, 168, 142) + rng.normal(0, 45, n),
            "chol_total":    rng.normal(200, 35, n),
            "hdl":           rng.normal(55, 12, n),
            "ldl":           rng.normal(130, 30, n),
            "crp":           np.where(flag == 1, 2.6, 2.0) + rng.normal(0, 1.4, n),
        })

        vitals = pd.DataFrame({
            "subject_id": patients["subject_id"],
            "heart_rate": rng.normal(75, 12, n),
            "sbp":        np.where(flag == 1, 128, 121) + rng.normal(0, 16, n),
            "dbp":        rng.normal(80, 10, n),
            "resp_rate":  rng.normal(16, 3, n),
            "temperature": rng.normal(36.8, 0.5, n),
        })

        # medications: diabetes drugs correlate with the label (as in reality).
        # These are STRIPPED downstream to avoid leakage; left here for realism.
        meds = pd.DataFrame({
            "subject_id":   patients["subject_id"],
            "metformin":    (rng.random(n) < (0.40 * flag + 0.03)).astype(int),
            "insulin":      (rng.random(n) < (0.25 * flag + 0.02)).astype(int),
            "glipizide":    (rng.random(n) < (0.15 * flag + 0.02)).astype(int),
            "lisinopril":   (rng.random(n) < 0.22).astype(int),
            "atorvastatin": (rng.random(n) < 0.20).astype(int),
            "aspirin":      (rng.random(n) < 0.25).astype(int),
            "amlodipine":   (rng.random(n) < 0.14).astype(int),
            "metoprolol":   (rng.random(n) < 0.16).astype(int),
            "omeprazole":   (rng.random(n) < 0.12).astype(int),
            "furosemide":   (rng.random(n) < 0.09).astype(int),
        })

        # ICD table (long): diabetics -> E11.9, others -> a non-diabetes code.
        # Comorbidity codes are added as legitimate, non-leaky features.
        icd_rows = []
        for pid, f in zip(patients["subject_id"].values, flag):
            icd_rows.append({"subject_id": pid,
                             "icd_code": "E11.9" if f else "I10", "icd_version": 10})
            if rng.random() < (0.35 * f + 0.18):
                icd_rows.append({"subject_id": pid, "icd_code": "I10", "icd_version": 10})    # HTN
            if rng.random() < (0.30 * f + 0.15):
                icd_rows.append({"subject_id": pid, "icd_code": "E78.5", "icd_version": 10})  # lipids
            if rng.random() < 0.20:
                icd_rows.append({"subject_id": pid, "icd_code": "E66.9", "icd_version": 10})  # obesity
        diagnoses_icd = pd.DataFrame(icd_rows)

        d_icd = pd.DataFrame({
            "icd_code": ["E11.9", "I10", "E78.5", "E66.9", "250.00"],
            "icd_version": [10, 10, 10, 10, 9],
            "long_title": [
                "Type 2 diabetes mellitus without complications",
                "Essential (primary) hypertension",
                "Hyperlipidemia, unspecified",
                "Obesity, unspecified",
                "Diabetes mellitus without complication (ICD-9)",
            ],
        })
        admissions = pd.DataFrame({
            "subject_id": patients["subject_id"],
            "hadm_id": np.arange(10001, 10001 + n),
        })
        prevalence = 100 * flag.mean()
        logger.info(f"Synthetic prevalence: {prevalence:.1f}% diabetic (overlapping signal)")

        return {
            "patients": patients,
            "labevents": labs,
            "chartevents": vitals,
            "pharmacy": meds,
            "diagnoses_icd": diagnoses_icd,
            "d_icd_diagnoses": d_icd,
            "admissions": admissions,
        }
