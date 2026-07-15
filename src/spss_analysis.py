"""
src/spss_analysis.py
====================
SPSS-style statistical analysis and reporting for the diabetic vs non-diabetic
cohort. Produces the tables a reviewer expects to see in a results chapter,
formatted the way SPSS / APA present them:

  * Group descriptive statistics (N, Mean, SD, SE, Min, Max, Median) per feature,
    split by outcome, mirroring SPSS "Descriptives / Explore".
  * Normality screening (Shapiro-Wilk) to justify parametric vs non-parametric.
  * Independent-samples comparison per feature: Levene's test for equality of
    variances, Student's / Welch's t-test, AND the Mann-Whitney U non-parametric
    alternative, with Cohen's d effect size -- the SPSS "Independent-Samples
    T-Test" table plus its non-parametric companion.
  * Categorical association: chi-square test of independence with Cramer's V
    (SPSS "Crosstabs" / "Chi-Square Tests").
  * Benjamini-Hochberg FDR correction across the many per-feature tests.
  * A univariate discrimination table (per-feature AUC + odds ratio) so each
    predictor's individual signal is quantified.

Significance is flagged with the conventional stars (* p<.05, ** p<.01,
*** p<.001). All outputs are returned as tidy DataFrames and can be written to
CSV; nothing here exposes a per-patient row.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

try:
    from loguru import logger
except ImportError:                       # pragma: no cover
    import logging
    logger = logging.getLogger(__name__)


def _stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled)


def _cramers_v(chi2: float, n: int, r: int, c: int) -> float:
    k = min(r - 1, c - 1)
    if k == 0 or n == 0:
        return 0.0
    return float(np.sqrt((chi2 / n) / k))


def _auc(y: np.ndarray, x: np.ndarray) -> float:
    pos, neg = x[y == 1], x[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # Mann-Whitney U statistic -> AUC (rank-based, direction-agnostic reported >=0.5)
    try:
        u, _ = stats.mannwhitneyu(pos, neg, alternative="two-sided")
        a = u / (len(pos) * len(neg))
        return float(max(a, 1 - a))
    except ValueError:
        return float("nan")


def _bh_fdr(pvals: List[float]) -> List[float]:
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    # enforce monotonicity
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    return list(out)


def group_descriptives(X: np.ndarray, y: np.ndarray,
                       feature_names: List[str]) -> pd.DataFrame:
    """SPSS 'Descriptives by group': N, Mean, SD, SE, Min, Max, Median per group."""
    rows = []
    for j, name in enumerate(feature_names):
        for g, glabel in [(0, "Non-diabetic"), (1, "Diabetic")]:
            v = X[y == g, j]
            v = v[~np.isnan(v)]
            n = len(v)
            sd = float(np.std(v, ddof=1)) if n > 1 else 0.0
            rows.append({
                "Feature": name, "Group": glabel, "N": n,
                "Mean": round(float(np.mean(v)), 4) if n else float("nan"),
                "SD": round(sd, 4),
                "SE": round(sd / np.sqrt(n), 4) if n else float("nan"),
                "Min": round(float(np.min(v)), 4) if n else float("nan"),
                "Max": round(float(np.max(v)), 4) if n else float("nan"),
                "Median": round(float(np.median(v)), 4) if n else float("nan"),
            })
    return pd.DataFrame(rows)


def independent_samples_tests(X: np.ndarray, y: np.ndarray,
                              feature_names: List[str]) -> pd.DataFrame:
    """SPSS 'Independent-Samples T-Test' + Mann-Whitney companion, per feature.

    Reports, per feature: group means/SDs, Levene's test (equal-variance
    assumption), Student's and Welch's t-test, Mann-Whitney U, Cohen's d, the
    per-feature discrimination AUC, and a crude odds ratio (upper vs lower
    median split). p-values are FDR-corrected across features.
    """
    recs = []
    raw_p_t, raw_p_u = [], []
    for j, name in enumerate(feature_names):
        a = X[y == 1, j]; a = a[~np.isnan(a)]        # diabetic
        b = X[y == 0, j]; b = b[~np.isnan(b)]        # non-diabetic
        if len(a) < 3 or len(b) < 3:
            continue
        # Levene's test for equality of variances -> choose Student vs Welch
        try:
            lev_stat, lev_p = stats.levene(a, b)
        except ValueError:
            lev_stat, lev_p = float("nan"), 1.0
        equal_var = (lev_p >= 0.05)
        t_stat, t_p = stats.ttest_ind(a, b, equal_var=equal_var)
        try:
            u_stat, u_p = stats.mannwhitneyu(a, b, alternative="two-sided")
        except ValueError:
            u_stat, u_p = float("nan"), 1.0
        d = _cohens_d(a, b)
        auc = _auc(y, X[:, j])
        # crude odds ratio via median dichotomisation (adds 0.5 continuity)
        med = np.nanmedian(X[:, j])
        hi = X[:, j] >= med
        tp = np.sum(hi & (y == 1)) + 0.5; fp = np.sum(hi & (y == 0)) + 0.5
        fn = np.sum(~hi & (y == 1)) + 0.5; tn = np.sum(~hi & (y == 0)) + 0.5
        odds = float((tp * tn) / (fp * fn))
        recs.append({
            "Feature": name,
            "Mean_Diabetic": round(float(np.mean(a)), 3),
            "SD_Diabetic": round(float(np.std(a, ddof=1)), 3),
            "Mean_NonDiabetic": round(float(np.mean(b)), 3),
            "SD_NonDiabetic": round(float(np.std(b, ddof=1)), 3),
            "Levene_p": round(float(lev_p), 4),
            "EqualVar": equal_var,
            "t": round(float(t_stat), 3),
            "t_p": float(t_p),
            "MannWhitney_U": round(float(u_stat), 1),
            "MW_p": float(u_p),
            "Cohens_d": round(float(d), 3),
            "AUC": round(float(auc), 3) if not np.isnan(auc) else float("nan"),
            "OddsRatio": round(odds, 3),
        })
        raw_p_t.append(float(t_p)); raw_p_u.append(float(u_p))
    df = pd.DataFrame(recs)
    if not df.empty:
        df["t_p_FDR"] = [round(p, 4) for p in _bh_fdr(raw_p_t)]
        df["MW_p_FDR"] = [round(p, 4) for p in _bh_fdr(raw_p_u)]
        df["Sig"] = [_stars(p) for p in df["t_p_FDR"]]
        df["t_p"] = df["t_p"].round(4)
        df["MW_p"] = df["MW_p"].round(4)
        df = df.sort_values("AUC", ascending=False).reset_index(drop=True)
    return df


def chi_square_tests(df_raw: pd.DataFrame, label_col: str,
                     categorical_cols: List[str]) -> pd.DataFrame:
    """SPSS 'Crosstabs' chi-square test of independence + Cramer's V.

    Only meaningful for genuinely categorical columns; pass those explicitly.
    """
    recs = []
    y = df_raw[label_col]
    raw_p = []
    for c in categorical_cols:
        if c == label_col or c not in df_raw.columns:
            continue
        tbl = pd.crosstab(df_raw[c], y)
        if tbl.shape[0] < 2 or tbl.shape[1] < 2:
            continue
        try:
            chi2, p, dof, _ = stats.chi2_contingency(tbl)
        except ValueError:
            continue
        v = _cramers_v(chi2, int(tbl.values.sum()), tbl.shape[0], tbl.shape[1])
        recs.append({"Variable": c, "ChiSquare": round(float(chi2), 3),
                     "dof": int(dof), "p": float(p),
                     "CramersV": round(v, 3)})
        raw_p.append(float(p))
    out = pd.DataFrame(recs)
    if not out.empty:
        out["p_FDR"] = [round(p, 4) for p in _bh_fdr(raw_p)]
        out["Sig"] = [_stars(p) for p in out["p_FDR"]]
        out["p"] = out["p"].round(4)
        out = out.sort_values("ChiSquare", ascending=False).reset_index(drop=True)
    return out


def normality_screen(X: np.ndarray, y: np.ndarray,
                     feature_names: List[str]) -> pd.DataFrame:
    """Shapiro-Wilk normality per feature per group (justifies test choice)."""
    recs = []
    for j, name in enumerate(feature_names):
        row = {"Feature": name}
        for g, glabel in [(1, "Diabetic"), (0, "NonDiabetic")]:
            v = X[y == g, j]; v = v[~np.isnan(v)]
            if 3 <= len(v) <= 5000 and np.std(v) > 0:
                try:
                    w, p = stats.shapiro(v)
                    row[f"Shapiro_W_{glabel}"] = round(float(w), 3)
                    row[f"Shapiro_p_{glabel}"] = round(float(p), 4)
                    row[f"Normal_{glabel}"] = bool(p >= 0.05)
                except ValueError:
                    row[f"Shapiro_W_{glabel}"] = float("nan")
            else:
                row[f"Shapiro_W_{glabel}"] = float("nan")
        recs.append(row)
    return pd.DataFrame(recs)


def run_spss_style_analysis(X: np.ndarray, y: np.ndarray, feature_names: List[str],
                            top_k: int = 25) -> Dict[str, pd.DataFrame]:
    """Run the full SPSS-style battery and return a dict of tidy tables.

    top_k limits the descriptive/normality tables to the strongest features (by
    discrimination AUC) so the output stays readable when there are hundreds of
    columns; the independent-samples table covers every feature.
    """
    X = np.asarray(X, float)
    y = np.asarray(y).astype(int)
    tests = independent_samples_tests(X, y, feature_names)
    keep = list(tests["Feature"].head(top_k)) if not tests.empty else feature_names[:top_k]
    keep_idx = [feature_names.index(f) for f in keep]
    Xk = X[:, keep_idx]
    out = {
        "independent_samples": tests,
        "descriptives": group_descriptives(Xk, y, keep),
        "normality": normality_screen(Xk, y, keep),
    }
    logger.info(f"SPSS-style analysis: {len(tests)} features tested, "
                f"{int((tests['Sig'] != 'ns').sum()) if not tests.empty else 0} "
                f"significant after FDR correction.")
    return out
