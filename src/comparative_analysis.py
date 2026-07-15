"""
src/comparative_analysis.py
============================
QSQ-FS vs GA, PSO, RIME, PLO, HGS, SVM, RandomForest, XGBoost.

Design notes:
  * Every feature-selection method is evaluated under the identical leak-free
    protocol: select on the outer train fold, score the chosen subset on the
    held-out test fold (src.evaluation.select_then_score), so no method is
    scored on the data it optimised.
  * Equal evaluation budget: all metaheuristics use the same population
    (`n_agents`) and `max_iter`; QSQ-FS is configured to the same single-run
    budget so iteration counts match.
  * Classical baselines (SVM/RF/XGB) use the same outer CV folds.
  * Significance uses the Wilcoxon signed-rank test with Bonferroni correction
    on paired per-fold vectors of equal length.
"""

from __future__ import annotations

import time
from typing import Dict

import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    from loguru import logger
except ImportError:                       # pragma: no cover
    import logging
    logger = logging.getLogger(__name__)

from src.qsfs import QSQFS
from src.evaluation import select_then_score


# --------------------------------------------------------------------------- #
# Internal KNN-CV fitness shared by all metaheuristic baselines               #
# (used only DURING selection, on the training fold)                          #
# --------------------------------------------------------------------------- #
def _fitness(mask: np.ndarray, X: np.ndarray, y: np.ndarray, k=3, cv=3) -> float:
    if mask.sum() == 0:
        return 0.5
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=k)
    return float(np.mean(cross_val_score(knn, X[:, mask.astype(bool)], y, cv=skf)))


# --------------------------------------------------------------------------- #
# Simplified binary metaheuristics (equal-budget)                            #
# --------------------------------------------------------------------------- #
class _BaseMH:
    def __init__(self, n_agents=30, max_iter=20, random_state=42):
        self.n, self.max_iter, self.rng = n_agents, max_iter, np.random.default_rng(random_state)
        self.best_mask = None
        self.best_fit = -np.inf

    def _init_pop(self, d):
        pop = (self.rng.random((self.n, d)) > 0.5).astype(int)
        fit = np.array([_fitness(p, self._X, self._y) for p in pop])
        b = int(np.argmax(fit))
        self.best_mask, self.best_fit = pop[b].copy(), float(fit[b])
        return pop, fit

    def fit(self, X, y):
        self._X, self._y = X, y
        self._run(X.shape[1])
        return self

    def selected(self):
        return np.where(self.best_mask == 1)[0]


class _GA(_BaseMH):
    def _run(self, d):
        pop, fit = self._init_pop(d)
        for _ in range(self.max_iter):
            new = []
            for _ in range(self.n):
                i, j = self.rng.integers(0, self.n, 2)
                p1 = pop[i if fit[i] >= fit[j] else j]
                i, j = self.rng.integers(0, self.n, 2)
                p2 = pop[i if fit[i] >= fit[j] else j]
                pt = self.rng.integers(1, d)
                child = np.concatenate([p1[:pt], p2[pt:]])
                flip = self.rng.random(d) < 0.02
                child[flip] = 1 - child[flip]
                new.append(child)
            pop = np.array(new)
            fit = np.array([_fitness(p, self._X, self._y) for p in pop])
            b = int(np.argmax(fit))
            if fit[b] > self.best_fit:
                self.best_mask, self.best_fit = pop[b].copy(), float(fit[b])


class _PSO(_BaseMH):
    def _sig(self, v):
        return 1 / (1 + np.exp(-np.clip(v, -500, 500)))

    def _run(self, d):
        pos = (self.rng.random((self.n, d)) > 0.5).astype(float)
        vel = self.rng.uniform(-4, 4, (self.n, d))
        pf = np.array([_fitness(p.astype(int), self._X, self._y) for p in pos])
        pbest = pos.copy()
        gb = int(np.argmax(pf))
        gpos = pbest[gb].copy()
        self.best_mask, self.best_fit = (gpos > 0.5).astype(int), float(pf[gb])
        for _ in range(self.max_iter):
            r1, r2 = self.rng.random((self.n, d)), self.rng.random((self.n, d))
            vel = 0.7 * vel + 1.5 * r1 * (pbest - pos) + 1.5 * r2 * (gpos - pos)
            pos = (self.rng.random((self.n, d)) < self._sig(vel)).astype(float)
            f = np.array([_fitness(p.astype(int), self._X, self._y) for p in pos])
            imp = f > pf
            pbest[imp], pf[imp] = pos[imp], f[imp]
            gb = int(np.argmax(pf))
            if pf[gb] > self.best_fit:
                gpos = pbest[gb].copy()
                self.best_mask, self.best_fit = (gpos > 0.5).astype(int), float(pf[gb])


class _RIME(_BaseMH):
    def _run(self, d):
        pop, fit = self._init_pop(d)
        for t in range(1, self.max_iter + 1):
            rf = t / self.max_iter
            for i in range(self.n):
                if self.rng.random() < 1 - rf:
                    child = pop[i].copy()
                    fl = self.rng.random(d) < 0.15
                    child[fl] = 1 - child[fl]
                else:
                    cr = self.rng.random(d) < rf * 0.5
                    child = np.where(cr, self.best_mask, pop[i])
                f = _fitness(child, self._X, self._y)
                if f > fit[i]:
                    pop[i], fit[i] = child, f
                    if f > self.best_fit:
                        self.best_mask, self.best_fit = child.copy(), float(f)


class _PLO(_BaseMH):
    def _run(self, d):
        pop, fit = self._init_pop(d)
        for _ in range(self.max_iter):
            for i in range(self.n):
                if self.rng.random() < 0.5:
                    child = pop[i].copy()
                    fl = self.rng.random(d) < 0.08
                    child[fl] = 1 - child[fl]
                else:
                    j = self.rng.integers(0, self.n)
                    cr = self.rng.random(d) < 0.3
                    child = np.where(cr, pop[j], pop[i])
                f = _fitness(child, self._X, self._y)
                if f >= fit[i]:
                    pop[i], fit[i] = child, f
                    if f > self.best_fit:
                        self.best_mask, self.best_fit = child.copy(), float(f)


class _HGS(_BaseMH):
    def _run(self, d):
        pop, fit = self._init_pop(d)
        for _ in range(self.max_iter):
            hunger = 1 - (fit / (fit.max() + 1e-9))
            for i in range(self.n):
                if self.rng.random() < hunger[i] * 0.5:
                    child = pop[i].copy()
                    fl = self.rng.random(d) < 0.15
                    child[fl] = 1 - child[fl]
                else:
                    j = self.rng.integers(0, self.n)
                    cr = self.rng.random(d) < 0.2
                    child = np.where(cr, pop[j], pop[i])
                f = _fitness(child, self._X, self._y)
                if f >= fit[i]:
                    pop[i], fit[i] = child, f
                    if f > self.best_fit:
                        self.best_mask, self.best_fit = child.copy(), float(f)


_MH = {"GA": _GA, "PSO": _PSO, "RIME": _RIME, "PLO": _PLO, "HGS": _HGS}


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def run_comparative_analysis(X, y, n_trials=5, random_state=42,
                             n_agents=30, max_iter=20):
    """
    n_trials is reused as the number of outer CV folds (paired across methods).
    Returns per-method per-fold accuracy/auc/f1/n_features + Wilcoxon vs QSQ-FS.
    """
    X = np.asarray(X, float)
    y = np.asarray(y).astype(int)
    n_outer = max(3, n_trials)
    results: Dict[str, dict] = {}

    # --- QSQ-FS selector (equal budget: same n_agents/max_iter, single-run) ---
    def qsqfs_selector(Xtr, ytr):
        m = QSQFS(n_colonies=n_agents, max_iter_stage1=max_iter, max_iter_stage2=0,
                  cv_folds=3, random_state=random_state, verbose=False)
        m.fit(Xtr, ytr)
        return m.get_selected_features()

    logger.info("Comparative: QSQ-FS (leak-free, equal budget)...")
    t0 = time.time()
    results["QSQ-FS"] = select_then_score(qsqfs_selector, X, y, n_outer=n_outer,
                                          random_state=random_state)
    results["QSQ-FS"]["runtime"] = np.array([(time.time() - t0) / n_outer] * n_outer)

    # --- metaheuristic baselines (same budget, same protocol) ---
    for name, Cls in _MH.items():
        logger.info(f"Comparative: {name}...")
        t0 = time.time()

        def sel(Xtr, ytr, Cls=Cls):
            return Cls(n_agents=n_agents, max_iter=max_iter,
                       random_state=random_state).fit(Xtr, ytr).selected()

        results[name] = select_then_score(sel, X, y, n_outer=n_outer, random_state=random_state)
        results[name]["runtime"] = np.array([(time.time() - t0) / n_outer] * n_outer)

    # --- classical ML baselines (all features, same outer folds) ---
    skf = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=random_state)
    classifiers = [
        ("SVM", lambda: SVC(kernel="rbf", probability=True, random_state=random_state)),
        ("RandomForest", lambda: RandomForestClassifier(n_estimators=200, random_state=random_state)),
    ]
    if XGB_AVAILABLE:
        classifiers.append(
            ("XGBoost", lambda: XGBClassifier(n_estimators=200, random_state=random_state,
                                              eval_metric="logloss")))
    for name, make in classifiers:
        logger.info(f"Comparative: {name}...")
        accs, aucs, f1s, rts = [], [], [], []
        for tr, te in skf.split(X, y):
            scaler = StandardScaler().fit(X[tr])
            Xtr, Xte = scaler.transform(X[tr]), scaler.transform(X[te])
            clf = make()
            t0 = time.time()
            clf.fit(Xtr, y[tr])
            rts.append(time.time() - t0)
            proba = clf.predict_proba(Xte)[:, 1]
            pred = (proba >= 0.5).astype(int)
            accs.append(accuracy_score(y[te], pred))
            aucs.append(roc_auc_score(y[te], proba) if len(np.unique(y[te])) > 1 else np.nan)
            f1s.append(f1_score(y[te], pred, zero_division=0))
        results[name] = {
            "accuracy": np.array(accs), "auc": np.array(aucs), "f1": np.array(f1s),
            "n_features": np.full(n_outer, X.shape[1]), "runtime": np.array(rts),
        }

    # --- Wilcoxon vs QSQ-FS (paired per-fold, equal length) ---
    base = results["QSQ-FS"]["accuracy"]
    comps = [k for k in results if k != "QSQ-FS"]
    alpha_bonf = 0.05 / max(1, len(comps))
    stat_rows = []
    for name in comps:
        comp = results[name]["accuracy"]
        diff = base - comp
        if np.allclose(diff, 0) or len(diff) < 5:
            w, p = np.nan, 1.0
        else:
            w, p = stats.wilcoxon(diff, alternative="greater", zero_method="wilcox")
        stat_rows.append({
            "algorithm": name,
            "mean_acc_ours": float(np.mean(base)),
            "mean_acc_theirs": float(np.mean(comp)),
            "delta_acc": float(np.mean(diff)),
            "w_statistic": round(float(w), 3) if not np.isnan(w) else "n/a",
            "p_value": round(float(p), 4),
            "significant": bool(p < alpha_bonf),
            "alpha_bonf": round(alpha_bonf, 4),
        })
    results["_stats"] = stat_rows
    results["_n_outer"] = n_outer
    return results
