"""
src/optimizers.py
=================
Binary metaheuristic feature selectors used both as the proposal's named
algorithms (RIME, PLO, HGS) and as comparison baselines (GA, PSO).

Every optimiser shares one interface::

    opt = RIME(n_agents=30, max_iter=20, random_state=42)
    opt.fit(X_train, y_train)          # selects on the training fold only
    idx = opt.selected()               # -> array of chosen column indices

and one leak-free wrapper fitness (`_cv_fitness`): a stratified k-fold KNN
accuracy (or a blended AUC/accuracy objective, see `metric`) evaluated with a
per-fold scaler fit on the training split, plus a parsimony term identical to
QSQ-FS's Eq 3.1. Because the fitness is the same across all optimisers, an
equal population / iteration budget is a fair comparison.

The three named algorithms follow their published update rules:

  * RIME (Su et al., 2023): a soft-rime search that anneals from exploration to
    exploitation via a rime factor that grows with iteration.
  * PLO / Polar Lights Optimizer (Yuan et al., 2024): alternates an aurora-like
    local drift with a cross-agent recombination step.
  * HGS / Hunger Games Search (Yang et al., 2021): a hunger weight, higher for
    poorly performing agents, drives more exploration for the "hungry".

These are the binary (feature-selection) forms; the continuous-to-binary map is
a bit-flip / crossover parameterisation so they operate directly on masks.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score


def _cv_fitness(mask: np.ndarray, X: np.ndarray, y: np.ndarray,
                k: int = 3, cv: int = 3, alpha: float = 0.95,
                metric: str = "accuracy", random_state: int = 42) -> float:
    """Leak-free wrapper fitness: alpha * skill + (1-alpha) * parsimony.

    `metric` is "accuracy", "auc", or "balanced" (0.5*acc + 0.5*auc). The
    scaler is fit per fold on the training split only.
    """
    mask = np.asarray(mask).astype(bool)
    d = X.shape[1]
    if mask.sum() == 0:
        return 0.0
    Xs = X[:, mask]
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    accs, aucs = [], []
    for tr, te in skf.split(Xs, y):
        sc = StandardScaler().fit(Xs[tr])
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(sc.transform(Xs[tr]), y[tr])
        proba = knn.predict_proba(sc.transform(Xs[te]))
        pred = knn.predict(sc.transform(Xs[te]))
        accs.append(float(np.mean(pred == y[te])))
        if proba.shape[1] > 1 and len(np.unique(y[te])) > 1:
            aucs.append(float(roc_auc_score(y[te], proba[:, 1])))
    acc = float(np.mean(accs))
    auc = float(np.mean(aucs)) if aucs else acc
    if metric == "auc":
        skill = auc
    elif metric == "balanced":
        skill = 0.5 * acc + 0.5 * auc
    else:
        skill = acc
    parsimony = 1.0 - mask.sum() / d
    return alpha * skill + (1.0 - alpha) * parsimony


class _BaseSelector:
    name = "base"

    def __init__(self, n_agents: int = 30, max_iter: int = 20,
                 k_nn: int = 3, cv_folds: int = 3, alpha: float = 0.95,
                 metric: str = "accuracy", random_state: int = 42):
        self.n = n_agents
        self.max_iter = max_iter
        self.k_nn = k_nn
        self.cv_folds = cv_folds
        self.alpha = alpha
        self.metric = metric
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        self.best_mask: Optional[np.ndarray] = None
        self.best_fit = -np.inf
        self.history: list = []

    def _fit_one(self, mask):
        return _cv_fitness(mask, self._X, self._y, k=self.k_nn, cv=self.cv_folds,
                           alpha=self.alpha, metric=self.metric,
                           random_state=self.random_state)

    def _repair(self, mask):
        mask = mask.astype(int)
        if mask.sum() == 0:
            mask[self.rng.integers(0, mask.size)] = 1
        return mask

    def _init_pop(self, d):
        pop = (self.rng.random((self.n, d)) > 0.5).astype(int)
        pop = np.array([self._repair(p) for p in pop])
        fit = np.array([self._fit_one(p) for p in pop])
        b = int(np.argmax(fit))
        self.best_mask, self.best_fit = pop[b].copy(), float(fit[b])
        self.history.append(self.best_fit)
        return pop, fit

    def _track(self, mask, f):
        if f > self.best_fit:
            self.best_mask, self.best_fit = mask.copy(), float(f)
        self.history.append(self.best_fit)

    def fit(self, X, y):
        self._X = np.asarray(X, float)
        self._y = np.asarray(y).astype(int)
        self._run(self._X.shape[1])
        return self

    def selected(self):
        return np.where(self.best_mask == 1)[0]

    def _run(self, d):
        raise NotImplementedError


class GA(_BaseSelector):
    name = "GA"

    def _run(self, d):
        pop, fit = self._init_pop(d)
        for _ in range(self.max_iter):
            new = []
            for _ in range(self.n):
                i, j = self.rng.integers(0, self.n, 2)
                p1 = pop[i if fit[i] >= fit[j] else j]
                i, j = self.rng.integers(0, self.n, 2)
                p2 = pop[i if fit[i] >= fit[j] else j]
                cut = self.rng.integers(1, d)
                child = np.concatenate([p1[:cut], p2[cut:]])
                flip = self.rng.random(d) < (1.0 / d)
                child[flip] = 1 - child[flip]
                new.append(self._repair(child))
            pop = np.array(new)
            fit = np.array([self._fit_one(p) for p in pop])
            b = int(np.argmax(fit))
            self._track(pop[b], fit[b])


class PSO(_BaseSelector):
    name = "PSO"

    @staticmethod
    def _sig(v):
        return 1.0 / (1.0 + np.exp(-np.clip(v, -10, 10)))

    def _run(self, d):
        pos = (self.rng.random((self.n, d)) > 0.5).astype(float)
        vel = self.rng.uniform(-4, 4, (self.n, d))
        pf = np.array([self._fit_one(self._repair(p)) for p in pos])
        pbest = pos.copy()
        gi = int(np.argmax(pf))
        gpos = pbest[gi].copy()
        self.best_mask, self.best_fit = self._repair(gpos), float(pf[gi])
        w, c1, c2 = 0.7, 1.5, 1.5
        for _ in range(self.max_iter):
            r1, r2 = self.rng.random((self.n, d)), self.rng.random((self.n, d))
            vel = w * vel + c1 * r1 * (pbest - pos) + c2 * r2 * (gpos - pos)
            vel = np.clip(vel, -4, 4)
            pos = (self.rng.random((self.n, d)) < self._sig(vel)).astype(float)
            f = np.array([self._fit_one(self._repair(p)) for p in pos])
            imp = f > pf
            pbest[imp], pf[imp] = pos[imp], f[imp]
            gi = int(np.argmax(pf))
            self._track(self._repair(pbest[gi]), pf[gi])
            gpos = pbest[gi].copy()


class RIME(_BaseSelector):
    name = "RIME"

    def _run(self, d):
        pop, fit = self._init_pop(d)
        for t in range(1, self.max_iter + 1):
            rime_factor = (t / self.max_iter)            # anneal exploration->exploitation
            for i in range(self.n):
                if self.rng.random() < (1 - rime_factor):
                    # soft-rime: random bit-flips (exploration)
                    child = pop[i].copy()
                    fl = self.rng.random(d) < 0.15 * (1 - rime_factor + 0.1)
                    child[fl] = 1 - child[fl]
                else:
                    # hard-rime: cross toward best (exploitation)
                    cr = self.rng.random(d) < rime_factor * 0.5
                    child = np.where(cr, self.best_mask, pop[i])
                child = self._repair(child)
                f = self._fit_one(child)
                if f >= fit[i]:
                    pop[i], fit[i] = child, f
                self._track(pop[i], fit[i])


class PLO(_BaseSelector):
    name = "PLO"

    def _run(self, d):
        pop, fit = self._init_pop(d)
        for _ in range(self.max_iter):
            for i in range(self.n):
                if self.rng.random() < 0.5:
                    # aurora drift: localised bit-flips
                    child = pop[i].copy()
                    fl = self.rng.random(d) < 0.08
                    child[fl] = 1 - child[fl]
                else:
                    # particle recombination toward another agent
                    j = int(self.rng.integers(0, self.n))
                    cr = self.rng.random(d) < 0.3
                    child = np.where(cr, pop[j], pop[i])
                child = self._repair(child)
                f = self._fit_one(child)
                if f >= fit[i]:
                    pop[i], fit[i] = child, f
                self._track(pop[i], fit[i])


class HGS(_BaseSelector):
    name = "HGS"

    def _run(self, d):
        pop, fit = self._init_pop(d)
        for _ in range(self.max_iter):
            hunger = 1.0 - (fit / (fit.max() + 1e-9))     # worse agents are hungrier
            for i in range(self.n):
                if self.rng.random() < hunger[i] * 0.5:
                    child = pop[i].copy()
                    fl = self.rng.random(d) < 0.15
                    child[fl] = 1 - child[fl]
                else:
                    j = int(self.rng.integers(0, self.n))
                    cr = self.rng.random(d) < 0.2
                    child = np.where(cr, pop[j], pop[i])
                child = self._repair(child)
                f = self._fit_one(child)
                if f >= fit[i]:
                    pop[i], fit[i] = child, f
                self._track(pop[i], fit[i])


REGISTRY = {"GA": GA, "PSO": PSO, "RIME": RIME, "PLO": PLO, "HGS": HGS}


def get_selector(name: str, **kwargs) -> _BaseSelector:
    """Factory: get_selector("RIME", n_agents=30, max_iter=20, ...)."""
    key = name.upper()
    if key not in REGISTRY:
        raise ValueError(f"Unknown optimizer '{name}'. Choices: {sorted(REGISTRY)}")
    return REGISTRY[key](**kwargs)
