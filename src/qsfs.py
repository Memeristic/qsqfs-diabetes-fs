"""
src/qsfs.py
===========
QSQ-FS: Quorum Sensing & Quorum Quenching Feature Selection.

Implements thesis equations 3.1-3.12, with the following mechanisms:

  [A] Elite preservation       -- the global-best colony is re-injected into the
      population every generation (config: `elitism`), so the monotonic best
      fitness is produced by the population dynamics themselves.
  [B] Fitness caching          -- fitness is a deterministic function of the
      feature bitmask, so the cache acts as a pure memoiser (config:
      `use_cache`); disabling it changes runtime only, not the selected subset.
  [C] Diversity injection      -- on stagnation the search (1) raises a
      background mutation rate to `injection_mutation_rate` (0.40), (2) lowers
      w_AI to `injection_w_AI` (0.20), and (3) replaces the bottom quartile,
      sustained for `recovery_iters` generations (thesis Sec 3.4.3, mechanisms
      1-3).
  [D] Quorum Quenching on raw fitness -- the Eq 3.8 suppression term is computed
      from the un-penalised fitness, so penalties do not compound.
  [E] Deterministic seeding    -- per-colony RNGs are seeded from
      (random_state, stage, iteration, colony_index) for reproducibility.
  [F] Frequency-weighted Stage-2 -- pool sampling honours per-feature weights
      (Sec 3.5.1) when supplied by the caller.
  [G] Ablation switches        -- `use_qs`, `use_qq`, `use_cache` turn each
      mechanism off independently for the ablation study.

NOTE: scaling is performed *inside* each CV fold (fit on the train split only)
to avoid leakage. Callers must not pre-scale on data that includes an outer
test fold (see src/evaluation.py for the evaluation protocol).
"""

from __future__ import annotations

import time
import warnings
from typing import List, Optional, Sequence, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")


def _roc_auc(y_true, proba) -> float:
    try:
        return float(roc_auc_score(y_true, proba))
    except ValueError:
        return 0.5


class QSQFS:
    """Quorum Sensing & Quorum Quenching wrapper feature selector."""

    def __init__(
        self,
        n_colonies: int = 50,
        max_iter_stage1: int = 30,
        max_iter_stage2: int = 70,
        alpha: float = 0.85,
        max_frac: float = 0.50,
        min_frac: float = 0.05,
        w_AI: float = 0.50,
        delta1: float = 0.97,
        delta2: float = 0.95,
        strong_thresh1: Optional[float] = None,   # None -> dynamic mean(F)
        weak_thresh1: float = 0.30,
        strong_thresh2: Optional[float] = None,   # None -> dynamic 85th pct
        weak_thresh2: Optional[float] = None,     # None -> dynamic mean(F)
        age_limit1: int = 5,                       # kept for config compatibility
        age_limit2: int = 10,                      # (cache is now pure memoisation)
        stagnation_window: int = 15,
        diversity_thresh: float = 0.05,
        rho: float = 0.80,
        mutation_rate1: float = 0.15,
        mutation_rate2: float = 0.25,
        injection_mutation_rate: float = 0.40,
        injection_w_AI: float = 0.20,
        recovery_iters: int = 5,
        ema_beta: float = 0.70,
        k_nn: int = 3,
        cv_folds: int = 10,
        fitness_metric: str = "accuracy",   # "accuracy" | "auc" | "balanced"
        random_state: int = 42,
        verbose: bool = True,
        # --- ablation switches ---
        use_qs: bool = True,
        use_qq: bool = True,
        use_cache: bool = True,
        elitism: bool = True,
    ) -> None:
        self.n_colonies = n_colonies
        self.max_iter_stage1 = max_iter_stage1
        self.max_iter_stage2 = max_iter_stage2
        self.alpha = alpha
        self.max_frac = max_frac
        self.min_frac = min_frac
        self.w_AI = w_AI
        self._base_w_AI = w_AI
        self.delta1 = delta1
        self.delta2 = delta2
        self.strong_thresh1 = strong_thresh1
        self.weak_thresh1 = weak_thresh1
        self.strong_thresh2 = strong_thresh2
        self.weak_thresh2 = weak_thresh2
        self.age_limit1 = age_limit1
        self.age_limit2 = age_limit2
        self.stagnation_window = stagnation_window
        self.diversity_thresh = diversity_thresh
        self.rho = rho
        self.mutation_rate1 = mutation_rate1
        self.mutation_rate2 = mutation_rate2
        self.injection_mutation_rate = injection_mutation_rate
        self.injection_w_AI = injection_w_AI
        self.recovery_iters = recovery_iters
        self.ema_beta = ema_beta
        self.k_nn = k_nn
        self.cv_folds = cv_folds
        self.fitness_metric = fitness_metric
        self.random_state = random_state
        self.verbose = verbose
        self.use_qs = use_qs
        self.use_qq = use_qq
        self.use_cache = use_cache
        self.elitism = elitism

        # State
        self.X: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None
        self.d: int = 0
        self.n_min: int = 1
        self.n_max: int = 0
        self.population: List[np.ndarray] = []
        self._data_key: tuple = ()
        self.fitness: np.ndarray = np.array([])          # penalised (selection) fitness
        self.global_best_colony: Optional[np.ndarray] = None
        self.global_best_fitness: float = -np.inf
        self.archive: dict = {}
        self._raw_cache: dict = {}                       # pure memoiser: key -> raw fitness
        self.n_evals: int = 0                            # true KNN-CV evaluations performed
        self.ai_hist: Optional[np.ndarray] = None
        self.stage: int = 1
        self.iteration: int = 0
        self.injection_active: bool = False
        self._recovery_left: int = 0
        self.convergence_history: List[float] = []
        self.feature_count_history: List[int] = []
        self.stage1_boundary: int = 0
        self.runtime: float = 0.0

    # ------------------------------------------------------------------ #
    # Fitness (Eq 3.1) with optional pure memoisation                    #
    # ------------------------------------------------------------------ #
    def _raw_fitness(self, colony: np.ndarray) -> float:
        # The key binds the mask to the partition it was scored on, so a cached
        # value is only ever reused for the data that produced it.
        key = (self._data_key, tuple(int(b) for b in colony))
        if self.use_cache and key in self._raw_cache:
            return self._raw_cache[key]

        selected = np.where(colony == 1)[0]
        if len(selected) == 0:
            fit = 0.5
        else:
            X_sel = self.X[:, selected]
            knn = KNeighborsClassifier(n_neighbors=self.k_nn)
            skf = StratifiedKFold(
                n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
            )
            accs, aucs = [], []
            for tr, va in skf.split(X_sel, self.y):
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_sel[tr])   # fit on train fold only
                X_va = scaler.transform(X_sel[va])
                knn.fit(X_tr, self.y[tr])
                accs.append(knn.score(X_va, self.y[va]))
                if self.fitness_metric in ("auc", "balanced"):
                    proba = knn.predict_proba(X_va)
                    if proba.shape[1] > 1 and len(np.unique(self.y[va])) > 1:
                        aucs.append(_roc_auc(self.y[va], proba[:, 1]))
            mean_acc = float(np.mean(accs))
            # skill term honours the requested metric so imbalanced cohorts can
            # be optimised for ranking (AUC) rather than raw accuracy.
            if self.fitness_metric == "auc" and aucs:
                skill = float(np.mean(aucs))
            elif self.fitness_metric == "balanced" and aucs:
                skill = 0.5 * mean_acc + 0.5 * float(np.mean(aucs))
            else:
                skill = mean_acc
            parsimony = 1.0 - len(selected) / self.d
            fit = self.alpha * skill + (1.0 - self.alpha) * parsimony  # Eq 3.1
            self.n_evals += 1

        if self.use_cache:
            self._raw_cache[key] = fit
        return fit

    def _evaluate_population(self) -> None:
        raw = np.array([self._raw_fitness(c) for c in self.population])
        if self.use_qq and self.archive:
            penalties = np.array(
                [self.archive.get(tuple(int(b) for b in c), 0.0) for c in self.population]
            )
        else:
            penalties = np.zeros(len(self.population))
        self.fitness = raw - penalties

        # [A] true elite preservation: keep best-so-far, re-inject into population
        best_idx = int(np.argmax(self.fitness))
        if self.fitness[best_idx] > self.global_best_fitness:
            self.global_best_fitness = float(self.fitness[best_idx])
            self.global_best_colony = self.population[best_idx].copy()
        if self.elitism and self.global_best_colony is not None:
            worst_idx = int(np.argmin(self.fitness))
            if not np.array_equal(self.population[worst_idx], self.global_best_colony):
                self.population[worst_idx] = self.global_best_colony.copy()
                # reflect the elite's (cached) fitness in the vector
                self.fitness[worst_idx] = self._raw_fitness(self.global_best_colony)

    # ------------------------------------------------------------------ #
    # Colony classification (Sec 3.3.1 / 3.5)                            #
    # ------------------------------------------------------------------ #
    def _classify(self) -> Tuple[List[int], List[int], float]:
        if self.stage == 1:
            theta_s = (
                self.strong_thresh1 if self.strong_thresh1 is not None
                else float(np.mean(self.fitness))
            )
            theta_w = self.weak_thresh1
        else:
            theta_s = (
                self.strong_thresh2 if self.strong_thresh2 is not None
                else float(np.percentile(self.fitness, 85))
            )
            theta_w = (
                self.weak_thresh2 if self.weak_thresh2 is not None
                else float(np.mean(self.fitness))
            )
        strong = [i for i, f in enumerate(self.fitness) if f >= theta_s]
        weak = [i for i, f in enumerate(self.fitness) if f < theta_w]
        return strong, weak, theta_s

    # ------------------------------------------------------------------ #
    # Autoinducer scoring (Eq 3.5-3.7a)                                  #
    # ------------------------------------------------------------------ #
    def _compute_ai(self, strong_idx: List[int], rng: np.random.Generator) -> np.ndarray:
        if not self.use_qs:
            return np.zeros(self.d)                       # [G] QS disabled
        if not strong_idx:
            return rng.random(self.d) * 0.1

        sf = self.fitness[strong_idx]
        f_min, f_max = float(sf.min()), float(sf.max())
        norm = (sf - f_min) / (f_max - f_min + 1e-10)     # Eq 3.5

        new_ai = np.zeros(self.d)
        for idx, nf in zip(strong_idx, norm):
            colony = self.population[idx].astype(float)
            rnd = rng.random(self.d)
            contrib = colony * (nf * self.w_AI + (1.0 - self.w_AI) * rnd)  # Eq 3.6
            new_ai = np.maximum(new_ai, contrib)          # Eq 3.7

        if self.ai_hist is None:                          # Eq 3.7a EMA
            self.ai_hist = new_ai.copy()
        else:
            self.ai_hist = self.ema_beta * new_ai + (1.0 - self.ema_beta) * self.ai_hist
        return self.ai_hist

    # ------------------------------------------------------------------ #
    # Three-way mutation (Eq 3.10-3.12) + injection background mutation  #
    # ------------------------------------------------------------------ #
    def _mutate(self, colony: np.ndarray, ai: np.ndarray, strong_best: np.ndarray,
                colony_index: int) -> np.ndarray:
        mutant = colony.copy()
        f_w = self._raw_fitness(colony)
        # [E] deterministic per-colony rng
        seed = (self.random_state * 1_000_003
                + self.stage * 100_003
                + self.iteration * 1009
                + colony_index)
        rng = np.random.default_rng(seed % (2**32))

        base_mut = self.mutation_rate1 if self.stage == 1 else self.mutation_rate2
        mut_rate = self.injection_mutation_rate if self.injection_active else base_mut

        U1 = rng.random(self.d)
        U2 = rng.random(self.d)
        U0 = rng.random(self.d)
        for j in range(self.d):
            if U1[j] < ai[j]:                 # Eq 3.10 inherit (exploit)
                mutant[j] = strong_best[j]
            elif U2[j] < f_w * (self.w_AI if self.use_qs else 0.0):   # Eq 3.11 retain
                pass
            else:                             # Eq 3.12 explore
                mutant[j] = int(rng.integers(0, 2))
            # background mutation overlay (thesis Sec 3.4.3 mech 1; active mainly on injection)
            if U0[j] < mut_rate:
                mutant[j] = int(rng.integers(0, 2))

        return self._repair(mutant, rng)

    # ------------------------------------------------------------------ #
    # QQ archive (Eq 3.8-3.9) using RAW fitness                          #
    # ------------------------------------------------------------------ #
    def _update_archive(self, weak_idx: List[int], theta_s: float) -> None:
        if not self.use_qq:
            return
        delta = self.delta1 if self.stage == 1 else self.delta2
        for key in list(self.archive.keys()):             # Eq 3.9 decay
            self.archive[key] *= delta
            if self.archive[key] < 1e-3:
                del self.archive[key]
        for idx in weak_idx:                              # Eq 3.8 (RAW fitness -> [D])
            colony = self.population[idx]
            raw = self._raw_fitness(colony)
            supp = max(0.0, theta_s - raw)
            key = tuple(int(b) for b in colony)
            if key not in self.archive or self.archive[key] < supp:
                self.archive[key] = supp

    # ------------------------------------------------------------------ #
    # Stagnation detection + diversity injection (Sec 3.4.3)             #
    # ------------------------------------------------------------------ #
    def _check_stagnation(self) -> None:
        hist = self.convergence_history
        if len(hist) < self.stagnation_window:
            return
        window = hist[-self.stagnation_window:]
        stagnating = window[-1] <= window[0] + 1e-9

        if stagnating and not self.injection_active:
            if self.verbose:
                print(f"  [Stagnation] iter={self.iteration} -> inject diversity")
            self.injection_active = True
            self._recovery_left = self.recovery_iters
            self.w_AI = self.injection_w_AI               # mech 2
            self._replace_bottom_quartile()               # mech 3
        elif self.injection_active:
            self._recovery_left -= 1
            if not stagnating or self._recovery_left <= 0:
                self.w_AI = self._base_w_AI
                self.injection_active = False
                if self.verbose:
                    print("  [Stagnation] recovery complete")

    def _replace_bottom_quartile(self) -> None:
        n_replace = max(1, int(0.25 * self.n_colonies))
        order = np.argsort(self.fitness)
        rng = np.random.default_rng(self.random_state + 7919 * self.iteration)
        for idx in order[:n_replace]:
            if self.elitism and np.array_equal(self.population[idx], self.global_best_colony):
                continue                                  # never overwrite the elite
            self.population[idx] = self._repair(rng.integers(0, 2, self.d), rng)

    # ------------------------------------------------------------------ #
    # Feasibility repair (Eq 3.2)                                        #
    # ------------------------------------------------------------------ #
    def _repair(self, col: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        col = col.astype(int)
        # Any bit outside the active pool is cleared before the cardinality
        # repair, making the pool a binding constraint on every candidate.
        if self.pool_mask is not None:
            col = col * self.pool_mask
        cur = int(col.sum())
        if cur < self.n_min:
            zeros = np.where(col == 0)[0]
            if self.pool_mask is not None:
                zeros = np.array([z for z in zeros if self.pool_mask[z] == 1], dtype=int)
            need = self.n_min - cur
            if len(zeros) >= need:
                col[rng.choice(zeros, need, replace=False)] = 1
        elif cur > self.n_max:
            ones = np.where(col == 1)[0]
            surplus = cur - self.n_max
            if len(ones) >= surplus:
                col[rng.choice(ones, surplus, replace=False)] = 0
        return col

    def _make_colony(self, rng: np.random.Generator) -> np.ndarray:
        return self._repair(rng.integers(0, 2, self.d), rng)

    def _should_switch(self) -> bool:
        if self.iteration >= self.max_iter_stage1:
            return True
        return float(np.std(self.fitness)) < self.diversity_thresh

    # ------------------------------------------------------------------ #
    # Main fit                                                           #
    # ------------------------------------------------------------------ #
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_pool: Optional[Sequence[int]] = None,
        pool_weights: Optional[Sequence[float]] = None,
    ) -> "QSQFS":
        t0 = time.time()
        self.X, self.y = np.asarray(X, dtype=float), np.asarray(y).astype(int)
        self._data_key = (int(self.X.shape[0]), int(self.X.shape[1]),
                          float(np.nansum(self.X)), int(self.y.sum()))
        self.d = self.X.shape[1]
        self.n_min = max(1, int(np.ceil(self.d * self.min_frac)))
        self.n_max = max(self.n_min + 1, int(np.ceil(self.d * self.max_frac)))
        self.pool_mask = None
        self.convergence_history, self.feature_count_history = [], []
        self.stage1_boundary = 0
        self.stage, self.iteration = 1, 0
        self.archive, self._raw_cache, self.n_evals = {}, {}, 0
        self.ai_hist = None
        self.injection_active = False
        self.w_AI = self._base_w_AI
        self.global_best_colony, self.global_best_fitness = None, -np.inf

        rng = np.random.default_rng(self.random_state)
        strong_idx: List[int] = []

        # ---- Stage 1 ----
        self.population = [self._make_colony(rng) for _ in range(self.n_colonies)]
        self._evaluate_population()
        self.convergence_history.append(self.global_best_fitness)
        self.feature_count_history.append(int(self.global_best_colony.sum()))
        if self.verbose:
            print(f"Stage 1 | Iter   0 | Best {self.global_best_fitness:.4f}")

        while self.iteration < self.max_iter_stage1:
            self.iteration += 1
            ai_rng = np.random.default_rng(self.random_state + 31 * self.iteration)
            strong_idx, weak_idx, theta_s = self._classify()
            ai = self._compute_ai(strong_idx, ai_rng)
            strong_best = (
                self.population[strong_idx[int(np.argmax(self.fitness[strong_idx]))]]
                if strong_idx else self.global_best_colony
            )
            weak_set = set(weak_idx)
            self.population = [
                self._mutate(c, ai, strong_best, i) if i in weak_set else c.copy()
                for i, c in enumerate(self.population)
            ]
            self._update_archive(weak_idx, theta_s)
            self._evaluate_population()
            self.convergence_history.append(self.global_best_fitness)
            self.feature_count_history.append(int(self.global_best_colony.sum()))
            self._check_stagnation()
            if self.verbose and self.iteration % 5 == 0:
                print(f"Stage 1 | Iter {self.iteration:3d} | Best {self.global_best_fitness:.4f} "
                      f"| Feat {int(self.global_best_colony.sum())} | Arch {len(self.archive)} "
                      f"| Evals {self.n_evals}")
            if self._should_switch():
                break

        self.stage1_boundary = len(self.convergence_history) - 1

        if self.max_iter_stage2 <= 0:
            self.runtime = time.time() - t0
            return self

        # ---- Stage 2 ----
        self.stage, self.iteration = 2, 0
        self.archive, self.ai_hist = {}, None
        self.injection_active = False
        self.w_AI = self._base_w_AI

        if feature_pool is not None and len(feature_pool) > 0:
            pool = list(feature_pool)
            weights = (np.asarray(pool_weights, dtype=float)
                       if pool_weights is not None and len(pool_weights) == len(pool) else None)
        else:
            freq = np.zeros(self.d)
            if strong_idx:
                for idx in strong_idx:
                    freq += self.population[idx]
                freq /= len(strong_idx)
            else:
                freq = np.ones(self.d) * 0.5
            pool = np.where(freq > 0.3)[0].tolist() or list(range(self.d))
            weights = freq[pool] if len(pool) else None

        # Stage 2 searches strictly within the Stage-1 candidate pool. The pool
        # is bound for the duration of the stage, and the incumbent global best
        # -- carried forward from a Stage-1 search over all d features -- is
        # projected onto the pool and re-scored so that elitism and the guided
        # blend operate inside the same feasible region as the population.
        self.pool_mask = np.zeros(self.d, dtype=int)
        self.pool_mask[np.asarray(pool, dtype=int)] = 1
        if self.global_best_colony is not None:
            projected = self._repair(self.global_best_colony.copy(),
                                     np.random.default_rng(self.random_state))
            self.global_best_colony = projected
            self.global_best_fitness = self._raw_fitness(projected)

        if weights is not None and weights.sum() > 0:     # [F] frequency-weighted init
            weights = weights / weights.sum()

        if self.verbose:
            print(f"\n--- Stage 2 | pool={len(pool)} | weighted={'yes' if weights is not None else 'no'} ---")

        for i in range(self.n_colonies):
            n_draw = min(max(1, int(np.ceil(len(pool) * 0.30))), len(pool))
            chosen = rng.choice(pool, size=n_draw, replace=False, p=weights)
            col = np.zeros(self.d, dtype=int)
            col[chosen] = 1
            self.population[i] = self._repair(col, rng)

        self._evaluate_population()
        self.convergence_history.append(self.global_best_fitness)
        self.feature_count_history.append(int(self.global_best_colony.sum()))

        while self.iteration < self.max_iter_stage2:
            self.iteration += 1
            ai_rng = np.random.default_rng(self.random_state + 53 * self.iteration)
            strong_idx, weak_idx, theta_s = self._classify()
            ai = self._compute_ai(strong_idx, ai_rng)
            strong_best = (
                self.population[strong_idx[int(np.argmax(self.fitness[strong_idx]))]]
                if strong_idx else self.global_best_colony
            )
            weak_set = set(weak_idx)
            new_pop = []
            for i, c in enumerate(self.population):
                if i in weak_set:
                    m = self._mutate(c, ai, strong_best, i)
                    # guided refinement: per-position blend toward global best (Sec 3.5.2)
                    gap = max(0.0, self.global_best_fitness - float(self.fitness[i]))
                    grng = np.random.default_rng(
                        (self.random_state + 97 * self.iteration + i) % (2**32))
                    blend = grng.random(self.d) < gap * self.rho
                    m[blend] = self.global_best_colony[blend]
                    new_pop.append(self._repair(m, grng))
                else:
                    new_pop.append(c.copy())
            self.population = new_pop
            self._update_archive(weak_idx, theta_s)
            self._evaluate_population()
            self.convergence_history.append(self.global_best_fitness)
            self.feature_count_history.append(int(self.global_best_colony.sum()))
            self._check_stagnation()
            if self.verbose and self.iteration % 10 == 0:
                print(f"Stage 2 | Iter {self.iteration:3d} | Best {self.global_best_fitness:.4f} "
                      f"| Feat {int(self.global_best_colony.sum())} | Evals {self.n_evals}")
            if float(np.std(self.fitness)) < 0.005:
                if self.verbose:
                    print("  Early stop: population converged.")
                break

        self.runtime = time.time() - t0
        if self.verbose:
            print(f"\nQSQ-FS done in {self.runtime:.2f}s | best {self.global_best_fitness:.4f} "
                  f"| {int(self.global_best_colony.sum())} features | {self.n_evals} KNN-CV evals")
        return self

    # ------------------------------------------------------------------ #
    # Accessors                                                          #
    # ------------------------------------------------------------------ #
    def saturated(self, tol: int = 2) -> bool:
        """True if the returned subset sits on the cardinality ceiling.

        A subset at `n_max` reflects the cardinality constraint rather than the
        search: it indicates the parsimony term is too weakly weighted to offset
        the skill gain from retaining a feature. Callers should treat such a run
        as inconclusive and re-weight `alpha` before reporting the subset.
        """
        if self.global_best_colony is None:
            return False
        return int(self.global_best_colony.sum()) >= self.n_max - tol

    def get_selected_features(self) -> np.ndarray:
        if self.global_best_colony is None:
            return np.array([], dtype=int)
        return np.where(self.global_best_colony == 1)[0]

    def get_best_fitness(self) -> float:
        return self.global_best_fitness

    def get_convergence(self) -> List[float]:
        return self.convergence_history

    def get_feature_counts(self) -> List[int]:
        return self.feature_count_history
