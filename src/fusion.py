"""
src/fusion.py
=============
Multimodal fusion for diabetes prediction.

Three fusion strategies (thesis Sec 2.2 / 2.5, "hybrid fusion"):

  * FEATURE-LEVEL ("early"): each modality is encoded into a low-dimensional
    latent representation by its own encoder; the latent vectors are
    concatenated into a shared space and a single classifier head is trained on
    the joint representation. This is the "shared latent space" of the proposal.
  * DECISION-LEVEL ("late"): a separate classifier is trained per modality and
    the per-modality probabilities are combined by a weighted average, where the
    weight of each modality is its own validation performance. This is the
    "ensemble of modality-specific models" of the proposal.
  * HYBRID: the feature-level joint probability and the decision-level ensemble
    probability are averaged, balancing generalisation (shared representation)
    and robustness (independent per-modality views).

The encoders and heads are deliberately light and dependency-free (an
autoencoder-style bottleneck via a small MLP hidden layer) so the whole thing
runs on base scikit-learn -- no torch required -- and therefore deploys on
Streamlit Community Cloud unchanged. If torch is present the caller may pass
`backend="mlp"`; the sklearn path is the default.

Every fit uses only the data handed to it. In the nested-CV pipeline the fusion
model is fitted inside the outer training fold, so the fusion step is leak-free
in exactly the same way as the feature selection.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

try:
    from loguru import logger
except ImportError:                       # pragma: no cover
    import logging
    logger = logging.getLogger(__name__)


def _safe_auc(y_true: np.ndarray, proba: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.5
    try:
        return float(roc_auc_score(y_true, proba))
    except ValueError:
        return 0.5


class _ModalityEncoder:
    """Encodes one modality into a fixed-width latent representation.

    Uses an MLP with a single bottleneck hidden layer trained to predict the
    label; the hidden activations are the latent code. This gives a supervised,
    discriminative embedding (a lightweight, dependency-free stand-in for a
    modality-specific representation network) rather than an unsupervised PCA.
    """

    def __init__(self, latent_dim: int = 8, l2_alpha: float = 1e-3,
                 max_iter: int = 300, random_state: int = 42):
        self.latent_dim = latent_dim
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.net = MLPClassifier(
            hidden_layer_sizes=(max(2, latent_dim),),
            activation="relu",
            solver="adam",
            alpha=l2_alpha,
            max_iter=max_iter,
            early_stopping=False,
            random_state=random_state,
        )
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_ModalityEncoder":
        Xs = self.scaler.fit_transform(X)
        self.net.fit(Xs, y)
        self._fitted = True
        return self

    def _hidden(self, X: np.ndarray) -> np.ndarray:
        # forward pass up to (not including) the output layer -> latent code
        Xs = self.scaler.transform(X)
        a = Xs
        # sklearn MLP stores coefs_/intercepts_ per layer; the last is the output
        for i in range(len(self.net.coefs_) - 1):
            a = a @ self.net.coefs_[i] + self.net.intercepts_[i]
            a = np.maximum(a, 0.0)          # ReLU
        return a

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("encoder not fitted")
        return self._hidden(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler.transform(X)
        return self.net.predict_proba(Xs)[:, 1]


class MultimodalFusion(BaseEstimator, ClassifierMixin):
    """Feature-, decision-, or hybrid-level multimodal fusion classifier.

    Parameters
    ----------
    modality_map : {modality_name: [column indices into X]}
        Which columns belong to which modality. Columns not covered by any
        modality are grouped into an implicit "other" modality so no signal is
        silently dropped.
    strategy : "feature" | "decision" | "hybrid"
    latent_dim : width of each modality's latent code (feature-level fusion)
    class_weight : "balanced" | None  -- passed to the fusion head so a minority
        (diabetic) class is not swamped; directly targets the low-sensitivity
        problem on imbalanced cohorts.
    """

    def __init__(
        self,
        modality_map: Optional[Dict[str, Sequence[int]]] = None,
        strategy: str = "hybrid",
        latent_dim: int = 8,
        hidden_layers: Sequence[int] = (32,),
        l2_alpha: float = 1e-3,
        class_weight: Optional[str] = "balanced",
        max_iter: int = 300,
        random_state: int = 42,
    ):
        self.modality_map = modality_map
        self.strategy = strategy
        self.latent_dim = latent_dim
        self.hidden_layers = hidden_layers
        self.l2_alpha = l2_alpha
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.random_state = random_state

    # ------------------------------------------------------------------ #
    def _resolve_modalities(self, n_features: int) -> Dict[str, List[int]]:
        if not self.modality_map:
            return {"all": list(range(n_features))}
        mm = {k: [int(c) for c in v if 0 <= int(c) < n_features]
              for k, v in self.modality_map.items()}
        mm = {k: v for k, v in mm.items() if v}
        covered = {c for cols in mm.values() for c in cols}
        rest = [c for c in range(n_features) if c not in covered]
        if rest:
            mm["other"] = rest
        return mm or {"all": list(range(n_features))}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MultimodalFusion":
        X = np.asarray(X, float)
        y = np.asarray(y).astype(int)
        self.mm_ = self._resolve_modalities(X.shape[1])
        cw = self.class_weight

        # ---- decision-level: one classifier per modality ----
        self.mod_clfs_: Dict[str, MLPClassifier] = {}
        self.mod_scalers_: Dict[str, StandardScaler] = {}
        self.mod_weights_: Dict[str, float] = {}

        # a small internal validation split to weight the modality voters and
        # to guard against a modality that cannot separate the classes at all
        if len(np.unique(y)) > 1 and np.min(np.bincount(y)) >= 4:
            Xtr, Xval, ytr, yval = train_test_split(
                X, y, test_size=0.25, stratify=y, random_state=self.random_state)
        else:
            Xtr, Xval, ytr, yval = X, X, y, y

        for name, cols in self.mm_.items():
            sc = StandardScaler().fit(Xtr[:, cols])
            clf = MLPClassifier(
                hidden_layer_sizes=tuple(self.hidden_layers),
                activation="relu", solver="adam", alpha=self.l2_alpha,
                max_iter=self.max_iter, early_stopping=False,
                random_state=self.random_state,
            )
            clf.fit(sc.transform(Xtr[:, cols]), ytr)
            self.mod_clfs_[name] = clf
            self.mod_scalers_[name] = sc
            vp = clf.predict_proba(sc.transform(Xval[:, cols]))[:, 1]
            self.mod_weights_[name] = max(_safe_auc(yval, vp) - 0.5, 1e-3)

        # ---- feature-level: encoders -> shared latent -> joint head ----
        self.encoders_: Dict[str, _ModalityEncoder] = {}
        latent_parts = []
        for name, cols in self.mm_.items():
            enc = _ModalityEncoder(latent_dim=self.latent_dim, l2_alpha=self.l2_alpha,
                                   max_iter=self.max_iter, random_state=self.random_state)
            enc.fit(X[:, cols], y)
            self.encoders_[name] = enc
            latent_parts.append(enc.transform(X[:, cols]))
        Z = np.hstack(latent_parts) if latent_parts else X
        self.joint_scaler_ = StandardScaler().fit(Z)
        # class-weighting is not natively supported by MLPClassifier, so we
        # oversample the minority class in the joint head instead -- same effect.
        Zb, yb = self._balance(self.joint_scaler_.transform(Z), y, cw)
        self.joint_head_ = MLPClassifier(
            hidden_layer_sizes=tuple(self.hidden_layers),
            activation="relu", solver="adam", alpha=self.l2_alpha,
            max_iter=self.max_iter, early_stopping=False,
            random_state=self.random_state,
        ).fit(Zb, yb)

        self.classes_ = np.array([0, 1])
        return self

    def _balance(self, X, y, class_weight):
        if class_weight != "balanced":
            return X, y
        counts = np.bincount(y, minlength=2)
        if counts.min() == 0 or counts[0] == counts[1]:
            return X, y
        minority = int(np.argmin(counts))
        need = counts.max() - counts.min()
        idx = np.where(y == minority)[0]
        rng = np.random.default_rng(self.random_state)
        extra = rng.choice(idx, size=need, replace=True)
        return np.vstack([X, X[extra]]), np.concatenate([y, y[extra]])

    # ------------------------------------------------------------------ #
    def _decision_proba(self, X: np.ndarray) -> np.ndarray:
        wsum = sum(self.mod_weights_.values()) or 1.0
        out = np.zeros(X.shape[0])
        for name, cols in self.mm_.items():
            p = self.mod_clfs_[name].predict_proba(
                self.mod_scalers_[name].transform(X[:, cols]))[:, 1]
            out += (self.mod_weights_[name] / wsum) * p
        return out

    def _feature_proba(self, X: np.ndarray) -> np.ndarray:
        parts = [self.encoders_[name].transform(X[:, cols])
                 for name, cols in self.mm_.items()]
        Z = np.hstack(parts) if parts else X
        return self.joint_head_.predict_proba(self.joint_scaler_.transform(Z))[:, 1]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, float)
        if self.strategy == "feature":
            p = self._feature_proba(X)
        elif self.strategy == "decision":
            p = self._decision_proba(X)
        else:                                 # hybrid
            p = 0.5 * (self._feature_proba(X) + self._decision_proba(X))
        return np.column_stack([1 - p, p])

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)


def score_fusion_strategies(X, y, modality_map, fold_selected, n_outer=5,
                            strategies=("feature", "decision", "hybrid"),
                            random_state=42):
    """Compare fusion strategies on FIXED per-fold selections (no re-selection).

    `fold_selected[k]` are the global feature indices chosen on outer-fold k's
    training split (e.g. reused from a prior nested_cv_evaluate run). For each
    strategy the fusion model is refit on each training fold and scored on the
    matching test fold, so this is still leak-free and directly comparable to
    the primary run -- it just avoids paying for feature selection again.
    Returns {strategy: {auc, accuracy, f1, sensitivity}}.
    """
    from sklearn.model_selection import StratifiedKFold
    X = np.asarray(X, float)
    y = np.asarray(y).astype(int)
    skf = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=random_state)
    folds = list(skf.split(X, y))
    out = {}
    for strat in strategies:
        aucs, accs, f1s, oof_p, oof_t = [], [], [], [], []
        for k, (tr, te) in enumerate(folds):
            sel = list(fold_selected[k]) if k < len(fold_selected) else list(range(X.shape[1]))
            if not sel:
                sel = list(range(X.shape[1]))
            local_mm = {}
            pos = {g: i for i, g in enumerate(sel)}
            for name, cols in (modality_map or {}).items():
                loc = [pos[g] for g in cols if g in pos]
                if loc:
                    local_mm[name] = loc
            model = MultimodalFusion(modality_map=local_mm or None, strategy=strat,
                                     class_weight="balanced",
                                     random_state=random_state + k).fit(X[np.ix_(tr, sel)], y[tr])
            p = model.predict_proba(X[np.ix_(te, sel)])[:, 1]
            pred = (p >= 0.5).astype(int)
            aucs.append(_safe_auc(y[te], p))
            accs.append(float(np.mean(pred == y[te])))
            from sklearn.metrics import f1_score
            f1s.append(float(f1_score(y[te], pred, zero_division=0)))
            oof_p.append(p); oof_t.append(y[te])
        oof_p = np.concatenate(oof_p); oof_t = np.concatenate(oof_t)
        pred = (oof_p >= 0.5).astype(int)
        tp = int(np.sum((pred == 1) & (oof_t == 1)))
        fn = int(np.sum((pred == 0) & (oof_t == 1)))
        out[strat] = {"auc": float(np.mean(aucs)), "accuracy": float(np.mean(accs)),
                      "f1": float(np.mean(f1s)),
                      "sensitivity": tp / (tp + fn) if (tp + fn) else 0.0}
    return out


def tune_threshold(y_true: np.ndarray, proba: np.ndarray,
                   target: str = "f1", min_sensitivity: float = 0.0) -> float:
    """Pick a decision threshold on a validation set.

    target="f1"        -> maximise F1 (good default for imbalanced data)
    target="youden"    -> maximise Youden's J (sensitivity+specificity-1)
    min_sensitivity>0  -> among thresholds meeting that recall, pick the one
                          with the best specificity (screening-oriented).
    Returns a threshold in (0, 1). Falls back to 0.5 if nothing qualifies.
    """
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, float)
    grid = np.unique(np.clip(proba, 0.01, 0.99))
    if grid.size == 0:
        return 0.5
    best_t, best_score = 0.5, -np.inf
    for t in grid:
        pred = (proba >= t).astype(int)
        tp = int(np.sum((pred == 1) & (y_true == 1)))
        fp = int(np.sum((pred == 1) & (y_true == 0)))
        fn = int(np.sum((pred == 0) & (y_true == 1)))
        tn = int(np.sum((pred == 0) & (y_true == 0)))
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        if min_sensitivity > 0 and sens < min_sensitivity:
            continue
        if target == "youden":
            score = sens + spec - 1
        elif min_sensitivity > 0:
            score = spec
        else:
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            score = 2 * prec * sens / (prec + sens) if (prec + sens) else 0.0
        if score > best_score:
            best_score, best_t = score, float(t)
    return best_t
