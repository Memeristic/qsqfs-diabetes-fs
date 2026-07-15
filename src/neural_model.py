"""
src/neural_model.py
===================
MLP evaluator for the final diabetes classifier.

Design notes:
  * Early stopping restores the best-validation weights.
  * L2 regularisation is controlled by a dedicated `l2_alpha` parameter,
    separate from the dropout rate.
  * `torch.manual_seed` is set for reproducibility.
  * The API is split into `train_mlp(...) -> model` and
    `evaluate_model(model, ...)` so the caller controls the train/test boundary
    (see src/evaluation.py). `train_evaluate_holdout(...)` provides a
    single-split convenience wrapper.

Torch is OPTIONAL. If unavailable the code uses a regularised sklearn
MLPClassifier, which is also the default on Streamlit Cloud (lighter cold start).
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from loguru import logger
except ImportError:                       # pragma: no cover
    import logging
    logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# --------------------------------------------------------------------------- #
# Torch model                                                                 #
# --------------------------------------------------------------------------- #
if TORCH_AVAILABLE:
    class _TorchMLP(nn.Module):
        def __init__(self, input_dim, hidden_layers, dropout):
            super().__init__()
            layers, prev = [], input_dim
            for h in hidden_layers:
                layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
                prev = h
            layers.append(nn.Linear(prev, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return torch.sigmoid(self.net(x))


class _SklearnWrapper:
    """Uniform predict_proba interface around a fitted sklearn MLP."""
    def __init__(self, model):
        self.model = model

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class _TorchWrapper:
    """Uniform predict_proba interface around a fitted torch MLP."""
    def __init__(self, model, device):
        self.model, self.device = model, device

    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            p = self.model(torch.FloatTensor(np.asarray(X)).to(self.device)).cpu().numpy().flatten()
        return np.column_stack([1 - p, p])


# --------------------------------------------------------------------------- #
# Training                                                                    #
# --------------------------------------------------------------------------- #
def train_mlp(X_train, y_train, config: dict, random_state: int = 42):
    """
    Train an MLP on the (already-scaled) training data and return a wrapped model
    exposing `predict_proba`. An internal train/val split drives early stopping;
    the test set is never touched here.
    """
    if TORCH_AVAILABLE:
        return _train_torch(X_train, y_train, config, random_state)
    return _train_sklearn(X_train, y_train, config, random_state)


def _train_torch(X_train, y_train, config, random_state):
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # internal validation split for early stopping
    Xtr, Xval, ytr, yval = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=random_state)

    model = _TorchMLP(
        input_dim=Xtr.shape[1],
        hidden_layers=config.get("hidden_layers", [64, 32]),
        dropout=config.get("dropout_rate", 0.3),
    ).to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.get("learning_rate", 1e-3),
        weight_decay=config.get("l2_alpha", 1e-4),
    )
    criterion = nn.BCELoss()
    bs = config.get("batch_size", 32)
    patience = config.get("early_stopping_patience", 15)

    tr_loader = DataLoader(
        TensorDataset(torch.FloatTensor(Xtr), torch.FloatTensor(ytr).view(-1, 1)),
        batch_size=bs, shuffle=True, drop_last=len(Xtr) > bs)
    val_x = torch.FloatTensor(Xval).to(device)
    val_y = torch.FloatTensor(yval).view(-1, 1).to(device)

    best_val, best_state, wait = np.inf, None, 0
    for epoch in range(config.get("max_epochs", 200)):
        model.train()
        for bx, by in tr_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            criterion(model(bx), by).backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            vl = criterion(model(val_x), val_y).item()
        if vl < best_val - 1e-5:
            best_val, wait = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}  # checkpoint best
        else:
            wait += 1
            if wait >= patience:
                logger.info(f"Early stopping at epoch {epoch} (best val {best_val:.4f})")
                break
    if best_state is not None:
        model.load_state_dict(best_state)                # RESTORE best weights
    return _TorchWrapper(model, device)


def _train_sklearn(X_train, y_train, config, random_state):
    from sklearn.neural_network import MLPClassifier
    mlp = MLPClassifier(
        hidden_layer_sizes=tuple(config.get("hidden_layers", [64, 32])),
        activation="relu",
        solver="adam",
        alpha=config.get("l2_alpha", 1e-4),               # proper L2 (NOT dropout)
        learning_rate_init=config.get("learning_rate", 1e-3),
        max_iter=config.get("max_epochs", 200),
        early_stopping=True,                              # restores best automatically
        validation_fraction=0.2,
        n_iter_no_change=config.get("early_stopping_patience", 15),
        random_state=random_state,
    )
    mlp.fit(X_train, y_train)
    return _SklearnWrapper(mlp)


# --------------------------------------------------------------------------- #
# Evaluation                                                                  #
# --------------------------------------------------------------------------- #
def evaluate_model(model, X_test, y_test) -> Dict[str, float]:
    proba = model.predict_proba(np.asarray(X_test))[:, 1]
    pred = (proba >= 0.5).astype(int)
    auc = roc_auc_score(y_test, proba) if len(np.unique(y_test)) > 1 else float("nan")
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "auc": float(auc),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "proba": proba,
        "y_test": np.asarray(y_test),
    }


def train_evaluate_holdout(
    X, y, config: dict, random_state: int = 42, test_size: float = 0.2,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Convenience single-split trainer/evaluator (scales train-only, no leakage).
    Returns (auc, accuracy, proba, y_test). For unbiased reporting prefer
    src.evaluation.nested_cv_evaluate.
    """
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state)
    scaler = StandardScaler().fit(X_tr)
    model = train_mlp(scaler.transform(X_tr), y_tr, config, random_state)
    m = evaluate_model(model, scaler.transform(X_te), y_te)
    return m["auc"], m["accuracy"], m["proba"], m["y_test"]
