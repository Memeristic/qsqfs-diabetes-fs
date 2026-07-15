"""
ui/captions.py — plain-language interpretation helpers
======================================================
Every metric and chart in the dashboard gets a one-line, non-technical
explanation aimed at a reader who is NOT a statistician. Keeping
these here (rather than inline) keeps ``app.py`` readable and makes the wording
easy to tune in one place.

All functions return a short plain string. None of them do any Streamlit calls,
so they're trivially unit-testable.
"""

from __future__ import annotations


def auc_meaning(auc: float) -> str:
    if auc >= 0.9:
        band = "excellent separation"
    elif auc >= 0.8:
        band = "good separation"
    elif auc >= 0.7:
        band = "fair separation"
    elif auc >= 0.6:
        band = "weak separation"
    else:
        band = "close to guessing"
    return (f"AUC {auc:.2f} — pick one diabetic and one non-diabetic patient at "
            f"random, and the model gives the diabetic the higher risk score about "
            f"{auc*100:.0f}% of the time ({band}). 0.5 is a coin-flip, 1.0 is perfect.")


def accuracy_meaning(acc: float, prevalence: float) -> str:
    baseline = max(prevalence, 1 - prevalence)
    verdict = ("better than" if acc > baseline + 0.02 else
               "about the same as" if abs(acc - baseline) <= 0.02 else
               "worse than")
    return (f"Accuracy {acc:.2f} — it labels {acc*100:.0f}% of patients correctly. "
            f"Always-guess-the-majority would score {baseline*100:.0f}%, so this is "
            f"{verdict} the naive baseline.")


def f1_meaning(f1: float) -> str:
    return (f"F1 {f1:.2f} — a single balance score between catching the diabetics "
            f"(recall) and not false-alarming on healthy patients (precision). "
            f"Higher is better; 1.0 is perfect.")


def prevalence_meaning(prev: float) -> str:
    return (f"{prev*100:.0f}% of patients in this cohort are diabetic. This matters "
            f"because accuracy alone can look high just by predicting the majority.")


def n_features_meaning(n: float, total: int) -> str:
    return (f"On average the method keeps {n:.0f} of the {total} available "
            f"measurements — fewer features means a simpler, easier-to-explain model.")


def selection_frequency_meaning() -> str:
    return ("How often each feature was chosen across the cross-validation folds. "
            "Features picked in most folds (bars past the dashed line) are the "
            "stable, trustworthy signals; ones picked rarely may be noise.")


def roc_meaning() -> str:
    return ("The ROC curve shows the trade-off between catching true diabetics "
            "(up) and raising false alarms (right) as you move the decision "
            "threshold. A curve hugging the top-left corner is good.")


def confusion_meaning() -> str:
    return ("Rows are the truth, columns are the prediction. The top-left and "
            "bottom-right cells are correct calls; the other two are mistakes "
            "(missed diabetics and false alarms).")


def convergence_meaning() -> str:
    return ("The search improving over time: each step tries new feature "
            "combinations and keeps the best. A curve that flattens out means the "
            "search has settled on a good subset.")


def comparison_meaning() -> str:
    return ("Each bar is a different feature-selection method scored under the "
            "exact same fair protocol. Taller is better. Error bars show "
            "fold-to-fold variability.")


def leakage_note() -> str:
    return ("These numbers come from nested cross-validation: features are chosen "
            "using only the training part of each split, then judged on unseen "
            "patients — so the score reflects real-world performance, not memorised "
            "answers.")
