"""Scoring helpers. Thin wrappers over sklearn so the CV loop stays readable."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(accuracy_score(y_true, y_pred))


def kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Cohen's kappa: accuracy corrected for chance agreement.

    Reported alongside accuracy because with k classes, chance is 1/k; kappa
    makes 'better than guessing' explicit and comparable across the binary and
    4-class experiments.
    """
    return float(cohen_kappa_score(y_true, y_pred))


def confusion(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
