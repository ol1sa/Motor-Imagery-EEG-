"""Result figures: per-subject distribution and confusion matrices.

The per-subject distribution is the most important plot in the project: mean
accuracy hides that some subjects are near-perfect and others near chance ("BCI
illiteracy"). Showing the spread is part of being honest about what the model
can and cannot do.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..logging_utils import get_logger

log = get_logger()


def plot_per_subject_distribution(acc_matrix: pd.DataFrame, n_classes: int,
                                  out_path: str | Path) -> Path:
    """Box + jittered points of per-subject accuracy for each model."""
    models = list(acc_matrix.columns)
    data = [acc_matrix[m].dropna().to_numpy() for m in models]

    fig, ax = plt.subplots(figsize=(1.6 * len(models) + 2, 5))
    ax.boxplot(data, labels=models, showmeans=True)
    for i, d in enumerate(data, start=1):
        x = np.random.default_rng(0).normal(i, 0.05, size=len(d))
        ax.scatter(x, d, alpha=0.5, s=18, color="tab:blue")
    ax.axhline(1.0 / n_classes, color="red", ls="--", label=f"chance = {1/n_classes:.2f}")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Per-subject accuracy distribution (spread = BCI variability)")
    ax.legend()
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("saved per-subject distribution -> %s", out_path)
    return out_path


def plot_confusions(confusions: dict[str, tuple[np.ndarray, np.ndarray]],
                    class_names: list[str], out_path: str | Path) -> Path:
    """One normalised confusion matrix per model."""
    from sklearn.metrics import confusion_matrix

    models = list(confusions.keys())
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.2))
    axes = np.atleast_1d(axes)
    k = len(class_names)
    for ax, m in zip(axes, models):
        y_true, y_pred = confusions[m]
        cm = confusion_matrix(y_true, y_pred, labels=list(range(k)))
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
        im = ax.imshow(cm, vmin=0, vmax=1, cmap="Blues")
        ax.set_title(m, fontsize=9)
        ax.set_xticks(range(k)); ax.set_yticks(range(k))
        ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(class_names, fontsize=7)
        for i in range(k):
            for j in range(k):
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if cm[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=axes.tolist(), fraction=0.025)
    fig.suptitle("Confusion matrices (row-normalised)")
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("saved confusion matrices -> %s", out_path)
    return out_path
