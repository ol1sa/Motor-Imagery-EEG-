"""Riemannian-geometry feature pipeline.

Each epoch is summarised by its spatial covariance matrix. Covariance matrices
live on a curved (Riemannian) manifold of symmetric positive-definite matrices,
not in flat Euclidean space, so we project them to the tangent space at the
geometric mean — a principled linearisation — and classify the tangent vectors.
This family is a strong, near-parameter-free baseline that often beats CSP and
is notably more robust across sessions/subjects.
"""

from __future__ import annotations

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def build_riemann_lr(seed: int) -> Pipeline:
    """Covariances (OAS-regularised) -> tangent space -> logistic regression."""
    return Pipeline([
        # OAS shrinkage keeps covariance estimates well-conditioned when epochs
        # are short relative to the channel count.
        ("cov", Covariances(estimator="oas")),
        ("ts", TangentSpace(metric="riemann")),
        ("lr", LogisticRegression(max_iter=1000, random_state=seed)),
    ])
