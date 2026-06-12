"""Classical model pipelines: CSP+LDA, CSP+SVM, Riemann+LR.

All three are plain scikit-learn estimators (fit/predict), so the CV loop treats
them identically to the deep models, which expose the same interface.
"""

from __future__ import annotations

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from ..config import ModelConfig
from ..features.csp import build_csp
from ..features.riemann import build_riemann_lr


def build_csp_lda(cfg: ModelConfig) -> Pipeline:
    """The reference baseline. Solid, fast, interpretable."""
    return Pipeline([
        ("csp", build_csp(cfg.csp_components)),
        ("lda", LinearDiscriminantAnalysis()),
    ])


def build_csp_svm(cfg: ModelConfig, seed: int) -> Pipeline:
    """CSP features into an RBF-kernel SVM (non-linear decision boundary)."""
    return Pipeline([
        ("csp", build_csp(cfg.csp_components)),
        ("svm", SVC(kernel="rbf", C=cfg.svm_c, gamma=cfg.svm_gamma,
                    random_state=seed)),
    ])


def build_classical(name: str, cfg: ModelConfig, seed: int) -> Pipeline:
    if name == "csp_lda":
        return build_csp_lda(cfg)
    if name == "csp_svm":
        return build_csp_svm(cfg, seed)
    if name == "riemann_lr":
        return build_riemann_lr(seed)
    raise ValueError(f"unknown classical model '{name}'")
