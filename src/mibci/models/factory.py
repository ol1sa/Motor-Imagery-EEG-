"""One factory that returns a fit/predict estimator for any model name.

Classical models are sklearn pipelines; deep models are wrapped in TorchEstimator
so the CV loop can treat every model identically.
"""

from __future__ import annotations

from typing import Any

from ..config import Config
from .classical import build_classical
from .denoising_eegnet import build_denoising_eegnet
from .eegnet import build_eegnet
from .torch_utils import TorchEstimator

CLASSICAL = {"csp_lda", "csp_svm", "riemann_lr"}
DEEP_BUILDERS = {
    "eegnet": build_eegnet,
    "denoising_eegnet": build_denoising_eegnet,
}


def build_model(name: str, cfg: Config, n_classes: int,
                n_channels: int, n_times: int) -> Any:
    """Return a fresh, unfitted estimator with .fit/.predict."""
    if name in CLASSICAL:
        return build_classical(name, cfg.model, cfg.seed)
    if name in DEEP_BUILDERS:
        return TorchEstimator(
            build_fn=DEEP_BUILDERS[name],
            deep_cfg=cfg.model.deep,
            n_classes=n_classes,
            n_channels=n_channels,
            n_times=n_times,
            seed=cfg.seed,
        )
    raise ValueError(f"unknown model '{name}'")
