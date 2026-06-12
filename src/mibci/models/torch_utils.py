"""Training harness that makes a PyTorch model look like an sklearn estimator.

The CV loop calls ``.fit(X, y)`` / ``.predict(X)`` on every model regardless of
family. This wrapper provides that interface for torch modules and centralises
the things that must be identical across deep models for a fair comparison:
  * deterministic seeding,
  * per-channel z-score standardisation fit on the TRAINING data only (no leak),
  * the Adam + cross-entropy training loop,
  * an optional unsupervised ``pretrain`` hook (used by the denoising variant).
"""

from __future__ import annotations

import random
from typing import Callable

import numpy as np
import torch
from torch import nn

from ..config import DeepConfig
from ..logging_utils import get_logger

log = get_logger()


def set_torch_seed(seed: int) -> None:
    """Seed every RNG that affects training so runs are reproducible.

    Note: exact bit-for-bit reproducibility on GPU also depends on
    deterministic cuDNN kernels; on CPU this is sufficient.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class _ChannelStandardizer:
    """Per-channel z-score using stats from training epochs only."""

    def fit(self, X: np.ndarray) -> "_ChannelStandardizer":
        # X: (N, C, T) -> mean/std per channel over epochs and time.
        self.mean_ = X.mean(axis=(0, 2), keepdims=True)
        self.std_ = X.std(axis=(0, 2), keepdims=True) + 1e-7
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_


class TorchEstimator:
    """Wrap a torch-module builder into a fit/predict classifier."""

    def __init__(
        self,
        build_fn: Callable[[int, int, int, DeepConfig], nn.Module],
        deep_cfg: DeepConfig,
        n_classes: int,
        n_channels: int,
        n_times: int,
        seed: int,
        device: torch.device | None = None,
    ) -> None:
        self.build_fn = build_fn
        self.cfg = deep_cfg
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.n_times = n_times
        self.seed = seed
        self.device = device or get_device()

    def _to_tensor(self, X: np.ndarray) -> torch.Tensor:
        # EEGNet expects (N, 1, C, T): a single-"image" with channels as height.
        return torch.from_numpy(X[:, None, :, :].astype(np.float32)).to(self.device)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TorchEstimator":
        set_torch_seed(self.seed)
        self.scaler_ = _ChannelStandardizer().fit(X)
        Xs = self.scaler_.transform(X)

        self.model_ = self.build_fn(self.n_channels, self.n_times,
                                    self.n_classes, self.cfg).to(self.device)

        xt = self._to_tensor(Xs)
        yt = torch.from_numpy(y.astype(np.int64)).to(self.device)

        # Optional unsupervised pretraining (denoising front-end).
        if hasattr(self.model_, "pretrain"):
            self.model_.pretrain(xt, self.cfg, self.device)

        opt = torch.optim.Adam(self.model_.parameters(), lr=self.cfg.lr,
                               weight_decay=self.cfg.weight_decay)
        loss_fn = nn.CrossEntropyLoss()
        n = xt.shape[0]
        self.model_.train()
        for epoch in range(self.cfg.epochs):
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, self.cfg.batch_size):
                idx = perm[start:start + self.cfg.batch_size]
                opt.zero_grad()
                out = self.model_(xt[idx])
                loss = loss_fn(out, yt[idx])
                loss.backward()
                opt.step()
        return self

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler_.transform(X)
        xt = self._to_tensor(Xs)
        self.model_.eval()
        logits = self.model_(xt)
        return logits.argmax(dim=1).cpu().numpy()
