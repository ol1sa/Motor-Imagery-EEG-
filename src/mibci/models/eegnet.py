"""EEGNet (Lawhern et al., 2018) — a compact convolutional net for EEG.

The architecture mirrors EEG analysis intuition:
  * Block 1 temporal conv  = learnable band-pass filters (frequency features).
  * Block 1 depthwise conv  = spatial filters across electrodes, per temporal
    filter — analogous to CSP, but learned end-to-end.
  * Block 2 separable conv  = how those spatio-temporal features evolve in time.
Average pooling and dropout keep the parameter count tiny, which matters because
single-subject EEG datasets are small.

We omit the original max-norm weight constraints to keep the model easy to read;
dropout + small capacity already regularise it for this dataset size.
"""

from __future__ import annotations

import torch
from torch import nn

from ..config import DeepConfig


class EEGNet(nn.Module):
    def __init__(self, n_channels: int, n_times: int, n_classes: int, cfg: DeepConfig) -> None:
        super().__init__()
        f1, d = cfg.f1, cfg.d
        f2 = cfg.f2 if cfg.f2 is not None else f1 * d
        k = cfg.kernel_length

        # Block 1: temporal filtering then spatial (depthwise across channels).
        self.temporal = nn.Sequential(
            nn.Conv2d(1, f1, (1, k), padding="same", bias=False),
            nn.BatchNorm2d(f1),
        )
        self.spatial = nn.Sequential(
            # Depthwise: groups=f1 so each temporal filter gets its own spatial
            # filters. Kernel (C,1) collapses the electrode dimension.
            nn.Conv2d(f1, f1 * d, (n_channels, 1), groups=f1, bias=False),
            nn.BatchNorm2d(f1 * d),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),         # downsample time x4
            nn.Dropout(cfg.dropout),
        )
        # Block 2: separable conv = depthwise temporal + pointwise mixing.
        self.separable = nn.Sequential(
            nn.Conv2d(f1 * d, f1 * d, (1, 16), padding="same", groups=f1 * d, bias=False),
            nn.Conv2d(f1 * d, f2, (1, 1), bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),         # downsample time x8
            nn.Dropout(cfg.dropout),
        )

        # Infer the flattened feature size with a dummy forward (robust to the
        # configured window length / sampling rate).
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_times)
            feat = self._features(dummy)
            flat = feat.shape[1] * feat.shape[2] * feat.shape[3]
        self.classifier = nn.Linear(flat, n_classes)

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.temporal(x)
        x = self.spatial(x)
        x = self.separable(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._features(x)
        x = x.flatten(start_dim=1)
        return self.classifier(x)


def build_eegnet(n_channels: int, n_times: int, n_classes: int, cfg: DeepConfig) -> EEGNet:
    return EEGNet(n_channels, n_times, n_classes, cfg)
