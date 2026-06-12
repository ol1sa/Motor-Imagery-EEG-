"""Visualise EEGNet's learned filters.

EEGNet is interpretable by construction:
  * Block-1 temporal filters act as learned band-pass filters; we plot their
    frequency responses to see which rhythms (mu/beta?) the net latched onto.
  * Block-1 depthwise spatial filters are scalp projections; we render them as
    topomaps and again expect sensorimotor (C3/C4) weighting.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np

from ..logging_utils import get_logger
from ..models.eegnet import EEGNet

log = get_logger()


def plot_eegnet_filters(model: EEGNet, info: mne.Info, sfreq: float,
                        out_path: str | Path) -> Path:
    """Plot temporal-filter frequency responses + spatial-filter topomaps."""
    # Temporal filters: (F1, 1, 1, k)
    temporal_w = model.temporal[0].weight.detach().cpu().numpy()[:, 0, 0, :]
    # Spatial filters: depthwise (F1*D, 1, C, 1) -> (n_filters, C)
    spatial_w = model.spatial[0].weight.detach().cpu().numpy()[:, 0, :, 0]

    n_temporal = temporal_w.shape[0]
    n_spatial = min(spatial_w.shape[0], 4)

    fig, axes = plt.subplots(2, max(n_temporal, n_spatial),
                             figsize=(3 * max(n_temporal, n_spatial), 6))
    axes = np.atleast_2d(axes)

    # Row 0: magnitude frequency response of each temporal filter.
    for i in range(n_temporal):
        spec = np.abs(np.fft.rfft(temporal_w[i], n=256))
        freqs = np.fft.rfftfreq(256, d=1.0 / sfreq)
        ax = axes[0, i]
        ax.plot(freqs, spec)
        ax.set_xlim(0, 40)
        ax.set_title(f"temporal #{i}")
        ax.set_xlabel("Hz")
    for j in range(n_temporal, axes.shape[1]):
        axes[0, j].axis("off")

    # Row 1: spatial filters as scalp topomaps.
    for i in range(n_spatial):
        ax = axes[1, i]
        mne.viz.plot_topomap(spatial_w[i], info, axes=ax, show=False)
        ax.set_title(f"spatial #{i}")
    for j in range(n_spatial, axes.shape[1]):
        axes[1, j].axis("off")

    fig.suptitle("EEGNet learned filters: temporal (top) + spatial (bottom)")
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("saved EEGNet filters -> %s", out_path)
    return out_path
