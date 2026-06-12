"""CSP spatial-pattern topomaps — the neurophysiology sanity check.

CSP *patterns* (not filters) show where on the scalp each component projects.
For left-vs-right hand motor imagery we EXPECT the extreme components to localise
over the hand area of contralateral sensorimotor cortex — around electrodes C3
(right hand) and C4 (left hand). Seeing that lateralisation is strong evidence
the pipeline is learning real physiology rather than artifacts.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: write figures without a display
import matplotlib.pyplot as plt
import numpy as np

from ..config import Config
from ..dataset import SubjectData
from ..features.csp import build_csp
from ..logging_utils import get_logger

log = get_logger()


def plot_csp_patterns(cfg: Config, subjects: list[SubjectData], experiment: str,
                      out_path: str | Path) -> Path:
    """Fit CSP on pooled epochs and save its spatial-pattern topomaps."""
    X = np.concatenate([s.X for s in subjects], axis=0)
    y = np.concatenate([s.y for s in subjects], axis=0)
    info = subjects[0].info

    csp = build_csp(cfg.model.csp_components)
    csp.fit(X, y)

    n = min(cfg.model.csp_components, 4)  # show the most discriminative few
    fig = csp.plot_patterns(info, components=range(n), ch_type="eeg",
                            show=False, colorbar=True)
    fig.suptitle("CSP spatial patterns (expect C3/C4 lateralisation)")
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("saved CSP patterns -> %s", out_path)
    return out_path
