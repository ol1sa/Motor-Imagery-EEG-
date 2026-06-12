"""Cue-locked epoching with baseline correction and amplitude rejection.

We convert annotations to events using a FIXED class->integer map (shared across
all subjects, so the labels mean the same thing in LOSO), cut epochs around cue
onset, baseline-correct on the pre-cue slice, and drop epochs whose peak-to-peak
amplitude exceeds a threshold (a simple, transparent reject of residual
artifacts that survived ICA).
"""

from __future__ import annotations

import mne
import numpy as np
from mne.io import BaseRaw

from ..config import EpochConfig
from ..data.loader import class_event_id
from ..logging_utils import get_logger

log = get_logger()


def make_epochs(raw: BaseRaw, cfg: EpochConfig, experiment: str, subject: int) -> mne.Epochs:
    """Build cue-locked, baseline-corrected, amplitude-rejected epochs."""
    event_id = class_event_id(experiment)
    # Only annotations whose description is a class name become events; 'rest'
    # (and anything else) is ignored.
    events, _ = mne.events_from_annotations(raw, event_id=event_id, verbose="ERROR")

    reject = None
    if cfg.reject_uv is not None:
        reject = {"eeg": cfg.reject_uv * 1e-6}  # config is in microvolts; MNE wants volts

    epochs = mne.Epochs(
        raw,
        events=events,
        event_id=event_id,
        tmin=cfg.tmin,
        tmax=cfg.tmax,
        baseline=cfg.baseline,
        reject=reject,
        preload=True,
        picks="eeg",
        verbose="ERROR",
    )
    n_before = len(events)
    n_after = len(epochs)
    log.info(
        "subject %03d | epochs: %d kept / %d cued (%d rejected by %s)",
        subject, n_after, n_before, n_before - n_after,
        f"{cfg.reject_uv} uV p2p" if cfg.reject_uv else "no reject",
    )
    return epochs


def epochs_to_xy(epochs: mne.Epochs, experiment: str) -> tuple[np.ndarray, np.ndarray]:
    """Return X (n_epochs, n_channels, n_times) in volts and integer labels y.

    Labels are remapped to a dense 0..K-1 range in the canonical class order.
    """
    event_id = class_event_id(experiment)
    code_to_label = {code: idx for idx, code in enumerate(event_id.values())}
    X = epochs.get_data(copy=True)
    y = np.array([code_to_label[c] for c in epochs.events[:, 2]], dtype=int)
    return X, y
