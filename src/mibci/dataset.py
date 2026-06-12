"""Per-subject preprocessing pipeline + cached dataset assembly.

This ties the stages together: load -> filter -> ICA -> epoch -> (X, y), and
caches the result per (preprocessing-config-hash, subject). The expensive work
happens once; swapping classifiers or CV schemes reuses the cache.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mne
import numpy as np

from .cache import EpochCache
from .config import Config
from .data.loader import load_subject
from .logging_utils import get_logger
from .preprocess.epochs import epochs_to_xy, make_epochs
from .preprocess.filters import apply_filters
from .preprocess.ica import apply_ica

log = get_logger()


@dataclass
class SubjectData:
    subject: int
    X: np.ndarray          # (n_epochs, n_channels, n_times), volts
    y: np.ndarray          # (n_epochs,), dense 0..K-1 labels
    sfreq: float
    ch_names: list[str]
    info: mne.Info         # kept for topographic plotting (CSP patterns, etc.)
    ica_removed: int
    n_epochs: int


def preprocess_subject(cfg: Config, subject: int, experiment: str) -> SubjectData | None:
    """Run the full preprocessing chain for one subject. None if excluded."""
    raw = load_subject(cfg, subject, experiment)
    if raw is None:
        return None

    apply_filters(raw, cfg.preprocess, subject=subject)
    ica_res = apply_ica(raw, cfg.ica, seed=cfg.seed, subject=subject)
    epochs = make_epochs(ica_res.raw, cfg.epoch, experiment, subject)

    if len(epochs) == 0:
        log.warning("EXCLUDE subject %03d | no epochs survived rejection", subject)
        return None

    X, y = epochs_to_xy(epochs, experiment)
    return SubjectData(
        subject=subject,
        X=X.astype(np.float64),
        y=y,
        sfreq=float(epochs.info["sfreq"]),
        ch_names=list(epochs.ch_names),
        info=epochs.info,
        ica_removed=ica_res.n_total_removed,
        n_epochs=len(epochs),
    )


def _payload_to_subjectdata(payload: dict[str, Any] | None) -> SubjectData | None:
    if payload is None:
        return None
    # Normalize X to a clean, native, C-contiguous float64 array. Arrays revived
    # from the joblib cache can otherwise trip MNE's strict `copy=None` guard
    # during CSP rank estimation (asanyarray returns a copy for such arrays),
    # which would crash only on cache *hits*. This makes both paths identical.
    payload = dict(payload)
    payload["X"] = np.ascontiguousarray(payload["X"], dtype=np.float64)
    return SubjectData(**payload)


def build_dataset(cfg: Config, experiment: str) -> list[SubjectData]:
    """Preprocess all configured subjects (cached). Returns surviving subjects."""
    cache = EpochCache(cfg.artifacts_dir, cfg.preprocessing_hash())
    subjects: list[SubjectData] = []

    for subject in cfg.data.subjects:
        def _compute() -> dict[str, Any] | None:
            sd = preprocess_subject(cfg, subject, experiment)
            return sd.__dict__ if sd is not None else None

        sd = _payload_to_subjectdata(cache.get_or_compute(subject, _compute))
        if sd is not None:
            subjects.append(sd)

    log.info("dataset ready | %d/%d subjects usable for experiment '%s'",
             len(subjects), len(cfg.data.subjects), experiment)
    if not subjects:
        raise RuntimeError("no usable subjects after preprocessing/exclusions")
    return subjects
