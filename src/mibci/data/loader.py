"""Load subjects, enforce integrity checks, and attach semantic class labels.

EEGMMIDB annotates every imagery run with the same three codes — T0 (rest),
T1, T2 — but their *meaning* depends on the run:
  * runs 4/8/12 : T1 = left fist (imagined),  T2 = right fist (imagined)
  * runs 6/10/14: T1 = both fists (imagined), T2 = both feet  (imagined)

So we cannot just concatenate runs and read T1/T2 — we must relabel each run
group to a semantic name *before* concatenation. The 4-class task therefore
needs BOTH run groups; runs 6/10/14 alone only provide two of the four classes.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from mne.io import BaseRaw

from ..config import Config
from ..logging_utils import get_logger
from . import exclusions
from .download import fetch_raw

log = get_logger()

# Per-run-group annotation -> semantic class name.
_LEFT_RIGHT = {"T1": "left_fist", "T2": "right_fist"}
_FISTS_FEET = {"T1": "both_fists", "T2": "both_feet"}

# Stable class ordering -> integer label. Fixed across subjects so the y-encoding
# is identical for every subject (essential for valid LOSO).
CLASS_ORDER: dict[str, tuple[str, ...]] = {
    "binary": ("left_fist", "right_fist"),
    "fourclass": ("left_fist", "right_fist", "both_fists", "both_feet"),
}


def class_event_id(experiment: str) -> dict[str, int]:
    # 1-based: MNE event codes are non-zero by convention. The dense 0..K-1
    # labels used by classifiers are derived from these in epochs_to_xy.
    return {name: i + 1 for i, name in enumerate(CLASS_ORDER[experiment])}


def _run_groups(cfg: Config, experiment: str) -> list[tuple[Sequence[int], dict[str, str]]]:
    if experiment == "binary":
        return [(cfg.data.runs_binary, _LEFT_RIGHT)]
    if experiment == "fourclass":
        return [
            (cfg.data.runs_binary, _LEFT_RIGHT),      # left/right fist
            (cfg.data.runs_fourclass, _FISTS_FEET),   # both fists/both feet
        ]
    raise ValueError(f"unknown experiment '{experiment}'")


def verify_recording(cfg: Config, raw: BaseRaw, subject: int) -> tuple[bool, str]:
    """Independent integrity check beyond the static known-bad list.

    Catches sampling-rate or duration anomalies in any subject so that a *new*
    bad recording is flagged and logged, not silently epoched into noise.
    """
    if abs(raw.info["sfreq"] - cfg.data.expected_sfreq) > 1e-6:
        return False, (f"sfreq {raw.info['sfreq']:.1f} != expected "
                       f"{cfg.data.expected_sfreq:.1f} Hz")
    # Annotations must be present and contain the task codes we will epoch on.
    descs = set(raw.annotations.description)
    if not ({"left_fist", "right_fist"} & descs or {"both_fists", "both_feet"} & descs):
        return False, f"no task annotations after relabel (have: {sorted(descs)})"
    return True, ""


def _relabel(raw: BaseRaw, mapping: dict[str, str]) -> BaseRaw:
    """Rewrite annotation descriptions to semantic class names; T0 -> 'rest'."""
    new = np.array([mapping.get(d, "rest") for d in raw.annotations.description])
    raw.annotations.description = new
    return raw


def load_subject(cfg: Config, subject: int, experiment: str) -> BaseRaw | None:
    """Return the relabelled, montaged Raw for a subject, or None if excluded.

    Every exclusion path logs *what* was dropped and *why*.
    """
    if cfg.data.exclude_known_bad and exclusions.is_known_bad(subject):
        log.warning("EXCLUDE subject %03d | known-bad: %s",
                    subject, exclusions.reason(subject))
        return None

    group_raws = []
    for runs, mapping in _run_groups(cfg, experiment):
        raw = fetch_raw(subject, runs)
        _relabel(raw, mapping)
        ok, why = verify_recording(cfg, raw, subject)
        if not ok:
            log.warning("EXCLUDE subject %03d | integrity check failed on runs %s: %s",
                        subject, list(runs), why)
            return None
        group_raws.append(raw)

    if len(group_raws) == 1:
        return group_raws[0]
    # Concatenate the run groups (e.g. left/right + fists/feet for 4-class).
    import mne
    return mne.concatenate_raws(group_raws, verbose="ERROR")
