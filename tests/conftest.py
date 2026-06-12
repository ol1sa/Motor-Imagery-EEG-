"""Shared fixtures. Synthetic data so unit tests run fast and offline."""

from __future__ import annotations

import mne
import numpy as np
import pytest


@pytest.fixture
def sfreq() -> float:
    return 160.0


@pytest.fixture
def synth_raw(sfreq):
    """A short multi-channel RawArray with EEG channel types (so CAR works)."""
    rng = np.random.default_rng(0)
    n_ch, n_sec = 8, 20
    n_times = int(sfreq * n_sec)
    data = rng.standard_normal((n_ch, n_times)) * 1e-5  # ~10 uV scale
    ch_names = [f"EEG{i:02d}" for i in range(n_ch)]
    info = mne.create_info(ch_names, sfreq, ch_types="eeg")
    return mne.io.RawArray(data, info, verbose="ERROR")


@pytest.fixture
def annotated_raw(synth_raw, sfreq):
    """synth_raw with alternating left/right fist annotations every 2 s."""
    onsets, descs = [], []
    for i in range(8):
        onsets.append(1.0 + i * 2.0)
        descs.append("left_fist" if i % 2 == 0 else "right_fist")
    ann = mne.Annotations(onset=onsets, duration=[0.0] * len(onsets), description=descs)
    synth_raw.set_annotations(ann)
    return synth_raw
