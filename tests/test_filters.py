"""Filtering preserves shape/sfreq, applies CAR, attenuates out-of-band power,
and interpolates a clearly-bad channel."""

from __future__ import annotations

import mne
import numpy as np

from mibci.config import PreprocessConfig
from mibci.preprocess.filters import _interpolate_bad_channels, apply_filters

# Interpolation needs a montage; the synthetic fixture has none, so disable it
# for the pure filtering tests and exercise it separately below.
_NO_INTERP = dict(interpolate_bads=False)


def test_filter_preserves_shape_and_sfreq(synth_raw, sfreq):
    n_ch, n_times = synth_raw.get_data().shape
    apply_filters(synth_raw, PreprocessConfig(**_NO_INTERP))
    out = synth_raw.get_data()
    assert out.shape == (n_ch, n_times)          # filtering must not resample
    assert synth_raw.info["sfreq"] == sfreq


def test_car_zeroes_channel_average(synth_raw):
    apply_filters(synth_raw, PreprocessConfig(apply_car=True, **_NO_INTERP))
    # After Common Average Reference, the mean across channels is ~0 everywhere.
    avg = synth_raw.get_data().mean(axis=0)
    assert np.allclose(avg, 0.0, atol=1e-12)


def test_bandpass_attenuates_low_frequency(sfreq):
    # 2 Hz sine sits below the 8 Hz high-pass edge -> should be strongly reduced.
    t = np.arange(0, 20, 1 / sfreq)
    data = (np.sin(2 * np.pi * 2.0 * t) * 1e-5)[None, :]
    info = mne.create_info(["EEG00"], sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    power_before = np.var(raw.get_data())
    apply_filters(raw, PreprocessConfig(apply_car=False, **_NO_INTERP))
    power_after = np.var(raw.get_data())
    assert power_after < 0.2 * power_before


def test_interpolates_bad_channel(sfreq):
    # Real 10-20 names so a montage (and thus interpolation) is available.
    names = ["Fp1", "Fp2", "C3", "C4", "Cz", "P3", "P4", "O1", "O2", "Pz"]
    rng = np.random.default_rng(0)
    data = rng.standard_normal((len(names), int(sfreq * 10))) * 1e-5
    data[3] *= 50.0  # make C4 a wildly high-variance "bad" channel
    info = mne.create_info(names, sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    raw.set_montage("standard_1020", verbose="ERROR")

    var_before = raw.get_data()[3].var()
    bads = _interpolate_bad_channels(raw, zscore=4.0, subject=1)
    assert "C4" in bads                                  # the bad channel is detected
    assert raw.get_data()[3].var() < var_before          # and replaced by interpolation
