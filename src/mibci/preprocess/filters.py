"""Spectral filtering and re-referencing of the continuous signal.

Order matters and is deliberate:
  1. Notch at the mains frequency (60 Hz) — remove line noise first so it does
     not leak into the band-pass transition band or contaminate the CAR.
  2. FIR band-pass to the motor band (default 8-30 Hz: mu ~8-13 Hz + beta
     ~13-30 Hz, the rhythms that desynchronise over sensorimotor cortex during
     movement and imagery — "event-related desynchronisation").
  3. Common Average Reference LAST, so the reference is computed from already
     clean, band-limited channels rather than from line-noise-laden ones.
"""

from __future__ import annotations

import numpy as np
from mne.io import BaseRaw

from ..config import PreprocessConfig
from ..logging_utils import get_logger

log = get_logger()


def _interpolate_bad_channels(raw: BaseRaw, zscore: float, subject: int | None) -> list[str]:
    """Flag channels with outlier band power and interpolate them from neighbours.

    Why: with 64 channels, a single dead/noisy electrode (often a peripheral
    eye/muscle site like FT8 or AF7) would (a) veto every epoch under the
    peak-to-peak reject and (b) poison the Common Average Reference. We flag a
    channel as bad if the robust z-score of its log-variance is extreme — i.e.
    it carries far more or far less power than its peers — then interpolate it
    using the montage geometry. This is standard practice and is fully logged.
    """
    data = raw.get_data(picks="eeg")
    logv = np.log(data.var(axis=1) + 1e-30)
    med = np.median(logv)
    mad = np.median(np.abs(logv - med)) * 1.4826 + 1e-30  # robust std estimate
    z = (logv - med) / mad
    names = [raw.ch_names[i] for i in range(len(raw.ch_names))
             if raw.get_channel_types()[i] == "eeg"]
    bads = [ch for ch, zz in zip(names, z) if abs(zz) > zscore]
    if bads:
        raw.info["bads"] = bads
        # Interpolation needs sensor positions; the montage was set at load time.
        raw.interpolate_bads(reset_bads=True, verbose="ERROR")
        log.info("subject %s | interpolated %d bad channel(s): %s",
                 f"{subject:03d}" if subject is not None else "?", len(bads), bads)
    return bads


def apply_filters(raw: BaseRaw, cfg: PreprocessConfig, subject: int | None = None) -> BaseRaw:
    """Notch -> band-pass -> interpolate bads -> CAR, in place."""
    # Notch only if the mains frequency is below Nyquist for this recording.
    nyquist = raw.info["sfreq"] / 2.0
    if cfg.notch_freq < nyquist:
        raw.notch_filter(freqs=cfg.notch_freq, verbose="ERROR")

    # Zero-phase FIR (firwin) band-pass. FIR is the MNE default and gives a
    # linear phase response — no frequency-dependent time shifts, which matters
    # when we later epoch tightly around a cue.
    raw.filter(l_freq=cfg.l_freq, h_freq=cfg.h_freq, method="fir",
               fir_design="firwin", verbose="ERROR")

    # Interpolate bad channels BEFORE the CAR so a single bad electrode does not
    # contaminate the reference computed from all channels.
    if cfg.interpolate_bads:
        _interpolate_bad_channels(raw, cfg.bad_channel_zscore, subject)

    if cfg.apply_car:
        # projection=False applies CAR directly to the data so downstream ICA /
        # CSP see the referenced signal.
        raw.set_eeg_reference("average", projection=False, verbose="ERROR")

    return raw
