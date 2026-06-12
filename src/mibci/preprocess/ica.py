"""Automatic ICA-based artifact removal.

Rationale and the key methodological choice:
  ICA separates the signal into maximally-independent components; eye blinks and
  muscle bursts tend to isolate into a few components we can zero out. BUT ICA's
  decomposition is unstable on a narrow band — fit it on the 8-30 Hz analysis
  signal and the components are dominated by the same rhythm we care about.
  The MNE-recommended fix (and what we do): FIT the ICA on a 1 Hz high-passed
  copy of the data (broadband, stable decomposition), then APPLY the resulting
  unmixing to the narrowband analysis data.

Artifact components are flagged automatically:
  * EOG (blinks / saccades): correlated with frontal channels Fp1/Fp2, which we
    use as EOG proxies since this dataset has no dedicated EOG electrode.
  * Muscle (EMG): high-frequency, broadly-distributed components, via
    ICA.find_bads_muscle.

Every subject logs how many components were removed — automatic cleaning that
isn't logged isn't auditable.
"""

from __future__ import annotations

from dataclasses import dataclass

import mne
from mne.io import BaseRaw
from mne.preprocessing import ICA

from ..config import ICAConfig
from ..logging_utils import get_logger

log = get_logger()


@dataclass
class ICAResult:
    raw: BaseRaw
    n_eog: int
    n_muscle: int
    n_total_removed: int
    n_components: int


def apply_ica(raw: BaseRaw, cfg: ICAConfig, seed: int, subject: int) -> ICAResult:
    """Fit ICA on a high-passed copy, flag EOG+muscle, apply to ``raw`` in place."""
    if not cfg.enabled:
        return ICAResult(raw=raw, n_eog=0, n_muscle=0, n_total_removed=0, n_components=0)

    # Fit on a broadband copy: high-pass only, so the analysis band-pass already
    # applied to `raw` does not destabilise the decomposition.
    raw_for_fit = raw.copy().filter(l_freq=cfg.fit_l_freq, h_freq=None,
                                    method="fir", fir_design="firwin", verbose="ERROR")

    ica = ICA(
        n_components=cfg.n_components,
        method=cfg.method,
        max_iter="auto",
        random_state=seed,           # deterministic decomposition
        verbose="ERROR",
    )
    ica.fit(raw_for_fit, verbose="ERROR")

    exclude: set[int] = set()

    # --- EOG via frontal proxy channels ---
    eog_idx: list[int] = []
    present = [ch for ch in cfg.eog_proxy_channels if ch in raw.ch_names]
    if present:
        try:
            eog_idx, _ = ica.find_bads_eog(
                raw, ch_name=present, threshold=cfg.eog_z_threshold, verbose="ERROR"
            )
        except Exception as exc:  # robustness: never let auto-detection abort a run
            log.warning("subject %03d | EOG detection skipped: %s", subject, exc)
    exclude.update(eog_idx)

    # --- Muscle ---
    muscle_idx: list[int] = []
    if cfg.flag_muscle:
        try:
            muscle_idx, _ = ica.find_bads_muscle(
                raw, threshold=cfg.muscle_threshold, verbose="ERROR"
            )
        except Exception as exc:
            log.warning("subject %03d | muscle detection skipped: %s", subject, exc)
    exclude.update(muscle_idx)

    ica.exclude = sorted(exclude)
    ica.apply(raw, verbose="ERROR")  # zero the flagged components, reconstruct

    result = ICAResult(
        raw=raw,
        n_eog=len(set(eog_idx)),
        n_muscle=len(set(muscle_idx)),
        n_total_removed=len(exclude),
        n_components=ica.n_components_,
    )
    log.info(
        "subject %03d | ICA: %d/%d components removed (EOG=%d, muscle=%d)",
        subject, result.n_total_removed, result.n_components,
        result.n_eog, result.n_muscle,
    )
    return result
