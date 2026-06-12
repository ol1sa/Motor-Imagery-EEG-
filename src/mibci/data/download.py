"""Programmatic download + montage standardization via MNE.

`mne.datasets.eegbci` fetches the EDF files from PhysioNet and caches them under
~/mne_data, so the first run downloads and every subsequent run is offline. We
also standardize channel names to the 10-05/10-10 convention and attach a
montage so that topographic plots (CSP patterns, ICA components) are spatially
correct.
"""

from __future__ import annotations

from typing import Sequence

import mne
from mne.datasets import eegbci
from mne.io import BaseRaw

from ..logging_utils import get_logger

log = get_logger()


def fetch_raw(subject: int, runs: Sequence[int]) -> BaseRaw:
    """Download (if needed) and return the concatenated Raw for given runs.

    The EEGMMIDB channel labels (e.g. 'Fc5.', 'Cz..') are non-standard; we run
    `eegbci.standardize` to map them to canonical 10-10 names and then set the
    'standard_1005' montage. on_missing='warn' tolerates the handful of
    electrodes that are not in the template without aborting the run.
    """
    # update_path=True records the download dir in MNE's config without an
    # interactive prompt (which would crash a non-interactive/batch run).
    paths = eegbci.load_data(subject, list(runs), update_path=True, verbose="ERROR")
    raws = [mne.io.read_raw_edf(p, preload=True, verbose="ERROR") for p in paths]
    raw = mne.concatenate_raws(raws, verbose="ERROR")

    eegbci.standardize(raw)  # rename channels in place to 10-10 convention
    montage = mne.channels.make_standard_montage("standard_1005")
    raw.set_montage(montage, on_missing="warn", verbose="ERROR")
    return raw
