"""Epoching produces the right count and array shape from annotations."""

from __future__ import annotations

from mibci.config import EpochConfig
from mibci.preprocess.epochs import epochs_to_xy, make_epochs


def test_epoch_count_matches_annotations(annotated_raw):
    # No amplitude rejection so every cued trial survives.
    cfg = EpochConfig(tmin=-0.5, tmax=1.5, baseline_start=-0.5, baseline_end=0.0,
                      reject_uv=None)
    epochs = make_epochs(annotated_raw, cfg, experiment="binary", subject=1)
    # 8 annotations, 4 left + 4 right.
    assert len(epochs) == 8


def test_xy_shape_and_labels(annotated_raw, sfreq):
    cfg = EpochConfig(tmin=0.0, tmax=2.0, baseline_start=None, baseline_end=None,
                      reject_uv=None)
    epochs = make_epochs(annotated_raw, cfg, experiment="binary", subject=1)
    X, y = epochs_to_xy(epochs, "binary")
    n_epochs = len(epochs)
    expected_times = int(round((cfg.tmax - cfg.tmin) * sfreq)) + 1
    assert X.shape == (n_epochs, 8, expected_times)
    assert set(y.tolist()) <= {0, 1}          # binary labels, dense 0/1
    assert y.shape == (n_epochs,)
