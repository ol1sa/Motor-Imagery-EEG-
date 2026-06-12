"""CSP transform dims + deep-model forward/fit shapes on synthetic data."""

from __future__ import annotations

import numpy as np

from mibci.config import DeepConfig
from mibci.features.csp import build_csp
from mibci.models.eegnet import build_eegnet
from mibci.models.torch_utils import TorchEstimator


def _synth_xy(n=40, n_ch=8, n_t=160, seed=0):
    """Two classes that differ in the variance of channel 0 (CSP-separable)."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_ch, n_t))
    y = np.array([0, 1] * (n // 2))
    X[y == 1, 0, :] *= 3.0  # class 1 has higher variance on channel 0
    return X, y


def test_csp_transform_dims():
    X, y = _synth_xy()
    csp = build_csp(n_components=4)
    feats = csp.fit_transform(X, y)
    # log-variance CSP -> one feature per spatial filter.
    assert feats.shape == (X.shape[0], 4)


def test_csp_handles_rank_deficient_data():
    """CAR + interpolation make the covariance singular; regularised CSP must
    still fit. Here we emulate CAR by forcing channels to sum to zero."""
    X, y = _synth_xy(n=20, n_ch=8, n_t=160)
    X = X - X.mean(axis=1, keepdims=True)   # Common-Average-Reference -> rank 7
    feats = build_csp(n_components=4).fit_transform(X, y)
    assert feats.shape == (X.shape[0], 4)
    assert np.isfinite(feats).all()


def test_eegnet_forward_shape():
    import torch

    n_ch, n_t, n_cls = 8, 160, 2
    model = build_eegnet(n_ch, n_t, n_cls, DeepConfig())
    out = model(torch.zeros(5, 1, n_ch, n_t))
    assert out.shape == (5, n_cls)


def test_torch_estimator_fit_predict():
    X, y = _synth_xy(n=24, n_t=160)
    cfg = DeepConfig(epochs=2, batch_size=8, ae_pretrain_epochs=1)
    est = TorchEstimator(build_eegnet, cfg, n_classes=2, n_channels=8,
                         n_times=160, seed=0)
    est.fit(X, y)
    pred = est.predict(X)
    assert pred.shape == (X.shape[0],)
    assert set(np.unique(pred).tolist()) <= {0, 1}
