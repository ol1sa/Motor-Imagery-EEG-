"""Config loading, validation, and cache-key stability."""

from __future__ import annotations

import dataclasses

import pytest

from mibci.config import Config, PreprocessConfig


def test_load_smoke_config():
    cfg = Config.from_yaml("configs/smoke.yaml")
    assert cfg.name == "smoke"
    assert cfg.preprocess.l_freq == 8.0 and cfg.preprocess.h_freq == 30.0
    assert "csp_lda" in cfg.model.names
    assert cfg.data.subjects == (1, 2, 3, 4)


def test_subjects_all_expands():
    cfg = Config.from_dict({
        "name": "t",
        "data": {"subjects": "all", "runs_binary": [4, 8, 12],
                 "runs_fourclass": [6, 10, 14]},
    })
    assert cfg.data.subjects == tuple(range(1, 110))
    assert len(cfg.data.subjects) == 109


def test_invalid_band_raises():
    with pytest.raises(ValueError):
        PreprocessConfig(l_freq=30.0, h_freq=8.0)  # low >= high


def test_preprocessing_hash_is_stable_and_selective():
    cfg = Config.from_yaml("configs/smoke.yaml")
    h1 = cfg.preprocessing_hash()
    # Changing a MODEL param must NOT change the preprocessing hash (cache reuse).
    cfg_model = dataclasses.replace(
        cfg, model=dataclasses.replace(cfg.model, csp_components=99))
    assert cfg_model.preprocessing_hash() == h1
    # Changing a PREPROCESS param MUST change it (cache invalidation).
    cfg_pre = dataclasses.replace(
        cfg, preprocess=dataclasses.replace(cfg.preprocess, l_freq=7.0))
    assert cfg_pre.preprocessing_hash() != h1
