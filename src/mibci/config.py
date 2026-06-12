"""Typed, validated configuration — the single source of truth for a run.

Why a config object instead of constants scattered through the code:
  * Reproducibility. A run is fully described by one YAML file; archive the YAML
    and you can reproduce the numbers.
  * Honest ablations. Changing the mu/beta band or the epoch window is a
    one-line edit, not a code change, so experiments stay comparable.
  * Caching. The preprocessing-relevant fields hash to a stable key, so cached
    epochs are reused only when the parameters that produced them are identical.

Everything is frozen (immutable) so a config cannot be mutated halfway through a
run — a class of bug that silently invalidates cached artifacts.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    """Which subjects/runs to load and the integrity checks to enforce."""

    subjects: tuple[int, ...]                 # 1..109
    runs_binary: tuple[int, ...]              # imagery L/R fist: 4, 8, 12
    runs_fourclass: tuple[int, ...]           # imagery L/R fist, both fists, feet: 6, 10, 14
    exclude_known_bad: bool = True            # drop S088/089/092/100 (see exclusions.py)
    expected_sfreq: float = 160.0             # EEGMMIDB nominal sampling rate (Hz)
    expected_run_seconds: float = 123.0       # nominal run length; tolerance applied in loader

    def __post_init__(self) -> None:
        if not self.subjects:
            raise ValueError("data.subjects must be non-empty")
        if any(not (1 <= s <= 109) for s in self.subjects):
            raise ValueError("data.subjects must be in 1..109 (EEGMMIDB has 109 subjects)")


@dataclass(frozen=True)
class PreprocessConfig:
    """Spectral filtering and referencing applied to the continuous signal."""

    l_freq: float = 8.0                       # band-pass low edge (mu rhythm onset)
    h_freq: float = 30.0                       # band-pass high edge (beta rhythm)
    notch_freq: float = 60.0                   # US mains line noise
    interpolate_bads: bool = True              # detect + interpolate globally-bad channels
    bad_channel_zscore: float = 4.0            # |robust z| of log-variance to flag a channel
    apply_car: bool = True                     # Common Average Reference

    def __post_init__(self) -> None:
        if not (0 < self.l_freq < self.h_freq):
            raise ValueError("preprocess: require 0 < l_freq < h_freq")


@dataclass(frozen=True)
class ICAConfig:
    """Automatic ICA artifact removal.

    ICA is *fit* on a 1 Hz high-passed copy (decomposition is unstable on the
    narrow 8-30 Hz analysis band) and then *applied* to the analysis data.
    """

    enabled: bool = True
    n_components: float = 0.95                 # keep enough PCs to explain this variance fraction
    fit_l_freq: float = 1.0                    # high-pass used only for the ICA fit
    method: str = "fastica"
    eog_proxy_channels: tuple[str, ...] = ("Fp1", "Fp2")  # no dedicated EOG channel in this dataset
    eog_z_threshold: float = 3.0               # z-score cutoff for find_bads_eog
    flag_muscle: bool = True
    muscle_threshold: float = 0.6              # correlation cutoff for find_bads_muscle (0..1)
    max_pca_components: int | None = 25        # cap fit cost; None = MNE default


@dataclass(frozen=True)
class EpochConfig:
    """Cue-locked epoching and quality control."""

    tmin: float = -0.5                         # include a pre-cue slice for baseline
    tmax: float = 4.0                          # motor-imagery window length post-cue
    baseline_start: float | None = -0.5        # baseline-correct on [start, 0]; None disables
    baseline_end: float | None = 0.0
    reject_uv: float | None = 200.0            # peak-to-peak rejection threshold (microvolts); None disables
                                               # (~10x physiological in the 8-30 Hz band; pairs with
                                               # bad-channel interpolation so single electrodes don't veto epochs)

    @property
    def baseline(self) -> tuple[float | None, float | None] | None:
        if self.baseline_start is None and self.baseline_end is None:
            return None
        return (self.baseline_start, self.baseline_end)


@dataclass(frozen=True)
class CVConfig:
    """Cross-validation regimes. Both are reported; LOSO is the honest one."""

    scheme: str = "within"                     # "within" (per-subject k-fold) or "loso"
    within_folds: int = 5
    loso_min_subjects: int = 2                 # LOSO needs >=2 subjects to be meaningful

    def __post_init__(self) -> None:
        if self.scheme not in ("within", "loso"):
            raise ValueError("cv.scheme must be 'within' or 'loso'")
        if self.within_folds < 2:
            raise ValueError("cv.within_folds must be >= 2")


@dataclass(frozen=True)
class DeepConfig:
    """Hyper-parameters shared by EEGNet and the Denoising-EEGNet variant."""

    epochs: int = 100
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 0.0
    # EEGNet structure (Lawhern et al. 2018 defaults, scaled to this dataset).
    f1: int = 8                                # temporal filters
    d: int = 2                                 # depth multiplier (spatial filters per temporal)
    f2: int | None = None                      # separable filters; None -> f1*d
    kernel_length: int = 64                    # ~half the sfreq -> ~0.4 s temporal kernel
    dropout: float = 0.5
    # Denoising autoencoder front-end (custom model only).
    ae_pretrain_epochs: int = 30
    ae_noise_std: float = 0.2                  # Gaussian corruption std (in standardized units)
    ae_latent_channels: int = 16


@dataclass(frozen=True)
class ModelConfig:
    """Which models to benchmark and their classical hyper-parameters."""

    names: tuple[str, ...] = (
        "csp_lda",
        "csp_svm",
        "riemann_lr",
        "eegnet",
        "denoising_eegnet",
    )
    csp_components: int = 6                     # spatial filters; even number, half per class extreme
    svm_c: float = 1.0
    svm_gamma: str = "scale"
    deep: DeepConfig = field(default_factory=DeepConfig)


@dataclass(frozen=True)
class Config:
    """Top-level run configuration."""

    name: str
    data: DataConfig
    preprocess: PreprocessConfig
    ica: ICAConfig
    epoch: EpochConfig
    cv: CVConfig
    model: ModelConfig
    seed: int = 42
    n_jobs: int = 1
    artifacts_dir: str = "artifacts"

    # --- construction -----------------------------------------------------
    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path, "r") as fh:
            raw: dict[str, Any] = yaml.safe_load(fh)
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Config":
        # Tuples (not lists) so the config is hashable and order is fixed.
        def t(x: Any) -> Any:
            return tuple(x) if isinstance(x, list) else x

        data_raw = dict(raw["data"])
        # Convenience: `subjects: all` expands to the full 109-subject cohort.
        if data_raw.get("subjects") == "all":
            data_raw["subjects"] = list(range(1, 110))
        data = DataConfig(**{k: t(v) for k, v in data_raw.items()})
        preprocess = PreprocessConfig(**raw.get("preprocess", {}))
        ica = ICAConfig(**{k: t(v) for k, v in raw.get("ica", {}).items()})
        epoch = EpochConfig(**raw.get("epoch", {}))
        cv = CVConfig(**raw.get("cv", {}))

        model_raw = dict(raw.get("model", {}))
        deep_raw = model_raw.pop("deep", {})
        model = ModelConfig(
            **{k: t(v) for k, v in model_raw.items()},
            deep=DeepConfig(**deep_raw),
        )
        top = {k: v for k, v in raw.items()
               if k not in ("data", "preprocess", "ica", "epoch", "cv", "model")}
        return cls(data=data, preprocess=preprocess, ica=ica, epoch=epoch,
                   cv=cv, model=model, **top)

    # --- caching ----------------------------------------------------------
    def preprocessing_hash(self) -> str:
        """Stable hash over the fields that determine preprocessed epochs.

        Used as a cache key. Deliberately excludes model/CV/seed so that
        swapping classifiers does not invalidate the (expensive) epoch cache.
        """
        relevant = {
            "data": asdict(self.data),
            "preprocess": asdict(self.preprocess),
            "ica": asdict(self.ica),
            "epoch": asdict(self.epoch),
        }
        blob = json.dumps(relevant, sort_keys=True, default=str)
        return hashlib.sha1(blob.encode()).hexdigest()[:12]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
