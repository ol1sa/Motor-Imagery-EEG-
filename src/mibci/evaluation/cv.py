"""Cross-validation: within-subject k-fold AND Leave-One-Subject-Out (LOSO).

Both regimes are reported because they answer different questions:
  * Within-subject: "if I calibrate on this person, how well do I decode them?"
    (the easy, optimistic number).
  * LOSO: "can a model trained on other people decode a brand-new person with no
    calibration?" (the hard, honest number — usually much lower because of
    inter-subject variability in anatomy and rhythm topography).

Critical fairness property: for a given subject/fold, EVERY model is trained and
tested on the exact same indices, so accuracy differences are about the model,
not the split — and the paired Wilcoxon test downstream is valid.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold

from ..config import Config
from ..dataset import SubjectData
from ..logging_utils import get_logger
from ..models.factory import build_model
from .metrics import accuracy, confusion, kappa

log = get_logger()


@dataclass
class EvalResult:
    scheme: str
    n_classes: int
    # Long-form: one row per (model, subject) with mean accuracy/kappa.
    per_subject: pd.DataFrame
    # model -> accumulated (y_true, y_pred) for confusion matrices.
    confusions: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)

    def accuracy_matrix(self) -> pd.DataFrame:
        """subjects x models accuracy table (for stats + plots)."""
        return self.per_subject.pivot(index="subject", columns="model", values="accuracy")


def _dims(subjects: list[SubjectData]) -> tuple[int, int, int]:
    n_classes = int(np.unique(np.concatenate([s.y for s in subjects])).size)
    n_channels = subjects[0].X.shape[1]
    n_times = subjects[0].X.shape[2]
    return n_classes, n_channels, n_times


def evaluate_within_subject(cfg: Config, subjects: list[SubjectData],
                            experiment: str) -> EvalResult:
    n_classes, n_ch, n_t = _dims(subjects)
    models = cfg.model.names
    rows: list[dict] = []
    conf: dict[str, list] = {m: ([], []) for m in models}

    for sd in subjects:
        skf = StratifiedKFold(n_splits=cfg.cv.within_folds, shuffle=True,
                              random_state=cfg.seed)
        splits = list(skf.split(sd.X, sd.y))  # computed ONCE, reused by all models

        for m in models:
            accs, kaps = [], []
            for tr, te in splits:
                est = build_model(m, cfg, n_classes, n_ch, n_t)
                est.fit(sd.X[tr], sd.y[tr])
                pred = est.predict(sd.X[te])
                accs.append(accuracy(sd.y[te], pred))
                kaps.append(kappa(sd.y[te], pred))
                conf[m][0].extend(sd.y[te]); conf[m][1].extend(pred)
            rows.append({"model": m, "subject": sd.subject,
                         "accuracy": float(np.mean(accs)),
                         "kappa": float(np.mean(kaps))})
            log.info("within | subject %03d | %-17s acc=%.3f", sd.subject, m, np.mean(accs))

    return _finalize("within", n_classes, rows, conf)


def evaluate_loso(cfg: Config, subjects: list[SubjectData],
                  experiment: str) -> EvalResult:
    if len(subjects) < cfg.cv.loso_min_subjects:
        raise ValueError(f"LOSO needs >= {cfg.cv.loso_min_subjects} subjects, "
                         f"have {len(subjects)}")
    n_classes, n_ch, n_t = _dims(subjects)
    models = cfg.model.names

    X = np.concatenate([s.X for s in subjects], axis=0)
    y = np.concatenate([s.y for s in subjects], axis=0)
    groups = np.concatenate([np.full(len(s.y), s.subject) for s in subjects])

    logo = LeaveOneGroupOut()
    splits = list(logo.split(X, y, groups))  # one fold per held-out subject

    rows: list[dict] = []
    conf: dict[str, list] = {m: ([], []) for m in models}
    for tr, te in splits:
        held_out = int(groups[te][0])
        for m in models:
            est = build_model(m, cfg, n_classes, n_ch, n_t)
            est.fit(X[tr], y[tr])
            pred = est.predict(X[te])
            acc = accuracy(y[te], pred)
            rows.append({"model": m, "subject": held_out,
                         "accuracy": acc, "kappa": kappa(y[te], pred)})
            conf[m][0].extend(y[te]); conf[m][1].extend(pred)
            log.info("loso | held-out %03d | %-17s acc=%.3f", held_out, m, acc)

    return _finalize("loso", n_classes, rows, conf)


def _finalize(scheme: str, n_classes: int, rows: list[dict],
              conf: dict[str, list]) -> EvalResult:
    confusions = {m: (np.asarray(t), np.asarray(p)) for m, (t, p) in conf.items()}
    return EvalResult(scheme=scheme, n_classes=n_classes,
                      per_subject=pd.DataFrame(rows), confusions=confusions)


def evaluate(cfg: Config, subjects: list[SubjectData], experiment: str) -> EvalResult:
    if cfg.cv.scheme == "loso":
        return evaluate_loso(cfg, subjects, experiment)
    return evaluate_within_subject(cfg, subjects, experiment)
