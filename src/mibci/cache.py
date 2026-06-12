"""Disk caching of expensive intermediates (preprocessed epochs).

Preprocessing a subject (filter + ICA + epoch) is the slow step and is pure with
respect to the preprocessing-relevant config. We therefore memoize per
(preprocessing_hash, subject) so that re-running the *modelling* side — different
classifiers, different CV — never re-pays the preprocessing cost.

We use a tiny joblib-backed helper rather than joblib.Memory's function
decorator because the cache key must be the explicit config hash, not joblib's
source-hash of the function (which would invalidate on unrelated code edits).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import joblib

from .logging_utils import get_logger

log = get_logger()


class EpochCache:
    def __init__(self, artifacts_dir: str | Path, preprocessing_hash: str) -> None:
        self.root = Path(artifacts_dir) / "cache" / preprocessing_hash
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, subject: int) -> Path:
        return self.root / f"sub-{subject:03d}.joblib"

    def get_or_compute(self, subject: int, compute: Callable[[], Any]) -> Any:
        """Return cached payload for ``subject`` or compute, store, and return it.

        ``compute`` may return ``None`` to signal "subject excluded"; that
        decision is cached too, so excluded subjects are not re-evaluated.
        """
        path = self._path(subject)
        if path.exists():
            log.info("cache hit  | subject %03d (%s)", subject, path.name)
            return joblib.load(path)
        payload = compute()
        joblib.dump(payload, path, compress=3)
        log.info("cache store | subject %03d -> %s", subject, path.name)
        return payload
