"""Structured logging.

A single configured logger so that *every* decision the pipeline makes —
which subjects were dropped and why, how many ICA components were removed, how
many epochs survived rejection — is visible and auditable. For a project whose
whole point is "defensible," silent behaviour is the enemy.
"""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def get_logger(name: str = "mibci") -> logging.Logger:
    global _CONFIGURED
    logger = logging.getLogger(name)
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                              datefmt="%H:%M:%S")
        )
        root = logging.getLogger("mibci")
        root.addHandler(handler)
        root.setLevel(logging.INFO)
        root.propagate = False
        # MNE is chatty; keep its warnings but quiet the routine info spam.
        logging.getLogger("mne").setLevel(logging.WARNING)
        _CONFIGURED = True
    return logger
