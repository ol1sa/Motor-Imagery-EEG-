"""Registry of known-problematic EEGMMIDB subjects, with reasons.

The PhysioNet EEG Motor Movement/Imagery Database is widely used but has a small
number of recordings with acquisition inconsistencies that corrupt epoching if
loaded naively. The community (e.g. MOABB) consistently excludes the same set.

We do TWO things, and log both:
  1. Maintain this explicit registry so a reader knows *which* subjects and
     *why* — the exclusion is documented, not folklore.
  2. Independently verify each loaded recording's sampling rate and run length
     (see loader.verify_recording) so that any *new* inconsistency is also
     caught and logged, rather than silently producing garbage epochs.
"""

from __future__ import annotations

# Subject id -> human-readable reason. These four are the canonical bad set:
# their interleaved task runs were recorded with timing/sampling inconsistent
# with the nominal 160 Hz / ~123 s-per-run protocol, which desynchronises the
# event annotations from the signal.
KNOWN_BAD_SUBJECTS: dict[int, str] = {
    88: "inconsistent run timing / sampling vs. 160 Hz protocol (corrupts event alignment)",
    89: "inconsistent run timing / sampling vs. 160 Hz protocol (corrupts event alignment)",
    92: "inconsistent run timing / sampling vs. 160 Hz protocol (corrupts event alignment)",
    100: "inconsistent run timing / sampling vs. 160 Hz protocol (corrupts event alignment)",
}


def is_known_bad(subject: int) -> bool:
    return subject in KNOWN_BAD_SUBJECTS


def reason(subject: int) -> str:
    return KNOWN_BAD_SUBJECTS.get(subject, "")
