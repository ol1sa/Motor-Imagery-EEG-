"""Common Spatial Patterns (CSP) helpers.

CSP learns spatial filters that maximise the variance ratio between classes —
for motor imagery it discovers the lateralised sensorimotor sources (C3 for the
right hand, C4 for the left). We use MNE's implementation, which handles the
multiclass case via approximate joint diagonalisation.
"""

from __future__ import annotations

from mne.decoding import CSP


def build_csp(n_components: int) -> CSP:
    """CSP transformer returning log-variance features per spatial filter.

    log-variance (the classic CSP feature) linearises the band-power so a simple
    linear classifier downstream can separate the classes.

    reg='ledoit_wolf' shrinks the covariance estimate toward a well-conditioned
    target. This is essential here: Common Average Reference and bad-channel
    interpolation both reduce the data rank, so an unregularised covariance is
    singular and CSP's generalised eigendecomposition fails ("not positive
    definite"). Shrinkage restores positive-definiteness.
    """
    return CSP(n_components=n_components, reg="ledoit_wolf", log=True, norm_trace=False)
