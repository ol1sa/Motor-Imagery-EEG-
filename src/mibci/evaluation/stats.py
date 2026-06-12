"""Statistical comparison of models across subjects.

We compare models *pairwise* with the Wilcoxon signed-rank test — the paired,
non-parametric counterpart to a t-test. Paired because every model saw the same
subjects/splits; non-parametric because per-subject accuracies are not assumed
Gaussian (and there are often few subjects). Because we run many pairwise tests,
raw p-values would inflate the false-positive rate, so we apply Holm-Bonferroni
correction.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests


def pairwise_wilcoxon(acc_matrix: pd.DataFrame) -> pd.DataFrame:
    """Holm-corrected pairwise Wilcoxon over a subjects x models accuracy table.

    Returns one row per model pair with the median accuracy difference, raw and
    corrected p-values, and a significance flag at corrected alpha = 0.05.
    """
    models = list(acc_matrix.columns)
    pairs, stats, pvals, diffs = [], [], [], []

    for a, b in combinations(models, 2):
        xa = acc_matrix[a].to_numpy()
        xb = acc_matrix[b].to_numpy()
        # Drop subjects missing either model (e.g. a model that failed to fit).
        mask = ~(np.isnan(xa) | np.isnan(xb))
        xa, xb = xa[mask], xb[mask]
        diff = float(np.median(xa - xb))
        if len(xa) < 1 or np.allclose(xa, xb):
            stat, p = np.nan, 1.0  # no detectable difference / too few samples
        else:
            try:
                stat, p = wilcoxon(xa, xb)
            except ValueError:
                stat, p = np.nan, 1.0
        pairs.append(f"{a} vs {b}"); stats.append(stat); pvals.append(p); diffs.append(diff)

    corrected = multipletests(pvals, alpha=0.05, method="holm")[1] if pvals else []
    return pd.DataFrame({
        "pair": pairs,
        "median_acc_diff": diffs,
        "statistic": stats,
        "p_raw": pvals,
        "p_holm": corrected,
        "significant_0.05": [p < 0.05 for p in corrected],
    })
