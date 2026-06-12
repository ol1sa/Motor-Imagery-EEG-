"""Command-line entry point.

    python -m mibci.run --config configs/binary.yaml --experiment binary --cv within
    python -m mibci.run --config configs/binary.yaml --experiment binary --cv loso

Runs the full pipeline: preprocess (cached) -> evaluate all models under the
chosen CV -> stats -> save results table + figures.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .data.loader import CLASS_ORDER
from .dataset import build_dataset
from .evaluation.cv import evaluate
from .evaluation.stats import pairwise_wilcoxon
from .interpret.csp_patterns import plot_csp_patterns
from .logging_utils import get_logger
from .models.torch_utils import set_torch_seed
from .viz.plots import plot_confusions, plot_per_subject_distribution

log = get_logger()


def summarize(per_subject: pd.DataFrame) -> pd.DataFrame:
    """Mean +/- std accuracy and kappa per model, across subjects."""
    g = per_subject.groupby("model")
    out = pd.DataFrame({
        "acc_mean": g["accuracy"].mean(),
        "acc_std": g["accuracy"].std(ddof=0),
        "kappa_mean": g["kappa"].mean(),
        "kappa_std": g["kappa"].std(ddof=0),
        "n_subjects": g["accuracy"].count(),
    }).sort_values("acc_mean", ascending=False)
    return out.reset_index()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Motor-imagery EEG classification benchmark")
    p.add_argument("--config", required=True, help="path to a YAML config")
    p.add_argument("--experiment", choices=["binary", "fourclass"], default="binary")
    p.add_argument("--cv", choices=["within", "loso"], default=None,
                   help="override cv.scheme from the config")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = Config.from_yaml(args.config)
    if args.cv is not None:
        # frozen dataclasses -> rebuild with the overridden scheme
        cfg = dataclasses.replace(cfg, cv=dataclasses.replace(cfg.cv, scheme=args.cv))

    set_torch_seed(cfg.seed)
    np.random.seed(cfg.seed)
    experiment = args.experiment
    scheme = cfg.cv.scheme
    class_names = list(CLASS_ORDER[experiment])
    tag = f"{cfg.name}_{experiment}_{scheme}"
    out_dir = Path(cfg.artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("=== run '%s' | experiment=%s | cv=%s | models=%s ===",
             cfg.name, experiment, scheme, ", ".join(cfg.model.names))

    subjects = build_dataset(cfg, experiment)
    result = evaluate(cfg, subjects, experiment)

    # --- results tables ---
    result.per_subject.to_csv(out_dir / f"per_subject_{tag}.csv", index=False)
    summary = summarize(result.per_subject)
    summary.to_csv(out_dir / f"summary_{tag}.csv", index=False)

    acc_matrix = result.accuracy_matrix()
    stats = pairwise_wilcoxon(acc_matrix)
    stats.to_csv(out_dir / f"stats_{tag}.csv", index=False)

    # --- figures (guarded: a plotting hiccup must not lose the numbers) ---
    try:
        plot_per_subject_distribution(acc_matrix, result.n_classes,
                                      out_dir / f"per_subject_{tag}.png")
        plot_confusions(result.confusions, class_names,
                        out_dir / f"confusions_{tag}.png")
        plot_csp_patterns(cfg, subjects, experiment, out_dir / f"csp_patterns_{tag}.png")
        # EEGNet filter visualisation: train one net on the pooled epochs purely
        # for interpretability (separate from the CV scoring above).
        if "eegnet" in cfg.model.names:
            import numpy as _np

            from .interpret.eegnet_filters import plot_eegnet_filters
            from .models.factory import build_model
            Xp = _np.concatenate([s.X for s in subjects], axis=0)
            yp = _np.concatenate([s.y for s in subjects], axis=0)
            est = build_model("eegnet", cfg, result.n_classes,
                              Xp.shape[1], Xp.shape[2])
            est.fit(Xp, yp)
            plot_eegnet_filters(est.model_, subjects[0].info, subjects[0].sfreq,
                                out_dir / f"eegnet_filters_{tag}.png")
    except Exception as exc:  # noqa: BLE001
        log.warning("figure generation issue (results still saved): %s", exc)

    # --- console report ---
    print("\n=== SUMMARY (%s) ===" % tag)
    print(summary.to_string(index=False))
    print("\n=== PAIRWISE WILCOXON (Holm-corrected) ===")
    print(stats.to_string(index=False))
    print(f"\nartifacts written to: {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
