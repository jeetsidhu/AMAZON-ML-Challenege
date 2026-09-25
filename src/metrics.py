"""Macro F_0.5 exactly as defined by the challenge (singletons included)."""
import numpy as np
import polars as pl


def macro_f05(pred, truth, beta=0.5):
    """pred/truth: DataFrames (s1 id column 's', list[str] column 'm'); rows of truth define the eval set.

    Per Source 1 entity: empty truth & empty pred -> 1, empty truth & non-empty pred -> 0,
    otherwise F_beta of the predicted vs. true id sets. Returns (macro F, stats dict).
    """
    b2 = beta * beta
    j = truth.rename({"m": "true"}).join(pred.rename({"m": "pred"}), on="s", how="left").with_columns(
        pl.col("pred").fill_null(pl.lit([], dtype=pl.List(pl.String)))
    )
    j = j.with_columns(
        pl.col("true").list.len().alias("nt"),
        pl.col("pred").list.len().alias("np"),
        pl.col("true").list.set_intersection("pred").list.len().alias("tp"),
    )
    nt = j["nt"].to_numpy().astype(float)
    npd = j["np"].to_numpy().astype(float)
    tp = j["tp"].to_numpy().astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(npd > 0, tp / npd, 0.0)
        r = np.where(nt > 0, tp / nt, 0.0)
        f = np.where((p + r) > 0, (1 + b2) * p * r / (b2 * p + r), 0.0)
    f = np.where((nt == 0) & (npd == 0), 1.0, f)
    f = np.where((nt == 0) & (npd > 0), 0.0, f)
    stats = {
        "n": len(f),
        "macro_f05": float(f.mean()),
        "micro_precision": float(tp.sum() / max(npd.sum(), 1)),
        "micro_recall": float(tp.sum() / max(nt.sum(), 1)),
        "singleton_acc": float(((nt == 0) & (npd == 0)).sum() / max((nt == 0).sum(), 1)),
    }
    return stats["macro_f05"], stats
