"""Vectorised threshold sweep for the challenge metric (macro F0.5 over Source 1 entities).

train.py used to call assign() + macro_f05() once per threshold, i.e. one full sort of every
candidate pair and several joins per grid point. The best candidate of each Source 2/3
record does not depend on the threshold, so it is computed once (`link_table`), after which
every grid point is a couple of np.bincount calls over one row per Source 2/3 record.

A decoy weight r > 1 reproduces select_threshold.py's density adjustment: each false link
caused by an unmatched ("decoy") record counts r times, which is what happens when the test
split has r times as many decoys per Source 1 entity as training.
"""
import numpy as np
import polars as pl

BETA2 = 0.25


def link_table(meta, p, truth, s1_rids):
    """meta: (t_rid, s_rid) aligned with p; truth: (t_rid, s_rid) true pairs; s1_rids: rids of ALL
    Source 1 records (the evaluation set, singletons included).
    Returns (links DataFrame [t_rid, s_rid, p, s_idx, tp, decoy], nt array per s_idx)."""
    s_index = pl.DataFrame({"s_rid": s1_rids.astype(np.uint32)}).with_row_index("s_idx")
    best = (
        meta.select("t_rid", "s_rid").with_columns(pl.Series("p", p))
        .sort("p", descending=True)
        .unique("t_rid", keep="first")
        .join(truth.select("t_rid", pl.col("s_rid").alias("true_s")), on="t_rid", how="left")
        .join(s_index, on="s_rid", how="left")
        .with_columns(
            (pl.col("true_s") == pl.col("s_rid")).fill_null(False).alias("tp"),
            pl.col("true_s").is_null().alias("decoy"),
        )
        .drop("true_s")
    )
    nt = np.bincount(truth.join(s_index, on="s_rid", how="inner")["s_idx"].to_numpy(), minlength=len(s1_rids))
    return best, nt


def metrics_at(links, nt, thr, decoy_weight=1.0):
    """Macro F0.5 and companions at one threshold (links/nt from link_table)."""
    n_s = len(nt)
    keep = links["p"].to_numpy() >= thr
    s_idx = links["s_idx"].to_numpy()[keep]
    tp_m = links["tp"].to_numpy()[keep]
    dec_m = links["decoy"].to_numpy()[keep]
    tp = np.bincount(s_idx, weights=tp_m, minlength=n_s)
    fp = np.bincount(s_idx, weights=(~tp_m) & (~dec_m), minlength=n_s) + decoy_weight * np.bincount(
        s_idx, weights=(~tp_m) & dec_m, minlength=n_s)
    npred = tp + fp
    ntf = nt.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(npred > 0, tp / npred, 0.0)
        rec = np.where(ntf > 0, tp / ntf, 0.0)
        f = np.where(prec + rec > 0, (1 + BETA2) * prec * rec / (BETA2 * prec + rec), 0.0)
    f = np.where((ntf == 0) & (npred == 0), 1.0, f)
    f = np.where((ntf == 0) & (npred > 0), 0.0, f)
    n_links = int(keep.sum())
    return {
        "threshold": float(thr),
        "macro_f05": float(f.mean()),
        "micro_precision": float(tp.sum() / max(npred.sum(), 1)),
        "micro_recall": float(tp.sum() / max(ntf.sum(), 1)),
        "singleton_acc": float(((ntf == 0) & (npred == 0)).sum() / max((ntf == 0).sum(), 1)),
        "links": n_links,
        "pair_accuracy": float((links["tp"].to_numpy() == keep).mean()) if len(keep) else None,
    }


def sweep(links, nt, grid, decoy_weight=1.0):
    return [metrics_at(links, nt, float(t), decoy_weight) for t in grid]


def best_threshold(rows):
    return max(rows, key=lambda r: r["macro_f05"])


def default_grid():
    return np.round(np.arange(0.05, 0.99, 0.01), 2)
