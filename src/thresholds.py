"""Vectorised threshold sweep for the challenge metric (macro F0.5 over Source 1 entities).

train.py used to call assign() + macro_f05() once per threshold, i.e. one full sort of every
candidate pair and several joins per grid point. The best candidate of each Source 2/3
record does not depend on the threshold, so it is computed once (`link_table`), after which
every grid point is a couple of np.bincount calls over one row per Source 2/3 record.

A decoy weight r > 1 reproduces select_threshold.py's density adjustment: each false link
caused by an unmatched ("decoy") record counts r times, which is what happens when the test
split has r times as many decoys per Source 1 entity as training.

Thresholds and decoy weights may be scalars or arrays aligned with the rows of `links`, which
is what per-class thresholding (threshold_policy.py) needs: row i is accepted iff p_i >= thr_i.
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


def subset_links(links, nt, entity_mask):
    """Restricts a link table to the Source 1 entities where entity_mask (bool per s_idx) is True,
    re-indexing s_idx so that the result is a self-contained (links, nt) pair."""
    entity_mask = np.asarray(entity_mask, dtype=bool)
    idx_map = -np.ones(len(entity_mask), dtype=np.int64)
    idx_map[entity_mask] = np.arange(int(entity_mask.sum()))
    s_idx = links["s_idx"].to_numpy()
    keep = entity_mask[s_idx]
    lk = links.filter(pl.Series(keep)).with_columns(pl.Series("s_idx", idx_map[s_idx[keep]]).cast(pl.UInt32))
    return lk, nt[entity_mask]


def _as_rows(x, n):
    x = np.asarray(x, dtype=np.float64)
    return x if x.ndim else np.full(n, float(x))


def entity_scores(links, nt, thr, decoy_weight=1.0, keep=None):
    """Per-entity F0.5 (challenge definition, singletons included) at threshold(s) `thr`.

    thr / decoy_weight: scalar or one value per row of `links`. `keep` overrides the acceptance
    mask (bool per row) when the caller has already computed it.
    Returns (f, tp, npred, keep) with f/tp/npred per Source 1 entity and keep per link row."""
    n_s = len(nt)
    p = links["p"].to_numpy()
    if keep is None:
        keep = p >= _as_rows(thr, len(p))
    s_idx = links["s_idx"].to_numpy()[keep]
    tp_m = links["tp"].to_numpy()[keep]
    dec_m = links["decoy"].to_numpy()[keep]
    w = _as_rows(decoy_weight, len(p))[keep]
    tp = np.bincount(s_idx, weights=tp_m, minlength=n_s)
    fp = np.bincount(s_idx, weights=(~tp_m) & (~dec_m), minlength=n_s) + np.bincount(
        s_idx, weights=w * ((~tp_m) & dec_m), minlength=n_s)
    npred = tp + fp
    ntf = nt.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(npred > 0, tp / npred, 0.0)
        rec = np.where(ntf > 0, tp / ntf, 0.0)
        f = np.where(prec + rec > 0, (1 + BETA2) * prec * rec / (BETA2 * prec + rec), 0.0)
    f = np.where((ntf == 0) & (npred == 0), 1.0, f)
    f = np.where((ntf == 0) & (npred > 0), 0.0, f)
    return f, tp, npred, keep


def metrics_at(links, nt, thr, decoy_weight=1.0, entity_mask=None, link_mask=None):
    """Macro F0.5 and companions at one threshold (or one threshold per link row).

    entity_mask: restrict the entity-level numbers (macro F0.5, singleton accuracy) to these
    entities; link_mask: restrict the link-level numbers (micro precision / recall over the
    linked records, links, pair accuracy) to these link rows. Both default to everything, in
    which case the result is exactly the challenge metric over all Source 1 entities."""
    f, tp, npred, keep = entity_scores(links, nt, thr, decoy_weight)
    ntf = nt.astype(float)
    em = np.ones(len(nt), dtype=bool) if entity_mask is None else np.asarray(entity_mask, dtype=bool)
    lm = np.ones(len(keep), dtype=bool) if link_mask is None else np.asarray(link_mask, dtype=bool)
    tp_rows = links["tp"].to_numpy()
    # micro numbers over the link rows of interest: true positives among the accepted rows; the recall
    # denominator is every true record (retrieval misses included) when no link mask is given, else the
    # matched (non-decoy) records among the masked rows
    acc = keep & lm
    n_tp = float((acc & tp_rows).sum())
    n_true_total = float(ntf[em].sum()) if link_mask is None else float((lm & ~links["decoy"].to_numpy()).sum())
    singles = (ntf[em] == 0)
    return {
        "threshold": float(thr) if np.ndim(thr) == 0 else None,
        "macro_f05": float(f[em].mean()) if em.any() else None,
        "micro_precision": float(n_tp / max(acc.sum(), 1)),
        "micro_recall": float(n_tp / max(n_true_total, 1)),
        "singleton_acc": float(((ntf[em] == 0) & (npred[em] == 0)).sum() / max(singles.sum(), 1)),
        "links": int(acc.sum()),
        "pair_accuracy": float((tp_rows[lm] == keep[lm]).mean()) if lm.any() else None,
        "n_entities": int(em.sum()),
    }


def sweep(links, nt, grid, decoy_weight=1.0):
    return [metrics_at(links, nt, float(t), decoy_weight) for t in grid]


def best_threshold(rows):
    return max(rows, key=lambda r: r["macro_f05"])


def default_grid():
    return np.round(np.arange(0.05, 0.99, 0.01), 2)
