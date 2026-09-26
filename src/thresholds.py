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

Decision rule beyond the threshold (model.assign applies the same rule at prediction time):
`margin` requires p_i - p2nd_i >= margin, where p2nd is the probability of the record's runner-up
candidate (confidence-based rejection of close calls), and `contra_penalty` raises the threshold
of a link whose pair carries a strong contradiction (model.contradiction_flag); a penalty >= 1 is
a hard veto. Both are stored in the link table columns p2nd / contra, so every sweep stays a few
np.bincount calls per grid point.
"""
import numpy as np
import polars as pl

BETA2 = 0.25


def link_table(meta, p, truth, s1_rids, contra=None):
    """meta: (t_rid, s_rid) aligned with p; truth: (t_rid, s_rid) true pairs; s1_rids: rids of ALL
    Source 1 records (the evaluation set, singletons included); contra: optional bool per row of
    meta (strong contradiction of the pair).
    Returns (links DataFrame [t_rid, s_rid, p, p2nd, contra, s_idx, tp, decoy], nt array per s_idx)."""
    from model import best_candidates  # local import: model.py imports lightgbm
    s_index = pl.DataFrame({"s_rid": s1_rids.astype(np.uint32)}).with_row_index("s_idx")
    best = (
        best_candidates(meta, p, contra).drop("i")
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


def accept_mask(links, thr, margin=0.0, contra_penalty=0.0):
    """Acceptance of every link row under (threshold(s), margin, contradiction penalty)."""
    p = links["p"].to_numpy().astype(np.float64)
    t = _as_rows(thr, len(p))
    keep = p >= t
    if margin > 0:
        keep &= (p - links["p2nd"].to_numpy()) >= margin
    if contra_penalty > 0 and "contra" in links.columns:
        keep &= (~links["contra"].to_numpy()) | (p >= t + contra_penalty)
    return keep


def entity_scores(links, nt, thr, decoy_weight=1.0, keep=None, margin=0.0, contra_penalty=0.0):
    """Per-entity F0.5 (challenge definition, singletons included) at threshold(s) `thr`.

    thr / decoy_weight: scalar or one value per row of `links`. `keep` overrides the acceptance
    mask (bool per row) when the caller has already computed it; margin / contra_penalty are the
    decision-rule extras of accept_mask.
    Returns (f, tp, npred, keep) with f/tp/npred per Source 1 entity and keep per link row."""
    n_s = len(nt)
    p = links["p"].to_numpy()
    if keep is None:
        keep = accept_mask(links, thr, margin, contra_penalty)
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


def metrics_at(links, nt, thr, decoy_weight=1.0, entity_mask=None, link_mask=None, margin=0.0, contra_penalty=0.0):
    """Macro F0.5 and companions at one threshold (or one threshold per link row).

    entity_mask: restrict the entity-level numbers (macro F0.5, singleton accuracy) to these
    entities; link_mask: restrict the link-level numbers (micro precision / recall over the
    linked records, links, pair accuracy) to these link rows. Both default to everything, in
    which case the result is exactly the challenge metric over all Source 1 entities.
    Also reports the singleton false positives (singletons that received a link) and the
    false-positive links split into decoy records and wrong-entity links."""
    f, tp, npred, keep = entity_scores(links, nt, thr, decoy_weight, margin=margin, contra_penalty=contra_penalty)
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
    dec_rows = links["decoy"].to_numpy()
    return {
        "threshold": float(thr) if np.ndim(thr) == 0 else None,
        "macro_f05": float(f[em].mean()) if em.any() else None,
        "micro_precision": float(n_tp / max(acc.sum(), 1)),
        "micro_recall": float(n_tp / max(n_true_total, 1)),
        "singleton_acc": float(((ntf[em] == 0) & (npred[em] == 0)).sum() / max(singles.sum(), 1)),
        "singleton_fp": int(((ntf[em] == 0) & (npred[em] > 0)).sum()),
        "singletons": int(singles.sum()),
        "links": int(acc.sum()),
        "fp_decoy": int((acc & ~tp_rows & dec_rows).sum()),
        "fp_wrong_entity": int((acc & ~tp_rows & ~dec_rows).sum()),
        "pair_accuracy": float((tp_rows[lm] == keep[lm]).mean()) if lm.any() else None,
        "n_entities": int(em.sum()),
    }


def sweep(links, nt, grid, decoy_weight=1.0, margin=0.0, contra_penalty=0.0):
    return [metrics_at(links, nt, float(t), decoy_weight, margin=margin, contra_penalty=contra_penalty) for t in grid]


def sweep_f05(links, nt, grid, decoy_weight=1.0, margin=0.0, contra_penalty=0.0):
    """Macro F0.5 only, at every grid threshold: the arrays are extracted once and the rows that the
    margin rejects are dropped up front; a contradiction penalty lowers the effective probability of
    the contradicted rows, so every grid point is three np.bincount calls (fit loops call this)."""
    n_s = len(nt)
    p = links["p"].to_numpy().astype(np.float64)
    ok = np.ones(len(p), dtype=bool)
    if margin > 0:
        ok &= (p - links["p2nd"].to_numpy()) >= margin
    if contra_penalty > 0 and "contra" in links.columns:
        p = np.where(links["contra"].to_numpy(), p - contra_penalty, p)
    s_idx = links["s_idx"].to_numpy()[ok]
    tp_m = links["tp"].to_numpy()[ok]
    dec_m = links["decoy"].to_numpy()[ok]
    w = _as_rows(decoy_weight, len(links["p"]))[ok]
    p = p[ok]
    fp_w = np.where(dec_m, w, 1.0) * (~tp_m)
    ntf = nt.astype(float)
    single = ntf == 0
    out = np.empty(len(grid))
    for i, t in enumerate(grid):
        keep = p >= t
        tp = np.bincount(s_idx[keep], weights=tp_m[keep], minlength=n_s)
        fp = np.bincount(s_idx[keep], weights=fp_w[keep], minlength=n_s)
        npred = tp + fp
        with np.errstate(divide="ignore", invalid="ignore"):
            prec = np.where(npred > 0, tp / npred, 0.0)
            rec = np.where(ntf > 0, tp / ntf, 0.0)
            f = np.where(prec + rec > 0, (1 + BETA2) * prec * rec / (BETA2 * prec + rec), 0.0)
        f = np.where(single & (npred == 0), 1.0, f)
        f = np.where(single & (npred > 0), 0.0, f)
        out[i] = f.mean()
    return out


def best_f05_threshold(links, nt, grid, decoy_weight=1.0, margin=0.0, contra_penalty=0.0):
    """(threshold, macro F0.5) maximising macro F0.5 over the grid (first maximum, like best_threshold)."""
    f = sweep_f05(links, nt, grid, decoy_weight, margin, contra_penalty)
    i = int(np.argmax(f))
    return float(grid[i]), float(f[i])


DEFAULT_MARGINS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7)
DEFAULT_PENALTIES = (0.0, 0.1, 0.2, 1.0)  # 1.0 = hard veto of contradicted pairs


def nested_rule_f05(links, nt, s_fold, grid, margin=0.0, contra_penalty=0.0, decoy_weight=1.0):
    """Leak-free macro F0.5 of a decision rule: the threshold is fit on the entities of K-1 folds and
    applied to the held-out fold; the concatenated held-out links give one number. Returns
    (nested macro F0.5, per-fold thresholds)."""
    s_fold = np.asarray(s_fold)
    link_fold = s_fold[links["s_idx"].to_numpy()]
    thr = np.full(links.height, np.nan)
    fold_thr = []
    dw = np.asarray(decoy_weight, dtype=np.float64)
    for k in np.unique(s_fold):
        lk, nt_k = subset_links(links, nt, s_fold != k)
        dw_k = dw[link_fold != k] if dw.ndim else float(dw)
        t, _ = best_f05_threshold(lk, nt_k, grid, dw_k, margin, contra_penalty)
        thr[link_fold == k] = t
        fold_thr.append(float(t))
    return metrics_at(links, nt, thr, decoy_weight, margin=margin, contra_penalty=contra_penalty)["macro_f05"], fold_thr


def select_rule(links, nt, s_fold, grid, margins=DEFAULT_MARGINS, penalties=DEFAULT_PENALTIES, decoy_weight=1.0, min_gain=1e-4):
    """Chooses (margin, contra_penalty) by nested macro F0.5; the plain threshold rule is kept unless
    a richer rule beats it by at least min_gain. Returns (rule dict, table of all rules)."""
    rows = []
    for m in margins:
        for c in penalties:
            f, ft = nested_rule_f05(links, nt, s_fold, grid, m, c, decoy_weight)
            rows.append({"margin": float(m), "contra_penalty": float(c), "nested_macro_f05": float(f), "fold_thresholds": ft})
    base = next(r for r in rows if r["margin"] == 0.0 and r["contra_penalty"] == 0.0)
    best = max(rows, key=lambda r: r["nested_macro_f05"])
    chosen = best if best["nested_macro_f05"] - base["nested_macro_f05"] >= min_gain else base
    rule = {"margin": chosen["margin"], "contra_penalty": chosen["contra_penalty"], "nested_macro_f05": chosen["nested_macro_f05"],
            "nested_macro_f05_threshold_only": base["nested_macro_f05"], "gain": chosen["nested_macro_f05"] - base["nested_macro_f05"],
            "min_gain": min_gain}
    return rule, rows


def best_threshold(rows):
    return max(rows, key=lambda r: r["macro_f05"])


def default_grid():
    return np.round(np.arange(0.05, 0.99, 0.01), 2)
