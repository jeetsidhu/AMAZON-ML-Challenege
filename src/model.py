"""Matching model: two-stage LightGBM + many-to-one assignment with confidence-based rejection.

Stage 1 scores every candidate pair from its pairwise features.
Stage 2 re-scores each pair with the stage-1 probabilities of the competing
candidates around it (the other Source 1 records proposed for the same
Source 2/3 record, and the other Source 2/3 records proposed for the same
Source 1 record). Stage-2 training uses out-of-fold stage-1 predictions.

Assignment (many-to-one, no global one-to-one constraint): each Source 2/3 record belongs to
at most one Source 1 entity, while a Source 1 entity may receive any number of records. A record
is linked to its highest-probability candidate only when
  * that probability clears the global threshold (threshold_policy.py),
  * the margin over the record's second-best candidate is at least `margin`, and
  * the pair carries no strong contradiction (contradiction_flag), or its probability also clears
    the threshold raised by `contra_penalty` (penalty >= 1 is a hard veto).
Otherwise the record stays unmatched. The predictions are then aggregated per Source 1 entity
(predict.write_lists), which is what the metric scores.
"""
import glob
import os

import lightgbm as lgb
import numpy as np
import polars as pl

from common import left_join_ordered

NON_FEATURES = {"pid", "t_rid", "s_rid", "label", "fold"}
CONTRA_COLS = ["unit_conflict", "pc_conflict", "na_conflict", "hn_conflict", "st_eq"]

LGB_PARAMS = dict(
    objective="binary",
    learning_rate=0.1,
    num_leaves=127,
    min_data_in_leaf=200,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=1.0,
    max_bin=255,
    verbose=-1,
    num_threads=0,
)


def part_files(split_dir):
    return sorted(glob.glob(os.path.join(split_dir, "pairs", "part_*.parquet")))


def iter_parts(files, columns=None):
    for f in files:
        yield pl.read_parquet(f, columns=columns)


def stage1_features(files):
    cols = pl.read_parquet_schema(files[0]).keys()
    return [c for c in cols if c not in NON_FEATURES]


def house_numbers(split_dir):
    """rid -> house number (a_hn) of every record of a split."""
    return pl.read_parquet(os.path.join(split_dir, "records.parquet"), columns=["rid", "a_hn"]).select(
        pl.col("rid").cast(pl.UInt32), pl.col("a_hn").fill_null(""))


def contradiction_flag(df):
    """Strong contradiction of a candidate pair (bool per row of a pair-feature frame): different
    unit / flat numbers, different postal-like codes, identical names at contradicting addresses,
    or a house number that is neither equal nor a digit corruption of the other on the same street."""
    return (
        (df["unit_conflict"] == 1) | (df["pc_conflict"] == 1) | (df["na_conflict"] == 1)
        | ((df["hn_conflict"] == 1) & (df["st_eq"] == 1))
    ).to_numpy()


def stage2_context(meta, p1, hn):
    """Context features from stage-1 probabilities; meta is (t_rid, s_rid) aligned with p1.

    hn: (rid, a_hn) house numbers, used for the consensus features: among the *other* confident
    candidates (p1 >= 0.5) of the same Source 1 entity, how many share this record's house
    number, and how many share the entity's own house number.
    Candidate competition (per Source 2/3 record): best and second-best stage-1 score, the pair's
    margin over its strongest competitor (c_t_margin: p - best other candidate), rank, count.
    Returns (float32 matrix, column names), rows aligned with meta.
    """
    p = pl.col("p1")
    df = meta.select("t_rid", "s_rid").with_columns(pl.Series("p1", p1)).with_columns(
        p.max().over("t_rid").alias("c_t_pmax"),
        p.sum().over("t_rid").alias("c_t_psum"),
        p.rank("ordinal", descending=True).over("t_rid").alias("c_t_prank"),
        p.sort(descending=True).over("t_rid", mapping_strategy="join").list.get(1, null_on_oob=True).fill_null(0.0).alias("c_t_p2nd"),
        pl.len().over("t_rid").alias("c_t_n"),
        p.max().over("s_rid").alias("c_s_pmax"),
        p.sum().over("s_rid").alias("c_s_psum"),
        p.rank("ordinal", descending=True).over("s_rid").alias("c_s_prank"),
        (p > 0.5).sum().over("s_rid").alias("c_s_n05"),
        pl.len().over("s_rid").alias("c_s_n"),
    )
    df = df.with_columns(
        (pl.col("c_t_pmax") - p).alias("c_t_gap"),
        (2 * p - pl.col("c_t_psum")).alias("c_t_vs_rest"),
        (pl.col("c_s_psum") - p).alias("c_s_psum_other"),
        (pl.col("c_s_pmax") - p).alias("c_s_gap"),
        # margin over the strongest competitor: p - second best when this pair is the best, else p - best
        pl.when(pl.col("c_t_prank") == 1).then(p - pl.col("c_t_p2nd")).otherwise(p - pl.col("c_t_pmax")).alias("c_t_margin"),
    )
    df = left_join_ordered(df, hn.rename({"rid": "t_rid", "a_hn": "t_hn"}), "t_rid")
    df = left_join_ordered(df, hn.rename({"rid": "s_rid", "a_hn": "s_hn"}), "s_rid")
    strong = (p >= 0.5) & (pl.col("t_hn") != "")
    df = df.with_columns(
        (strong.cast(pl.Int32).sum().over("s_rid", "t_hn") - strong.cast(pl.Int32)).alias("c_hn_same_t"),
        ((strong & (pl.col("t_hn") == pl.col("s_hn"))).cast(pl.Int32).sum().over("s_rid")
         - (strong & (pl.col("t_hn") == pl.col("s_hn"))).cast(pl.Int32)).alias("c_hn_same_s"),
    ).with_columns(
        pl.when(pl.col("t_hn") == "").then(-1).otherwise(pl.col("c_hn_same_t")).alias("c_hn_same_t"),
        pl.when(pl.col("s_hn") == "").then(-1).otherwise(pl.col("c_hn_same_s")).alias("c_hn_same_s"),
    ).drop("t_rid", "s_rid", "p1", "t_hn", "s_hn")
    return df.to_numpy().astype(np.float32), df.columns


def train_lgb(X, y, rounds, params=None):
    params = dict(LGB_PARAMS, **(params or {}))
    return lgb.train(params, lgb.Dataset(X, label=y), num_boost_round=rounds)


def to_np(df, cols):
    return df.select(cols).to_numpy().astype(np.float32)


def best_candidates(meta, p, contra=None):
    """One row per Source 2/3 record: its best candidate (t_rid, s_rid, p), the second-best probability
    in its list (p2nd, 0 when there is none) and the best pair's contradiction flag (contra).
    Row order: the same as meta's first occurrence order is NOT kept; rows carry `i`, the row index
    of the best pair in meta."""
    df = meta.select("t_rid", "s_rid").with_row_index("i").with_columns(pl.Series("p", np.asarray(p, dtype=np.float32)))
    df = df.with_columns(pl.Series("contra", np.asarray(contra, dtype=bool)) if contra is not None else pl.lit(False).alias("contra"))
    df = df.sort("p", descending=True, maintain_order=True).with_columns(pl.int_range(pl.len()).over("t_rid").alias("j"))
    second = df.filter(pl.col("j") == 1).select("t_rid", pl.col("p").alias("p2nd"))
    best = df.filter(pl.col("j") == 0).drop("j").join(second, on="t_rid", how="left").with_columns(pl.col("p2nd").fill_null(0.0))
    return best


def accept(best, threshold, margin=0.0, contra_penalty=0.0):
    """Acceptance mask over the rows of best_candidates(): p >= thr AND p - p2nd >= margin AND
    (no contradiction OR p >= thr + contra_penalty). thr: scalar or one value per row."""
    p = best["p"].to_numpy().astype(np.float64)
    thr = np.asarray(threshold, dtype=np.float64)
    thr = thr if thr.ndim else np.full(len(p), float(thr))
    keep = (p >= thr) & (p - best["p2nd"].to_numpy() >= margin)
    if contra_penalty > 0:
        keep &= (~best["contra"].to_numpy()) | (p >= thr + contra_penalty)
    return keep


def assign(meta, p, threshold, margin=0.0, contra_penalty=0.0, contra=None):
    """Many-to-one assignment -> DataFrame (t_rid, s_rid, p): the best Source 1 candidate of every
    Source 2/3 record that passes `accept` (threshold, margin over the runner-up, contradiction rule).

    threshold: a scalar, or one threshold per row of meta (the pair is then accepted iff its own
    probability clears its own threshold).
    contra: optional bool per row of meta (model.contradiction_flag) used by contra_penalty."""
    best = best_candidates(meta, p, contra)
    thr = threshold if np.ndim(threshold) == 0 else np.asarray(threshold, dtype=np.float64)[best["i"].to_numpy()]
    keep = accept(best, thr, margin, contra_penalty)
    return best.filter(pl.Series(keep)).select("t_rid", "s_rid", "p")
