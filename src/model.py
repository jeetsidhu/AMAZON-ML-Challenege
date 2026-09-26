"""Matching model: two-stage LightGBM + one-to-one assignment.

Stage 1 scores every candidate pair from its pairwise features.
Stage 2 re-scores each pair with the stage-1 probabilities of the competing
candidates around it (the other Source 1 records proposed for the same
Source 2/3 record, and the other Source 2/3 records proposed for the same
Source 1 record). Stage-2 training uses out-of-fold stage-1 predictions.

Assignment: each Source 2/3 record belongs to at most one Source 1 entity, so it
is linked to its highest-probability candidate if that probability >= threshold.
"""
import glob
import os

import lightgbm as lgb
import numpy as np
import polars as pl

from common import left_join_ordered

NON_FEATURES = {"pid", "t_rid", "s_rid", "label", "fold"}

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


def stage2_context(meta, p1, hn):
    """Context features from stage-1 probabilities; meta is (t_rid, s_rid) aligned with p1.

    hn: (rid, a_hn) house numbers, used for the consensus features: among the *other* confident
    candidates (p1 >= 0.5) of the same Source 1 entity, how many share this record's house
    number, and how many share the entity's own house number.
    Returns (float32 matrix, column names), rows aligned with meta.
    """
    p = pl.col("p1")
    df = meta.select("t_rid", "s_rid").with_columns(pl.Series("p1", p1)).with_columns(
        p.max().over("t_rid").alias("c_t_pmax"),
        p.sum().over("t_rid").alias("c_t_psum"),
        p.rank("ordinal", descending=True).over("t_rid").alias("c_t_prank"),
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


def assign(meta, p, threshold):
    """Best Source 1 per Source 2/3 record if p >= threshold -> DataFrame (t_rid, s_rid, p)."""
    best = (
        meta.select("t_rid", "s_rid").with_columns(pl.Series("p", p))
        .sort("p", descending=True)
        .unique("t_rid", keep="first")
    )
    return best.filter(pl.col("p") >= threshold)
