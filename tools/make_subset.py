"""Builds a small dataset directory with the challenge layout for fast experiments.

Sampling is done by *name group* (lower-cased business name + country of the Source 1
record), not by individual entity: 38 % of Source 1 records share their exact name with
another Source 1 record of the same country (chains, franchises, namesakes). Sampling
whole groups keeps those hard-to-separate namesakes together, so the subset stays about as
hard as the full data instead of becoming artificially easy.

For every sampled Source 1 entity all of its true Source 2/3 records are kept. Unmatched
("decoy") Source 2/3 records are sampled independently with rate = --frac * --decoy-mult,
so --decoy-mult 2 produces a subset with roughly the test split's higher decoy density.

Usage: python tools/make_subset.py --data-dir dataset --out-dir subset --frac 0.05 [--decoy-mult 1]
The output has train/ and test/ folders: test/ is a *second, disjoint* sample of the training
data (rate --test-frac, decoy multiplier --test-decoy-mult) with its own ground truth kept in
test/subset_ground_truth.tsv, so a full run of the pipeline can be scored locally.
"""
import argparse
import os
import sys

import polars as pl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from common import read_tsv  # noqa: E402


def write_tsv(df, path):
    df.write_csv(path, separator="\t", quote_style="never", null_value="")


def sample(s1, s23, gt, ex, lo, hi, decoy_lo, decoy_hi, seed):
    """Keep S1 name groups whose hash lands in [lo, hi) (fractions of 2**64) and decoys in [decoy_lo, decoy_hi)."""
    key = (pl.col("business_name").str.to_lowercase().fill_null("") + "|" + pl.col("country").fill_null("")).hash(seed=seed)
    u = (key.cast(pl.Float64) / float(2**64)).alias("u")
    s1k = s1.with_columns(u).filter((pl.col("u") >= lo) & (pl.col("u") < hi)).drop("u")
    keep_s1 = s1k.select(pl.col("entity_id").alias("source1_entity_id"))
    matched = ex.join(keep_s1, on="source1_entity_id").select(pl.col("matched_entity_ids").alias("entity_id"))
    is_matched = s23.select("entity_id").join(ex.select(pl.col("matched_entity_ids").alias("entity_id")), on="entity_id", how="semi")
    du = (pl.col("entity_id").hash(seed=seed + 1).cast(pl.Float64) / float(2**64)).alias("u")
    decoys = s23.join(is_matched, on="entity_id", how="anti").with_columns(du).filter(
        (pl.col("u") >= decoy_lo) & (pl.col("u") < decoy_hi)).drop("u").select("entity_id")
    keep_23 = pl.concat([matched, decoys]).unique()
    s23k = s23.join(keep_23, on="entity_id", how="semi")
    gtk = gt.join(keep_s1, on="source1_entity_id", how="semi")
    return s1k, s23k, gtk


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--frac", type=float, default=0.05)
    ap.add_argument("--decoy-mult", type=float, default=1.0)
    ap.add_argument("--test-frac", type=float, default=0.02)
    ap.add_argument("--test-decoy-mult", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    s1 = read_tsv(os.path.join(args.data_dir, "train", "train_source1.tsv"))
    s2 = read_tsv(os.path.join(args.data_dir, "train", "train_source2.tsv"))
    s3 = read_tsv(os.path.join(args.data_dir, "train", "train_source3.tsv"))
    gt = read_tsv(os.path.join(args.data_dir, "train", "train_ground_truth.tsv"))
    ex = (gt.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
          .drop_nulls("matched_entity_ids"))
    s23 = pl.concat([s2, s3])

    plan = [
        ("train", 0.0, args.frac, 0.0, args.frac * args.decoy_mult),
        ("test", args.frac, args.frac + args.test_frac, args.frac * args.decoy_mult,
         args.frac * args.decoy_mult + args.test_frac * args.test_decoy_mult),
    ]
    for split, lo, hi, dlo, dhi in plan:
        d = os.path.join(args.out_dir, split)
        os.makedirs(d, exist_ok=True)
        s1k, s23k, gtk = sample(s1, s23, gt, ex, lo, hi, dlo, dhi, args.seed)
        write_tsv(s1k, os.path.join(d, f"{split}_source1.tsv"))
        write_tsv(s23k.filter(pl.col("entity_id").str.starts_with("S2-")), os.path.join(d, f"{split}_source2.tsv"))
        write_tsv(s23k.filter(pl.col("entity_id").str.starts_with("S3-")), os.path.join(d, f"{split}_source3.tsv"))
        gt_name = "train_ground_truth.tsv" if split == "train" else "subset_ground_truth.tsv"
        write_tsv(gtk, os.path.join(d, gt_name))
        n_m = ex.join(gtk.select("source1_entity_id"), on="source1_entity_id", how="semi").height
        print(f"{split}: S1={s1k.height} S2/3={s23k.height} matched={n_m} decoys={s23k.height - n_m} "
              f"(S2/3 per S1 = {s23k.height / max(s1k.height, 1):.3f}, decoys/S1 = {(s23k.height - n_m) / max(s1k.height, 1):.3f})")


if __name__ == "__main__":
    main()
