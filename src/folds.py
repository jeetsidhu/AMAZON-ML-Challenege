"""Step 0: validation folds, assigned BEFORE any label-derived preprocessing runs.

Why this exists (leakage + realism):
* The Indic-script lexicon (build_lexicon.py) is learned from the training ground truth. It
  used to be learned from *all* of it and then applied to every record, so the validation
  folds were normalised with a lexicon that had seen their own labels. Folds therefore have
  to be known first, so that each fold can be normalised with a lexicon learned from the
  other folds only.
* 38 % of Source 1 records share their exact name with another Source 1 record of the same
  country (chains, franchises, namesakes). Random per-entity folds put "Dent Diner #1" in
  training and "Dent Diner #2" in validation; the model then learns the namesake's
  address pattern from the training copy. Folds are therefore assigned per *name group*
  (normalised core name + country): all namesakes share a fold.
* Matched Source 2/3 records inherit the fold of their Source 1 entity. Unmatched records
  ("decoys") get fold -1 here; train.py assigns them the fold of their best blocking
  candidate so that as many candidate pairs as possible are within one fold.

Schemes (--scheme):
  group   : K folds by hashed name group (default, the honest i.i.d. estimate)
  country : one fold per country label (leave-one-country-out; simulates the unseen
            country of the test split -- France never appears in training)

Writes <work>/train/folds.parquet: entity_id, fold (Int8), group (UInt64 hash).
"""
import os

import polars as pl

import textnorm
from common import Stage, base_args, log, read_tsv, source_path, split_dir


def name_group_key(names, countries):
    """Namesake key: sorted core-name tokens + country. Computed without the Indic lexicon
    (Source 1 names are Latin script; for Source 2/3 the unidecode fallback is used), so it
    depends on no label-derived state."""
    keys = []
    for n, c in zip(names, countries):
        core = textnorm.norm_name(n, c)["n_core"].split()
        keys.append(" ".join(sorted(core)) + "|" + (c or ""))
    return keys


def main():
    ap = base_args(__doc__)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--scheme", choices=["group", "country"], default="group")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    with Stage(args.work_dir, "folds"):
        run(args)


def run(args):
    textnorm.set_lexicon({})
    out_dir = split_dir(args.work_dir, "train")

    s1 = read_tsv(source_path(args.data_dir, "train", "source1"))
    s1 = s1.with_columns(pl.Series("group_key", name_group_key(s1["business_name"].to_list(), s1["country"].to_list())))
    if args.scheme == "country":
        countries = sorted(s1["country"].fill_null("").unique().to_list())
        fold_expr = pl.col("country").fill_null("").replace_strict(countries, list(range(len(countries)))).cast(pl.Int8)
        log("country folds:", dict(enumerate(countries)))
    else:
        fold_expr = (pl.col("group_key").hash(seed=args.seed) % args.folds).cast(pl.Int8)
    s1 = s1.with_columns(fold_expr.alias("fold"), pl.col("group_key").hash(seed=args.seed).alias("group"))
    n_groups = s1["group_key"].n_unique()
    log(f"S1 {s1.height} records in {n_groups} name groups; fold sizes {s1['fold'].value_counts().sort('fold')['count'].to_list()}")

    gt = read_tsv(os.path.join(args.data_dir, "train", "train_ground_truth.tsv"))
    matched = (
        gt.with_columns(pl.col("matched_entity_ids").str.split(","))
        .explode("matched_entity_ids")
        .drop_nulls("matched_entity_ids")
        .join(s1.select(pl.col("entity_id").alias("source1_entity_id"), "fold", "group"), on="source1_entity_id")
        .select(pl.col("matched_entity_ids").alias("entity_id"), "fold", "group")
    )
    others = pl.concat([read_tsv(source_path(args.data_dir, "train", s)).select("entity_id") for s in ("source2", "source3")])
    decoys = others.join(matched.select("entity_id"), on="entity_id", how="anti").with_columns(
        pl.lit(-1, pl.Int8).alias("fold"), pl.lit(0, pl.UInt64).alias("group"))
    folds = pl.concat([s1.select("entity_id", "fold", "group"), matched, decoys])
    folds.write_parquet(os.path.join(out_dir, "folds.parquet"))
    log(f"wrote folds.parquet: {s1.height} S1, {matched.height} matched S2/S3, {decoys.height} decoys (fold -1)")


if __name__ == "__main__":
    main()
