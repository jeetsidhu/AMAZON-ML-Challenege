"""Scores a matching_results.tsv against a ground-truth file with the challenge metric.

    python src/evaluate.py --pred output/matching_results.tsv --truth subset/test/subset_ground_truth.tsv \
        [--source1 subset/test/test_source1.tsv]   # for per-country breakdown

Reports macro F0.5 (singletons included, exactly as scored), micro precision / recall,
per-entity mean precision / recall, singleton accuracy, and the same per country when the
Source 1 file is given. Writes JSON to --out when given.
"""
import argparse
import json

import polars as pl

from common import read_tsv
from metrics import macro_f05


def id_lists(df, col):
    return df.select(
        pl.col("source1_entity_id").alias("s"),
        pl.col(col).fill_null("").str.split(",").list.eval(pl.element().filter(pl.element() != "")).alias("m"),
    )


def per_entity(pred, truth):
    """Per-entity precision / recall / F0.5 as a frame (s, nt, np, tp, p, r, f)."""
    j = truth.rename({"m": "true"}).join(pred.rename({"m": "pred"}), on="s", how="left").with_columns(
        pl.col("pred").fill_null(pl.lit([], dtype=pl.List(pl.String))))
    j = j.with_columns(
        pl.col("true").list.len().alias("nt"), pl.col("pred").list.len().alias("np"),
        pl.col("true").list.set_intersection("pred").list.len().alias("tp"))
    j = j.with_columns(
        pl.when(pl.col("np") > 0).then(pl.col("tp") / pl.col("np")).otherwise(0.0).alias("p"),
        pl.when(pl.col("nt") > 0).then(pl.col("tp") / pl.col("nt")).otherwise(0.0).alias("r"))
    f = pl.when((pl.col("nt") == 0) & (pl.col("np") == 0)).then(1.0).when(pl.col("nt") == 0).then(0.0).when(
        pl.col("p") + pl.col("r") > 0).then(1.25 * pl.col("p") * pl.col("r") / (0.25 * pl.col("p") + pl.col("r"))).otherwise(0.0)
    return j.with_columns(f.alias("f")).drop("true", "pred")


def summary(pe):
    """The numbers people quote, side by side, so the discrepancies between them are explicit."""
    non_single = pe.filter(pl.col("nt") > 0)
    return {
        "n_entities": pe.height,
        "macro_f05": float(pe["f"].mean()),
        "macro_f05_non_singletons": float(non_single["f"].mean()) if non_single.height else None,
        "singleton_share": float((pe["nt"] == 0).mean()),
        "singleton_acc": float(((pe["nt"] == 0) & (pe["np"] == 0)).sum() / max((pe["nt"] == 0).sum(), 1)),
        "micro_precision": float(pe["tp"].sum() / max(pe["np"].sum(), 1)),
        "micro_recall": float(pe["tp"].sum() / max(pe["nt"].sum(), 1)),
        "mean_entity_precision": float(non_single["p"].mean()) if non_single.height else None,
        "mean_entity_recall": float(non_single["r"].mean()) if non_single.height else None,
        "entities_perfect": float((pe["f"] == 1.0).mean()),
        "entities_zero": float((pe["f"] == 0.0).mean()),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--source1", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    pred = id_lists(read_tsv(args.pred), "matched_entity_ids")
    truth = id_lists(read_tsv(args.truth), "matched_entity_ids")
    f, stats = macro_f05(pred, truth)
    pe = per_entity(pred, truth)
    res = {"overall": summary(pe)}
    assert abs(res["overall"]["macro_f05"] - f) < 1e-9
    if args.source1:
        s1 = read_tsv(args.source1).select(pl.col("entity_id").alias("s"), pl.col("country").fill_null(""))
        pe_c = pe.join(s1, on="s", how="left")
        res["per_country"] = {c: summary(pe_c.filter(pl.col("country") == c)) for c in sorted(pe_c["country"].unique().to_list())}
    print(json.dumps(res, indent=1))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
