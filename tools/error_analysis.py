"""Error analysis of the out-of-fold predictions at the tuned threshold.

Categorises every missed true record (false negative) and every wrong link (false positive)
by the failure mode that explains it, so that effort goes where the errors are:

  retrieval_miss    : the true Source 1 entity was not among the record's candidates (blocking)
  outranked         : it was a candidate but another candidate scored higher (namesake / assignment)
  below_threshold   : it was the best candidate but its probability was below the threshold
and for each category the share of records that are in Indic script, carry an alias, are a
domain-style name, have no address, or whose Source 1 entity has namesakes. Also lists the
worst 15 examples of each kind with the raw strings. Writes <work>/error_analysis.json.
"""
import argparse
import json
import os
import sys

import polars as pl

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
sys.path.insert(0, SRC)
from pair_features import truth_pairs  # noqa: E402


def share(df, col):
    return float(df[col].cast(pl.Float64).mean()) if df.height else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--examples", type=int, default=15)
    args = ap.parse_args()
    d = os.path.join(args.work_dir, "train")
    with open(os.path.join(args.work_dir, "model_meta.json")) as f:
        thr = json.load(f)["threshold"]
    rec = pl.read_parquet(os.path.join(d, "records.parquet"),
                          columns=["rid", "entity_id", "src", "country", "business_name", "business_address", "n_core",
                                   "f_indic", "f_alias", "n_domain", "a_norm", "group"])
    rec = rec.with_columns(pl.col("rid").cast(pl.UInt32), (pl.col("n_domain").fill_null("") != "").alias("f_domain"),
                           (pl.col("a_norm").fill_null("") == "").alias("f_noaddr"))
    s1 = rec.filter(pl.col("src") == 1)
    namesakes = s1.group_by("group").agg(pl.len().alias("n_namesakes"))
    s1 = s1.join(namesakes, on="group")
    truth = truth_pairs(rec, args.data_dir).drop("label")
    oof = pl.read_parquet(os.path.join(d, "oof.parquet"), columns=["t_rid", "s_rid", "label", "p2_cal"])
    best = oof.sort("p2_cal", descending=True).unique("t_rid", keep="first")
    links = best.filter(pl.col("p2_cal") >= thr)

    # ---- false negatives: every true pair, classified
    tp_pairs = truth.join(oof.select("t_rid", "s_rid", pl.col("p2_cal").alias("p_true")), on=["t_rid", "s_rid"], how="left")
    tp_pairs = tp_pairs.join(best.select("t_rid", pl.col("s_rid").alias("best_s"), pl.col("p2_cal").alias("p_best")), on="t_rid", how="left")
    fn = tp_pairs.with_columns(
        pl.when(pl.col("p_true").is_null()).then(pl.lit("retrieval_miss"))
        .when(pl.col("best_s") != pl.col("s_rid")).then(pl.lit("outranked"))
        .when(pl.col("p_best") < thr).then(pl.lit("below_threshold"))
        .otherwise(pl.lit("found")).alias("outcome"))
    t_side = rec.select(pl.col("rid").alias("t_rid"), pl.col("src").alias("t_src"), "f_indic", "f_alias", "f_domain", "f_noaddr",
                        pl.col("business_name").alias("t_name"), pl.col("business_address").alias("t_addr"))
    s_side = s1.select(pl.col("rid").alias("s_rid"), "country", "n_namesakes", pl.col("business_name").alias("s_name"),
                       pl.col("business_address").alias("s_addr"))
    fn = fn.join(t_side, on="t_rid").join(s_side, on="s_rid")
    out = {"threshold": thr, "true_pairs": truth.height, "false_negatives": {}, "false_positives": {}}
    for oc in ("retrieval_miss", "outranked", "below_threshold", "found"):
        sub = fn.filter(pl.col("outcome") == oc)
        out["false_negatives"][oc] = {
            "n": sub.height, "share_of_true_pairs": sub.height / max(truth.height, 1),
            "indic": share(sub, "f_indic"), "alias": share(sub, "f_alias"), "domain": share(sub, "f_domain"),
            "no_address": share(sub, "f_noaddr"), "s1_has_namesakes": float((sub["n_namesakes"] > 1).mean()) if sub.height else None,
            "source3_share": float((sub["t_src"] == 3).mean()) if sub.height else None,
            "by_country": dict(sub["country"].value_counts().iter_rows()) if sub.height else {},
        }
    # ---- false positives: links whose record is not a true match of that entity
    fp = links.join(truth.select("t_rid", pl.col("s_rid").alias("true_s")), on="t_rid", how="left").filter(
        (pl.col("true_s").is_null()) | (pl.col("true_s") != pl.col("s_rid")))
    fp = fp.with_columns(pl.when(pl.col("true_s").is_null()).then(pl.lit("decoy_linked")).otherwise(pl.lit("wrong_entity")).alias("kind"))
    fp = fp.join(t_side, on="t_rid").join(s_side, on="s_rid")
    for kd in ("decoy_linked", "wrong_entity"):
        sub = fp.filter(pl.col("kind") == kd)
        out["false_positives"][kd] = {
            "n": sub.height, "share_of_links": sub.height / max(links.height, 1),
            "indic": share(sub, "f_indic"), "alias": share(sub, "f_alias"), "domain": share(sub, "f_domain"),
            "no_address": share(sub, "f_noaddr"), "s1_has_namesakes": float((sub["n_namesakes"] > 1).mean()) if sub.height else None,
            "by_country": dict(sub["country"].value_counts().iter_rows()) if sub.height else {},
            "mean_p": float(sub["p2_cal"].mean()) if sub.height else None,
        }
    base = {"all_true_pairs": {k: share(fn, k) for k in ("f_indic", "f_alias", "f_domain", "f_noaddr")},
            "s1_has_namesakes": float((fn["n_namesakes"] > 1).mean())}
    out["base_rates"] = base
    ex = {}
    for oc in ("retrieval_miss", "outranked", "below_threshold"):
        sub = fn.filter(pl.col("outcome") == oc).head(args.examples)
        ex[oc] = [{"s1": r[0], "s1_addr": r[1], "record": r[2], "record_addr": r[3], "p_true": r[4], "p_best": r[5]}
                  for r in sub.select("s_name", "s_addr", "t_name", "t_addr", "p_true", "p_best").iter_rows()]
    for kd in ("decoy_linked", "wrong_entity"):
        sub = fp.filter(pl.col("kind") == kd).sort("p2_cal", descending=True).head(args.examples)
        ex[kd] = [{"s1": r[0], "s1_addr": r[1], "record": r[2], "record_addr": r[3], "p": r[4]}
                  for r in sub.select("s_name", "s_addr", "t_name", "t_addr", "p2_cal").iter_rows()]
    out["examples"] = ex
    with open(os.path.join(args.work_dir, "error_analysis.json"), "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"threshold {thr}; true pairs {truth.height}; links {links.height}")
    print("\n| false negatives | n | share of true pairs | indic | alias | domain | no address | S1 has namesakes | S3 share |")
    print("|---|---|---|---|---|---|---|---|---|")
    for oc, v in out["false_negatives"].items():
        f_ = lambda x: "-" if x is None else f"{x:.3f}"  # noqa: E731
        print(f"| {oc} | {v['n']} | {v['share_of_true_pairs']:.4f} | {f_(v['indic'])} | {f_(v['alias'])} | {f_(v['domain'])} | {f_(v['no_address'])} | {f_(v['s1_has_namesakes'])} | {f_(v['source3_share'])} |")
    print(f"\nbase rates over all true pairs: {json.dumps(base)}")
    print("\n| false positives | n | share of links | indic | alias | domain | no address | S1 has namesakes | mean p |")
    print("|---|---|---|---|---|---|---|---|---|")
    for kd, v in out["false_positives"].items():
        f_ = lambda x: "-" if x is None else f"{x:.3f}"  # noqa: E731
        print(f"| {kd} | {v['n']} | {v['share_of_links']:.4f} | {f_(v['indic'])} | {f_(v['alias'])} | {f_(v['domain'])} | {f_(v['no_address'])} | {f_(v['s1_has_namesakes'])} | {f_(v['mean_p'])} |")
    for k, rows in ex.items():
        print(f"\n### {k}")
        for r in rows[:6]:
            print("  ", json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main()
