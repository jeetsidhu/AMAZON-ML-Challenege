"""Entity-level error analysis of the out-of-fold predictions under the selected policy.

The metric is macro F0.5 over Source 1 entities, so errors are attributed per *entity* and weighted
by the F0.5 each one loses (1 - F0.5 of the entity, divided by the number of entities = the macro
F0.5 points the entity costs). Every entity with F0.5 < 1 is assigned to the first matching
category of

  singleton_fp     : a singleton (no true record) received a link             -> F0.5 = 0
  fp_wrong_entity  : it received a record that belongs to another entity
  fp_decoy         : it received an unmatched ("decoy") record
  blocking_miss    : one of its true records was never proposed as a candidate
  wrong_top        : a true record was proposed but another entity outscored it
  below_threshold  : a true record was the best candidate but rejected (by the threshold, the margin
                     over the runner-up, or the contradiction penalty - reported separately)

and the flag `missing_multi` (entity with >= 2 true records, some but not all found) is counted on
top. For every category: entities, macro F0.5 points lost, the share of the total loss, and the
composition of the affected records (Indic script, alias, domain-style name, no address, Source 1
with namesakes, Source 3 share) plus a per-country split; then the worst examples with raw strings.
Categories are printed in the order of their loss, which is the priority order for fixes.

    python tools/error_analysis.py --data-dir DATA --work-dir WORK [--policy FILE] [--examples 15]

Writes <work>/error_analysis.json and prints markdown.
"""
import argparse
import json
import os
import sys

import numpy as np
import polars as pl

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
sys.path.insert(0, SRC)
from checkpoint import Checkpoint  # noqa: E402
from model import best_candidates  # noqa: E402
from pair_features import truth_pairs  # noqa: E402
from threshold_policy import ThresholdPolicy, class_keys, record_columns  # noqa: E402

CATEGORIES = ("singleton_fp", "fp_wrong_entity", "fp_decoy", "blocking_miss", "wrong_top", "below_threshold")


def share(df, col):
    return float(df[col].cast(pl.Float64).mean()) if df.height else None


def load_policy(work_dir, path):
    if path:
        return ThresholdPolicy.load(path)
    ckpt = Checkpoint.resolve(work_dir, None)
    for cand in ([ckpt.path(os.path.join("thresholds", "selected.json")), ckpt.path(os.path.join("thresholds", "global_train.json"))] if ckpt else []) \
            + [os.path.join(work_dir, "threshold_policy.json")]:
        if os.path.exists(cand):
            return ThresholdPolicy.load(cand)
    with open(os.path.join(work_dir, "model_meta.json")) as f:
        meta = json.load(f)
    return ThresholdPolicy(meta["threshold"], name="model_meta_global", **meta.get("decision", {}))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--policy", default=None, help="threshold policy JSON (default: the selected policy of the latest checkpoint)")
    ap.add_argument("--examples", type=int, default=15)
    ap.add_argument("--out", default=None, help="output JSON (default <work>/error_analysis.json)")
    args = ap.parse_args()
    d = os.path.join(args.work_dir, "train")
    policy = load_policy(args.work_dir, args.policy)
    cols = sorted(set(record_columns(policy.class_by)) | {"rid", "entity_id", "src", "country", "business_name", "business_address", "n_core",
                                                          "f_indic", "f_alias", "n_domain", "a_norm", "group"})
    rec = pl.read_parquet(os.path.join(d, "records.parquet"), columns=cols)
    rec = rec.with_columns(pl.col("rid").cast(pl.UInt32), (pl.col("n_domain").fill_null("") != "").alias("f_domain"),
                           (pl.col("a_norm").fill_null("") == "").alias("f_noaddr"))
    s1 = rec.filter(pl.col("src") == 1)
    namesakes = s1.group_by("group").agg(pl.len().alias("n_namesakes"))
    s1 = s1.join(namesakes, on="group")
    truth = truth_pairs(rec, args.data_dir).drop("label")
    oof = pl.read_parquet(os.path.join(d, "oof.parquet"))
    contra = oof["contra"].to_numpy() if "contra" in oof.columns else None
    p = oof["p2_cal"].to_numpy()
    best = best_candidates(oof.select("t_rid", "s_rid"), p, contra)
    keys = class_keys(oof.select("t_rid", "s_rid"), rec, policy.class_by)
    thr = policy.thresholds_for(keys)[best["i"].to_numpy()]
    pb, p2 = best["p"].to_numpy().astype(np.float64), best["p2nd"].to_numpy()
    ok_thr = pb >= thr
    ok_margin = (pb - p2) >= policy.margin
    ok_contra = (~best["contra"].to_numpy()) | (pb >= thr + policy.contra_penalty) if policy.contra_penalty > 0 else np.ones(len(pb), dtype=bool)
    accepted = ok_thr & ok_margin & ok_contra
    best = best.with_columns(pl.Series("thr", thr), pl.Series("accepted", accepted), pl.Series("ok_thr", ok_thr),
                             pl.Series("ok_margin", ok_margin), pl.Series("ok_contra", ok_contra)).drop("i")
    links = best.filter(pl.col("accepted"))

    # ---- per true record: outcome
    tp_pairs = truth.join(oof.select("t_rid", "s_rid", pl.col("p2_cal").alias("p_true")), on=["t_rid", "s_rid"], how="left")
    tp_pairs = tp_pairs.join(best.select("t_rid", pl.col("s_rid").alias("best_s"), pl.col("p").alias("p_best"), "accepted", "ok_thr", "ok_margin", "ok_contra"), on="t_rid", how="left")
    outcome = (
        pl.when(pl.col("p_true").is_null()).then(pl.lit("blocking_miss"))
        .when(pl.col("best_s") != pl.col("s_rid")).then(pl.lit("wrong_top"))
        .when(~pl.col("accepted")).then(pl.lit("below_threshold"))
        .otherwise(pl.lit("found"))
    )
    reject_kind = (
        pl.when(pl.col("accepted") | (pl.col("best_s") != pl.col("s_rid")) | pl.col("p_true").is_null()).then(pl.lit(None))
        .when(~pl.col("ok_thr")).then(pl.lit("threshold"))
        .when(~pl.col("ok_margin")).then(pl.lit("margin"))
        .otherwise(pl.lit("contradiction"))
    )
    tp_pairs = tp_pairs.with_columns(outcome.alias("outcome"), reject_kind.alias("reject_kind"))

    # ---- per accepted link: kind
    lk = links.join(truth.select("t_rid", pl.col("s_rid").alias("true_s")), on="t_rid", how="left").with_columns(
        pl.when(pl.col("true_s") == pl.col("s_rid")).then(pl.lit("tp"))
        .when(pl.col("true_s").is_null()).then(pl.lit("fp_decoy")).otherwise(pl.lit("fp_wrong_entity")).alias("kind"))

    # ---- per entity: F0.5 and flags
    s_ids = s1.select("rid").rename({"rid": "s_rid"})
    nt = truth.group_by("s_rid").len().rename({"len": "nt"})
    npred = lk.group_by("s_rid").agg(pl.len().alias("np"), (pl.col("kind") == "tp").sum().alias("tp"),
                                     (pl.col("kind") == "fp_decoy").sum().alias("n_fp_decoy"), (pl.col("kind") == "fp_wrong_entity").sum().alias("n_fp_wrong"))
    per_true = tp_pairs.group_by("s_rid").agg(*[(pl.col("outcome") == oc).sum().alias("n_" + oc) for oc in ("blocking_miss", "wrong_top", "below_threshold")],
                                              *[(pl.col("reject_kind") == rk).sum().alias("n_rej_" + rk) for rk in ("threshold", "margin", "contradiction")])
    e = s_ids.join(nt, on="s_rid", how="left").join(npred, on="s_rid", how="left").join(per_true, on="s_rid", how="left").fill_null(0)
    e = e.with_columns(
        pl.when(pl.col("np") > 0).then(pl.col("tp") / pl.col("np")).otherwise(0.0).alias("prec"),
        pl.when(pl.col("nt") > 0).then(pl.col("tp") / pl.col("nt")).otherwise(0.0).alias("rec"))
    f = (pl.when((pl.col("nt") == 0) & (pl.col("np") == 0)).then(1.0).when(pl.col("nt") == 0).then(0.0)
         .when(pl.col("prec") + pl.col("rec") > 0).then(1.25 * pl.col("prec") * pl.col("rec") / (0.25 * pl.col("prec") + pl.col("rec"))).otherwise(0.0))
    e = e.with_columns(f.alias("f")).with_columns((1.0 - pl.col("f")).alias("loss"))
    cat = (pl.when(pl.col("f") >= 1.0).then(pl.lit("perfect"))
           .when((pl.col("nt") == 0) & (pl.col("np") > 0)).then(pl.lit("singleton_fp"))
           .when(pl.col("n_fp_wrong") > 0).then(pl.lit("fp_wrong_entity"))
           .when(pl.col("n_fp_decoy") > 0).then(pl.lit("fp_decoy"))
           .when(pl.col("n_blocking_miss") > 0).then(pl.lit("blocking_miss"))
           .when(pl.col("n_wrong_top") > 0).then(pl.lit("wrong_top"))
           .when(pl.col("n_below_threshold") > 0).then(pl.lit("below_threshold")).otherwise(pl.lit("other")))
    e = e.with_columns(cat.alias("category"), ((pl.col("nt") >= 2) & (pl.col("tp") > 0) & (pl.col("tp") < pl.col("nt"))).alias("missing_multi"))
    e = e.join(s1.select("rid", "country", "n_namesakes", pl.col("f_noaddr").alias("s_noaddr")).rename({"rid": "s_rid"}), on="s_rid")
    n_ent = e.height
    macro = float(e["f"].mean())
    total_loss = float(e["loss"].sum())

    t_side = rec.select(pl.col("rid").alias("t_rid"), pl.col("src").alias("t_src"), "f_indic", "f_alias", "f_domain", "f_noaddr",
                        pl.col("business_name").alias("t_name"), pl.col("business_address").alias("t_addr"))
    s_side = s1.select(pl.col("rid").alias("s_rid"), "country", "n_namesakes", pl.col("business_name").alias("s_name"), pl.col("business_address").alias("s_addr"))
    fn = tp_pairs.join(t_side, on="t_rid").join(s_side, on="s_rid")
    fp = lk.filter(pl.col("kind") != "tp").join(t_side, on="t_rid").join(s_side, on="s_rid")

    def composition(sub):
        return {"records": sub.height, "indic": share(sub, "f_indic"), "alias": share(sub, "f_alias"), "domain": share(sub, "f_domain"),
                "no_address": share(sub, "f_noaddr"), "s1_has_namesakes": float((sub["n_namesakes"] > 1).mean()) if sub.height else None,
                "source3_share": float((sub["t_src"] == 3).mean()) if sub.height else None}

    cats = {}
    for c in CATEGORIES:
        sub = e.filter(pl.col("category") == c)
        loss = float(sub["loss"].sum())
        rec_sub = {"singleton_fp": fp.join(sub.select("s_rid"), on="s_rid"), "fp_wrong_entity": fp.filter(pl.col("kind") == "fp_wrong_entity"),
                   "fp_decoy": fp.filter(pl.col("kind") == "fp_decoy"), "blocking_miss": fn.filter(pl.col("outcome") == "blocking_miss"),
                   "wrong_top": fn.filter(pl.col("outcome") == "wrong_top"), "below_threshold": fn.filter(pl.col("outcome") == "below_threshold")}[c]
        cats[c] = {"entities": sub.height, "entity_share": sub.height / n_ent, "macro_f05_points_lost": loss / n_ent,
                   "share_of_total_loss": loss / total_loss if total_loss else 0.0, "mean_entity_f05": float(sub["f"].mean()) if sub.height else None,
                   "by_country": {k: {"entities": int(v), "points_lost": float(sub.filter(pl.col("country") == k)["loss"].sum() / n_ent)}
                                  for k, v in sub["country"].value_counts().iter_rows()},
                   "missing_multi_entities": int(sub["missing_multi"].sum()),
                   "records": composition(rec_sub)}
        if c == "below_threshold":
            cats[c]["rejected_by"] = {k: int(fn.filter(pl.col("reject_kind") == k).height) for k in ("threshold", "margin", "contradiction")}
    order = sorted(CATEGORIES, key=lambda c: -cats[c]["macro_f05_points_lost"])
    out = {
        "policy": policy.to_dict(), "entities": n_ent, "macro_f05": macro, "total_points_lost": total_loss / n_ent,
        "perfect_entities": float((e["f"] >= 1.0).mean()), "links": int(links.height),
        "true_pairs": int(truth.height), "records_found": int(fn.filter(pl.col("outcome") == "found").height),
        "missing_multi": {"entities": int(e["missing_multi"].sum()), "points_lost": float(e.filter(pl.col("missing_multi"))["loss"].sum() / n_ent)},
        "priority": order, "categories": {c: cats[c] for c in order},
        "base_rates": {"true_pairs": composition(fn), "s1_has_namesakes": float((fn["n_namesakes"] > 1).mean())},
    }
    ex = {}
    for c in ("blocking_miss", "wrong_top", "below_threshold"):
        sub = fn.filter(pl.col("outcome") == c).sort("p_true", descending=True, nulls_last=True).head(args.examples)
        ex[c] = [{"s1": r[0], "s1_addr": r[1], "record": r[2], "record_addr": r[3], "p_true": r[4], "p_best": r[5], "n_namesakes": r[6]}
                 for r in sub.select("s_name", "s_addr", "t_name", "t_addr", "p_true", "p_best", "n_namesakes").iter_rows()]
    for c in ("fp_decoy", "fp_wrong_entity"):
        sub = fp.filter(pl.col("kind") == c).sort("p", descending=True).head(args.examples)
        ex[c] = [{"s1": r[0], "s1_addr": r[1], "record": r[2], "record_addr": r[3], "p": r[4], "p2nd": r[5], "n_namesakes": r[6]}
                 for r in sub.select("s_name", "s_addr", "t_name", "t_addr", "p", "p2nd", "n_namesakes").iter_rows()]
    sub = fp.join(e.filter(pl.col("category") == "singleton_fp").select("s_rid"), on="s_rid").sort("p", descending=True).head(args.examples)
    ex["singleton_fp"] = [{"s1": r[0], "s1_addr": r[1], "record": r[2], "record_addr": r[3], "p": r[4]}
                          for r in sub.select("s_name", "s_addr", "t_name", "t_addr", "p").iter_rows()]
    out["examples"] = ex
    with open(args.out or os.path.join(args.work_dir, "error_analysis.json"), "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)

    f_ = lambda x, nd=3: "-" if x is None else f"{x:.{nd}f}"  # noqa: E731
    print(f"policy {policy.describe()}\nentities {n_ent}; OOF macro F0.5 {macro:.5f}; perfect entities {out['perfect_entities']:.4f}; "
          f"links {links.height}; true records found {out['records_found']}/{truth.height}")
    print("\n| category (fix priority) | entities | share | macro F0.5 points lost | share of loss | records | indic | alias | domain | no address | S1 namesakes | S3 share |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for c in order:
        v, r = cats[c], cats[c]["records"]
        print(f"| {c} | {v['entities']} | {v['entity_share']:.4f} | {v['macro_f05_points_lost']:.5f} | {v['share_of_total_loss']:.3f} | {r['records']} | "
              f"{f_(r['indic'])} | {f_(r['alias'])} | {f_(r['domain'])} | {f_(r['no_address'])} | {f_(r['s1_has_namesakes'])} | {f_(r['source3_share'])} |")
    print(f"\nmissing multiple matches (entity with >= 2 true records, some found): {out['missing_multi']['entities']} entities, "
          f"{out['missing_multi']['points_lost']:.5f} points; below-threshold rejections by rule: {cats['below_threshold'].get('rejected_by')}")
    print(f"base rates over all true pairs: {json.dumps(out['base_rates'])}")
    print("\n| category | " + " | ".join(f"{c} entities / points" for c in sorted({k for v in cats.values() for k in v['by_country']})) + " |")
    countries = sorted({k for v in cats.values() for k in v["by_country"]})
    print("|---|" + "---|" * len(countries))
    for c in order:
        print(f"| {c} | " + " | ".join(f"{cats[c]['by_country'].get(k, {}).get('entities', 0)} / {cats[c]['by_country'].get(k, {}).get('points_lost', 0.0):.5f}" for k in countries) + " |")
    for k, rows in ex.items():
        print(f"\n### {k}")
        for r in rows[:6]:
            print("  ", json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main()
