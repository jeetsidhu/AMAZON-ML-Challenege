"""Automated leakage audit of a finished training run (exit code 1 on any violation).

Checks
  1. fold isolation      : every Source 1 name group lives in exactly one fold; every matched
                           Source 2/3 record carries the fold of its Source 1 entity; every
                           positive candidate pair is within one fold.
  2. lexicon isolation   : an Indic token whose only alignment evidence comes from fold k is
                           absent from the lexicon used to normalise fold k ("fold_k").
  3. train/eval overlap  : the OOF file scores each pair exactly once, with the model of its
                           own fold, and no pid is duplicated.
  4. label-shuffle canary: a small stage-1 model trained on shuffled labels with the same
                           strict folds must have OOF AUC ~ 0.5. Anything clearly above means
                           rows (or their duplicates) are shared between training and
                           evaluation folds.
  5. single-feature AUC  : no stage-1 feature separates the classes perfectly on its own
                           (a feature derived from the label would).
  6. test split hygiene  : test records / pairs carry no label or fold column, and the model's
                           feature list contains none of the bookkeeping columns.
Writes <work>/leakage_check.json.
"""
import collections
import json
import os
import sys

import numpy as np
import polars as pl

from build_lexicon import count_alignments
from common import Stage, base_args, log, read_tsv, source_path, split_dir
from model import NON_FEATURES, part_files, stage1_features, train_lgb, to_np
from textnorm import INDIC_RE


def auc(score, y):
    order = np.argsort(score)
    ranks = np.empty(len(score), dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1)
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def main():
    ap = base_args(__doc__)
    ap.add_argument("--canary-rows", type=int, default=1_000_000)
    ap.add_argument("--canary-rounds", type=int, default=50)
    args = ap.parse_args()
    with Stage(args.work_dir, "leakage_check"):
        ok = run(args)
    sys.exit(0 if ok else 1)


def run(args):
    dtr = split_dir(args.work_dir, "train")
    res, problems = {}, []

    # ---- 1. fold isolation
    rec = pl.read_parquet(os.path.join(dtr, "records.parquet"), columns=["rid", "entity_id", "src", "fold", "group", "country"])
    s1 = rec.filter(pl.col("src") == 1)
    g = s1.group_by("group").agg(pl.col("fold").n_unique().alias("nf"))
    multi = int((g["nf"] > 1).sum())
    res["name_groups"] = {"n": g.height, "split_across_folds": multi}
    if multi:
        problems.append(f"{multi} Source 1 name groups appear in more than one fold")
    gt = read_tsv(os.path.join(args.data_dir, "train", "train_ground_truth.tsv"))
    ex = (gt.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").drop_nulls("matched_entity_ids"))
    f = rec.select("entity_id", "fold")
    ex = ex.join(f.rename({"entity_id": "source1_entity_id", "fold": "fs"}), on="source1_entity_id").join(
        f.rename({"entity_id": "matched_entity_ids", "fold": "ft"}), on="matched_entity_ids")
    bad = int((ex["fs"] != ex["ft"]).sum())
    res["matched_records_fold_mismatch"] = bad
    if bad:
        problems.append(f"{bad} matched Source 2/3 records are not in their Source 1 entity's fold")

    # ---- 3. OOF overlap + positive pairs within fold
    oof = pl.read_parquet(os.path.join(dtr, "oof.parquet"))
    dup = oof.height - oof["pid"].n_unique()
    res["oof"] = {"pairs": oof.height, "duplicate_pids": dup,
                  "positive_pairs_cross_fold": int(oof.filter((pl.col("label") == 1) & (pl.col("s_fold") != pl.col("t_fold"))).height),
                  "cross_fold_pair_share": float((oof["s_fold"] != oof["t_fold"]).mean())}
    if dup:
        problems.append(f"{dup} duplicated pids in oof.parquet")
    if res["oof"]["positive_pairs_cross_fold"]:
        problems.append("positive pairs with mismatching folds")

    # ---- 2. lexicon isolation
    with open(os.path.join(args.work_dir, "lexicon.json"), encoding="utf-8") as fh:
        lexicons = json.load(fh)
    s1raw = read_tsv(source_path(args.data_dir, "train", "source1"))
    others = pl.concat([read_tsv(source_path(args.data_dir, "train", s)) for s in ("source2", "source3")])
    pairs = ex.select(pl.col("source1_entity_id").alias("s1"), pl.col("matched_entity_ids").alias("t"), pl.col("fs").alias("fold"))
    indic = others.filter(pl.col("business_name").str.contains(INDIC_RE.pattern)
                          | pl.col("business_address").fill_null("").str.contains(INDIC_RE.pattern))
    j = indic.join(pairs, left_on="entity_id", right_on="t").join(
        s1raw.select(pl.col("entity_id").alias("s1"), pl.col("business_name").alias("n1"), pl.col("business_address").alias("a1")), on="s1")
    per_fold = {k: count_alignments(j.filter(pl.col("fold") == k)) for k in sorted(j["fold"].unique().to_list())}
    lex_viol, lex_checked = 0, 0
    for k, (nc, ac) in per_fold.items():
        lex = lexicons.get(f"fold_{k}")
        if lex is None:
            continue
        others_tokens = collections.Counter()
        for kk, (nc2, ac2) in per_fold.items():
            if kk != k:
                others_tokens.update(nc2.keys())
                others_tokens.update(ac2.keys())
        for tok in list(nc.keys()) + list(ac.keys()):
            if tok not in others_tokens:  # evidence only in fold k
                lex_checked += 1
                if tok in lex["name"] or tok in lex["addr"]:
                    lex_viol += 1
    res["lexicon"] = {"fold_only_tokens_checked": lex_checked, "leaked_into_own_fold_lexicon": lex_viol,
                      "lexicons": {k: {"name": len(v["name"]), "addr": len(v["addr"])} for k, v in lexicons.items()}}
    if lex_viol:
        problems.append(f"{lex_viol} fold-specific Indic tokens present in their own fold's lexicon")

    # ---- 4./5. label-shuffle canary and single-feature AUC on a row sample
    files = part_files(dtr)
    feats = stage1_features(files)
    rng = np.random.default_rng(0)
    frames = []
    for fpath in files:
        df = pl.read_parquet(fpath, columns=["pid", "label"] + feats)
        frames.append(df)
    df = pl.concat(frames)
    if df.height > args.canary_rows:
        df = df.sample(args.canary_rows, seed=0)
    df = df.join(oof.select("pid", "s_fold", "t_fold"), on="pid", how="left")
    X = to_np(df, feats)
    y = df["label"].to_numpy().astype(np.int8)
    sf, tf = df["s_fold"].to_numpy(), df["t_fold"].to_numpy()
    y_sh = y.copy()
    for k in np.unique(sf):  # shuffle within fold: keeps each fold's positive rate
        idx = np.where(sf == k)[0]
        y_sh[idx] = y[rng.permutation(idx)]
    p = np.zeros(len(y), dtype=np.float32)
    for k in np.unique(sf):
        tr = (sf != k) & (tf != k)
        m = train_lgb(X[tr], y_sh[tr], args.canary_rounds, {"num_leaves": 31, "min_data_in_leaf": 50})
        p[sf == k] = m.predict(X[sf == k])
    canary = auc(p, y_sh)
    res["label_shuffle_canary_auc"] = canary
    if not (canary < 0.55):
        problems.append(f"label-shuffle canary OOF AUC {canary:.3f} (expected ~0.5): rows are shared between folds")
    single = sorted(((c, auc(X[:, i], y)) for i, c in enumerate(feats)), key=lambda x: -abs(x[1] - 0.5))[:8]
    res["top_single_feature_auc"] = [(c, round(a, 4)) for c, a in single]
    perfect = [c for c, a in single if a > 0.999 or a < 0.001]
    if perfect:
        problems.append(f"features separating the classes perfectly: {perfect}")

    # ---- 6. test split hygiene
    dte = split_dir(args.work_dir, "test")
    test_ok = True
    if os.path.exists(os.path.join(dte, "records.parquet")):
        cols = set(pl.read_parquet_schema(os.path.join(dte, "records.parquet")).keys())
        tfiles = part_files(dte)
        pcols = set(pl.read_parquet_schema(tfiles[0]).keys()) if tfiles else set()
        test_ok = "label" not in pcols and "fold" not in pcols and "label" not in cols
        res["test_split"] = {"records_have_fold_or_label": bool({"fold", "label"} & cols), "pairs_have_label": "label" in pcols}
    with open(os.path.join(args.work_dir, "model_meta.json")) as fh:
        meta = json.load(fh)
    leaky_feats = sorted(set(meta["f1"]) & NON_FEATURES)
    res["model_features_bookkeeping"] = leaky_feats
    if leaky_feats or not test_ok:
        problems.append(f"bookkeeping columns used as features / present in test: {leaky_feats}")

    res["problems"] = problems
    res["ok"] = not problems
    with open(os.path.join(args.work_dir, "leakage_check.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    log("leakage check", "OK" if not problems else "FAILED", json.dumps({k: v for k, v in res.items() if k != "lexicon"}, default=str)[:1500])
    for p_ in problems:
        log("  PROBLEM:", p_)
    return not problems


if __name__ == "__main__":
    main()
