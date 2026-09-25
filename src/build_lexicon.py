"""Learns an Indic-script -> Latin lexicon from the *training* ground truth.

Source 2/3 names are sometimes written in an Indic script (Devanagari, Telugu, ...)
while their Source 1 counterpart is in Latin script with the same number of tokens,
so tokens can be aligned by position. Addresses carry Indic-script state/city
components which are aligned against the Source 1 address components by
co-occurrence. Only the provided training data is used.

Leakage fix: the lexicon is label-derived preprocessing, so it must be fit on the training
folds only. This script therefore writes one lexicon per validation fold, learned from the
pairs of the *other* folds ("fold_k"), plus "all" (every training pair) which is the one used
for the test split. prepare.py normalises each training record with the lexicon of its own
fold, so no validation record is ever transliterated with knowledge of its own label.
Requires <work>/train/folds.parquet (folds.py).
"""
import collections
import json
import os

import polars as pl

from common import Stage, base_args, log, read_tsv, source_path, split_dir
from textnorm import INDIC_RE, indic_key, latin_tokens, to_ascii


def count_alignments(j):
    """Token / component co-occurrence counts of the Indic-script pairs in `j`.
    Returns per-pair-fold counters so that leave-one-fold-out lexicons are a subtraction."""
    name_cnt = collections.defaultdict(collections.Counter)
    addr_cnt = collections.defaultdict(collections.Counter)
    for name, addr, n1, a1 in j.select("business_name", "business_address", "n1", "a1").iter_rows():
        if name and INDIC_RE.search(name):
            tt, st = name.split(), n1.split()
            if len(tt) == len(st):
                for u, v in zip(tt, st):
                    if INDIC_RE.search(u):
                        lv = latin_tokens(to_ascii(v))
                        if len(lv) == 1:
                            name_cnt[indic_key(u)][lv[0]] += 1
        if addr and a1 and INDIC_RE.search(addr):
            s1_comps = {" ".join(latin_tokens(to_ascii(c))) for c in a1.split(",")}
            s1_comps.discard("")
            for c in addr.split(","):
                if INDIC_RE.search(c):
                    k = indic_key(c)
                    addr_cnt[k]["__n__"] += 1
                    for v in s1_comps:
                        addr_cnt[k][v] += 1
    return name_cnt, addr_cnt


def merge_counts(parts):
    name_cnt = collections.defaultdict(collections.Counter)
    addr_cnt = collections.defaultdict(collections.Counter)
    for nc, ac in parts:
        for k, c in nc.items():
            name_cnt[k].update(c)
        for k, c in ac.items():
            addr_cnt[k].update(c)
    return name_cnt, addr_cnt


def to_lexicon(name_cnt, addr_cnt, min_count):
    name_lex = {}
    for u, cnt in name_cnt.items():
        v, c = cnt.most_common(1)[0]
        if c >= min_count and c / sum(cnt.values()) >= 0.5:
            name_lex[u] = v
    addr_lex = {}
    for u, cnt in addr_cnt.items():
        cnt = cnt.copy()
        n = cnt.pop("__n__", 0)
        if not cnt or n == 0:
            continue
        v, c = cnt.most_common(1)[0]
        if c >= min_count and c / n >= 0.5:
            addr_lex[u] = v
    return {"name": name_lex, "addr": addr_lex}


def main():
    ap = base_args(__doc__)
    ap.add_argument("--min-count", type=int, default=2)
    args = ap.parse_args()
    with Stage(args.work_dir, "lexicon"):
        run(args)


def run(args):

    s1 = read_tsv(source_path(args.data_dir, "train", "source1"))
    others = pl.concat([read_tsv(source_path(args.data_dir, "train", s)) for s in ("source2", "source3")])
    gt = read_tsv(os.path.join(args.data_dir, "train", "train_ground_truth.tsv"))
    pairs = (
        gt.with_columns(pl.col("matched_entity_ids").str.split(","))
        .explode("matched_entity_ids")
        .drop_nulls("matched_entity_ids")
        .rename({"source1_entity_id": "s1", "matched_entity_ids": "t"})
    )
    folds = pl.read_parquet(os.path.join(split_dir(args.work_dir, "train"), "folds.parquet"), columns=["entity_id", "fold"])
    indic = others.filter(
        pl.col("business_name").str.contains(INDIC_RE.pattern)
        | pl.col("business_address").fill_null("").str.contains(INDIC_RE.pattern)
    )
    j = indic.join(pairs, left_on="entity_id", right_on="t").join(
        s1.select(pl.col("entity_id").alias("s1"), pl.col("business_name").alias("n1"),
                  pl.col("business_address").alias("a1")),
        on="s1",
    ).join(folds.rename({"entity_id": "s1"}), on="s1", how="left").with_columns(pl.col("fold").fill_null(-1))
    log("indic-script pairs", j.height)

    # counts per fold of the Source 1 entity; the lexicon for validation fold k uses every fold but k
    fold_ids = sorted(j["fold"].unique().to_list())
    per_fold = {k: count_alignments(j.filter(pl.col("fold") == k)) for k in fold_ids}
    lexicons = {"all": to_lexicon(*merge_counts(per_fold.values()), args.min_count)}
    for k in fold_ids:
        if k < 0:
            continue
        lexicons[f"fold_{k}"] = to_lexicon(*merge_counts(v for kk, v in per_fold.items() if kk != k), args.min_count)
    for name, lex in lexicons.items():
        log(f"lexicon {name}: name {len(lex['name'])} entries, address {len(lex['addr'])} entries")
    os.makedirs(args.work_dir, exist_ok=True)
    with open(os.path.join(args.work_dir, "lexicon.json"), "w", encoding="utf-8") as f:
        json.dump(lexicons, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
