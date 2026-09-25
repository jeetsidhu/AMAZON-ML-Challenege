"""Learns an Indic-script -> Latin lexicon from the *training* ground truth.

Source 2/3 names are sometimes written in an Indic script (Devanagari, Telugu, ...)
while their Source 1 counterpart is in Latin script with the same number of tokens,
so tokens can be aligned by position. Addresses carry Indic-script state/city
components which are aligned against the Source 1 address components by
co-occurrence. Only the provided training data is used.
"""
import collections
import json
import os

import polars as pl

from common import base_args, log, read_tsv, source_path
from textnorm import INDIC_RE, indic_key, latin_tokens, to_ascii


def main():
    ap = base_args(__doc__)
    ap.add_argument("--min-count", type=int, default=2)
    args = ap.parse_args()

    s1 = read_tsv(source_path(args.data_dir, "train", "source1"))
    others = pl.concat([read_tsv(source_path(args.data_dir, "train", s)) for s in ("source2", "source3")])
    gt = read_tsv(os.path.join(args.data_dir, "train", "train_ground_truth.tsv"))
    pairs = (
        gt.with_columns(pl.col("matched_entity_ids").str.split(","))
        .explode("matched_entity_ids")
        .drop_nulls("matched_entity_ids")
        .rename({"source1_entity_id": "s1", "matched_entity_ids": "t"})
    )
    indic = others.filter(
        pl.col("business_name").str.contains(INDIC_RE.pattern)
        | pl.col("business_address").fill_null("").str.contains(INDIC_RE.pattern)
    )
    j = indic.join(pairs, left_on="entity_id", right_on="t").join(
        s1.select(pl.col("entity_id").alias("s1"), pl.col("business_name").alias("n1"),
                  pl.col("business_address").alias("a1")),
        on="s1",
    )
    log("indic-script pairs", j.height)

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

    name_lex = {}
    for u, cnt in name_cnt.items():
        v, c = cnt.most_common(1)[0]
        if c >= args.min_count and c / sum(cnt.values()) >= 0.5:
            name_lex[u] = v
    addr_lex = {}
    for u, cnt in addr_cnt.items():
        n = cnt.pop("__n__")
        v, c = cnt.most_common(1)[0]
        if c >= args.min_count and c / n >= 0.5:
            addr_lex[u] = v
    log(f"name lexicon {len(name_lex)} entries, address lexicon {len(addr_lex)} entries")
    os.makedirs(args.work_dir, exist_ok=True)
    with open(os.path.join(args.work_dir, "lexicon.json"), "w", encoding="utf-8") as f:
        json.dump({"name": name_lex, "addr": addr_lex}, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
