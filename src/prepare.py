"""Step 1: normalise every record of a split (train or test) in parallel.

Writes <work>/<split>/records.parquet with one row per record of all three sources:
  rid (global row id), entity_id, src (1/2/3), country, raw fields, normalised fields
  and the blocking-feature strings.
"""
import json
import multiprocessing as mp
import os

import polars as pl

import textnorm
from common import SOURCES, base_args, log, n_workers, read_tsv, source_path, split_dir

CHUNK = 50_000


def _init(lex):
    textnorm.set_lexicon(lex)


def _work(args):
    names, addrs, countries = args
    rows = [textnorm.normalize_record(n, a, c) for n, a, c in zip(names, addrs, countries)]
    return {k: [r[k] for r in rows] for k in rows[0]}


def main():
    ap = base_args(__doc__)
    ap.add_argument("--split", required=True, choices=["train", "test"])
    args = ap.parse_args()
    out_dir = split_dir(args.work_dir, args.split)
    with open(os.path.join(args.work_dir, "lexicon.json"), encoding="utf-8") as f:
        lex = json.load(f)

    frames = []
    for i, s in enumerate(SOURCES):
        df = read_tsv(source_path(args.data_dir, args.split, s)).with_columns(pl.lit(i + 1, pl.Int8).alias("src"))
        frames.append(df)
    df = pl.concat(frames).with_row_index("rid")
    log(f"{args.split}: {df.height} records")

    names = df["business_name"].to_list()
    addrs = df["business_address"].to_list()
    countries = df["country"].to_list()
    jobs = [(names[i:i + CHUNK], addrs[i:i + CHUNK], countries[i:i + CHUNK]) for i in range(0, len(names), CHUNK)]
    del names, addrs, countries
    parts = []
    with mp.get_context("spawn").Pool(n_workers(), initializer=_init, initargs=(lex,)) as pool:
        for k, res in enumerate(pool.imap(_work, jobs, chunksize=1)):
            parts.append(pl.DataFrame(res))
            if k % 40 == 0:
                log(f"normalised {min((k + 1) * CHUNK, df.height)}/{df.height}")
    norm = pl.concat(parts)
    df = pl.concat([df, norm], how="horizontal")
    df.write_parquet(os.path.join(out_dir, "records.parquet"))
    log("wrote", os.path.join(out_dir, "records.parquet"), df.shape)


if __name__ == "__main__":
    main()
