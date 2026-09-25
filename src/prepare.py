"""Step 1: normalise every record of a split (train or test) in parallel.

Writes <work>/<split>/records.parquet with one row per record of all three sources:
  rid (global row id), entity_id, src (1/2/3), country, raw fields, normalised fields,
  the blocking-feature strings and (train only) the validation fold.

Leakage fix: a training record is transliterated with the Indic lexicon learned WITHOUT
its own fold ("fold_k" in lexicon.json); decoys (fold -1) and every test record use the
lexicon learned from all training pairs ("all"), exactly as at inference time.
"""
import json
import multiprocessing as mp
import os

import polars as pl

import textnorm
from common import Stage, SOURCES, base_args, log, n_workers, read_tsv, source_path, split_dir

CHUNK = 50_000


_LEX = {}


def _init(lexicons):
    _LEX.update(lexicons)


def _work(args):
    names, addrs, countries, lex_keys = args
    rows = []
    current = None
    for n, a, c, k in zip(names, addrs, countries, lex_keys):
        if k != current:  # records are grouped by lexicon key, so this switches rarely
            textnorm.set_lexicon(_LEX.get(k) or _LEX["all"])
            current = k
        rows.append(textnorm.normalize_record(n, a, c))
    return {k: [r[k] for r in rows] for k in rows[0]}


def main():
    ap = base_args(__doc__)
    ap.add_argument("--split", required=True, choices=["train", "test"])
    args = ap.parse_args()
    with Stage(args.work_dir, "prepare_" + args.split):
        run(args)


def run(args):
    out_dir = split_dir(args.work_dir, args.split)
    with open(os.path.join(args.work_dir, "lexicon.json"), encoding="utf-8") as f:
        lexicons = json.load(f)
    if "all" not in lexicons:  # lexicon written by the old single-lexicon build_lexicon.py
        lexicons = {"all": lexicons}

    frames = []
    for i, s in enumerate(SOURCES):
        df = read_tsv(source_path(args.data_dir, args.split, s)).with_columns(pl.lit(i + 1, pl.Int8).alias("src"))
        frames.append(df)
    df = pl.concat(frames).with_row_index("rid")
    if args.split == "train":
        folds = pl.read_parquet(os.path.join(out_dir, "folds.parquet"), columns=["entity_id", "fold", "group"])
        df = df.join(folds, on="entity_id", how="left", maintain_order="left").with_columns(
            pl.col("fold").fill_null(-1).cast(pl.Int8), pl.col("group").fill_null(0))
        lex_key = pl.when(pl.col("fold") >= 0).then("fold_" + pl.col("fold").cast(pl.String)).otherwise(pl.lit("all"))
    else:
        lex_key = pl.lit("all")
    df = df.with_columns(lex_key.alias("lex_key"))
    log(f"{args.split}: {df.height} records; lexicon keys {df['lex_key'].value_counts().sort('lex_key').rows()}")

    names = df["business_name"].to_list()
    addrs = df["business_address"].to_list()
    countries = df["country"].to_list()
    keys = df["lex_key"].to_list()
    jobs = [(names[i:i + CHUNK], addrs[i:i + CHUNK], countries[i:i + CHUNK], keys[i:i + CHUNK])
            for i in range(0, len(names), CHUNK)]
    del names, addrs, countries, keys
    parts = []
    with mp.get_context("spawn").Pool(n_workers(), initializer=_init, initargs=(lexicons,)) as pool:
        for k, res in enumerate(pool.imap(_work, jobs, chunksize=1)):
            parts.append(pl.DataFrame(res))
            if k % 40 == 0:
                log(f"normalised {min((k + 1) * CHUNK, df.height)}/{df.height}")
    norm = pl.concat(parts)
    df = pl.concat([df.drop("lex_key"), norm], how="horizontal")
    df.write_parquet(os.path.join(out_dir, "records.parquet"))
    log("wrote", os.path.join(out_dir, "records.parquet"), df.shape)


if __name__ == "__main__":
    main()
