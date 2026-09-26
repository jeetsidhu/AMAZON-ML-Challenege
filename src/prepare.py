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
import shutil

import polars as pl

import textnorm
from common import Stage, SOURCES, base_args, left_join_ordered, log, n_workers, read_tsv, source_path, split_dir

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
        df = left_join_ordered(df, folds, "entity_id").with_columns(
            pl.col("fold").fill_null(-1).cast(pl.Int8), pl.col("group").fill_null(0))
        lex_key = pl.when(pl.col("fold") >= 0).then("fold_" + pl.col("fold").cast(pl.String)).otherwise(pl.lit("all"))
    else:
        lex_key = pl.lit("all")
    df = df.with_columns(lex_key.alias("lex_key"))
    log(f"{args.split}: {df.height} records; lexicon keys {df['lex_key'].value_counts().sort('lex_key').rows()}")

    n_rows = df.height
    cols = df.select("business_name", "business_address", "country", "lex_key")

    def jobs():
        # one chunk at a time: materialising all 12.5M rows as Python lists costs several GB
        for i in range(0, n_rows, CHUNK):
            c = cols.slice(i, CHUNK)
            yield (c["business_name"].to_list(), c["business_address"].to_list(), c["country"].to_list(), c["lex_key"].to_list())

    # each normalised chunk is joined to its raw rows and written straight to disk; the parts are then
    # merged with a streaming write, so memory stays at ~one chunk instead of the whole table (the
    # blocking-feature strings of 12.5M records are ~10 GB in RAM)
    parts_dir = os.path.join(out_dir, "records_parts")
    shutil.rmtree(parts_dir, ignore_errors=True)
    os.makedirs(parts_dir)
    raw = df.drop("lex_key")
    with mp.get_context("spawn").Pool(n_workers(), initializer=_init, initargs=(lexicons,)) as pool:
        for k, res in enumerate(pool.imap(_work, jobs(), chunksize=1)):
            part = pl.concat([raw.slice(k * CHUNK, CHUNK), pl.DataFrame(res)], how="horizontal")
            part.write_parquet(os.path.join(parts_dir, f"part_{k:05d}.parquet"))
            if k % 40 == 0:
                log(f"normalised {min((k + 1) * CHUNK, n_rows)}/{n_rows}")
    del raw, df
    out_path = os.path.join(out_dir, "records.parquet")
    pl.scan_parquet(os.path.join(parts_dir, "part_*.parquet")).sink_parquet(out_path)
    shutil.rmtree(parts_dir, ignore_errors=True)
    df = pl.read_parquet(out_path, columns=["rid"])
    log("wrote", os.path.join(out_dir, "records.parquet"), df.shape)


if __name__ == "__main__":
    main()
