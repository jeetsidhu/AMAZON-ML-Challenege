"""Shared paths, IO helpers and logging."""
import argparse
import os
import sys
import time

import polars as pl

SOURCES = ("source1", "source2", "source3")
_T0 = time.time()


def log(*msg):
    print(f"[{time.time() - _T0:8.1f}s]", *msg, file=sys.stderr, flush=True)


def read_tsv(path):
    """Reads a challenge TSV. quote_char=None: fields are never quoted and may contain quotes."""
    df = pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False)
    return df.with_columns(pl.col(c).replace("", None) for c in df.columns)


def source_path(data_dir, split, source):
    return os.path.join(data_dir, split, f"{split}_{source}.tsv")


def base_args(desc):
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--data-dir", required=True, help="the challenge dataset/ directory (contains train/ and test/)")
    ap.add_argument("--work-dir", required=True, help="scratch directory for intermediate artefacts")
    return ap


def split_dir(work_dir, split):
    d = os.path.join(work_dir, split)
    os.makedirs(d, exist_ok=True)
    return d


def n_workers():
    return max(1, (os.cpu_count() or 2))
