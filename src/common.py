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


class Stage:
    """Context manager that appends wall-clock / CPU time of one stage to <work>/profile.json.

        with Stage(work_dir, "blocking_train"):
            ...

    CPU time is the process + children CPU seconds, so cpu/wall > 1 shows the stage actually
    ran in parallel; cpu/wall ~ 1 on a multi-core box means it was effectively sequential."""

    def __init__(self, work_dir, name):
        self.work_dir, self.name = work_dir, name

    def __enter__(self):
        self.t0, self.c0 = time.time(), _cpu_seconds()
        return self

    def __exit__(self, *exc):
        import json
        wall, cpu = time.time() - self.t0, _cpu_seconds() - self.c0
        rec = {"wall_s": round(wall, 2), "cpu_s": round(cpu, 2), "cpu_per_wall": round(cpu / max(wall, 1e-6), 2),
               "workers": n_workers(), "ok": exc[0] is None}
        os.makedirs(self.work_dir, exist_ok=True)
        path = os.path.join(self.work_dir, "profile.json")
        try:
            with open(path) as f:
                prof = json.load(f)
        except (OSError, ValueError):
            prof = {}
        prof[self.name] = rec
        with open(path, "w") as f:
            json.dump(prof, f, indent=1)
        log(f"stage {self.name}: {wall:.1f}s wall, {cpu:.1f}s cpu ({rec['cpu_per_wall']}x)")


def _cpu_seconds():
    t = os.times()
    return t.user + t.system + t.children_user + t.children_system
