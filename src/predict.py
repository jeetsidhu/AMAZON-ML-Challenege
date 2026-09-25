"""Step 5: score the test candidate pairs and write the two submission files.

output/matching_results.tsv : source1_entity_id, matched_entity_ids
output/candidate_pairs.tsv  : source1_entity_id, candidate_entity_ids (the exact pairs scored)
Every test Source 1 entity gets exactly one row in both files (empty list if nothing).
"""
import json
import os

import lightgbm as lgb
import numpy as np
import polars as pl

import calibrate
from common import Stage, base_args, log, split_dir
from decode import decode
from model import assign, house_numbers, iter_parts, part_files, stage2_context, to_np


def write_lists(s1_ids, links, id_col, path):
    """links: (source1_entity_id, entity_id). Writes one row per Source 1 id, tab separated."""
    agg = links.group_by("source1_entity_id").agg(pl.col("entity_id").unique().sort().str.join(",").alias(id_col))
    out = s1_ids.join(agg, on="source1_entity_id", how="left", maintain_order="left").with_columns(pl.col(id_col).fill_null(""))
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"source1_entity_id\t{id_col}\n")
        for sid, ids in out.iter_rows():
            f.write(f"{sid}\t{ids}\n")
    return out


def main():
    ap = base_args(__doc__)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--threshold", type=float, default=None, help="override the tuned threshold")
    ap.add_argument("--decoder", choices=["threshold", "expected_f"], default=None,
                    help="override the decision rule chosen by select_threshold.py")
    args = ap.parse_args()
    with Stage(args.work_dir, "predict"):
        run(args)


def run(args):
    d = split_dir(args.work_dir, "test")
    with open(os.path.join(args.work_dir, "model_meta.json")) as f:
        meta_json = json.load(f)
    thr = args.threshold if args.threshold is not None else meta_json["threshold"]
    f1 = meta_json["f1"]
    m1 = lgb.Booster(model_file=os.path.join(args.work_dir, "stage1.txt"))
    use_stage2 = meta_json.get("stage2", True)
    m2 = lgb.Booster(model_file=os.path.join(args.work_dir, "stage2.txt")) if use_stage2 else None
    cal_path = os.path.join(args.work_dir, "calibration.json")
    cal = calibrate.load(cal_path) if os.path.exists(cal_path) else None

    files = part_files(d)
    meta = pl.concat(list(iter_parts(files, ["pid", "t_rid", "s_rid"]))).sort("pid")
    n = meta.height
    log("test pairs", n)
    p1 = np.zeros(n, dtype=np.float32)
    for df in iter_parts(files):
        pid = df["pid"].to_numpy()
        p1[pid] = m1.predict(to_np(df, f1))
    if use_stage2:
        C, _ = stage2_context(meta, p1, house_numbers(d))
        p2 = np.zeros(n, dtype=np.float32)
        for df in iter_parts(files):
            pid = df["pid"].to_numpy()
            p2[pid] = m2.predict(np.hstack([to_np(df, f1), p1[pid, None], C[pid]]))
    else:
        p2 = p1
    # the threshold was tuned on the calibrated scale (train.py); apply the same calibrator here
    p2 = calibrate.apply(cal, p2).astype(np.float32) if meta_json.get("threshold_scale") == "calibrated" else p2
    decoder = args.decoder or meta_json.get("decoder", "threshold")
    if decoder == "expected_f":
        links = decode(meta, p2)
    else:
        links = assign(meta, p2, thr)
    log(f"decoder {decoder} (threshold {thr}): {links.height} links")

    rec = pl.read_parquet(os.path.join(d, "records.parquet"), columns=["rid", "entity_id", "src"])
    ids = rec.select(pl.col("rid").cast(pl.UInt32), "entity_id")
    s1_ids = rec.filter(pl.col("src") == 1).select(pl.col("entity_id").alias("source1_entity_id"))

    def named(df):
        return (
            df.select("t_rid", "s_rid")
            .join(ids.rename({"rid": "s_rid", "entity_id": "source1_entity_id"}), on="s_rid")
            .join(ids.rename({"rid": "t_rid"}), on="t_rid")
            .select("source1_entity_id", "entity_id")
        )

    os.makedirs(args.out_dir, exist_ok=True)
    m = write_lists(s1_ids, named(links), "matched_entity_ids", os.path.join(args.out_dir, "matching_results.tsv"))
    c = write_lists(s1_ids, named(meta), "candidate_entity_ids", os.path.join(args.out_dir, "candidate_pairs.tsv"))
    log(f"wrote {m.height} rows; {(m['matched_entity_ids'] != '').sum()} Source 1 entities matched; "
        f"{(c['candidate_entity_ids'] != '').sum()} with candidates")
    meta.with_columns(pl.Series("p1", p1), pl.Series("p2", p2)).write_parquet(os.path.join(d, "test_scores.parquet"))


if __name__ == "__main__":
    main()
