"""Step 5: score the test candidate pairs and write the two submission files.

output/matching_results.tsv : source1_entity_id, matched_entity_ids
output/candidate_pairs.tsv  : source1_entity_id, candidate_entity_ids (the exact pairs scored)
Every test Source 1 entity gets exactly one row in both files (empty list if nothing).

Model and decision rule are decoupled:
  --checkpoint  which trained checkpoint to score with (default: <work>/checkpoints/LATEST, falling
                back to the legacy <work>/stage*.txt files)
  --policy      threshold policy JSON (threshold_policy.py): global or per-class thresholds.
                Default: the checkpoint's thresholds/selected.json (written by tune_thresholds.py),
                else <work>/threshold_policy.json, else the checkpoint's OOF-optimal global threshold.
  --threshold   a plain global threshold, overriding the policy
  --reuse-scores  skip model inference when <work>/test/scores_<checkpoint>.parquet exists (written
                on the first run): evaluating one checkpoint under many policies then costs seconds.
output/prediction_meta.json records which checkpoint and policy produced the files.
"""
import json
import os

import lightgbm as lgb
import numpy as np
import polars as pl

import calibrate
from checkpoint import Checkpoint
from common import Stage, base_args, left_join_ordered, log, split_dir
from decode import decode
from model import assign, house_numbers, iter_parts, part_files, stage2_context, to_np
from threshold_policy import ThresholdPolicy, class_keys, record_columns


def write_lists(s1_ids, links, id_col, path):
    """links: (source1_entity_id, entity_id). Writes one row per Source 1 id, tab separated."""
    agg = links.group_by("source1_entity_id").agg(pl.col("entity_id").unique().sort().str.join(",").alias(id_col))
    out = left_join_ordered(s1_ids, agg, "source1_entity_id").with_columns(pl.col(id_col).fill_null(""))
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"source1_entity_id\t{id_col}\n")
        for sid, ids in out.iter_rows():
            f.write(f"{sid}\t{ids}\n")
    return out


def add_args(ap):
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--checkpoint", default=None, help="checkpoint name or directory (default: LATEST)")
    ap.add_argument("--policy", default=None, help="threshold policy JSON (default: the checkpoint's selected policy)")
    ap.add_argument("--threshold", type=float, default=None, help="override: one global threshold")
    ap.add_argument("--decoder", choices=["threshold", "expected_f"], default=None,
                    help="override the decision rule (default: threshold policy)")
    ap.add_argument("--reuse-scores", action="store_true", help="reuse cached test scores of this checkpoint if present")
    return ap


def main():
    ap = add_args(base_args(__doc__))
    args = ap.parse_args()
    with Stage(args.work_dir, "predict"):
        run(args)


def resolve_model(work_dir, checkpoint):
    """(checkpoint or None, path resolver) - legacy work dirs without checkpoints still work."""
    ckpt = Checkpoint.resolve(work_dir, checkpoint)
    if ckpt is not None and ckpt.exists():
        return ckpt, ckpt.path
    if checkpoint:
        raise SystemExit(f"checkpoint {checkpoint} not found under {work_dir}/checkpoints")
    return None, lambda f: os.path.join(work_dir, f)


def resolve_policy(args, ckpt, meta_json):
    if args.threshold is not None:
        return ThresholdPolicy(args.threshold, name=f"cli_{args.threshold}", scale=meta_json.get("threshold_scale", "calibrated"))
    if args.policy:
        if not os.path.exists(args.policy):
            raise SystemExit(f"policy file {args.policy} not found")
        return ThresholdPolicy.load(args.policy)
    for path in ([ckpt.path(os.path.join("thresholds", "selected.json"))] if ckpt else []) + [os.path.join(args.work_dir, "threshold_policy.json")]:
        if os.path.exists(path):
            return ThresholdPolicy.load(path)
    return ThresholdPolicy(meta_json["threshold"], name="model_meta_global", scale=meta_json.get("threshold_scale", "calibrated"))


def score_test(d, files, meta, n, m1, m2, f1, use_stage2, cal, calibrated):
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
    # the thresholds are tuned on the calibrated scale (train.py); apply the same calibrator here
    p2 = calibrate.apply(cal, p2).astype(np.float32) if calibrated else p2
    return p1, p2


def run(args):
    d = split_dir(args.work_dir, "test")
    ckpt, path = resolve_model(args.work_dir, args.checkpoint)
    with open(path("model_meta.json")) as f:
        meta_json = json.load(f)
    policy = resolve_policy(args, ckpt, meta_json)
    if policy.scale != meta_json.get("threshold_scale", "calibrated"):
        raise SystemExit(f"policy scale {policy.scale} does not match the model's {meta_json.get('threshold_scale')}")
    ckpt_name = ckpt.name if ckpt else "legacy"
    files = part_files(d)
    meta = pl.concat(list(iter_parts(files, ["pid", "t_rid", "s_rid"]))).sort("pid")
    n = meta.height
    log(f"test pairs {n}; checkpoint {ckpt_name}; {policy.describe()}")

    cache = os.path.join(d, f"scores_{ckpt_name}.parquet")
    if args.reuse_scores and os.path.exists(cache):
        sc = pl.read_parquet(cache)
        assert sc.height == n and (sc["pid"].to_numpy() == meta["pid"].to_numpy()).all(), "cached scores do not match the pair table"
        p1, p2 = sc["p1"].to_numpy(), sc["p2"].to_numpy()
        log(f"reused cached scores {cache}")
    else:
        f1 = meta_json["f1"]
        m1 = lgb.Booster(model_file=path("stage1.txt"))
        use_stage2 = meta_json.get("stage2", True)
        m2 = lgb.Booster(model_file=path("stage2.txt")) if use_stage2 else None
        cal = calibrate.load(path("calibration.json")) if os.path.exists(path("calibration.json")) else None
        p1, p2 = score_test(d, files, meta, n, m1, m2, f1, use_stage2, cal, meta_json.get("threshold_scale") == "calibrated")
        meta.with_columns(pl.Series("p1", p1), pl.Series("p2", p2)).write_parquet(cache)
        meta.with_columns(pl.Series("p1", p1), pl.Series("p2", p2)).write_parquet(os.path.join(d, "test_scores.parquet"))

    decoder = args.decoder or meta_json.get("decoder", "threshold")
    rec = pl.read_parquet(os.path.join(d, "records.parquet"), columns=sorted(set(record_columns(policy.class_by)) | {"rid", "entity_id", "src"}))
    keys = class_keys(meta, rec, policy.class_by)
    thr_rows = policy.thresholds_for(keys)
    if decoder == "expected_f":
        links = decode(meta, p2)
    else:
        links = assign(meta, p2, thr_rows)
    seen = set(policy.thresholds)
    unseen = sorted(set(keys.tolist()) - seen) if policy.thresholds else []
    if unseen:
        log(f"classes unseen when the policy was fit -> fallback threshold {policy.fallback():.3f}: {unseen}")
    log(f"decoder {decoder}: {links.height} links")

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
    per_class = {}
    if policy.class_by:
        kk = pl.DataFrame({"key": keys.astype(str), "thr": thr_rows}).with_row_index("i")
        best = meta.with_row_index("i").with_columns(pl.Series("p", p2)).sort("p", descending=True).unique("t_rid", keep="first").join(kk, on="i")
        agg = best.group_by("key").agg(pl.len().alias("records"), (pl.col("p") >= pl.col("thr")).sum().alias("links"), pl.col("thr").first())
        per_class = {r["key"]: {"records": int(r["records"]), "links": int(r["links"]), "threshold": float(r["thr"])} for r in agg.iter_rows(named=True)}
    info = {"checkpoint": ckpt_name, "checkpoint_dir": ckpt.dir if ckpt else None, "policy": policy.to_dict(), "decoder": decoder,
            "test_pairs": int(n), "links": int(links.height), "entities_matched": int((m["matched_entity_ids"] != "").sum()),
            "unseen_classes": unseen, "per_class": per_class, "out_dir": os.path.abspath(args.out_dir)}
    with open(os.path.join(args.out_dir, "prediction_meta.json"), "w") as f:
        json.dump(info, f, indent=1)
    if ckpt:
        ckpt.log_experiment({"kind": "predict", "policy": policy.name, "policy_thresholds": policy.thresholds, "default": policy.default,
                             "decoder": decoder, "links": int(links.height), "out_dir": os.path.abspath(args.out_dir)})
    return info


if __name__ == "__main__":
    main()
