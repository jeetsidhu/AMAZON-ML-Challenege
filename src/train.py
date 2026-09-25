"""Step 4: train the two-stage matcher on the training split and pick the threshold.

* Source 2/3 records are split into K folds (a matched record inherits the fold of its
  Source 1 entity, so an entity and all of its true matches share a fold).
* Stage 1 and stage 2 are trained with K-fold cross-fitting on a random row sample,
  giving out-of-fold (OOF) probabilities for *every* candidate pair of the training split.
* The acceptance threshold is chosen by maximising the challenge metric (macro F0.5 over
  all training Source 1 entities, singletons included) on the OOF stage-2 probabilities.
* Final stage-1 / stage-2 models are refit on the whole sample and saved for inference.
"""
import json
import os

import numpy as np
import polars as pl

from common import base_args, log, read_tsv, split_dir
from metrics import macro_f05
from model import (assign, house_numbers, iter_parts, part_files, stage1_features, stage2_context, to_np,
                   train_lgb)
from pair_features import truth_pairs


def evaluate(meta, p, thr, ids, truth_lists):
    links = assign(meta, p, thr)
    pred = (
        links.join(ids.rename({"rid": "t_rid", "entity_id": "t_id"}), on="t_rid")
        .join(ids.rename({"rid": "s_rid", "entity_id": "s"}), on="s_rid")
        .group_by("s").agg(pl.col("t_id").alias("m"))
    )
    return macro_f05(pred, truth_lists)


def oof_predict(files, feats_fn, models, folds_of_pid, n):
    """Streams the parts once; each row is scored by the model that did not see its fold."""
    out = np.zeros(n, dtype=np.float32)
    for part in files:
        df = pl.read_parquet(part)
        pid = df["pid"].to_numpy()
        X = feats_fn(df, pid)
        fk = folds_of_pid[pid]
        for k, m in enumerate(models):
            sel = fk == k
            if sel.any():
                out[pid[sel]] = m.predict(X[sel])
    return out


def main():
    ap = base_args(__doc__)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--rounds1", type=int, default=300)
    ap.add_argument("--rounds2", type=int, default=200)
    ap.add_argument("--sample-rows", type=int, default=5_000_000)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    d = split_dir(args.work_dir, "train")
    files = part_files(d)
    rec = pl.read_parquet(os.path.join(d, "records.parquet"), columns=["rid", "entity_id"])
    ids = rec.select(pl.col("rid").cast(pl.UInt32), "entity_id")
    truth = truth_pairs(rec, args.data_dir)
    gt = read_tsv(os.path.join(args.data_dir, "train", "train_ground_truth.tsv"))
    truth_lists = gt.select(
        pl.col("source1_entity_id").alias("s"),
        pl.col("matched_entity_ids").fill_null("").str.split(",").list.eval(pl.element().filter(pl.element() != "")).alias("m"),
    )

    meta = pl.concat(list(iter_parts(files, ["pid", "t_rid", "s_rid", "label"]))).sort("pid")
    n = meta.height
    assert meta["pid"][-1] == n - 1
    # folds on Source 2/3 records; matched records inherit their Source 1 entity's fold
    t_fold = truth.select("t_rid", (pl.col("s_rid").hash(seed=7) % args.folds).cast(pl.Int8).alias("tf"))
    meta = meta.join(t_fold, on="t_rid", how="left", maintain_order="left").with_columns(
        pl.coalesce("tf", (pl.col("t_rid").hash(seed=11) % args.folds).cast(pl.Int8)).alias("fold")
    ).drop("tf")
    folds = meta["fold"].to_numpy()
    y_all = meta["label"].to_numpy()
    log(f"pairs {n}, positives {int(y_all.sum())}, fold sizes {np.bincount(folds)}")

    f1 = stage1_features(files)
    log("stage-1 features", len(f1))
    rng = np.random.default_rng(args.seed)
    in_sample = np.zeros(n, dtype=bool)
    in_sample[rng.choice(n, min(args.sample_rows, n), replace=False)] = True
    parts = []
    for df in iter_parts(files, ["pid"] + f1):
        parts.append(df.filter(pl.Series(in_sample[df["pid"].to_numpy()])))
    sample = pl.concat(parts).sort("pid")
    del parts
    s_pid = sample["pid"].to_numpy()
    X1 = to_np(sample, f1)
    del sample
    ys, fs = y_all[s_pid], folds[s_pid]
    log("sample", X1.shape, "positive rate", ys.mean())

    # ---------------- stage 1 (OOF)
    m1s = []
    for k in range(args.folds):
        m1s.append(train_lgb(X1[fs != k], ys[fs != k], args.rounds1))
        log(f"stage-1 fold {k} trained")
    p1 = oof_predict(files, lambda df, pid: to_np(df, f1), m1s, folds, n)
    for thr in (0.3, 0.5, 0.7):
        log(f"stage-1 OOF thr={thr}", evaluate(meta, p1, thr, ids, truth_lists)[1])

    # ---------------- stage 2 (OOF) on stage-1 OOF context
    C, cnames = stage2_context(meta, p1, house_numbers(d))
    f2 = f1 + ["p1"] + cnames
    X2 = np.hstack([X1, p1[s_pid, None], C[s_pid]])
    m2s = []
    for k in range(args.folds):
        m2s.append(train_lgb(X2[fs != k], ys[fs != k], args.rounds2))
        log(f"stage-2 fold {k} trained")
    p2 = oof_predict(files, lambda df, pid: np.hstack([to_np(df, f1), p1[pid, None], C[pid]]), m2s, folds, n)

    best = (None, -1.0)
    for thr in np.round(np.arange(0.2, 0.95, 0.05), 2):
        f, st = evaluate(meta, p2, float(thr), ids, truth_lists)
        log(f"stage-2 OOF thr={thr:.2f}", st)
        if f > best[1]:
            best = (float(thr), f)
    log("best threshold", best)

    # ---------------- final models on the whole sample
    m1 = train_lgb(X1, ys, args.rounds1)
    m2 = train_lgb(X2, ys, args.rounds2)
    m1.save_model(os.path.join(args.work_dir, "stage1.txt"))
    m2.save_model(os.path.join(args.work_dir, "stage2.txt"))
    with open(os.path.join(args.work_dir, "model_meta.json"), "w") as f:
        json.dump({"f1": f1, "f2": f2, "threshold": best[0], "oof_macro_f05": best[1]}, f, indent=1)
    imp = sorted(zip(f2, m2.feature_importance("gain")), key=lambda x: -x[1])
    log("stage-2 top features", [(nm, round(float(g))) for nm, g in imp[:25]])
    meta.with_columns(pl.Series("p1", p1), pl.Series("p2", p2)).write_parquet(os.path.join(d, "oof.parquet"))


if __name__ == "__main__":
    main()
