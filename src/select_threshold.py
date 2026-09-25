"""Step 4b: pick the acceptance threshold for the test set.

The test split has noticeably more Source 2/3 records per Source 1 entity than the
training split (5.75 vs 4.68), i.e. more unmatched "decoy" records per entity. We
estimate the decoy-density ratio r from *record counts only* (no test labels):

    decoys per S1 = (#S2/S3 records / #S1 records) - (true matches per S1 in train)
    r = decoys_per_S1(test) / decoys_per_S1(train)

and choose the threshold that maximises the out-of-fold macro F0.5 on train after
weighting each false positive caused by a decoy record by r. With r = 1 this is
exactly the plain OOF macro F0.5 optimisation done in train.py.
Writes the chosen threshold into <work>/model_meta.json.
"""
import json
import os

import numpy as np
import polars as pl

from common import base_args, log, split_dir
from pair_features import truth_pairs


def adjusted_macro_f05(best, thr, s1, n_true, r):
    links = best.filter(pl.col("p2") >= thr).with_columns(
        (pl.col("true_s") == pl.col("s_rid")).fill_null(False).alias("tp"),
        pl.col("true_s").is_null().alias("decoy"),
    )
    agg = links.group_by("s_rid").agg(
        pl.col("tp").sum().alias("tp"),
        (~pl.col("tp") & pl.col("decoy")).sum().alias("fp_decoy"),
        (~pl.col("tp") & ~pl.col("decoy")).sum().alias("fp_other"),
    )
    a = s1.join(agg, on="s_rid", how="left").join(n_true, on="s_rid", how="left").fill_null(0)
    tp = a["tp"].to_numpy().astype(float)
    fp = a["fp_other"].to_numpy() + r * a["fp_decoy"].to_numpy()
    nt = a["nt"].to_numpy().astype(float)
    npred = tp + fp
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(npred > 0, tp / npred, 0.0)
        rc = np.where(nt > 0, tp / nt, 0.0)
        f = np.where(p + rc > 0, 1.25 * p * rc / (0.25 * p + rc), 0.0)
    f = np.where((nt == 0) & (npred == 0), 1.0, f)
    f = np.where((nt == 0) & (npred > 0), 0.0, f)
    return float(f.mean())


def main():
    ap = base_args(__doc__)
    ap.add_argument("--ratio", type=float, default=None, help="override the estimated decoy-density ratio")
    args = ap.parse_args()
    dtr, dte = split_dir(args.work_dir, "train"), split_dir(args.work_dir, "test")
    rec = pl.read_parquet(os.path.join(dtr, "records.parquet"), columns=["rid", "entity_id", "src"])
    truth = truth_pairs(rec, args.data_dir).drop("label")
    src_tr = rec["src"].value_counts().sort("src")["count"].to_list()
    src_te = pl.read_parquet(os.path.join(dte, "records.parquet"), columns=["src"])["src"].value_counts().sort("src")["count"].to_list()
    m = truth.height / src_tr[0]
    dec_tr = (src_tr[1] + src_tr[2]) / src_tr[0] - m
    dec_te = (src_te[1] + src_te[2]) / src_te[0] - m
    r = args.ratio if args.ratio is not None else max(1.0, dec_te / dec_tr)
    log(f"true matches/S1 (train) {m:.3f}; decoys/S1 train {dec_tr:.3f} test {dec_te:.3f}; ratio r={r:.3f}")

    oof = pl.read_parquet(os.path.join(dtr, "oof.parquet"), columns=["t_rid", "s_rid", "p2"])
    best = (
        oof.sort("p2", descending=True).unique("t_rid", keep="first")
        .join(truth.rename({"s_rid": "true_s"}), on="t_rid", how="left")
    )
    s1 = rec.filter(pl.col("src") == 1).select(pl.col("rid").cast(pl.UInt32).alias("s_rid"))
    n_true = truth.group_by("s_rid").agg(pl.len().alias("nt"))
    grid = np.round(np.arange(0.50, 0.96, 0.05), 2)
    scores = {float(t): (adjusted_macro_f05(best, t, s1, n_true, 1.0), adjusted_macro_f05(best, t, s1, n_true, r)) for t in grid}
    for t, (f1, fr) in scores.items():
        log(f"thr={t:.2f}  OOF macro F0.5={f1:.5f}  density-adjusted={fr:.5f}")
    thr = max(scores, key=lambda t: scores[t][1])
    meta_path = os.path.join(args.work_dir, "model_meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    meta.update({"threshold": thr, "decoy_ratio": r, "oof_macro_f05_at_threshold": scores[thr][0],
                 "adjusted_macro_f05_at_threshold": scores[thr][1]})
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=1)
    log(f"selected threshold {thr}")


if __name__ == "__main__":
    main()
