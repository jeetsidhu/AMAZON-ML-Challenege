"""Step 4b: pick the acceptance threshold for the test split under the expected prior shift.

The test split has more Source 2/3 records per Source 1 entity than training (5.75 vs
4.68), i.e. more unmatched "decoy" records per entity. Two ways to transfer the threshold
tuned on the training OOF predictions are compared here, both using record counts only
(no test labels):

  density  : re-weight every false link caused by a decoy by r = decoys_per_S1(test) /
             decoys_per_S1(train) in the OOF macro F0.5 and maximise (the previous method).
  prior    : treat the calibrated probability as P(match | pair) under the training prior and
             shift its odds by the change in the positive/negative pair ratio between
             train and test candidate pairs, then keep the training threshold
             (calibrate.shift_odds). Equivalent to moving the threshold on the training
             scale; reported as such.

Both are computed on the calibrated OOF probabilities (train.py writes p2_cal); the chosen
threshold (on the calibrated scale) and the full analysis go to <work>/model_meta.json and
<work>/threshold_analysis.json. --method chooses which one is written as `threshold`.
"""
import json
import os

import numpy as np
import polars as pl

import calibrate
from common import base_args, log, split_dir
from pair_features import truth_pairs
from thresholds import best_threshold, default_grid, link_table, metrics_at, sweep


def main():
    ap = base_args(__doc__)
    ap.add_argument("--ratio", type=float, default=None, help="override the estimated decoy-density ratio")
    ap.add_argument("--method", choices=["density", "prior", "train"], default="density")
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

    oof = pl.read_parquet(os.path.join(dtr, "oof.parquet"), columns=["t_rid", "s_rid", "label", "p2_cal"])
    s1_rids = rec.filter(pl.col("src") == 1)["rid"].to_numpy().astype(np.uint32)
    links, nt = link_table(oof.select("t_rid", "s_rid"), oof["p2_cal"].to_numpy(), truth, s1_rids)
    grid = default_grid()
    plain = sweep(links, nt, grid)
    density = sweep(links, nt, grid, decoy_weight=r)
    # prior shift: positive pairs / negative pairs among the candidates. Under the density model
    # the negatives grow by the decoy share; the odds ratio is (neg_train / neg_test_expected).
    y = oof["label"].to_numpy()
    n_pos, n_neg = float(y.sum()), float((1 - y).sum())
    # negative pairs that come from decoy records scale with r; the rest (wrong candidates of
    # matched records) do not
    matched_t = truth.select("t_rid").unique()
    n_decoy_pairs = float(oof.join(matched_t, on="t_rid", how="anti").height)
    neg_te = n_neg + (r - 1.0) * n_decoy_pairs
    odds_ratio = (n_pos / neg_te) / (n_pos / n_neg)
    shifted = calibrate.shift_odds(links["p"].to_numpy(), odds_ratio)
    links_shift = links.with_columns(pl.Series("p", shifted))
    prior_rows = sweep(links_shift, nt, grid, decoy_weight=r)
    thr_train = best_threshold(plain)["threshold"]
    prior_equiv = float(calibrate.sigmoid(calibrate.logit(thr_train) - np.log(odds_ratio)))  # same rule on the unshifted scale
    choice = {"train": thr_train, "density": best_threshold(density)["threshold"], "prior": round(prior_equiv, 4)}
    thr = choice[args.method]
    analysis = {
        "decoy_ratio": r, "odds_ratio_prior_shift": odds_ratio, "chosen_method": args.method, "threshold": thr,
        "candidates": choice,
        "oof_plain_at": {k: metrics_at(links, nt, v) for k, v in choice.items()},
        "oof_density_adjusted_at": {k: metrics_at(links, nt, v, decoy_weight=r) for k, v in choice.items()},
        "sweep_plain": plain, "sweep_density": density, "sweep_prior_shifted": prior_rows,
    }
    for k, v in choice.items():
        log(f"{k:8s} thr={v:.3f}  OOF macro F0.5={analysis['oof_plain_at'][k]['macro_f05']:.5f}  "
            f"density-adjusted={analysis['oof_density_adjusted_at'][k]['macro_f05']:.5f}")
    with open(os.path.join(args.work_dir, "threshold_analysis.json"), "w") as f:
        json.dump(analysis, f, indent=1)
    meta_path = os.path.join(args.work_dir, "model_meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    meta.update({"threshold": thr, "threshold_method": args.method, "decoy_ratio": r, "threshold_candidates": choice,
                 "oof_macro_f05_at_threshold": analysis["oof_plain_at"][args.method]["macro_f05"],
                 "adjusted_macro_f05_at_threshold": analysis["oof_density_adjusted_at"][args.method]["macro_f05"]})
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=1)
    log(f"selected threshold {thr} ({args.method})")


if __name__ == "__main__":
    main()
