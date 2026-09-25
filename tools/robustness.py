"""Robustness of the tuned threshold and of the calibration under simulated distribution shift.

Uses only the OOF predictions of the training split (<work>/train/oof.parquet), so nothing is
fit on test data. Scenarios are built by re-sampling the OOF candidate pairs:

  decoys_x{m}   : unmatched Source 2/3 records are up-weighted m times (the test split has
                  ~2x the decoy density of training; m = 3 is a stress test)
  matched_x{f}  : a random fraction f of the matched Source 2/3 records is removed, from the
                  candidates AND from the ground truth (fewer true matches per entity)
  singletons_x2 : singleton share doubled by removing all matches of a random set of entities
  source_{2|3}  : only Source 2 (or 3) records kept
  indic_only    : only Source 2/3 records in Indic script (India, transliteration-heavy)

For every scenario: macro F0.5 at the training threshold, at the scenario's own best
threshold (the "regret" is the gap), at the prior-shift-corrected threshold, and the
calibration (ECE / Brier) of the Platt probabilities with and without prior correction.
Writes <work>/robustness.json and prints a table.
"""
import argparse
import json
import os
import sys

import numpy as np
import polars as pl

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
sys.path.insert(0, SRC)
import calibrate  # noqa: E402
from pair_features import truth_pairs  # noqa: E402
from thresholds import best_threshold, default_grid, link_table, metrics_at, sweep  # noqa: E402


def evaluate(meta, p, truth, s1_rids, thr_train, grid, decoy_weight=1.0, odds_ratio=1.0):
    links, nt = link_table(meta, p, truth, s1_rids)
    rows = sweep(links, nt, grid, decoy_weight)
    own = best_threshold(rows)
    at_train = metrics_at(links, nt, thr_train, decoy_weight)
    thr_prior = float(calibrate.sigmoid(calibrate.logit(thr_train) - np.log(odds_ratio)))
    at_prior = metrics_at(links, nt, thr_prior, decoy_weight)
    return {"at_train_threshold": at_train, "own_best": own, "regret": own["macro_f05"] - at_train["macro_f05"],
            "prior_corrected_threshold": thr_prior, "at_prior_corrected": at_prior}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()
    d = os.path.join(args.work_dir, "train")
    rec = pl.read_parquet(os.path.join(d, "records.parquet"), columns=["rid", "entity_id", "src", "f_indic", "country"])
    truth = truth_pairs(rec, args.data_dir).drop("label")
    s1_rids = rec.filter(pl.col("src") == 1)["rid"].to_numpy().astype(np.uint32)
    oof = pl.read_parquet(os.path.join(d, "oof.parquet"))
    with open(os.path.join(args.work_dir, "model_meta.json")) as f:
        thr_train = json.load(f)["threshold"]
    grid = default_grid()
    p = oof["p2_cal"].to_numpy()
    y = oof["label"].to_numpy()
    t_info = rec.select(pl.col("rid").cast(pl.UInt32).alias("t_rid"), "src", "f_indic")
    oof = oof.join(t_info, on="t_rid", how="left")
    matched_t = truth.select("t_rid").unique()
    oof = oof.with_columns(pl.col("t_rid").is_in(matched_t["t_rid"].implode()).alias("t_matched"))
    rng = np.random.default_rng(args.seed)
    n_pos, n_neg = float(y.sum()), float((1 - y).sum())
    n_decoy_pairs = float(oof.filter(~pl.col("t_matched")).height)

    def odds_ratio_for(decoy_mult=1.0, matched_keep=1.0):
        pos = n_pos * matched_keep
        neg = (n_neg - n_decoy_pairs) * matched_keep + n_decoy_pairs * decoy_mult
        return (pos / neg) / (n_pos / n_neg)

    scenarios = {}
    base = evaluate(oof, p, truth, s1_rids, thr_train, grid)
    scenarios["train_distribution"] = {**base, "n_pairs": oof.height}
    for m in (2.0, 3.0):
        scenarios[f"decoys_x{m:g}"] = {**evaluate(oof, p, truth, s1_rids, thr_train, grid, decoy_weight=m, odds_ratio=odds_ratio_for(decoy_mult=m)), "n_pairs": oof.height}
    for keep in (0.7, 0.5):
        drop_t = matched_t.filter(pl.Series(rng.random(matched_t.height) >= keep))
        sub = oof.join(drop_t, on="t_rid", how="anti")
        tr = truth.join(drop_t, on="t_rid", how="anti")
        scenarios[f"matched_x{keep:g}"] = {**evaluate(sub, sub["p2_cal"].to_numpy(), tr, s1_rids, thr_train, grid, odds_ratio=odds_ratio_for(matched_keep=keep)), "n_pairs": sub.height}
    # singletons doubled: strip every true match of a random 6 % of entities (they become singletons whose
    # former matches turn into decoys)
    ent = pl.DataFrame({"s_rid": s1_rids}).filter(pl.Series(rng.random(len(s1_rids)) < 0.06))
    t_drop = truth.join(ent, on="s_rid").select("t_rid")
    tr = truth.join(t_drop, on="t_rid", how="anti")
    sub = oof  # candidates unchanged: those records are now decoys the model must reject
    scenarios["singletons_x2"] = {**evaluate(sub, p, tr, s1_rids, thr_train, grid), "n_pairs": sub.height,
                                  "note": "former matches kept as candidates but removed from truth (labels flipped -> hard decoys)"}
    for src in (2, 3):
        sub = oof.filter(pl.col("src") == src)
        tr = truth.join(sub.select("t_rid").unique(), on="t_rid")
        scenarios[f"source_{src}_only"] = {**evaluate(sub, sub["p2_cal"].to_numpy(), tr, s1_rids, thr_train, grid), "n_pairs": sub.height}
    sub = oof.filter(pl.col("f_indic"))
    if sub.height:
        tr = truth.join(sub.select("t_rid").unique(), on="t_rid")
        s1_in = rec.filter(pl.col("src") == 1).filter(pl.col("country") == "India")["rid"].to_numpy().astype(np.uint32)
        scenarios["indic_script_only"] = {**evaluate(sub, sub["p2_cal"].to_numpy(), tr, s1_in, thr_train, grid), "n_pairs": sub.height}
    # calibration under prior shift (pair level): re-weight negatives from decoys by m
    cal_rows = {}
    is_decoy_pair = (~oof["t_matched"]).to_numpy()
    for m in (1.0, 2.0, 3.0):
        w = np.where(is_decoy_pair, m, 1.0)
        idx = np.repeat(np.arange(len(y)), w.astype(int))
        pp, yy = p[idx], y[idx]
        q = calibrate.shift_odds(pp, odds_ratio_for(decoy_mult=m))
        cal_rows[f"decoys_x{m:g}"] = {"platt": {k: v for k, v in calibrate.calibration_metrics(pp, yy).items() if k != "reliability"},
                                      "platt_prior_corrected": {k: v for k, v in calibrate.calibration_metrics(q, yy).items() if k != "reliability"}}
    out = {"train_threshold": thr_train, "scenarios": scenarios, "calibration_under_shift": cal_rows}
    with open(os.path.join(args.work_dir, "robustness.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"training threshold {thr_train}")
    print("| scenario | pairs | F0.5 @train thr | own best thr | F0.5 @own | regret | F0.5 @prior-corrected thr | recall @train thr | precision @train thr |")
    print("|---|---|---|---|---|---|---|---|---|")
    for k, v in scenarios.items():
        a, o, pc = v["at_train_threshold"], v["own_best"], v["at_prior_corrected"]
        print(f"| {k} | {v['n_pairs']} | {a['macro_f05']:.5f} | {o['threshold']:.2f} | {o['macro_f05']:.5f} | {v['regret']:+.5f} | "
              f"{pc['macro_f05']:.5f} (thr {v['prior_corrected_threshold']:.2f}) | {a['micro_recall']:.4f} | {a['micro_precision']:.4f} |")
    print("\n| calibration scenario | ECE platt | ECE prior-corrected | Brier platt | Brier prior-corrected | logloss platt | logloss corrected |")
    print("|---|---|---|---|---|---|---|")
    for k, v in cal_rows.items():
        print(f"| {k} | {v['platt']['ece']:.4f} | {v['platt_prior_corrected']['ece']:.4f} | {v['platt']['brier']:.5f} | "
              f"{v['platt_prior_corrected']['brier']:.5f} | {v['platt']['log_loss']:.4f} | {v['platt_prior_corrected']['log_loss']:.4f} |")


if __name__ == "__main__":
    main()
