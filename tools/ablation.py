"""Feature-group ablation: retrain (cross-fitted, OOF) with one feature group removed at a time
and report the change in macro F0.5, micro recall / precision and the tuned threshold.

    python tools/ablation.py --data-dir DATA --work-dir WORK [--rounds1 150 --rounds2 100] [--groups name_string,legal]

Each run is `src/train.py --drop-features ... --tag ablate_<group>` (no final models are written)
and reads back <work>/validation_report_ablate_<group>.json. Results: <work>/ablation.json and a
markdown table on stdout. Groups are defined by the feature names produced by pair_features.py.
"""
import argparse
import json
import os
import subprocess
import sys

import polars as pl

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
sys.path.insert(0, SRC)
from model import NON_FEATURES  # noqa: E402

GROUPS = {
    "retrieval": lambda c: c in {"score", "rank", "cos_name", "cos_addr", "cos_cross"} or c.startswith(("gap_", "t_best", "t_second", "t_ncand", "s_ncand", "s_rank", "s_best", "s_ntop1", "s_gap")),
    "name_string": lambda c: c.startswith(("nm_ratio", "nm_core", "nm_tsort", "nm_tset", "nm_partial", "nm_jw", "nm_part", "nm_first", "cmp_")),
    "name_tokens": lambda c: c in {"t_ntok", "s_ntok", "nm_common", "nm_jacc", "nm_cov_s", "nm_cov_t", "nm_xt", "nm_xs", "t_n_namesake"},
    "domain": lambda c: c.startswith("dom_") or c == "t_domain",
    "address_string": lambda c: c.startswith(("ad_ratio", "ad_tsort", "ad_tset", "ad_partial", "street_", "st_")),
    "address_tokens": lambda c: c in {"t_natok", "s_natok", "ad_common", "ad_jacc"},
    "house_numbers": lambda c: c.startswith(("hn_", "num_")) or c in {"t_nnum", "s_nnum", "t_n_addr_exact"},
    "legal_form": lambda c: c.startswith("lg_"),
    "record_flags": lambda c: c in {"t_src", "t_indic", "t_alias", "t_noaddr", "s_noaddr", "t_noname", "t_nlen", "s_nlen"},
    "crowding": lambda c: c in {"s_addr_mult", "s_street_mult"},
    "channels": lambda c: c.startswith("r_") or c in {"n_ch", "cos_nchar"},
    "contradiction": lambda c: c in {"hn_conflict", "unit_conflict", "pc_conflict", "num_conflict", "na_conflict", "t_n_namesake_contra"},
    "token_idf": lambda c: c.endswith(("_maxidf", "_sumidf")),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--groups", default=None, help="comma separated subset of groups (default: all + stage2)")
    ap.add_argument("--rounds1", type=int, default=300)
    ap.add_argument("--rounds2", type=int, default=200)
    ap.add_argument("--sample-rows", type=int, default=5_000_000)
    args = ap.parse_args()
    parts = sorted(os.listdir(os.path.join(args.work_dir, "train", "pairs")))
    cols = [c for c in pl.read_parquet_schema(os.path.join(args.work_dir, "train", "pairs", parts[0])).keys() if c not in NON_FEATURES]
    members = {g: [c for c in cols if f(c)] for g, f in GROUPS.items()}
    unassigned = [c for c in cols if not any(f(c) for f in GROUPS.values())]
    if unassigned:
        print("features in no group (kept in every run):", unassigned)
    runs = ["baseline"] + (args.groups.split(",") if args.groups else list(GROUPS) + ["stage2"])
    common = ["python", os.path.join(SRC, "train.py"), "--data-dir", args.data_dir, "--work-dir", args.work_dir,
              "--rounds1", str(args.rounds1), "--rounds2", str(args.rounds2), "--sample-rows", str(args.sample_rows)]
    results = {}
    for g in runs:
        tag = f"ablate_{g}"
        cmd = common + ["--tag", tag]
        if g == "stage2":
            cmd += ["--no-stage2"]
        elif g != "baseline":
            if not members[g]:
                print(f"group {g}: no features present, skipped")
                continue
            cmd += ["--drop-features", ",".join(members[g])]
        print(f"== {g}: dropping {members.get(g, [])}", flush=True)
        subprocess.run(cmd, check=True)
        with open(os.path.join(args.work_dir, f"validation_report_{tag}.json")) as f:
            rep = json.load(f)
        at = rep["oof_at_threshold"]
        results[g] = {"dropped": members.get(g, ["stage-2 model"] if g == "stage2" else []), "threshold": rep["threshold"],
                      "macro_f05": at["macro_f05"], "micro_recall": at["micro_recall"], "micro_precision": at["micro_precision"],
                      "fold_std": rep["per_fold"]["macro_f05_std"], "fold_min": rep["per_fold"]["macro_f05_min"]}
    base = results.get("baseline")
    with open(os.path.join(args.work_dir, "ablation.json"), "w") as f:
        json.dump({"groups": members, "results": results}, f, indent=1)
    print("\n| removed group | #feats | macro F0.5 | delta F0.5 | recall | delta recall | precision | thr | fold std |")
    print("|---|---|---|---|---|---|---|---|---|")
    for g, r in results.items():
        d = r["macro_f05"] - base["macro_f05"] if base else 0.0
        dr = r["micro_recall"] - base["micro_recall"] if base else 0.0
        print(f"| {g} | {len(r['dropped'])} | {r['macro_f05']:.5f} | {d:+.5f} | {r['micro_recall']:.5f} | {dr:+.5f} | "
              f"{r['micro_precision']:.5f} | {r['threshold']:.2f} | {r['fold_std']:.5f} |")


if __name__ == "__main__":
    main()
