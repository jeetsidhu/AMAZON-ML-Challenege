"""Scores one checkpoint under several threshold policies on a *labelled* local hold-out.

The subset produced by tools/make_subset.py has a test/ split with its own ground truth
(test/subset_ground_truth.tsv, a disjoint sample of the training data with twice the decoy
density). This tool runs predict.py once per policy on that split - model inference happens
once, the cached scores are reused for every other policy - and evaluates each submission
with the challenge metric, overall and per country.

    python tools/policy_holdout_eval.py --data-dir subset --work-dir work_subset --out-dir out_subset/policies \
        [--checkpoint NAME] [--policies work_subset/checkpoints/NAME/thresholds/*.json] [--truth subset/test/subset_ground_truth.tsv]

Writes <work>/holdout_policy_eval.json and .md and appends to the checkpoint's experiments.jsonl.
Never run this on the real test split: it has no labels, and a policy chosen on a labelled
hold-out is a policy fit on that hold-out.
"""
import argparse
import glob
import json
import os
import sys

import polars as pl

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
sys.path.insert(0, SRC)
import predict  # noqa: E402
from checkpoint import Checkpoint  # noqa: E402
from common import read_tsv  # noqa: E402
from evaluate import id_lists, per_entity, summary  # noqa: E402
from threshold_policy import ThresholdPolicy  # noqa: E402


def evaluate_file(pred_path, truth_path, source1_path):
    pred = id_lists(read_tsv(pred_path), "matched_entity_ids")
    truth = id_lists(read_tsv(truth_path), "matched_entity_ids")
    pe = per_entity(pred, truth)
    res = {"overall": summary(pe)}
    s1 = read_tsv(source1_path).select(pl.col("entity_id").alias("s"), pl.col("country").fill_null(""))
    pe_c = pe.join(s1, on="s", how="left")
    res["per_country"] = {c: summary(pe_c.filter(pl.col("country") == c)) for c in sorted(pe_c["country"].unique().to_list())}
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--policies", nargs="*", default=None, help="policy JSON files (default: every policy of the checkpoint)")
    ap.add_argument("--truth", default=None, help="ground truth of the hold-out (default: <data>/test/subset_ground_truth.tsv)")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    ckpt = Checkpoint.resolve(args.work_dir, args.checkpoint)
    if ckpt is None or not ckpt.exists():
        raise SystemExit("no checkpoint found")
    truth = args.truth or os.path.join(args.data_dir, "test", "subset_ground_truth.tsv")
    if not os.path.exists(truth):
        raise SystemExit(f"no labelled hold-out at {truth}; this tool is for local subsets only")
    source1 = os.path.join(args.data_dir, "test", "test_source1.tsv")
    policies = args.policies or sorted(glob.glob(ckpt.path(os.path.join("thresholds", "*.json"))))
    policies = [p for p in policies if os.path.basename(p) != "selected.json" or len(policies) == 1]
    rows = {}
    for i, pol_path in enumerate(policies):
        pol = ThresholdPolicy.load(pol_path)
        name = os.path.splitext(os.path.basename(pol_path))[0]
        out_dir = os.path.join(args.out_dir, name)
        ns = argparse.Namespace(data_dir=args.data_dir, work_dir=args.work_dir, out_dir=out_dir, checkpoint=ckpt.name, policy=pol_path,
                                threshold=None, decoder=None, reuse_scores=True)
        info = predict.run(ns)
        res = evaluate_file(os.path.join(out_dir, "matching_results.tsv"), truth, source1)
        rows[name] = {"policy_path": pol_path, "policy": pol.to_dict(), "links": info["links"], "unseen_classes": info["unseen_classes"],
                      "per_class_links": info["per_class"], **res}
        o = res["overall"]
        print(f"{name:28s} macro F0.5 {o['macro_f05']:.5f}  P {o['micro_precision']:.5f}  R {o['micro_recall']:.5f}  "
              f"singletons {o['singleton_acc']:.4f}  links {info['links']}", flush=True)
    base = rows.get("global_train") or rows.get("global") or next(iter(rows.values()))
    tag = f"_{args.tag}" if args.tag else ""
    out = {"checkpoint": ckpt.name, "truth": truth, "baseline": "global_train" if "global_train" in rows else next(iter(rows)), "policies": rows}
    with open(os.path.join(args.work_dir, f"holdout_policy_eval{tag}.json"), "w") as f:
        json.dump(out, f, indent=1)
    countries = sorted({c for r in rows.values() for c in r["per_country"]})
    L = [f"# Hold-out evaluation of threshold policies (checkpoint `{ckpt.name}`)", "", f"truth: `{truth}`", "",
         "| policy | classes | macro F0.5 | delta vs global | precision | recall | singleton acc | perfect entities | links | " + " | ".join(f"F0.5 {c}" for c in countries) + " |",
         "|---|---|---|---|---|---|---|---|---|" + "---|" * len(countries)]
    for name, r in rows.items():
        o = r["overall"]
        L.append(f"| {name} | {len(r['policy']['thresholds'])} | {o['macro_f05']:.5f} | {o['macro_f05'] - base['overall']['macro_f05']:+.5f} | {o['micro_precision']:.5f} | "
                 f"{o['micro_recall']:.5f} | {o['singleton_acc']:.4f} | {o['entities_perfect']:.4f} | {r['links']} | "
                 + " | ".join(f"{r['per_country'][c]['macro_f05']:.5f}" if c in r["per_country"] else "-" for c in countries) + " |")
    md = "\n".join(L) + "\n"
    with open(os.path.join(args.work_dir, f"holdout_policy_eval{tag}.md"), "w") as f:
        f.write(md)
    ckpt.log_experiment({"kind": "holdout_eval", "truth": truth, "results": {n: {"macro_f05": r["overall"]["macro_f05"], "precision": r["overall"]["micro_precision"],
                                                                               "recall": r["overall"]["micro_recall"], "links": r["links"]} for n, r in rows.items()},
                         "outputs": [os.path.join(args.work_dir, f"holdout_policy_eval{tag}.json")]})
    print("\n" + md)


if __name__ == "__main__":
    main()
