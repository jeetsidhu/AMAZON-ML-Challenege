"""Side-by-side comparison of several work directories (before/after table for the report).

    python tools/compare_runs.py --runs base=work_base new=work_new [...] [--out FILE.md]

For every run: candidate audit (pair recall, complete-entity coverage, oracle macro F0.5, candidates
per record), OOF macro F0.5 at the tuned rule, nested macro F0.5, precision / recall, singleton and
other false positives, the selected decision rule and policy, and the labelled hold-out score
(evaluate.py output in <work>/holdout_eval.json) overall and per country.
"""
import argparse
import json
import os


def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def row(name, w):
    rep = load(os.path.join(w, "validation_report.json")) or {}
    ca = rep.get("candidate_audit") or rep.get("blocking_recall") or load(os.path.join(w, "train", "blocking_recall.json")) or {}
    at = rep.get("oof_at_threshold", {})
    dr = rep.get("decision_rule") or {}
    ho = load(os.path.join(w, "holdout_eval.json")) or {}
    hov = ho.get("overall", {})
    recall = ca.get("pair_recall")
    if recall is None and ca.get("recall_at_k"):
        recall = ca["recall_at_k"].get(str(ca.get("topk_kept", 3)))
    return {
        "run": name, "pairs": rep.get("n_pairs"), "features": rep.get("stage1_features"),
        "cand_recall": recall, "entity_coverage": ca.get("entity_complete_coverage"), "oracle": ca.get("oracle_macro_f05"),
        "cands_per_record": ca.get("candidates_per_record"), "missed_recoverable": (ca.get("misses") or {}).get("missed_recoverable"),
        "oof_f05": at.get("macro_f05"), "nested_f05": dr.get("nested_macro_f05"), "threshold": at.get("threshold"),
        "precision": at.get("micro_precision"), "recall": at.get("micro_recall"),
        "singleton_fp": at.get("singleton_fp"), "fp_decoy": at.get("fp_decoy"), "fp_wrong": at.get("fp_wrong_entity"), "links": at.get("links"),
        "rule": f"m={dr.get('margin', 0):.2f} c={dr.get('contra_penalty', 0):.2f}" if dr else "-",
        "policy": (load(os.path.join(w, "threshold_policy.json")) or {}).get("name"),
        "holdout_f05": hov.get("macro_f05"), "holdout_precision": hov.get("micro_precision"), "holdout_recall": hov.get("micro_recall"),
        "holdout_singleton_acc": hov.get("singleton_acc"),
        "holdout_by_country": {c: v.get("macro_f05") for c, v in (ho.get("per_country") or {}).items()},
        "train_wall_s": (load(os.path.join(w, "profile.json")) or {}).get("train", {}).get("wall_s"),
        "blocking_wall_s": (load(os.path.join(w, "profile.json")) or {}).get("blocking_train", {}).get("wall_s"),
    }


def fmt(x, nd=5):
    if x is None:
        return "-"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="name=work_dir ...")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rows = [row(*r.split("=", 1)) for r in args.runs]
    countries = sorted({c for r in rows for c in r["holdout_by_country"]})
    L = ["| run | pairs | feats | cand recall | entity coverage | oracle F0.5 | cands/rec | recoverable misses | OOF macro F0.5 | nested | thr | rule | P | R | singleton FP | FP decoy | FP wrong | policy | hold-out F0.5 | hold-out P | hold-out R | "
         + " | ".join(f"hold-out {c}" for c in countries) + " | blocking s | train s |",
         "|---" * (22 + len(countries)) + "|"]
    for r in rows:
        L.append(f"| {r['run']} | {fmt(r['pairs'])} | {fmt(r['features'])} | {fmt(r['cand_recall'], 4)} | {fmt(r['entity_coverage'], 4)} | {fmt(r['oracle'], 4)} | {fmt(r['cands_per_record'], 2)} | "
                 f"{fmt(r['missed_recoverable'])} | {fmt(r['oof_f05'])} | {fmt(r['nested_f05'])} | {fmt(r['threshold'], 2)} | {r['rule']} | {fmt(r['precision'])} | {fmt(r['recall'])} | "
                 f"{fmt(r['singleton_fp'])} | {fmt(r['fp_decoy'])} | {fmt(r['fp_wrong'])} | {fmt(r['policy'])} | {fmt(r['holdout_f05'])} | {fmt(r['holdout_precision'])} | {fmt(r['holdout_recall'])} | "
                 + " | ".join(fmt(r["holdout_by_country"].get(c)) for c in countries) + f" | {fmt(r['blocking_wall_s'], 1)} | {fmt(r['train_wall_s'], 1)} |")
    md = "\n".join(L) + "\n"
    print(md)
    if args.out:
        with open(args.out, "w") as f:
            f.write(md)
        with open(os.path.splitext(args.out)[0] + ".json", "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
