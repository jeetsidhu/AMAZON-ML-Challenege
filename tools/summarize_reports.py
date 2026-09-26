"""Prints the diagnostics of a work directory as markdown tables.

    python tools/summarize_reports.py --work-dir work [--baseline-profile other/profile.json]
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


def fmt(x, nd=5):
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", required=True)
    args = ap.parse_args()
    w = args.work_dir
    rep = load(os.path.join(w, "validation_report.json"))
    if rep:
        at = rep["oof_at_threshold"]
        print("## Out-of-fold validation\n")
        print(f"pairs {rep['n_pairs']}, positive pairs {rep['n_positive_pairs']}, Source 1 entities {rep['n_s1']}, "
              f"stage-1 features {rep['stage1_features']}, folds {rep['folds']}\n")
        print("| metric | value |\n|---|---|")
        for k in ("threshold", "macro_f05", "micro_precision", "micro_recall", "singleton_acc", "links"):
            print(f"| {k} | {fmt(at[k])} |")
        pl_ = rep["pair_level"]
        print(f"| pair-level accuracy | {fmt(pl_['accuracy'])} |\n| pair-level positive rate | {fmt(pl_['positive_rate'])} |"
              f"\n| pair-level precision | {fmt(pl_['precision'])} |\n| pair-level recall | {fmt(pl_['recall'])} |")
        if rep.get("blocking_recall"):
            r = rep["blocking_recall"]["recall_at_k"]
            print("| retrieval recall@1 / @3 / @5 / @10 | " + " / ".join(fmt(r[str(k)], 4) for k in (1, 3, 5, 10) if str(k) in r) + " |")
        print("\n### Per fold (at the global threshold)\n")
        print("| fold | entities | macro F0.5 | precision | recall | singleton acc | own best thr | F0.5 @ own thr |\n|---|---|---|---|---|---|---|---|")
        for r in rep["per_fold"]["folds"]:
            print(f"| {r['fold']} | {r['n_entities']} | {fmt(r['macro_f05_at_global_thr'])} | {fmt(r['micro_precision_at_global_thr'])} | "
                  f"{fmt(r['micro_recall_at_global_thr'])} | {fmt(r['singleton_acc_at_global_thr'], 4)} | {r['best_threshold']:.2f} | {fmt(r['macro_f05_at_own_thr'])} |")
        pf = rep["per_fold"]
        print(f"\nmean {fmt(pf['macro_f05_mean'])}, std {fmt(pf['macro_f05_std'])}, min {fmt(pf['macro_f05_min'])}; "
              f"fold-optimal thresholds {pf['best_threshold_range']} (std {fmt(pf['best_threshold_std'], 4)})\n")
        print("### Per country\n")
        print("| country | entities-with-links | macro F0.5 | precision | recall |\n|---|---|---|---|---|")
        for c, r in rep["per_country"].items():
            print(f"| {c or '(none)'} | {r['links']} | {fmt(r['macro_f05'])} | {fmt(r['micro_precision'])} | {fmt(r['micro_recall'])} |")
        print("\n### Calibration (nested: fit on K-1 folds, scored on the held-out fold)\n")
        print("| probabilities | log loss | Brier | ECE |\n|---|---|---|---|")
        for k, v in rep["calibration"]["nested"].items():
            print(f"| {k} | {fmt(v['log_loss'])} | {fmt(v['brier'])} | {fmt(v['ece'])} |")
        print(f"\nPlatt: a={rep['calibration']['platt']['a']:.3f}, b={rep['calibration']['platt']['b']:.3f}\n")
        au = rep["leakage_audit"]
        print("### Cross-fold audit\n")
        print(f"cross-fold pair share {fmt(au['cross_fold_pair_share'], 4)}, entities with a cross-fold pair {fmt(au['entities_with_cross_fold_pair'], 4)}\n")
        if "within_fold_only" in au:
            print("| entity set | macro F0.5 | precision | recall |\n|---|---|---|---|")
            for k in ("within_fold_only", "with_cross_fold_pairs"):
                print(f"| {k} | {fmt(au[k]['macro_f05'])} | {fmt(au[k]['micro_precision'])} | {fmt(au[k]['micro_recall'])} |")
        print("\n### Threshold sweep (calibrated scale, OOF)\n")
        print("| thr | macro F0.5 | precision | recall | links |\n|---|---|---|---|---|")
        for r in rep["threshold_sweep"]:
            if round(r["threshold"] * 100) % 10 == 0 or abs(r["threshold"] - rep["threshold"]) < 1e-9:
                print(f"| {r['threshold']:.2f} | {fmt(r['macro_f05'])} | {fmt(r['micro_precision'])} | {fmt(r['micro_recall'])} | {r['links']} |")
    lk = load(os.path.join(w, "leakage_check.json"))
    if lk:
        print("\n## Leakage check\n")
        print(f"result: **{'OK' if lk['ok'] else 'FAILED'}**; problems: {lk['problems'] or 'none'}\n")
        print("| check | value |\n|---|---|")
        print(f"| name groups split across folds | {lk['name_groups']['split_across_folds']} / {lk['name_groups']['n']} |")
        print(f"| matched records outside their entity's fold | {lk['matched_records_fold_mismatch']} |")
        print(f"| duplicated pids in OOF | {lk['oof']['duplicate_pids']} |")
        print(f"| positive pairs across folds | {lk['oof']['positive_pairs_cross_fold']} |")
        print(f"| fold-only Indic tokens leaked into own-fold lexicon | {lk['lexicon']['leaked_into_own_fold_lexicon']} / {lk['lexicon']['fold_only_tokens_checked']} checked |")
        print(f"| label-shuffle canary OOF AUC | {fmt(lk['label_shuffle_canary_auc'], 4)} |")
        print(f"| strongest single features (AUC) | {lk['top_single_feature_auc'][:5]} |")
        print(f"| bookkeeping columns among model features | {lk['model_features_bookkeeping'] or 'none'} |")
    ta = load(os.path.join(w, "threshold_analysis.json"))
    if ta:
        print("\n## Threshold transfer to the test split\n")
        print(f"decoy-density ratio r = {ta['decoy_ratio']:.3f}; pair-level odds ratio for the prior shift = {ta['odds_ratio_prior_shift']:.3f}; chosen: {ta['chosen_method']}\n")
        print("| method | threshold | OOF macro F0.5 (train mix) | OOF macro F0.5 (test-like mix) | precision | recall |\n|---|---|---|---|---|---|")
        for k, v in ta["candidates"].items():
            a, b = ta["oof_plain_at"][k], ta["oof_density_adjusted_at"][k]
            print(f"| {k} | {v:.3f} | {fmt(a['macro_f05'])} | {fmt(b['macro_f05'])} | {fmt(b['micro_precision'])} | {fmt(b['micro_recall'])} |")
    ho = load(os.path.join(w, "holdout_eval.json"))
    if ho:
        print("\n## Hold-out score (labelled split never used for training or thresholds)\n")
        std = rep["per_fold"]["macro_f05_std"] if rep else None
        oof = rep["oof_at_threshold"]["macro_f05"] if rep else None
        print("| set | entities | macro F0.5 | precision | recall | singleton acc |\n|---|---|---|---|---|---|")
        for k, v in [("overall", ho["overall"])] + sorted(ho.get("per_country", {}).items()):
            print(f"| {k or '(none)'} | {v['n_entities']} | {fmt(v['macro_f05'])} | {fmt(v['micro_precision'])} | {fmt(v['micro_recall'])} | {fmt(v['singleton_acc'], 4)} |")
        if oof is not None:
            d = ho["overall"]["macro_f05"] - oof
            print(f"\nhold-out macro F0.5 {fmt(ho['overall']['macro_f05'])} vs out-of-fold {fmt(oof)}: delta {d:+.5f} "
                  f"({'within' if std is not None and abs(d) <= 2 * std else 'outside'} 2x the fold std {fmt(std, 5) if std is not None else '?'})")
    te = load(os.path.join(w, "threshold_experiments.json"))
    if te:
        print("\n## Threshold policies: global vs per-class (tune_thresholds.py)\n")
        print(f"checkpoint `{te['checkpoint']}`; decoy ratio r = {te['decoy_ratio_global']:.3f} ({te['density']}); "
              f"global threshold {te['global_threshold_density_adjusted']:.2f} (train-optimal {te['global_threshold_train_optimal']:.2f}); "
              f"**selected: {te['selected']}** ({te['selection_rule']})\n")
        print("| config | classes | nested macro F0.5 | gain vs global | folds better | in-sample macro F0.5 | precision | recall | thresholds |\n|---|---|---|---|---|---|---|---|---|")
        for name, r in te["configs"].items():
            n, o = r["nested"], r["in_sample"]["overall"]
            thr = ", ".join(f"{k}={v:.2f}" for k, v in r["policy"]["thresholds"].items()) or f"{r['policy']['default']:.2f}"
            print(f"| {name} | {r['n_classes']} | {fmt(n['nested_macro_f05'])} | {n['gain_vs_global']:+.5f} | {n['folds_better_than_global']}/{len(n['per_fold'])} | "
                  f"{fmt(o['macro_f05'])} | {fmt(o['micro_precision'])} | {fmt(o['micro_recall'])} | {thr} |")
    he = load(os.path.join(w, "holdout_policy_eval.json"))
    if he:
        print("\n## Threshold policies on the labelled hold-out (tools/policy_holdout_eval.py)\n")
        base = he["policies"][he["baseline"]]["overall"]["macro_f05"]
        print("| policy | classes | macro F0.5 | delta vs global | precision | recall | singleton acc | links |\n|---|---|---|---|---|---|---|---|")
        for name, r in he["policies"].items():
            o = r["overall"]
            print(f"| {name} | {len(r['policy']['thresholds'])} | {fmt(o['macro_f05'])} | {o['macro_f05'] - base:+.5f} | {fmt(o['micro_precision'])} | "
                  f"{fmt(o['micro_recall'])} | {fmt(o['singleton_acc'], 4)} | {r['links']} |")
    ck_root = os.path.join(w, "checkpoints")
    if os.path.isdir(ck_root):
        print("\n## Checkpoints\n")
        latest = open(os.path.join(ck_root, "LATEST")).read().strip() if os.path.exists(os.path.join(ck_root, "LATEST")) else None
        print("| checkpoint | created | git | stages | rounds | OOF macro F0.5 | global thr | policies |\n|---|---|---|---|---|---|---|---|")
        for name in sorted(os.listdir(ck_root)):
            m = load(os.path.join(ck_root, name, "manifest.json"))
            if not m:
                continue
            ta = m.get("train_args") or {}
            pols = sorted(f[:-5] for f in os.listdir(os.path.join(ck_root, name, "thresholds"))) if os.path.isdir(os.path.join(ck_root, name, "thresholds")) else []
            print(f"| {name}{' (LATEST)' if name == latest else ''} | {m.get('created')} | {m.get('git_commit')} | {','.join(m.get('stages_done', []))} | "
                  f"{ta.get('rounds1')}/{ta.get('rounds2')} | {fmt(m.get('oof_macro_f05')) if m.get('oof_macro_f05') is not None else '-'} | "
                  f"{m.get('global_threshold', '-')} | {', '.join(pols)} |")
    prof = load(os.path.join(w, "profile.json"))
    if prof:
        print("\n## Runtime profile\n")
        print("| stage | wall s | cpu s | cpu/wall | workers |\n|---|---|---|---|---|")
        tot = 0.0
        for k, v in prof.items():
            tot += v["wall_s"]
            print(f"| {k} | {v['wall_s']:.1f} | {v['cpu_s']:.1f} | {v['cpu_per_wall']:.2f} | {v['workers']} |")
        print(f"| **total** | {tot:.1f} | | | |")
    ab = load(os.path.join(w, "ablation.json"))
    if ab:
        base = ab["results"].get("baseline")
        print("\n## Feature ablation (OOF, one group removed at a time)\n")
        print("| removed group | #feats | macro F0.5 | delta | recall | delta recall | precision | thr | fold std |\n|---|---|---|---|---|---|---|---|---|")
        for g, r in ab["results"].items():
            d = r["macro_f05"] - base["macro_f05"] if base else 0
            dr = r["micro_recall"] - base["micro_recall"] if base else 0
            print(f"| {g} | {len(r['dropped'])} | {fmt(r['macro_f05'])} | {d:+.5f} | {fmt(r['micro_recall'])} | {dr:+.5f} | {fmt(r['micro_precision'])} | {r['threshold']:.2f} | {fmt(r['fold_std'])} |")


if __name__ == "__main__":
    main()
