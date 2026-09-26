"""Step 4: train the two-stage matcher on the training split, with leak-free validation.

Folds come from folds.py (records.parquet carries `fold` / `group`): all namesakes of a
Source 1 entity share its fold, matched Source 2/3 records inherit it, and unmatched
("decoy") records are put in the fold of their best blocking candidate so that nearly every
candidate pair is inside one fold.

* The evaluation fold of a candidate pair is the fold of its Source 1 entity, so every
  entity's whole candidate list is scored by one model.
* The model for fold k is trained on pairs whose Source 1 entity AND Source 2/3 record are
  both outside fold k (strict exclusion; the remaining "cross-fold" pairs are counted and
  reported in validation_report.json).
* Stage 1 and stage 2 are cross-fitted; stage 2 uses out-of-fold stage-1 probabilities.
* The stage-2 OOF probabilities are calibrated (Platt, calibrate.py) - fit on OOF only -
  and the acceptance threshold is chosen on the calibrated scale by maximising macro F0.5
  over ALL training Source 1 entities (singletons included). The sweep is vectorised
  (thresholds.py) instead of re-sorting every pair per grid point.
* Decision rule (thresholds.select_rule): besides the threshold, a margin over the record's
  runner-up candidate and a threshold penalty for contradicted pairs are considered; the rule is
  chosen by *nested* macro F0.5 (threshold fit on K-1 folds, scored on the held-out fold) and only
  kept when it beats the plain threshold by --rule-min-gain. The report also carries the nested
  macro F0.5 of the final rule, so the headline number is one the thresholds were not tuned on.
* Reporting: macro F0.5, singleton false positives, false positives by kind, per-source
  (Source 2 vs 3) and per-country metrics, and the candidate-generation audit of blocking.py.
* Per-fold metrics, threshold stability across folds, calibration metrics and the
  cross-fold audit go to <work>/validation_report.json; OOF predictions to
  <work>/train/oof.parquet; final models (refit on all sampled rows) to <work>/stage*.txt.
* Checkpointing (checkpoint.py): every artefact of the run is written to
  <work>/checkpoints/<name>/ - each cross-fitted fold model as soon as it is trained, the
  stage-1 OOF probabilities, the final models, calibration, OOF predictions, report and a
  manifest (arguments, data fingerprint, git commit). `--resume` continues an interrupted run
  from the last saved fold model. The top-level <work>/ files are links to the latest
  checkpoint, so one checkpoint can be evaluated under any number of threshold policies
  (tune_thresholds.py / predict.py --checkpoint --policy) without retraining.
"""
import json
import os
import shutil

import numpy as np
import polars as pl

import calibrate
from checkpoint import Checkpoint, data_fingerprint
from common import Stage, base_args, left_join_ordered, log, split_dir
from model import CONTRA_COLS, contradiction_flag, house_numbers, iter_parts, part_files, stage1_features, stage2_context, to_np, train_lgb
from pair_features import truth_pairs
from threshold_policy import ThresholdPolicy
from thresholds import best_threshold, default_grid, link_table, metrics_at, nested_rule_f05, select_rule, subset_links, sweep

TRAIN_ARGS = ("rounds1", "rounds2", "sample_rows", "seed", "drop_features", "no_stage2", "tag")
NO_RULE = {"margin": 0.0, "contra_penalty": 0.0}


def oof_predict(files, feats_fn, models, eval_fold, n):
    """Streams the parts once; each row is scored by the model that did not see its fold."""
    out = np.zeros(n, dtype=np.float32)
    for part in files:
        df = pl.read_parquet(part)
        pid = df["pid"].to_numpy()
        X = feats_fn(df, pid)
        fk = eval_fold[pid]
        for k, m in models.items():
            sel = fk == k
            if sel.any():
                out[pid[sel]] = m.predict(X[sel])
    return out


def assign_folds(meta, rec):
    """Adds s_fold (eval fold), t_fold and the strict train mask helpers to meta."""
    f = rec.select(pl.col("rid").cast(pl.UInt32), "fold")
    meta = left_join_ordered(meta, f.rename({"rid": "s_rid", "fold": "s_fold"}), "s_rid")
    meta = left_join_ordered(meta, f.rename({"rid": "t_rid", "fold": "t_fold"}), "t_rid")
    # decoys (t_fold == -1): fold of the best-scoring candidate of the record
    best_s_fold = (
        meta.filter(pl.col("t_fold") < 0)
        .sort("score", descending=True)
        .unique("t_rid", keep="first")
        .select("t_rid", pl.col("s_fold").alias("decoy_fold"))
    )
    meta = left_join_ordered(meta, best_s_fold, "t_rid").with_columns(
        pl.when(pl.col("t_fold") < 0).then(pl.col("decoy_fold")).otherwise(pl.col("t_fold")).cast(pl.Int8).alias("t_fold")
    ).drop("decoy_fold")
    return meta


def fold_report(links, nt, s_fold_of_s1, thr, grid, rule=NO_RULE):
    """Per-fold macro F0.5 at the global threshold, plus each fold's own best threshold."""
    rows = []
    for k in np.unique(s_fold_of_s1):
        sel_s = s_fold_of_s1 == k
        lk, nt_k = subset_links(links, nt, sel_s)
        at = metrics_at(lk, nt_k, thr, **rule)
        own = best_threshold(sweep(lk, nt_k, grid, **rule))
        rows.append({"fold": int(k), "n_entities": int(sel_s.sum()), **{f"{a}_at_global_thr": b for a, b in at.items() if a != "threshold"},
                     "best_threshold": own["threshold"], "macro_f05_at_own_thr": own["macro_f05"]})
    f = np.array([r["macro_f05_at_global_thr"] for r in rows])
    t = np.array([r["best_threshold"] for r in rows])
    return {"folds": rows, "macro_f05_mean": float(f.mean()), "macro_f05_std": float(f.std(ddof=0)),
            "macro_f05_min": float(f.min()), "best_threshold_std": float(t.std(ddof=0)), "best_threshold_range": [float(t.min()), float(t.max())]}


def per_source_report(links, nt, thr, rec, rule=NO_RULE):
    """Link-level precision / recall and false positives per Source 2/3 record source (entity-level
    macro F0.5 cannot be split by source: one entity receives records of both sources)."""
    src = links.select("t_rid").join(rec.select(pl.col("rid").cast(pl.UInt32).alias("t_rid"), "src"), on="t_rid", how="left")["src"].to_numpy()
    out = {}
    for s in sorted(set(src.tolist())):
        lm = src == s
        r = metrics_at(links, nt, thr, link_mask=lm, **rule)
        out[f"source{int(s)}"] = {k: r[k] for k in ("micro_precision", "micro_recall", "links", "fp_decoy", "fp_wrong_entity", "pair_accuracy")}
        out[f"source{int(s)}"]["link_rows"] = int(lm.sum())
    return out


def main():
    ap = base_args(__doc__)
    ap.add_argument("--rounds1", type=int, default=300)
    ap.add_argument("--rounds2", type=int, default=200)
    ap.add_argument("--sample-rows", type=int, default=5_000_000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--drop-features", default="", help="comma separated feature names to exclude (ablations)")
    ap.add_argument("--no-stage2", action="store_true", help="ablation: use stage-1 probabilities directly")
    ap.add_argument("--no-rule-search", action="store_true", help="plain threshold only (no margin / contradiction rule search)")
    ap.add_argument("--rule-min-gain", type=float, default=1e-4, help="nested macro F0.5 gain a margin / contradiction rule needs")
    ap.add_argument("--tag", default="", help="suffix for the report / oof file names (ablations)")
    ap.add_argument("--checkpoint-name", default=None,
                    help="name of the checkpoint directory under <work>/checkpoints (default: ckpt_<timestamp>[_<tag>])")
    ap.add_argument("--resume", action="store_true",
                    help="reuse the fold models / stage-1 OOF already saved in the checkpoint (same arguments required)")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing checkpoint of the same name")
    args = ap.parse_args()
    with Stage(args.work_dir, "train" + (f"_{args.tag}" if args.tag else "")):
        run(args)


def open_checkpoint(args):
    """Creates (or, with --resume, reopens) the checkpoint of this run and records its arguments."""
    name = args.checkpoint_name or (Checkpoint.default_name() + (f"_{args.tag}" if args.tag else ""))
    ckpt = Checkpoint(args.work_dir, name)
    train_args = {k: getattr(args, k) for k in TRAIN_ARGS}
    if ckpt.exists() and args.overwrite and not args.resume:
        log(f"overwriting checkpoint {ckpt.dir}")
        shutil.rmtree(ckpt.dir)
    if ckpt.exists():
        prev = ckpt.manifest().get("train_args")
        if not args.resume:
            raise SystemExit(f"checkpoint {ckpt.dir} exists; --resume continues it, --overwrite replaces it, or choose another --checkpoint-name")
        if prev != train_args:
            raise SystemExit(f"cannot resume {ckpt.dir}: it was trained with {prev}, now {train_args}")
        log(f"resuming checkpoint {ckpt.dir}: stages done {ckpt.manifest().get('stages_done')}")
    ckpt.update_manifest(train_args=train_args, data_dir=os.path.abspath(args.data_dir), data=data_fingerprint(args.data_dir),
                         resumed=bool(args.resume and ckpt.exists()))
    return ckpt


def load_or_train(ckpt, fname, resume, fn):
    """Fold model from the checkpoint when resuming, else trained by fn() and saved immediately."""
    if resume and ckpt.has(fname):
        log(f"  {fname}: loaded from checkpoint")
        return ckpt.load_model(fname)
    m = fn()
    ckpt.save_model(m, fname)
    return m


def run(args):
    d = split_dir(args.work_dir, "train")
    ckpt = open_checkpoint(args)
    files = part_files(d)
    rec = pl.read_parquet(os.path.join(d, "records.parquet"), columns=["rid", "entity_id", "src", "fold", "country"])
    truth = truth_pairs(rec, args.data_dir).drop("label")
    s1 = rec.filter(pl.col("src") == 1)
    s1_rids = s1["rid"].to_numpy().astype(np.uint32)
    s1_fold = s1["fold"].to_numpy()

    meta = pl.concat(list(iter_parts(files, ["pid", "t_rid", "s_rid", "label", "score", *CONTRA_COLS]))).sort("pid")
    n = meta.height
    assert meta["pid"][-1] == n - 1
    contra = contradiction_flag(meta)
    meta = assign_folds(meta, rec).drop(CONTRA_COLS)
    s_fold = meta["s_fold"].to_numpy()
    t_fold = meta["t_fold"].to_numpy()
    y_all = meta["label"].to_numpy()
    fold_ids = sorted(int(k) for k in np.unique(s_fold))
    cross = s_fold != t_fold
    log(f"pairs {n}, positives {int(y_all.sum())}, eval-fold sizes {np.bincount(s_fold)}, "
        f"cross-fold pairs {cross.mean():.4f} (positives among them {int(y_all[cross].sum())})")

    drop = {x for x in args.drop_features.split(",") if x}
    f1 = [c for c in stage1_features(files) if c not in drop]
    log("stage-1 features", len(f1), "dropped", sorted(drop))
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
    ys, sf, tf = y_all[s_pid], s_fold[s_pid], t_fold[s_pid]
    log("sample", X1.shape, "positive rate", ys.mean())

    def train_mask(k):
        return (sf != k) & (tf != k)  # strict: neither side of the pair is in the held-out fold

    # ---------------- stage 1 (OOF)
    m1s = {}
    for k in fold_ids:
        m = train_mask(k)
        m1s[k] = load_or_train(ckpt, f"stage1_fold{k}.txt", args.resume, lambda: train_lgb(X1[m], ys[m], args.rounds1))
        log(f"stage-1 fold {k} ready ({int(m.sum())} training pairs)")
    if args.resume and ckpt.has("oof_stage1.npy") and ckpt.stage_done("stage1"):
        p1 = np.load(ckpt.path("oof_stage1.npy"))
        assert len(p1) == n, "checkpointed stage-1 OOF does not match the pair table"
        log("stage-1 OOF loaded from checkpoint")
    else:
        p1 = oof_predict(files, lambda df, pid: to_np(df, f1), m1s, s_fold, n)
        np.save(ckpt.path("oof_stage1.npy"), p1)
    ckpt.mark_stage("stage1")
    links1, nt = link_table(meta, p1, truth, s1_rids, contra)
    s1_best = best_threshold(sweep(links1, nt, default_grid()))
    log("stage-1 OOF", {k: s1_best[k] for k in ("threshold", "macro_f05", "micro_precision", "micro_recall")})

    # ---------------- stage 2 (OOF) on stage-1 OOF context
    if args.no_stage2:
        p2, f2, C = p1, f1, None
        m2s = None
    else:
        C, cnames = stage2_context(meta, p1, house_numbers(d))
        f2 = f1 + ["p1"] + cnames
        X2 = np.hstack([X1, p1[s_pid, None], C[s_pid]])
        m2s = {}
        for k in fold_ids:
            m = train_mask(k)
            m2s[k] = load_or_train(ckpt, f"stage2_fold{k}.txt", args.resume, lambda: train_lgb(X2[m], ys[m], args.rounds2))
            log(f"stage-2 fold {k} ready")
        p2 = oof_predict(files, lambda df, pid: np.hstack([to_np(df, f1), p1[pid, None], C[pid]]), m2s, s_fold, n)
    ckpt.mark_stage("stage2")

    # ---------------- calibration (fit on OOF only) + threshold on the calibrated scale
    cal = calibrate.fit_platt(p2, y_all)
    cal_report = calibrate.nested_calibration_report(p2, y_all, s_fold)
    p2c = calibrate.apply(cal, p2).astype(np.float32)
    log(f"Platt a={cal['a']:.3f} b={cal['b']:.3f}; nested calibration: " +
        ", ".join(f"{k}: ll={v['log_loss']:.4f} brier={v['brier']:.5f} ece={v['ece']:.4f}" for k, v in cal_report.items()))
    grid = default_grid()
    links, nt = link_table(meta, p2c, truth, s1_rids, contra)
    # ---------------- decision rule (margin over the runner-up, contradiction penalty): nested selection
    if args.no_rule_search:
        rule_sel, rule_rows = dict(NO_RULE, nested_macro_f05=nested_rule_f05(links, nt, s1_fold, grid)[0]), []
        rule_sel["nested_macro_f05_threshold_only"] = rule_sel["nested_macro_f05"]
    else:
        rule_sel, rule_rows = select_rule(links, nt, s1_fold, grid, min_gain=args.rule_min_gain)
        log("decision rules (nested macro F0.5):", [(r["margin"], r["contra_penalty"], round(r["nested_macro_f05"], 5)) for r in rule_rows])
    rule = {"margin": rule_sel["margin"], "contra_penalty": rule_sel["contra_penalty"]}
    log(f"decision rule: {rule} (nested macro F0.5 {rule_sel['nested_macro_f05']:.5f} vs threshold-only {rule_sel['nested_macro_f05_threshold_only']:.5f})")
    rows = sweep(links, nt, grid, **rule)
    best = best_threshold(rows)
    best_plain = best_threshold(sweep(links, nt, grid)) if rule != NO_RULE else best
    log("stage-2 OOF best", {k: best[k] for k in ("threshold", "macro_f05", "micro_precision", "micro_recall", "singleton_fp", "fp_decoy", "fp_wrong_entity")})
    folds_rep = fold_report(links, nt, s1_fold, best["threshold"], grid, rule)
    log("per-fold macro F0.5 at global thr", [round(r["macro_f05_at_global_thr"], 4) for r in folds_rep["folds"]],
        "std", round(folds_rep["macro_f05_std"], 5), "fold-best thresholds", [r["best_threshold"] for r in folds_rep["folds"]])
    # metrics on the entities whose candidate lists are entirely within their fold (no cross-fold pair)
    within_s = np.ones(len(s1_rids), dtype=bool)
    s_index = pl.DataFrame({"s_rid": s1_rids}).with_row_index("s_idx")
    cross_s = meta.filter(pl.Series(cross)).select("s_rid").unique().join(s_index, on="s_rid")["s_idx"].to_numpy()
    within_s[cross_s] = False
    audit = {"cross_fold_pair_share": float(cross.mean()), "entities_with_cross_fold_pair": float((~within_s).mean())}
    if within_s.any() and (~within_s).any():
        for name, sel in (("within_fold_only", within_s), ("with_cross_fold_pairs", ~within_s)):
            audit[name] = metrics_at(*subset_links(links, nt, sel), best["threshold"], **rule)
    # per-country (entity level) and per-source (link level)
    country = s1["country"].fill_null("").to_numpy()
    per_country = {}
    for c in sorted(set(country.tolist())):
        per_country[c] = metrics_at(*subset_links(links, nt, country == c), best["threshold"], **rule)
    per_source = per_source_report(links, nt, best["threshold"], rec, rule)
    # pair-level metrics: why "accuracy" looks great while F0.5 does not
    pred_pos = p2c >= best["threshold"]
    pair = {"accuracy": float((pred_pos == (y_all == 1)).mean()), "positive_rate": float(y_all.mean()),
            "precision": float((pred_pos & (y_all == 1)).sum() / max(pred_pos.sum(), 1)),
            "recall": float((pred_pos & (y_all == 1)).sum() / max((y_all == 1).sum(), 1))}
    blocking_recall = None
    br = os.path.join(d, "blocking_recall.json")
    if os.path.exists(br):
        with open(br) as fh:
            blocking_recall = json.load(fh)
    report = {
        "checkpoint": ckpt.name,
        "n_pairs": int(n), "n_positive_pairs": int(y_all.sum()), "n_s1": int(len(s1_rids)), "folds": fold_ids,
        "stage1_features": len(f1), "dropped_features": sorted(drop), "stage2": not args.no_stage2,
        "threshold": best["threshold"], "oof_at_threshold": best, "threshold_sweep": rows,
        "decision_rule": {**rule_sel, "candidates": rule_rows, "oof_at_threshold_only": best_plain},
        "stage1_oof_at_threshold": s1_best,
        "per_fold": folds_rep, "per_country": per_country, "per_source": per_source, "pair_level": pair,
        "calibration": {"platt": cal, "nested": cal_report}, "leakage_audit": audit,
        "blocking_recall": blocking_recall, "candidate_audit": blocking_recall,
    }
    tag = f"_{args.tag}" if args.tag else ""
    with open(ckpt.path("validation_report.json"), "w") as fh:
        json.dump(report, fh, indent=1)
    meta.select("pid", "t_rid", "s_rid", "label", "s_fold", "t_fold").with_columns(
        pl.Series("p1", p1), pl.Series("p2", p2), pl.Series("p2_cal", p2c), pl.Series("contra", contra)).write_parquet(ckpt.path("oof.parquet"))
    calibrate.save(cal, ckpt.path("calibration.json"))
    # the OOF-optimal global threshold (+ the selected decision rule) as a policy file: the baseline every
    # per-class policy is compared with
    ThresholdPolicy(best["threshold"], name="global_train", fit={"source": "train.py OOF sweep", "objective": "macro_f05",
                    "oof_macro_f05": best["macro_f05"], "nested_macro_f05": rule_sel["nested_macro_f05"]}, **rule).save(
        ckpt.path(os.path.join("thresholds", "global_train.json")))
    ckpt.update_manifest(oof_macro_f05=best["macro_f05"], nested_macro_f05=rule_sel["nested_macro_f05"], global_threshold=best["threshold"],
                         decision=rule, n_pairs=int(n), n_s1=int(len(s1_rids)), folds=fold_ids, calibration=cal, stage2=not args.no_stage2)
    ckpt.mark_stage("oof")
    if args.tag:
        # ablation run: report + OOF only (no final models); keep the legacy tagged copies at the top level
        with open(os.path.join(args.work_dir, f"validation_report{tag}.json"), "w") as fh:
            json.dump(report, fh, indent=1)
        pl.read_parquet(ckpt.path("oof.parquet")).write_parquet(os.path.join(d, f"oof{tag}.parquet"))
        log(f"ablation checkpoint {ckpt.dir}")
        return

    # ---------------- final models on the whole sample
    m1 = load_or_train(ckpt, "stage1.txt", args.resume, lambda: train_lgb(X1, ys, args.rounds1))
    if not args.no_stage2:
        m2 = load_or_train(ckpt, "stage2.txt", args.resume, lambda: train_lgb(X2, ys, args.rounds2))
        imp = sorted(zip(f2, m2.feature_importance("gain")), key=lambda x: -x[1])
    else:
        imp = sorted(zip(f1, m1.feature_importance("gain")), key=lambda x: -x[1])
    with open(ckpt.path("model_meta.json"), "w") as fh:
        json.dump({"checkpoint": ckpt.name, "f1": f1, "f2": f2, "stage2": not args.no_stage2, "threshold": best["threshold"],
                   "threshold_scale": "calibrated", "oof_macro_f05": best["macro_f05"], "nested_macro_f05": rule_sel["nested_macro_f05"],
                   "decision": rule, "feature_importance_gain": [(nm, float(g)) for nm, g in imp]}, fh, indent=1)
    ckpt.mark_stage("final")
    ckpt.set_latest()
    log(f"checkpoint {ckpt.dir} complete; <work>/ files now point at it")
    log("top features", [(nm, round(float(g))) for nm, g in imp[:25]])


if __name__ == "__main__":
    main()
