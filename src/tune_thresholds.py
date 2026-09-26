"""Step 4c: fit, compare and select threshold policies (global vs per-class) on one checkpoint.

Reads the out-of-fold calibrated probabilities of a checkpoint (never the test split's labels:
the test split only contributes record *counts* for the decoy-density adjustment, exactly as
in select_threshold.py) and, for every configuration:

  1. fits the global threshold (macro F0.5 maximiser on the OOF predictions, decoy-weighted);
  2. fits per-class thresholds on top of it (threshold_policy.fit_policy);
  3. reports per-class metrics before (global) and after (per-class) on the whole OOF set;
  4. re-fits both on K-1 folds and scores the held-out fold (nested_comparison), so the
     per-class-vs-global comparison is measured on entities whose thresholds were never
     tuned on them. Per-class thresholds always win in-sample by construction; the nested
     number is the one that decides.

Selection (--select auto): the configuration with the best nested macro F0.5 whose gain over
the global policy is at least --min-gain (default 1e-4, i.e. below the fold-to-fold noise a
per-class policy is not worth its extra parameters); otherwise the global policy. The chosen
policy is written to <checkpoint>/thresholds/selected.json and linked as <work>/threshold_policy.json,
which predict.py picks up. Every configuration's policy is kept in <checkpoint>/thresholds/, every
run is appended to <checkpoint>/experiments.jsonl, and the comparison tables go to
<work>/threshold_experiments.json / .md.

Configurations: --class-by for one, --configs FILE (JSON list of {name, class_by, objective, beta,
floor, min_support, unseen, passes}) for many, or the built-in set (--configs default).
"""
import json
import os

import numpy as np
import polars as pl

from checkpoint import Checkpoint
from common import base_args, log, split_dir
from pair_features import truth_pairs
from select_threshold import decoy_ratio, source_counts
from threshold_policy import ATTRS, ThresholdPolicy, class_keys, evaluate_policy, fit_global, fit_policy, nested_comparison, parse_class_by, record_attrs, record_columns
from thresholds import default_grid, link_table, metrics_at

DEFAULT_CONFIGS = [
    {"name": "global", "class_by": []},
    {"name": "country", "class_by": ["country"]},
    {"name": "src", "class_by": ["src"]},
    {"name": "country|src", "class_by": ["country", "src"]},
    {"name": "indic", "class_by": ["indic"]},
    {"name": "noaddr", "class_by": ["noaddr"]},
    {"name": "country|src|indic", "class_by": ["country", "src", "indic"]},
    {"name": "country|src|noaddr", "class_by": ["country", "src", "noaddr"]},
]


def load_configs(spec, class_by):
    if class_by:
        return [{"name": "global", "class_by": []}, {"name": "|".join(parse_class_by(class_by)), "class_by": parse_class_by(class_by)}]
    if spec in (None, "default"):
        return [dict(c) for c in DEFAULT_CONFIGS]
    with open(spec) as f:
        cfgs = json.load(f)
    if not any(c.get("name") == "global" for c in cfgs):
        cfgs = [{"name": "global", "class_by": []}] + cfgs
    return cfgs


def class_decoy_ratios(rec_tr, rec_te, truth, class_by, r_global):
    """Decoy-density ratio per class from record counts only (no test labels).

    For class c = (S1-side attributes c_s, Source 2/3-side attributes c_t):
      matches per S1 entity   m_c = #true pairs in c / #S1(c_s)            (train)
      decoys per S1 entity    d_c = #T records in c / #S1(c_s) - m_c        (train and test)
      ratio                   r_c = max(1, d_c(test) / d_c(train))
    #T records in c needs c_s to be readable off the Source 2/3 record, which is the case for
    'both' attributes (country); a class with any other S1-side attribute keeps r_global.
    Classes absent from the test split, or with no training decoys, also keep r_global."""
    class_by = parse_class_by(class_by)
    if not class_by:
        return {}, {}
    if any(ATTRS[a].side == "s" for a in class_by):
        log(f"per-class decoy ratios not available for S1-only attributes {[a for a in class_by if ATTRS[a].side == 's']}; using r={r_global:.3f}")
        return {}, {}
    s_attrs = [a for a in class_by if ATTRS[a].side == "both"]

    def counts(rec):
        ra = record_attrs(rec, class_by).join(rec.select(pl.col("rid").cast(pl.UInt32), pl.col("src").alias("__src")), on="rid")
        key_t = pl.concat_str([pl.lit(f"{a}=") + pl.col(a) for a in class_by], separator="|")
        key_s = pl.concat_str([pl.lit(f"{a}=") + pl.col(a) for a in s_attrs], separator="|") if s_attrs else pl.lit("")
        n_t = ra.filter(pl.col("__src") != 1).group_by(key_t.alias("key")).len()
        n_s = ra.filter(pl.col("__src") == 1).group_by(key_s.alias("key_s")).len()
        return dict(zip(n_t["key"], n_t["len"])), dict(zip(n_s["key_s"], n_s["len"]))

    nt_tr, ns_tr = counts(rec_tr)
    nt_te, ns_te = counts(rec_te)
    tk = class_keys(truth.select("t_rid", "s_rid"), rec_tr, class_by)
    n_true = {k: int(v) for k, v in zip(*np.unique(tk, return_counts=True))}

    def s_part(key):
        return "|".join(x for x in key.split("|") if x.split("=")[0] in s_attrs)

    ratios, detail = {}, {}
    for key in sorted(set(nt_tr) | set(nt_te)):
        ks = s_part(key)
        s1_tr, s1_te = ns_tr.get(ks, 0), ns_te.get(ks, 0)
        if s1_tr == 0 or s1_te == 0 or key not in nt_tr or key not in nt_te:
            detail[key] = {"ratio": r_global, "note": "class absent from one split -> global ratio"}
            continue
        m = n_true.get(key, 0) / s1_tr
        d_tr = nt_tr[key] / s1_tr - m
        d_te = nt_te[key] / s1_te - m
        r = max(1.0, d_te / d_tr) if d_tr > 1e-9 else r_global
        ratios[key] = r
        detail[key] = {"ratio": r, "matches_per_s1": m, "decoys_per_s1_train": d_tr, "decoys_per_s1_test": d_te,
                       "n_t_train": int(nt_tr[key]), "n_t_test": int(nt_te[key]), "n_s1_train": int(s1_tr), "n_s1_test": int(s1_te)}
    return ratios, detail


def markdown(res):
    L = [f"# Threshold policies on checkpoint `{res['checkpoint']}`", "",
         f"OOF entities {res['n_entities']}, link rows {res['n_link_rows']}, global decoy ratio r = {res['decoy_ratio_global']:.3f} "
         f"(density mode `{res['density']}`), objective macro F0.5 on the calibrated scale, grid step {res['grid_step']}.", "",
         "## Configurations (nested = fit on K-1 folds, scored on the held-out fold; the honest comparison)", "",
         "| config | classes | nested macro F0.5 | gain vs global | folds better | fold std | in-sample macro F0.5 | precision | recall | singleton acc | links |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in res["configs"].items():
        n, o = r["nested"], r["in_sample"]["overall"]
        L.append(f"| {name} | {r['n_classes']} | {n['nested_macro_f05']:.5f} | {n['gain_vs_global']:+.5f} | {n['folds_better_than_global']}/{len(n['per_fold'])} | "
                 f"{n['fold_macro_f05_std']:.5f} | {o['macro_f05']:.5f} | {o['micro_precision']:.5f} | {o['micro_recall']:.5f} | {o['singleton_acc']:.4f} | {o['links']} |")
    L += ["", f"**Selected: `{res['selected']}`** ({res['selection_rule']})", ""]
    for name, r in res["configs"].items():
        if not r["policy"]["thresholds"]:
            continue
        L += [f"## {name}: per-class thresholds and metrics (OOF, in-sample)", "",
              "| class | entities | link rows | decoy ratio | thr global | thr class | F0.5 global | F0.5 class | delta | P global | P class | R global | R class |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for cls, a in r["in_sample"]["per_class"].items():
            b = r["in_sample"]["per_class_at_global"].get(cls, a)
            rr = r["decoy_ratios"].get(cls, res["decoy_ratio_global"])
            L.append(f"| {cls} | {a['n_entities']} | {a['n_link_rows']} | {rr:.3f} | {r['policy']['default']:.2f} | {a['threshold']:.2f} | {b['macro_f05']:.5f} | {a['macro_f05']:.5f} | "
                     f"{a['macro_f05'] - b['macro_f05']:+.5f} | {b['micro_precision']:.5f} | {a['micro_precision']:.5f} | {b['micro_recall']:.5f} | {a['micro_recall']:.5f} |")
        fold_thr = r["nested"]["fold_policies"]
        L += ["", "Per-fold refits (thresholds chosen without that fold): " + "; ".join(
            f"fold {fp['fold']}: " + ", ".join(f"{k}={v:.2f}" for k, v in fp["policy"].items()) for fp in fold_thr), ""]
    return "\n".join(L) + "\n"


def main():
    ap = base_args(__doc__)
    ap.add_argument("--checkpoint", default=None, help="checkpoint name or directory (default: LATEST)")
    ap.add_argument("--configs", default="default", help="JSON file with a list of configurations, or 'default'")
    ap.add_argument("--class-by", default=None, help="one configuration, e.g. country,src (overrides --configs)")
    ap.add_argument("--density", choices=["global", "per_class", "none"], default="global",
                    help="decoy-density adjustment: one ratio for all classes, one per class, or none")
    ap.add_argument("--ratio", type=float, default=None, help="override the global decoy-density ratio")
    ap.add_argument("--min-support", type=int, default=200, help="classes with fewer OOF entities keep the global threshold")
    ap.add_argument("--min-gain", type=float, default=1e-4, help="nested gain a per-class policy needs to be selected")
    ap.add_argument("--select", default="auto", help="auto | global | <config name>")
    ap.add_argument("--unseen", choices=["default", "max", "min", "mean"], default="default", help="threshold of a class unseen at fit time")
    ap.add_argument("--tag", default="", help="suffix for the output file names")
    args = ap.parse_args()
    run(args)


def run(args):
    ckpt = Checkpoint.resolve(args.work_dir, args.checkpoint)
    if ckpt is None or not ckpt.exists():
        raise SystemExit("no checkpoint found: run train.py first (or pass --checkpoint)")
    dtr = split_dir(args.work_dir, "train")
    configs = load_configs(args.configs, args.class_by)
    attrs = sorted({a for c in configs for a in parse_class_by(c.get("class_by", []))})
    cols = sorted(set(record_columns(attrs)) | {"rid", "entity_id", "src", "fold", "country"})
    rec = pl.read_parquet(os.path.join(dtr, "records.parquet"), columns=cols)
    truth = truth_pairs(rec, args.data_dir).drop("label")
    s1 = rec.filter(pl.col("src") == 1)
    s1_rids = s1["rid"].to_numpy().astype(np.uint32)
    s1_fold = s1["fold"].to_numpy()
    oof_path = ckpt.path("oof.parquet") if ckpt.has("oof.parquet") else os.path.join(dtr, "oof.parquet")
    oof = pl.read_parquet(oof_path, columns=["t_rid", "s_rid", "label", "p2_cal"])
    links, nt = link_table(oof.select("t_rid", "s_rid"), oof["p2_cal"].to_numpy(), truth, s1_rids)
    grid = default_grid()
    log(f"checkpoint {ckpt.name}: {links.height} link rows, {len(nt)} entities, {len(configs)} configurations")

    # decoy-density adjustment from record counts only
    src_tr, src_te = source_counts(args.work_dir)
    r_global = 1.0
    if args.density != "none" and src_te is not None:
        _, dec_tr, dec_te, r_global = decoy_ratio(src_tr, src_te, truth.height)
        r_global = args.ratio if args.ratio is not None else r_global
        log(f"decoys per S1: train {dec_tr:.3f}, test {dec_te:.3f}; global ratio r={r_global:.3f}")
    rec_te = None
    if args.density == "per_class":
        te_path = os.path.join(split_dir(args.work_dir, "test"), "records.parquet")
        rec_te = pl.read_parquet(te_path, columns=[c for c in cols if c not in ("fold", "entity_id")]) if os.path.exists(te_path) else None
        if rec_te is None:
            log("no test records: per-class density falls back to the global ratio")

    # class keys of every link row (its Source 1 side and Source 2/3 side) and of every S1 entity
    link_pairs = links.select("t_rid", "s_rid")
    g_all = fit_global(links, nt, grid, r_global)
    g_train = fit_global(links, nt, grid, 1.0)
    log(f"global threshold: train-optimal {g_train:.2f}, density-adjusted {g_all:.2f}")
    results, policies, nested_cfgs, keys_by_cfg, dw_by_cfg = {}, {}, {}, {}, {}
    for cfg in configs:
        name = cfg["name"]
        class_by = parse_class_by(cfg.get("class_by", []))
        keys = class_keys(link_pairs, rec, class_by)
        entity_keys = class_keys(pl.DataFrame({"t_rid": s1_rids, "s_rid": s1_rids}), rec, class_by) if class_by and all(ATTRS[a].side != "t" for a in class_by) else None
        ratios, ratio_detail = ({}, {})
        if rec_te is not None and class_by:
            ratios, ratio_detail = class_decoy_ratios(rec, rec_te, truth, class_by, r_global)
        dw = np.fromiter((ratios.get(k, r_global) for k in keys), dtype=np.float64, count=len(keys)) if ratios else r_global
        fit_kw = {k: cfg[k] for k in ("objective", "beta", "floor", "passes", "unseen", "class_objectives") if k in cfg}
        fit_kw.setdefault("unseen", args.unseen)
        pol, rep = fit_policy(links, nt, keys, class_by, grid, dw, min_support=cfg.get("min_support", args.min_support),
                              name=name, global_threshold=g_all, entity_keys=entity_keys, **fit_kw)
        pol.fit.update({"checkpoint": ckpt.name, "density": args.density, "decoy_ratio_global": r_global, "decoy_ratios": ratios,
                        "global_threshold_train_optimal": g_train})
        rep["at_train_optimal_global"] = metrics_at(links, nt, g_train, dw)
        policies[name] = pol
        keys_by_cfg[name], dw_by_cfg[name] = keys, dw
        results[name] = {"policy": pol.to_dict(), "n_classes": len(pol.thresholds), "in_sample": rep, "decoy_ratios": ratios, "decoy_ratio_detail": ratio_detail}
        nested_cfgs[name] = {k: v for k, v in cfg.items() if k != "name"}
        nested_cfgs[name]["min_support"] = cfg.get("min_support", args.min_support)
        log(f"{pol.describe()}  in-sample macro F0.5 {rep['overall']['macro_f05']:.5f}")

    # nested (leak-free) comparison: one shared key set is needed -> run per config with its own keys
    for name, cfg in nested_cfgs.items():
        if name == "global":
            continue
        ek = None
        cb = parse_class_by(cfg.get("class_by", []))
        if cb and all(ATTRS[a].side != "t" for a in cb):
            ek = class_keys(pl.DataFrame({"t_rid": s1_rids, "s_rid": s1_rids}), rec, cb)
        nc = nested_comparison(links, nt, keys_by_cfg[name], s1_fold, {name: cfg}, grid, dw_by_cfg[name], entity_keys=ek)
        results[name]["nested"] = nc[name]
        if "nested" not in results["global"]:
            results["global"]["nested"] = nc["global"]
        log(f"nested: {name} {nc[name]['nested_macro_f05']:.5f} vs global {nc['global']['nested_macro_f05']:.5f} "
            f"({nc[name]['gain_vs_global']:+.5f}, better in {nc[name]['folds_better_than_global']}/{len(nc[name]['per_fold'])} folds)")
    if "nested" not in results["global"]:
        results["global"]["nested"] = nested_comparison(links, nt, keys_by_cfg["global"], s1_fold, {}, grid, dw_by_cfg["global"])["global"]

    # selection
    if args.select == "auto":
        cands = [(r["nested"]["nested_macro_f05"], n) for n, r in results.items() if n != "global" and r["nested"]["gain_vs_global"] >= args.min_gain]
        selected = max(cands)[1] if cands else "global"
        rule = f"auto: best nested macro F0.5 with gain >= {args.min_gain} over global, else global"
    else:
        selected = args.select if args.select in results else "global"
        rule = f"--select {args.select}"
    log(f"selected policy: {selected} ({rule})")

    # outputs
    tag = f"_{args.tag}" if args.tag else ""
    for name, pol in policies.items():
        pol.save(ckpt.path(os.path.join("thresholds", f"{name.replace('|', '_')}{tag}.json")))
    sel_path = ckpt.path(os.path.join("thresholds", f"selected{tag}.json"))
    policies[selected].save(sel_path)
    latest = Checkpoint.resolve(args.work_dir, None)
    if not args.tag and latest is not None and latest.name == ckpt.name:
        ckpt.set_latest()  # refreshes <work>/threshold_policy.json
    out = {"checkpoint": ckpt.name, "checkpoint_dir": ckpt.dir, "n_entities": int(len(nt)), "n_link_rows": int(links.height),
           "density": args.density, "decoy_ratio_global": r_global, "grid_step": float(np.round(grid[1] - grid[0], 4)),
           "global_threshold_train_optimal": g_train, "global_threshold_density_adjusted": g_all,
           "selected": selected, "selection_rule": rule, "selected_policy_path": sel_path,
           "configs": results}
    with open(os.path.join(args.work_dir, f"threshold_experiments{tag}.json"), "w") as f:
        json.dump(out, f, indent=1)
    md = markdown(out)
    with open(os.path.join(args.work_dir, f"threshold_experiments{tag}.md"), "w") as f:
        f.write(md)
    ckpt.log_experiment({"kind": "tune_thresholds", "density": args.density, "decoy_ratio_global": r_global, "selected": selected,
                         "configs": {n: {"nested_macro_f05": r["nested"]["nested_macro_f05"], "gain_vs_global": r["nested"]["gain_vs_global"],
                                         "in_sample_macro_f05": r["in_sample"]["overall"]["macro_f05"], "thresholds": r["policy"]["thresholds"],
                                         "default": r["policy"]["default"]} for n, r in results.items()},
                         "outputs": [os.path.join(args.work_dir, f"threshold_experiments{tag}.json"), sel_path]})
    print(md)


if __name__ == "__main__":
    main()
