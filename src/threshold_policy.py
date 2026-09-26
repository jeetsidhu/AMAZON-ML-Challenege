"""Configurable acceptance thresholds: one global threshold or one threshold per class.

A *class* is a value of one or more record attributes that are known at prediction time
without labels (the Source 1 entity's country, the Source 2/3 record's source / script /
missing-address flag, ...). Every candidate link (Source 2/3 record -> its best Source 1
candidate) gets the threshold of its class; a class never seen when the policy was fit (the
test split's France, for example) falls back to the policy's default threshold.

    policy = fit_policy(links, nt, cls, grid, ...)          # on out-of-fold predictions
    policy.save(path); policy = ThresholdPolicy.load(path)
    thr = policy.thresholds_for(keys)                        # one threshold per link row
    links_kept = assign(meta, p, thr)                        # model.assign accepts arrays

Objective. The challenge scores macro F0.5 over Source 1 entities, so the per-class thresholds
are chosen to maximise exactly that, with the thresholds of the other classes held fixed
(coordinate ascent from the global optimum; it converges in 2-3 passes because the classes
interact only through entities whose links fall in several classes). When the classes
partition the *entities* (country only), the ascent is exact in one pass and each class's
threshold is simply the maximiser of its own entities' mean F0.5. Alternatives are available
per class through `objective`: "fbeta" with a class-specific beta (beta < 0.5 = even more
precision-first, e.g. for a class whose calibration is doubtful) and "precision_floor"
(maximise recall subject to the class's link precision >= floor). Two safeguards against
fitting noise: a class with fewer than `min_support` entities keeps the global threshold, and
`nested_comparison` refits everything on K-1 folds and scores the held-out fold, so the
reported gain of per-class over global thresholds is measured on entities the thresholds were
never tuned on (both policies are always evaluated on the same held-out entities).
"""
import json
import os
import time

import numpy as np
import polars as pl

from thresholds import accept_mask, best_f05_threshold, entity_scores, metrics_at, subset_links

POLICY_VERSION = 2  # 2: decision-rule extras (margin, contra_penalty)
SEP = "|"


# ------------------------------------------------------------------ class attributes
class Attr:
    """A record attribute that may define a class. side: 's' (taken from the Source 1 record of the
    pair), 't' (from the Source 2/3 record) or 'both' (present on both records and equal within a
    pair, e.g. country: used for record counts on either side)."""

    def __init__(self, side, expr, columns):
        self.side, self.expr, self.columns = side, expr, columns


ATTRS = {
    "country": Attr("both", pl.col("country").fill_null(""), ["country"]),
    "src": Attr("t", pl.col("src").cast(pl.Int64).cast(pl.String), ["src"]),
    "indic": Attr("t", pl.col("f_indic").cast(pl.Int8).cast(pl.String), ["f_indic"]),
    "alias": Attr("t", pl.col("f_alias").cast(pl.Int8).cast(pl.String), ["f_alias"]),
    "noaddr": Attr("t", (pl.col("a_norm").fill_null("") == "").cast(pl.Int8).cast(pl.String), ["a_norm"]),
    "s_noaddr": Attr("s", (pl.col("a_norm").fill_null("") == "").cast(pl.Int8).cast(pl.String), ["a_norm"]),
    "domain": Attr("t", (pl.col("n_domain").fill_null("") != "").cast(pl.Int8).cast(pl.String), ["n_domain"]),
}


def parse_class_by(spec):
    """'country,src' / 'country|src' / list -> list of attribute names (validated)."""
    if isinstance(spec, str):
        spec = [x for x in spec.replace("|", ",").split(",") if x]
    spec = list(spec or [])
    unknown = [a for a in spec if a not in ATTRS]
    if unknown:
        raise ValueError(f"unknown class attributes {unknown}; known: {sorted(ATTRS)}")
    return spec


def record_columns(class_by):
    cols = {"rid"}
    for a in class_by:
        cols.update(ATTRS[a].columns)
    return sorted(cols)


def record_attrs(rec, class_by):
    """rid -> one string column per attribute (evaluated on the record itself)."""
    return rec.select(pl.col("rid").cast(pl.UInt32), *[ATTRS[a].expr.alias(a) for a in class_by])


def class_keys(pairs, rec, class_by):
    """Class key of every row of `pairs` (t_rid, s_rid): 'country=US|src=2' (empty string when
    class_by is empty). S-side attributes come from the Source 1 record, t-side ones from the
    Source 2/3 record; 'both' attributes are read from the Source 1 side."""
    class_by = parse_class_by(class_by)
    n = pairs.height
    if not class_by:
        return np.full(n, "", dtype=object)
    ra = record_attrs(rec, class_by)
    out = pairs.select("t_rid", "s_rid").with_row_index("__i")
    s_attrs = [a for a in class_by if ATTRS[a].side in ("s", "both")]
    t_attrs = [a for a in class_by if ATTRS[a].side == "t"]
    if s_attrs:
        out = out.join(ra.select("rid", *s_attrs).rename({"rid": "s_rid"}), on="s_rid", how="left")
    if t_attrs:
        out = out.join(ra.select("rid", *t_attrs).rename({"rid": "t_rid"}), on="t_rid", how="left")
    out = out.sort("__i")
    key = pl.concat_str([pl.lit(f"{a}=") + pl.col(a).fill_null("") for a in class_by], separator=SEP)
    return out.select(key.alias("key"))["key"].to_numpy().astype(object)


def encode_keys(keys, names=None):
    """String keys -> (int codes, sorted list of class names). Unknown keys (not in names) -> -1."""
    keys = np.asarray(keys, dtype=object)
    if names is None:
        names = sorted(set(keys.tolist()))
    lut = {k: i for i, k in enumerate(names)}
    codes = np.fromiter((lut.get(k, -1) for k in keys), dtype=np.int64, count=len(keys))
    return codes, list(names)


# ------------------------------------------------------------------ the policy object
class ThresholdPolicy:
    """scheme 'global' (one threshold) or 'per_class' (thresholds[class key], default for the rest)."""

    def __init__(self, default, thresholds=None, class_by=(), scale="calibrated", unseen="default", name=None, fit=None,
                 margin=0.0, contra_penalty=0.0):
        self.default = float(default)
        self.thresholds = {k: float(v) for k, v in (thresholds or {}).items()}
        self.class_by = parse_class_by(class_by)
        self.scale = scale
        self.unseen = unseen  # 'default' | 'max' | 'min' | 'mean': what an unseen class gets
        self.name = name or ("global" if not self.thresholds else SEP.join(self.class_by))
        self.fit = fit or {}
        # decision-rule extras shared by every class (thresholds.accept_mask / model.assign)
        self.margin = float(margin or 0.0)
        self.contra_penalty = float(contra_penalty or 0.0)

    @property
    def rule(self):
        return {"margin": self.margin, "contra_penalty": self.contra_penalty}

    @property
    def scheme(self):
        return "per_class" if self.thresholds else "global"

    def fallback(self):
        if self.unseen == "default" or not self.thresholds:
            return self.default
        vals = np.array(list(self.thresholds.values()))
        return float({"max": vals.max(), "min": vals.min(), "mean": vals.mean()}[self.unseen])

    def thresholds_for(self, keys):
        """One threshold per key (str array); classes absent from the policy get the fallback."""
        fb = self.fallback()
        if not self.thresholds:
            return np.full(len(keys), self.default)
        return np.fromiter((self.thresholds.get(k, fb) for k in keys), dtype=np.float64, count=len(keys))

    def to_dict(self):
        return {"version": POLICY_VERSION, "name": self.name, "scheme": self.scheme, "scale": self.scale,
                "class_by": self.class_by, "default": self.default, "unseen": self.unseen,
                "thresholds": dict(sorted(self.thresholds.items())), "margin": self.margin, "contra_penalty": self.contra_penalty,
                "fit": self.fit}

    @classmethod
    def from_dict(cls, d):
        return cls(d["default"], d.get("thresholds"), d.get("class_by", []), d.get("scale", "calibrated"),
                   d.get("unseen", "default"), d.get("name"), d.get("fit"), d.get("margin", 0.0), d.get("contra_penalty", 0.0))

    def save(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=1)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def describe(self):
        extra = ""
        if self.margin or self.contra_penalty:
            extra = f" (margin {self.margin:.2f}, contradiction penalty {self.contra_penalty:.2f})"
        if not self.thresholds:
            return f"{self.name}: global threshold {self.default:.3f}{extra}"
        return f"{self.name}: per-class thresholds on {self.class_by} " + ", ".join(
            f"{k}={v:.2f}" for k, v in sorted(self.thresholds.items())) + f"; unseen -> {self.fallback():.3f}{extra}"


# ------------------------------------------------------------------ objectives
def make_objective(kind="macro_f05", beta=0.5, floor=None):
    """Returns objective(f, tp, npred, nt, keep, links, cls_mask) -> float to maximise.

    macro_f05      : mean per-entity F0.5 over all entities (the challenge metric)
    fbeta          : mean per-entity F_beta over all entities (beta < 0.5 = more precision-first)
    precision_floor: link recall of the class subject to its link precision >= floor (-inf otherwise)
    """
    if kind == "macro_f05" or (kind == "fbeta" and abs(beta - 0.5) < 1e-12):
        return lambda f, tp, npred, nt, keep, links, m: float(f.mean())
    if kind == "fbeta":
        b2 = beta * beta

        def fb(f, tp, npred, nt, keep, links, m):
            ntf = nt.astype(float)
            with np.errstate(divide="ignore", invalid="ignore"):
                prec = np.where(npred > 0, tp / npred, 0.0)
                rec = np.where(ntf > 0, tp / ntf, 0.0)
                g = np.where(prec + rec > 0, (1 + b2) * prec * rec / (b2 * prec + rec), 0.0)
            g = np.where((ntf == 0) & (npred == 0), 1.0, g)
            g = np.where((ntf == 0) & (npred > 0), 0.0, g)
            return float(g.mean())
        return fb
    if kind == "precision_floor":
        floor = 0.999 if floor is None else float(floor)

        def pf(f, tp, npred, nt, keep, links, m):
            acc = keep & m
            tpr = links["tp"].to_numpy()
            n_acc = acc.sum()
            prec = (acc & tpr).sum() / n_acc if n_acc else 1.0
            rec = (acc & tpr).sum() / max((m & ~links["decoy"].to_numpy()).sum(), 1)
            return float(rec) if prec >= floor else -1.0 + float(prec)
        return pf
    raise ValueError(kind)


# ------------------------------------------------------------------ fitting
def fit_global(links, nt, grid, decoy_weight=1.0, objective=None, margin=0.0, contra_penalty=0.0):
    """Global threshold: the grid point maximising the objective (default macro F0.5)."""
    if objective is None:
        return best_f05_threshold(links, nt, grid, decoy_weight, margin, contra_penalty)[0]
    p = links["p"].to_numpy()
    m = np.ones(len(p), dtype=bool)
    best_v, best_t = -np.inf, None
    for t in grid:
        f, tp, npred, keep = entity_scores(links, nt, float(t), decoy_weight, margin=margin, contra_penalty=contra_penalty)
        v = objective(f, tp, npred, nt, keep, links, m)
        if v > best_v:
            best_v, best_t = v, float(t)
    return best_t


def fit_per_class(links, nt, cls, n_classes, grid, init, decoy_weight=1.0, objective=None, min_support=200,
                  passes=4, tol=1e-12, class_objectives=None, margin=0.0, contra_penalty=0.0):
    """Coordinate ascent: sweep one class's threshold over the grid with all others fixed.

    cls: int class code per link row (-1 = no class); init: starting threshold (the global one);
    min_support: classes touching fewer entities keep `init`; class_objectives: optional
    {code: objective} overriding `objective` for particular classes.
    Returns (thr per class, support per class, trace)."""
    objective = objective or make_objective()
    class_objectives = class_objectives or {}
    p = links["p"].to_numpy()
    s_idx = links["s_idx"].to_numpy()
    dw = decoy_weight if np.ndim(decoy_weight) else float(decoy_weight)
    thr = np.full(n_classes, float(init))
    support = np.array([len(np.unique(s_idx[cls == c])) for c in range(n_classes)])
    active = [c for c in range(n_classes) if support[c] >= min_support]
    trace = []
    grid = np.asarray(grid, dtype=np.float64)
    for it in range(passes):
        changed = 0
        for c in active:
            m = cls == c
            obj = class_objectives.get(c, objective)
            base_keep = (~m) & accept_mask(links, thr[cls.clip(0)], margin, contra_penalty)  # other classes at their current thresholds
            base_keep &= cls >= 0
            best_v, best_t = -np.inf, thr[c]
            vals = []
            for t in grid:
                keep = base_keep | (m & accept_mask(links, float(t), margin, contra_penalty))
                f, tp, npred, _ = entity_scores(links, nt, 0.0, dw, keep=keep)
                v = obj(f, tp, npred, nt, keep, links, m)
                vals.append(v)
                # ties: prefer the threshold closest to the starting point (stability), then the higher one
                if v > best_v + tol or (abs(v - best_v) <= tol and abs(t - init) < abs(best_t - init) - 1e-12):
                    best_v, best_t = v, float(t)
            if abs(best_t - thr[c]) > 1e-12:
                changed += 1
            thr[c] = best_t
            trace.append({"pass": it, "class": int(c), "threshold": float(best_t), "objective": float(best_v)})
        if changed == 0:
            break
    return thr, support, trace


def class_report(links, nt, cls, names, thr_rows, decoy_weight=1.0, entity_cls=None, margin=0.0, contra_penalty=0.0):
    """Per-class metrics at per-row thresholds thr_rows. For each class: the link-level precision /
    recall over its rows and the macro F0.5 over the entities it touches (exactly the class's
    entities when entity_cls, a code per Source 1 entity, is given)."""
    out = {}
    s_idx = links["s_idx"].to_numpy()
    for c, name in enumerate(names):
        lm = cls == c
        if entity_cls is not None:
            em = entity_cls == c
        else:
            em = np.zeros(len(nt), dtype=bool)
            em[np.unique(s_idx[lm])] = True
        if not lm.any() and not em.any():
            continue
        r = metrics_at(links, nt, thr_rows, decoy_weight, entity_mask=em, link_mask=lm, margin=margin, contra_penalty=contra_penalty)
        r["threshold"] = float(np.unique(thr_rows[lm])[0]) if lm.any() and len(np.unique(thr_rows[lm])) == 1 else None
        r["n_link_rows"] = int(lm.sum())
        out[name] = r
    return out


def fit_policy(links, nt, keys, class_by, grid, decoy_weight=1.0, objective="macro_f05", beta=0.5, floor=None,
               min_support=200, passes=4, unseen="default", name=None, global_threshold=None, class_objectives=None,
               entity_keys=None, margin=0.0, contra_penalty=0.0):
    """Fits a global threshold and, when class_by is non-empty, per-class thresholds on top of it.
    keys: class key per link row (class_keys); entity_keys: optional class key per Source 1 entity
    (only meaningful when every attribute is s-side), used for exact per-class entity metrics.
    margin / contra_penalty: the decision-rule extras (thresholds.accept_mask) the policy carries.
    Returns (policy, report)."""
    class_by = parse_class_by(class_by)
    obj = make_objective(objective, beta, floor)
    t0 = time.time()
    rule = {"margin": margin, "contra_penalty": contra_penalty}
    g = float(global_threshold) if global_threshold is not None else fit_global(links, nt, grid, decoy_weight, None if objective == "macro_f05" else obj, **rule)
    fit_info = {"objective": objective, "beta": beta, "floor": floor, "min_support": min_support, "passes": passes,
                "grid": [float(grid[0]), float(grid[-1]), float(np.round(grid[1] - grid[0], 6)) if len(grid) > 1 else None],
                "global_threshold": g, "n_entities": int(len(nt)), "n_link_rows": int(links.height),
                "decoy_weight": (float(decoy_weight) if np.ndim(decoy_weight) == 0 else "per_row"), **rule}
    if not class_by:
        pol = ThresholdPolicy(g, {}, [], name=name or "global", fit=fit_info, **rule)
        rep = {"overall": metrics_at(links, nt, g, decoy_weight, **rule), "per_class": {}}
        fit_info["seconds"] = round(time.time() - t0, 2)
        return pol, rep
    cls, names = encode_keys(keys)
    per_class_obj = None
    if class_objectives:
        per_class_obj = {names.index(k): make_objective(**v) for k, v in class_objectives.items() if k in names}
    thr, support, trace = fit_per_class(links, nt, cls, len(names), grid, g, decoy_weight, obj, min_support, passes,
                                        class_objectives=per_class_obj, **rule)
    thresholds = {names[c]: float(thr[c]) for c in range(len(names))}
    pol = ThresholdPolicy(g, thresholds, class_by, unseen=unseen, name=name or SEP.join(class_by), fit=fit_info, **rule)
    thr_rows = pol.thresholds_for(keys)
    ecls = None
    if entity_keys is not None:
        ecls, _ = encode_keys(entity_keys, names)
    fit_info.update({"support": {names[c]: int(support[c]) for c in range(len(names))},
                     "classes_below_min_support": [names[c] for c in range(len(names)) if support[c] < min_support],
                     "trace": trace, "seconds": round(time.time() - t0, 2)})
    rep = {"overall": metrics_at(links, nt, thr_rows, decoy_weight, **rule),
           "overall_at_global": metrics_at(links, nt, g, decoy_weight, **rule),
           "per_class": class_report(links, nt, cls, names, thr_rows, decoy_weight, ecls, **rule),
           "per_class_at_global": class_report(links, nt, cls, names, np.full(len(keys), g), decoy_weight, ecls, **rule)}
    return pol, rep


def evaluate_policy(policy, links, nt, keys, decoy_weight=1.0, entity_keys=None):
    """Overall + per-class metrics of a policy on a link table (keys per link row)."""
    thr_rows = policy.thresholds_for(keys)
    rep = {"overall": metrics_at(links, nt, thr_rows, decoy_weight, **policy.rule)}
    if policy.class_by:
        names = sorted(set(policy.thresholds) | set(np.unique(keys).tolist()))
        cls, names = encode_keys(keys, names)
        ecls = encode_keys(entity_keys, names)[0] if entity_keys is not None else None
        rep["per_class"] = class_report(links, nt, cls, names, thr_rows, decoy_weight, ecls, **policy.rule)
    return rep


def nested_comparison(links, nt, keys, s_fold, configs, grid, decoy_weight=1.0, entity_keys=None, margin=0.0, contra_penalty=0.0):
    """Leak-free comparison of threshold configurations.

    For every fold k: fit each configuration (global threshold + per-class thresholds) on the
    entities of the other folds only, then apply it to the entities of fold k. The concatenated
    held-out predictions give one honest macro F0.5 per configuration, computed on the same
    entities for all of them. configs: {name: dict(class_by=..., objective=..., ...)}; the
    'global' configuration is always included.
    Returns {name: {"nested_macro_f05", "per_fold": [...], "fold_thresholds": [...]}}."""
    s_fold = np.asarray(s_fold)
    folds = sorted(int(k) for k in np.unique(s_fold))
    keys = np.asarray(keys, dtype=object)
    link_fold = s_fold[links["s_idx"].to_numpy()]
    configs = {"global": {"class_by": []}, **{k: v for k, v in configs.items() if k != "global"}}
    dw_rows = np.asarray(decoy_weight, dtype=np.float64)
    per_row_dw = dw_rows.ndim == 1
    thr_oof = {name: np.full(links.height, np.nan) for name in configs}
    fold_rows = {name: [] for name in configs}
    for k in folds:
        tr_e = s_fold != k
        lk_tr, nt_tr = subset_links(links, nt, tr_e)
        tr_rows = link_fold != k
        te_rows = link_fold == k
        dw_tr = dw_rows[tr_rows] if per_row_dw else float(dw_rows)
        rule = {"margin": margin, "contra_penalty": contra_penalty}
        g = fit_global(lk_tr, nt_tr, grid, dw_tr, **rule)
        ek_tr = entity_keys[tr_e] if entity_keys is not None else None
        for name, cfg in configs.items():
            cfg = {kk: vv for kk, vv in cfg.items() if kk in ("class_by", "objective", "beta", "floor", "min_support", "passes", "unseen", "class_objectives")}
            pol, _ = fit_policy(lk_tr, nt_tr, keys[tr_rows], cfg.get("class_by", []), grid, dw_tr, global_threshold=g,
                                entity_keys=ek_tr, **rule, **{kk: vv for kk, vv in cfg.items() if kk != "class_by"})
            thr_oof[name][te_rows] = pol.thresholds_for(keys[te_rows])
            fold_rows[name].append({"fold": k, "policy": pol.to_dict()["thresholds"] or {"global": pol.default}, "default": pol.default})
    out = {}
    rule = {"margin": margin, "contra_penalty": contra_penalty}
    for name in configs:
        thr = thr_oof[name]
        assert not np.isnan(thr).any()
        overall = metrics_at(links, nt, thr, decoy_weight, **rule)
        per_fold = []
        for k in folds:
            em = s_fold == k
            per_fold.append({"fold": k, **{a: b for a, b in metrics_at(links, nt, thr, decoy_weight, entity_mask=em, link_mask=link_fold == k, **rule).items() if a != "threshold"}})
        out[name] = {"nested_macro_f05": overall["macro_f05"], "nested_overall": overall, "per_fold": per_fold,
                     "fold_policies": fold_rows[name], "fold_macro_f05_std": float(np.std([r["macro_f05"] for r in per_fold]))}
    base = out["global"]
    for name, r in out.items():
        r["gain_vs_global"] = r["nested_macro_f05"] - base["nested_macro_f05"]
        r["folds_better_than_global"] = int(sum(a["macro_f05"] > b["macro_f05"] + 1e-12 for a, b in zip(r["per_fold"], base["per_fold"])))
    return out


def assign_with_policy(meta, p, keys, policy, contra=None):
    """Links (t_rid, s_rid, p) accepted under a policy: each Source 2/3 record's best candidate iff
    p >= the threshold of that pair's class, the margin over the runner-up is >= policy.margin and
    the contradiction rule holds (the same rule as model.assign with per-row thresholds)."""
    from model import assign  # local import: model.py imports lightgbm, which the tuning tools do not need
    return assign(meta, p, policy.thresholds_for(np.asarray(keys, dtype=object)), policy.margin, policy.contra_penalty, contra)
