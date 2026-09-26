"""Per-class thresholds: fitting, evaluation, serialisation and the leak-free comparison."""
import json
import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import threshold_policy as tp  # noqa: E402
from model import assign  # noqa: E402
from thresholds import entity_scores, link_table, metrics_at, subset_links  # noqa: E402


def synthetic(seed=0, n_entities=3000, classes=("A", "B")):
    """Two classes of Source 1 entities whose matcher is calibrated differently: class B's decoys
    score higher, so its optimal threshold is higher than class A's."""
    rng = np.random.default_rng(seed)
    s_rids = np.arange(n_entities, dtype=np.uint32)
    cls = np.array([classes[i % 2] for i in range(n_entities)], dtype=object)
    t_rid, s_rid, p, tp_, decoy = [], [], [], [], []
    t = 10_000
    truth_t, truth_s = [], []
    for s in s_rids:
        c = cls[s]
        n_true = rng.integers(0, 4)
        for _ in range(n_true):
            truth_t.append(t); truth_s.append(s)
            t_rid.append(t); s_rid.append(s); p.append(min(0.999, rng.beta(9, 1.2)))
            t += 1
        n_dec = rng.integers(1, 5)
        for _ in range(n_dec):
            hi = 0.35 if c == "A" else 0.75
            t_rid.append(t); s_rid.append(s); p.append(float(np.clip(rng.uniform(0.0, hi), 0, 0.999)))
            t += 1
    meta = pl.DataFrame({"t_rid": np.array(t_rid, dtype=np.uint32), "s_rid": np.array(s_rid, dtype=np.uint32)})
    truth = pl.DataFrame({"t_rid": np.array(truth_t, dtype=np.uint32), "s_rid": np.array(truth_s, dtype=np.uint32)})
    links, nt = link_table(meta, np.array(p, dtype=np.float32), truth, s_rids)
    keys = np.array([f"c={cls[s]}" for s in links["s_rid"].to_numpy()], dtype=object)
    entity_keys = np.array([f"c={c}" for c in cls], dtype=object)
    s_fold = np.arange(n_entities) % 3
    return meta, np.array(p, dtype=np.float32), truth, s_rids, links, nt, keys, entity_keys, s_fold


GRID = np.round(np.arange(0.05, 0.99, 0.01), 2)


def test_per_row_threshold_matches_scalar():
    *_, links, nt, keys, entity_keys, s_fold = synthetic()
    a = metrics_at(links, nt, 0.6)
    b = metrics_at(links, nt, np.full(links.height, 0.6))
    assert a["macro_f05"] == b["macro_f05"] and a["links"] == b["links"]
    f, tp_, npred, keep = entity_scores(links, nt, 0.6)
    assert len(f) == len(nt) and keep.sum() == a["links"]


def test_per_class_fit_beats_global_and_finds_different_thresholds():
    *_, links, nt, keys, entity_keys, s_fold = synthetic()
    pol, rep = tp.fit_policy(links, nt, keys, ["country"], GRID, min_support=10, entity_keys=entity_keys)
    # (class attribute name is irrelevant to the fit; keys are what matter)
    assert pol.scheme == "per_class" and set(pol.thresholds) == {"c=A", "c=B"}
    assert pol.thresholds["c=B"] > pol.thresholds["c=A"]
    assert rep["overall"]["macro_f05"] >= rep["overall_at_global"]["macro_f05"]
    for c in ("c=A", "c=B"):
        assert rep["per_class"][c]["n_entities"] == 1500
        assert rep["per_class"][c]["macro_f05"] >= rep["per_class_at_global"][c]["macro_f05"] - 1e-12


def test_min_support_keeps_global():
    *_, links, nt, keys, entity_keys, s_fold = synthetic()
    pol, _ = tp.fit_policy(links, nt, keys, ["country"], GRID, min_support=10_000)
    assert all(v == pol.default for v in pol.thresholds.values())
    assert pol.fit["classes_below_min_support"] == ["c=A", "c=B"]


def test_unseen_class_fallback_and_roundtrip(tmp_path):
    pol = tp.ThresholdPolicy(0.7, {"country=US": 0.6, "country=India": 0.8}, ["country"], unseen="max")
    thr = pol.thresholds_for(np.array(["country=US", "country=France", "country=India"], dtype=object))
    assert thr.tolist() == [0.6, 0.8, 0.8]
    pol.unseen = "default"
    assert pol.thresholds_for(np.array(["country=France"], dtype=object)).tolist() == [0.7]
    path = tmp_path / "p.json"
    pol.save(str(path))
    back = tp.ThresholdPolicy.load(str(path))
    assert back.to_dict() == pol.to_dict()
    assert json.load(open(path))["scheme"] == "per_class"
    g = tp.ThresholdPolicy(0.79)
    assert g.scheme == "global" and g.thresholds_for(np.array(["x", "y"], dtype=object)).tolist() == [0.79, 0.79]


def test_assign_accepts_per_row_thresholds():
    meta, p, truth, s_rids, links, nt, keys, entity_keys, s_fold = synthetic(n_entities=200)
    pol = tp.ThresholdPolicy(0.5, {"c=A": 0.3, "c=B": 0.9}, ["country"])
    pair_keys = np.array([f"c={'A' if s % 2 == 0 else 'B'}" for s in meta["s_rid"].to_numpy()], dtype=object)
    thr = pol.thresholds_for(pair_keys)
    out = assign(meta, p, thr)
    # every accepted link clears its own class threshold, and every rejected best candidate does not
    best = meta.with_columns(pl.Series("p", p), pl.Series("thr", thr)).sort("p", descending=True).unique("t_rid", keep="first")
    assert out.height == int((best["p"] >= best["thr"]).sum())
    assert out.height == tp.assign_with_policy(meta, p, pair_keys, pol).height
    scalar = assign(meta, p, 0.5)
    assert scalar.height == int((best["p"] >= 0.5).sum())


def test_nested_comparison_is_honest_and_consistent():
    *_, links, nt, keys, entity_keys, s_fold = synthetic()
    res = tp.nested_comparison(links, nt, keys, s_fold, {"cls": {"class_by": ["country"], "min_support": 10}}, GRID, entity_keys=entity_keys)
    assert set(res) == {"global", "cls"}
    assert len(res["cls"]["per_fold"]) == 3 and len(res["cls"]["fold_policies"]) == 3
    # thresholds of fold k were fit without fold k: the per-fold policies differ from an in-sample fit at most slightly
    assert res["cls"]["gain_vs_global"] > 0  # the synthetic classes really need different thresholds
    assert res["global"]["gain_vs_global"] == 0.0
    ent = np.array([metrics_at(*subset_links(links, nt, s_fold == k), 0.5)["n_entities"] for k in range(3)])
    assert ent.sum() == len(nt)


def test_class_keys_from_records():
    rec = pl.DataFrame({"rid": [0, 1, 2, 3], "country": ["US", "US", "India", None], "src": [1, 2, 1, 3],
                        "f_indic": [False, False, False, True], "a_norm": ["x", "", "y", None], "f_alias": [False] * 4, "n_domain": [""] * 4})
    pairs = pl.DataFrame({"t_rid": [1, 3], "s_rid": [0, 2]}).cast(pl.UInt32)
    k = tp.class_keys(pairs, rec, "country,src,indic,noaddr")
    assert k.tolist() == ["country=US|src=2|indic=0|noaddr=1", "country=India|src=3|indic=1|noaddr=1"]
    assert tp.class_keys(pairs, rec, []).tolist() == ["", ""]
    codes, names = tp.encode_keys(np.array(["b", "a", "b"], dtype=object))
    assert names == ["a", "b"] and codes.tolist() == [1, 0, 1]
    assert tp.encode_keys(np.array(["zz"], dtype=object), names)[0].tolist() == [-1]


def test_objectives():
    *_, links, nt, keys, entity_keys, s_fold = synthetic(n_entities=600)
    f05 = tp.make_objective("macro_f05")
    f1 = tp.make_objective("fbeta", beta=1.0)
    pf = tp.make_objective("precision_floor", floor=0.99)
    f, tp_, npred, keep = entity_scores(links, nt, 0.5)
    m = np.ones(links.height, dtype=bool)
    assert 0 <= f05(f, tp_, npred, nt, keep, links, m) <= 1
    assert 0 <= f1(f, tp_, npred, nt, keep, links, m) <= 1
    assert pf(f, tp_, npred, nt, keep, links, m) <= 1
    # a recall-leaning objective never picks a higher threshold than the precision-leaning one
    t_f1 = tp.fit_global(links, nt, GRID, objective=f1)
    t_f05 = tp.fit_global(links, nt, GRID)
    assert t_f1 <= t_f05
