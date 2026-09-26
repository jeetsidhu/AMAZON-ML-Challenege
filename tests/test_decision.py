"""Many-to-one assignment with confidence-based rejection, the vectorised link table and the
nested decision-rule selection."""
import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model import accept, assign, best_candidates, contradiction_flag  # noqa: E402
from thresholds import accept_mask, link_table, metrics_at, select_rule  # noqa: E402


def _meta():
    # record 10: candidates 1 (0.9) and 2 (0.8) -> close call; record 11: 1 (0.95), 3 (0.1);
    # record 12: single candidate 2 (0.6); record 13: 3 (0.85, contradicted) and 1 (0.2)
    meta = pl.DataFrame({"t_rid": np.array([10, 10, 11, 11, 12, 13, 13], dtype=np.uint32),
                         "s_rid": np.array([1, 2, 1, 3, 2, 3, 1], dtype=np.uint32)})
    p = np.array([0.9, 0.8, 0.95, 0.1, 0.6, 0.85, 0.2], dtype=np.float32)
    contra = np.array([False, False, False, False, False, True, False])
    return meta, p, contra


def test_best_candidates_runner_up_and_flags():
    meta, p, contra = _meta()
    b = best_candidates(meta, p, contra).sort("t_rid")
    assert b["t_rid"].to_list() == [10, 11, 12, 13]
    assert b["s_rid"].to_list() == [1, 1, 2, 3]
    assert np.allclose(b["p2nd"].to_list(), [0.8, 0.1, 0.0, 0.2])
    assert b["contra"].to_list() == [False, False, False, True]
    assert b["i"].to_list() == [0, 2, 4, 5]  # row index of the best pair in meta


def test_assign_is_many_to_one_not_one_to_one():
    meta, p, contra = _meta()
    links = assign(meta, p, 0.5)
    # Source 1 entity 1 receives records 10 and 11 (many-to-one); nobody receives two links per record
    assert links.filter(pl.col("s_rid") == 1).height == 2
    assert links["t_rid"].n_unique() == links.height
    assert set(links["t_rid"].to_list()) == {10, 11, 12, 13}


def test_margin_rejects_close_calls():
    meta, p, contra = _meta()
    links = assign(meta, p, 0.5, margin=0.2)
    assert 10 not in links["t_rid"].to_list()  # 0.9 - 0.8 < 0.2
    assert set(links["t_rid"].to_list()) == {11, 12, 13}


def test_contradiction_penalty_and_hard_veto():
    meta, p, contra = _meta()
    assert 13 in assign(meta, p, 0.5, contra_penalty=0.3, contra=contra)["t_rid"].to_list()   # 0.85 >= 0.8
    assert 13 not in assign(meta, p, 0.5, contra_penalty=0.4, contra=contra)["t_rid"].to_list()
    assert 13 not in assign(meta, p, 0.5, contra_penalty=1.0, contra=contra)["t_rid"].to_list()  # hard veto
    assert 13 in assign(meta, p, 0.5, contra_penalty=1.0, contra=None)["t_rid"].to_list()     # no flags: no veto


def test_per_row_thresholds_follow_meta_rows():
    meta, p, contra = _meta()
    thr = np.full(meta.height, 0.5)
    thr[4] = 0.7  # the pair (12, 2) gets a stricter threshold
    links = assign(meta, p, thr)
    assert 12 not in links["t_rid"].to_list()


def test_accept_mask_matches_model_accept():
    meta, p, contra = _meta()
    truth = pl.DataFrame({"t_rid": np.array([10, 11], dtype=np.uint32), "s_rid": np.array([1, 1], dtype=np.uint32)})
    links, nt = link_table(meta, p, truth, np.array([1, 2, 3], dtype=np.uint32), contra)
    b = best_candidates(meta, p, contra)
    for rule in ({}, {"margin": 0.2}, {"contra_penalty": 0.4}, {"margin": 0.2, "contra_penalty": 1.0}):
        m1 = accept_mask(links, 0.5, **rule)
        m2 = accept(b, 0.5, **rule)
        # both tables are per record; compare as {t_rid: accepted}
        d1 = dict(zip(links["t_rid"].to_list(), m1.tolist()))
        d2 = dict(zip(b["t_rid"].to_list(), m2.tolist()))
        assert d1 == d2
    assert nt.tolist() == [2, 0, 0]
    m = metrics_at(links, nt, 0.5)
    assert m["singleton_fp"] == 2 and m["fp_decoy"] == 2 and m["links"] == 4  # records 12, 13 are decoys linked to singletons 2, 3
    assert abs(m["macro_f05"] - 1 / 3) < 1e-9  # entity 1 perfect, singletons 2 and 3 wrong


def test_contradiction_flag_definition():
    df = pl.DataFrame({"unit_conflict": [1, -1, -1, -1, -1], "pc_conflict": [-1, 1, -1, -1, -1], "na_conflict": [0, 0, 1, 0, 0],
                       "hn_conflict": [0, 0, 0, 1, 1], "st_eq": [1, 1, 1, 1, 0]})
    assert contradiction_flag(df).tolist() == [True, True, True, True, False]


def test_select_rule_prefers_margin_when_it_helps():
    """Close calls (runner-up within 0.1) are wrong 70 % of the time, so a margin rule must win the
    nested comparison; with no such structure the plain threshold must be kept."""
    rng = np.random.default_rng(3)
    n_ent = 3000
    t_rid, s_rid, p, truth_t, truth_s = [], [], [], [], []
    t = 100_000
    for s in range(n_ent):
        # one true record per entity, scored high
        truth_t.append(t); truth_s.append(s)
        t_rid.append(t); s_rid.append(s); p.append(rng.uniform(0.85, 0.99))
        t += 1
        # a close-call record: best candidate = this entity, runner-up almost as high; true only 30 % of the time
        other = (s + 1) % n_ent
        hi = rng.uniform(0.7, 0.9)
        t_rid += [t, t]; s_rid += [s, other]; p += [hi, hi - 0.05]
        if rng.random() < 0.3:
            truth_t.append(t); truth_s.append(s)
        t += 1
    meta = pl.DataFrame({"t_rid": np.array(t_rid, dtype=np.uint32), "s_rid": np.array(s_rid, dtype=np.uint32)})
    truth = pl.DataFrame({"t_rid": np.array(truth_t, dtype=np.uint32), "s_rid": np.array(truth_s, dtype=np.uint32)})
    links, nt = link_table(meta, np.array(p, dtype=np.float32), truth, np.arange(n_ent, dtype=np.uint32))
    folds = np.arange(n_ent) % 3
    grid = np.round(np.arange(0.5, 0.99, 0.01), 2)
    rule, rows = select_rule(links, nt, folds, grid, margins=(0.0, 0.1, 0.2), penalties=(0.0,))
    assert rule["margin"] > 0 and rule["gain"] > 0
    base = next(r for r in rows if r["margin"] == 0.0)
    assert rule["nested_macro_f05"] > base["nested_macro_f05"]


def test_threshold_policy_round_trip(tmp_path):
    from threshold_policy import ThresholdPolicy, assign_with_policy
    pol = ThresholdPolicy(0.76, name="global_density", fit={"method": "density"}, margin=0.1, contra_penalty=0.2)
    path = tmp_path / "p.json"
    pol.save(str(path))
    back = ThresholdPolicy.load(str(path))
    assert back.default == 0.76 and back.rule == {"margin": 0.1, "contra_penalty": 0.2} and back.name == "global_density"
    assert "global threshold 0.760" in back.describe()
    meta, p, contra = _meta()
    a = assign_with_policy(meta, p, back, contra)
    b = assign(meta, p, 0.76, margin=0.1, contra_penalty=0.2, contra=contra)
    assert a.sort("t_rid").rows() == b.sort("t_rid").rows()
    # older policy files with per-class fields still load as their default threshold
    legacy = ThresholdPolicy.from_dict({"default": 0.7, "thresholds": {"country=US": 0.6}, "class_by": ["country"]})
    assert legacy.default == 0.7 and legacy.rule == {"margin": 0.0, "contra_penalty": 0.0}
