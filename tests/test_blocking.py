"""Multi-channel candidate generation: channel parsing, per-row top-k, the candidate audit and the
char n-gram features."""
import os
import sys

import numpy as np
import polars as pl
import pytest
import scipy.sparse as sp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import blocking  # noqa: E402
import textnorm  # noqa: E402


def test_parse_channels():
    assert blocking.parse_channels("combined=3") == {"combined": 3}
    assert blocking.parse_channels("combined=3,nchar=2, rev=1") == {"combined": 3, "nchar": 2, "rev": 1}
    assert blocking.parse_channels("combined=3,name=0") == {"combined": 3}  # k = 0 switches a channel off
    with pytest.raises(ValueError):
        blocking.parse_channels("name=2")  # combined is required
    with pytest.raises(ValueError):
        blocking.parse_channels("combined=3,bogus=2")


def test_topk_per_row_orders_and_offsets():
    c = sp.csr_matrix(np.array([[0.5, 0.9, 0.1, 0.0], [0.0, 0.0, 0.05, 0.3]]))
    rows, cols, score, rank = blocking.topk_per_row(c, k=2, min_score=0.1, offset=10)
    assert rows.tolist() == [10, 10, 11]
    assert cols.tolist() == [1, 0, 3]
    assert rank.tolist() == [0, 1, 0]
    assert np.allclose(score, [0.9, 0.5, 0.3])


def test_char_ngrams():
    assert textnorm.char_ngrams("dent", 4) == ["#den", "dent", "ent#"]
    assert textnorm.char_ngrams("ab", 4) == ["#ab#"]
    r = textnorm.normalize_record("Dent Diner", "1 Main St, Springfield, IL", "US")
    assert "g:#den" in r["feat_nchar"].split() and "g:ner#" in r["feat_nchar"].split()
    # a typo keeps most grams in common with the correct spelling
    a = set(textnorm.char_ngrams("college")); b = set(textnorm.char_ngrams("co1lege"))
    assert len(a & b) >= 2


def test_candidate_audit_recall_coverage_oracle():
    truth = pl.DataFrame({"t_rid": np.array([10, 11, 12, 13], dtype=np.uint32), "s_rid": np.array([1, 1, 2, 3], dtype=np.uint32)})
    # entity 1: both records found (one by the combined channel only, one by nchar only); entity 2: found;
    # entity 3: missed; entity 4: singleton
    cand = pl.DataFrame({"t_rid": np.array([10, 11, 12, 13], dtype=np.uint32), "s_rid": np.array([1, 1, 2, 9], dtype=np.uint32),
                         "n_ch": [1, 1, 2, 1], "r_combined": [0, -1, 0, 0], "r_nchar": [-1, 0, 1, -1]})
    audit = blocking.candidate_audit(cand, truth, np.array([1, 2, 3, 4], dtype=np.uint32), {"combined": 3, "nchar": 2},
                                     ambiguous=np.array([False, False, False, True]))
    assert audit["pair_recall"] == 0.75
    assert audit["pair_recall_by_channel"] == {"combined": 0.5, "nchar": 0.5}
    assert audit["true_pairs_found_by_one_channel_only"] == {"combined": 1, "nchar": 1}
    assert audit["entity_complete_coverage"] == pytest.approx(2 / 3)
    assert audit["entity_zero_coverage"] == pytest.approx(1 / 3)
    # oracle: entity 1 -> 1, entity 2 -> 1, entity 3 -> 0, singleton -> 1
    assert audit["oracle_macro_f05"] == pytest.approx(0.75)
    assert audit["misses"] == {"ambiguous_true_pairs": 1, "missed_ambiguous": 1, "missed_recoverable": 0, "recall_on_recoverable": 1.0}


def test_union_keeps_rank_per_channel(tmp_path):
    """End to end on a tiny in-memory country: every channel proposes its top-k, the union carries one
    row per pair with the rank in each channel and the exact combined cosine."""
    import argparse
    recs = [
        (1, 1, "Dent Diner", "12 Main St, Springfield, IL"),
        (2, 1, "Dent Diner", "900 Oak Ave, Peoria, IL"),
        (3, 1, "Acme Tools Inc", "5 Elm Rd, Peoria, IL"),
        (10, 2, "Dent Diner", "12 Main Street, Springfield, IL"),
        (11, 3, "Acme Too1s", None),
        (12, 2, "Zebra Unrelated", "77 Nowhere, Peoria, IL"),
    ]
    rows = []
    for rid, src, name, addr in recs:
        r = textnorm.normalize_record(name, addr, "US")
        rows.append({"rid": rid, "src": src, "country": "US", **{k: r[k] for k in blocking.BLOCKS}})
    rec = pl.DataFrame(rows).with_columns(pl.col("rid").cast(pl.UInt32), pl.col("src").cast(pl.Int8))
    args = argparse.Namespace(df_cap=1000, rare_df=5, chunk=4000)
    # serial retrieval (no process pool) keeps the test fast and deterministic
    def serial_pool(tmp, mats, channels, reverse, bounds, tag):
        blocking._init(tmp, mats, channels, reverse)
        out = []
        for b in bounds:
            out.extend(blocking._retrieve(b))
        return out
    blocking.run_pool, orig = serial_pool, blocking.run_pool
    try:
        out = blocking.run_country(rec, "US", args, str(tmp_path), blocking.parse_channels(blocking.DEFAULT_CHANNELS))
    finally:
        blocking.run_pool = orig
    assert out.select("t_rid", "s_rid").unique().height == out.height
    pair = out.filter((pl.col("t_rid") == 10) & (pl.col("s_rid") == 1)).row(0, named=True)
    assert pair["rank"] == 0 and pair["r_combined"] == 0 and pair["n_ch"] >= 3
    assert pair["cos_nchar"] > 0.9  # identical compact name
    # the typo'd name still reaches its entity through the char n-gram / rev channels
    assert out.filter((pl.col("t_rid") == 11) & (pl.col("s_rid") == 3)).height == 1
    assert "r_rev" in out.columns and (out["r_rev"] >= 0).any()
