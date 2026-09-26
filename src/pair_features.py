"""Step 3: candidate selection + pairwise features.

Candidate set (this is exactly what the matching model scores, and what is written
to candidate_pairs.tsv): every pair retrieved by blocking.py (--topk / --min-score there).
--max-rank / --min-score here only exist to *tighten* the set for experiments; by default
nothing is filtered a second time.

Features (all country-agnostic; the country label itself is never a feature):
  retrieval  : score, rank, per-block TF-IDF cosines, gaps to the best candidate of the
               same Source 2/3 record, candidate-list statistics on both sides
  name       : rapidfuzz ratio / token_sort / token_set / partial / Jaro-Winkler on the
               normalised full & core names, alias-part max, compact-name and domain-stem
               similarities, token-set overlaps, length features
  address    : rapidfuzz similarities on the normalised address, token and bigram
               overlaps, house/unit number agreement, missing-field flags
  record     : source of the Source 2/3 record, Indic-script / alias / domain flags
  numbers    : relation between the two house numbers (equal / dropped leading digits / truncated /
               one digit substituted / transposed / other), their numeric offset, cross-containment
  street     : similarity of the street name without its number
  legal      : legal-form tokens on each side, overlap and conflict (LLC vs Inc, SARL vs SAS)
  crowding   : how many Source 1 records share the Source 1 address / street; how many candidates of
               the Source 2/3 record share its exact address or its name
  extra toks : typo-tolerant count of name tokens present on one side only
  channels   : rank of the pair in every retrieval channel (r_<channel>, -1 = not proposed), number of
               channels that proposed it (n_ch), char-4-gram name cosine (cos_nchar) - from blocking.py
  contradict : strong negative evidence: different house numbers on both sides (hn_conflict), different
               unit / flat / suite numbers (unit_conflict), different postal-like codes (pc_conflict),
               no digit group in common although both sides carry numbers (num_conflict), identical
               names at contradicting addresses (na_conflict), and a Source 1 - IDF weighted view of the
               name tokens: the rarest token present on one side only (xt_maxidf / xs_maxidf, "rare
               token conflict") and the rarest token both share (common_maxidf, rare positive evidence)

Output: <work>/<split>/pairs/part_XXXX.parquet (pid, t_rid, s_rid, features..., [label for train])
"""
import multiprocessing as mp
import os
import re

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

from common import Stage, base_args, log, n_workers, read_tsv, split_dir

REC_COLS = ["rid", "entity_id", "src", "country", "n_full", "n_core", "n_parts", "n_compact", "n_domain", "n_legal",
            "f_indic", "f_alias", "a_norm", "a_comp", "a_num", "a_hn", "a_street", "a_key"]
_POOL = None
# unit / flat / suite / shop numbers inside a normalised address component (textnorm canonical forms)
UNIT_RE = re.compile(r"\b(?:unit|apt|ste|fl|flat|shop|room|rm|office|door|gala|lot|cabin)\s*(?:no\s*)?([a-z]?\d+[a-z]?)\b")
PC_RE = re.compile(r"\b\d{5,6}\b")  # postal-like code: 5-6 digit group (US ZIP, Indian PIN, French code postal)


def select_candidates(cand, max_rank=None, min_score=None):
    if max_rank is not None:
        cand = cand.filter(pl.col("rank") < max_rank)
    if min_score is not None:
        cand = cand.filter(pl.col("score") >= min_score)
    return cand


def context_features(c):
    """Features describing the candidate lists around each pair."""
    c = c.with_columns(
        pl.col("score").max().over("t_rid").alias("t_best"),
        pl.len().over("t_rid").alias("t_ncand"),
        pl.col("cos_name").max().over("t_rid").alias("t_best_name"),
        pl.col("cos_addr").max().over("t_rid").alias("t_best_addr"),
        pl.col("cos_cross").max().over("t_rid").alias("t_best_cross"),
        pl.col("score").sort(descending=True).over("t_rid", mapping_strategy="join").list.get(1, null_on_oob=True).alias("t_second"),
        pl.len().over("s_rid").alias("s_ncand"),
        pl.col("score").rank("ordinal", descending=True).over("s_rid").alias("s_rank"),
        pl.col("score").max().over("s_rid").alias("s_best"),
        (pl.col("rank") == 0).sum().over("s_rid").alias("s_ntop1"),
    )
    return c.with_columns(
        (pl.col("t_best") - pl.col("score")).alias("gap_best"),
        (pl.col("score") - pl.col("t_second").fill_null(0)).alias("gap_second"),
        (pl.col("t_best_name") - pl.col("cos_name")).alias("gap_name"),
        (pl.col("t_best_addr") - pl.col("cos_addr")).alias("gap_addr"),
        (pl.col("t_best_cross") - pl.col("cos_cross")).alias("gap_cross"),
        (pl.col("s_best") - pl.col("score")).alias("s_gap_best"),
    )


def _sim(scorer, a, b):
    """Pairwise (row-wise) similarity of two equally long string lists as float32."""
    if hasattr(process, "cpdist"):  # rapidfuzz >= 3.9: parallel C++ implementation
        return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32)
    return np.fromiter((scorer(x, y) for x, y in zip(a, b)), dtype=np.float32, count=len(a))


def string_features(df):
    """df has columns t_* and s_* with the record fields of both sides."""
    out = {}
    g = lambda c: df[c].fill_null("").to_list()  # noqa: E731
    tf, sf = g("t_n_full"), g("s_n_full")
    tc, sc = g("t_n_core"), g("s_n_core")
    out["nm_ratio"] = _sim(fuzz.ratio, tf, sf)
    out["nm_core_ratio"] = _sim(fuzz.ratio, tc, sc)
    out["nm_tsort"] = _sim(fuzz.token_sort_ratio, tc, sc)
    out["nm_tset"] = _sim(fuzz.token_set_ratio, tc, sc)
    out["nm_partial"] = _sim(fuzz.partial_ratio, tc, sc)
    out["nm_jw"] = _sim(JaroWinkler.normalized_similarity, tc, sc)
    # alias parts: best / last part of the Source 2/3 name against the Source 1 core name
    parts = df.select(
        pl.col("t_n_parts").fill_null("").str.split("|").alias("p")
    ).with_columns(pl.col("p").list.first().alias("p0"), pl.col("p").list.last().alias("p1"))
    p0 = parts["p0"].fill_null("").to_list()
    p1 = parts["p1"].fill_null("").to_list()
    a0 = _sim(fuzz.token_set_ratio, p0, sc)
    a1 = _sim(fuzz.token_set_ratio, p1, sc)
    out["nm_part_max"] = np.maximum(a0, a1)
    out["nm_part_min"] = np.minimum(a0, a1)
    # compact names (handles concatenated / domain style names)
    comp = df.select(
        pl.col("t_n_compact").fill_null("").str.split("|").list.first().fill_null("").alias("tc"),
        pl.col("s_n_compact").fill_null("").str.split("|").list.first().fill_null("").alias("sc"),
        pl.col("s_n_compact").fill_null("").str.split("|").list.last().fill_null("").alias("sf"),
        pl.col("t_n_domain").fill_null("").str.split("|").list.first().fill_null("").alias("td"),
    )
    tcc, scc, sfc, td = (comp[c].to_list() for c in ("tc", "sc", "sf", "td"))
    out["cmp_ratio"] = _sim(fuzz.ratio, tcc, scc)
    out["cmp_jw"] = _sim(JaroWinkler.normalized_similarity, tcc, scc)
    has_dom = np.array([len(x) > 0 for x in td])
    dpre = np.maximum(
        _sim(fuzz.ratio, td, [s[:len(d)] for s, d in zip(scc, td)]),
        _sim(fuzz.ratio, td, [s[:len(d)] for s, d in zip(sfc, td)]),
    )
    dfull = np.maximum(_sim(fuzz.ratio, td, scc), _sim(fuzz.ratio, td, sfc))
    out["dom_prefix"] = np.where(has_dom, dpre, -1).astype(np.float32)
    out["dom_full"] = np.where(has_dom, dfull, -1).astype(np.float32)
    # address
    ta, sa = g("t_a_norm"), g("s_a_norm")
    out["ad_ratio"] = _sim(fuzz.ratio, ta, sa)
    out["ad_tsort"] = _sim(fuzz.token_sort_ratio, ta, sa)
    out["ad_tset"] = _sim(fuzz.token_set_ratio, ta, sa)
    out["ad_partial"] = _sim(fuzz.partial_ratio, ta, sa)
    # first address component containing a digit (street line) vs the same on the other side
    street = df.select(
        pl.col("t_a_comp").fill_null("").str.split(",").list.eval(pl.element().filter(pl.element().str.contains(r"\d"))).list.first().fill_null("").alias("ts"),
        pl.col("s_a_comp").fill_null("").str.split(",").list.eval(pl.element().filter(pl.element().str.contains(r"\d"))).list.first().fill_null("").alias("ss"),
    )
    out["street_ratio"] = _sim(fuzz.ratio, street["ts"].to_list(), street["ss"].to_list())
    out["street_tset"] = _sim(fuzz.token_set_ratio, street["ts"].to_list(), street["ss"].to_list())
    return pl.DataFrame(out)


# --------------------------------------------------------------- house numbers / tokens
def _hn_relation(a, b):
    """Relation between two house numbers (leading zeros already stripped).

    -1 missing, 0 equal, 1 one is a suffix of the other (dropped leading digits),
    2 prefix (truncated), 3 same length with one substituted digit, 4 transposition, 5 other.
    True matches mostly corrupt a number by dropping/truncating digits, while decoy records
    of a *different* business sit a few doors away (small numeric offset).
    """
    if not a or not b:
        return -1
    if a == b:
        return 0
    if a.endswith(b) or b.endswith(a):
        return 1
    if a.startswith(b) or b.startswith(a):
        return 2
    if len(a) == len(b):
        if sum(x != y for x, y in zip(a, b)) == 1:
            return 3
        if sorted(a) == sorted(b):
            return 4
    return 5


def _tok_unmatched(xs, ys):
    """Number of tokens of xs with no fuzzy counterpart in ys (typo-tolerant set difference)."""
    n = 0
    for x in xs:
        if x in ys:
            continue
        if not any(fuzz.ratio(x, y) >= 80 or (len(x) >= 3 and len(y) >= 3 and (x.startswith(y) or y.startswith(x)))
                   for y in ys):
            n += 1
    return n


def _conflict(xs, ys):
    """1 when both sides carry values and share none, 0 when both carry values and share one, -1 otherwise."""
    if not xs or not ys:
        return -1.0
    return 0.0 if set(xs) & set(ys) else 1.0


def _py_features(args):
    t_hn, s_hn, t_num, s_num, t_core, s_core, t_comp, s_comp = args
    out = np.empty((len(t_hn), 12), dtype=np.float32)
    for i in range(len(t_hn)):
        a, b = t_hn[i] or "", s_hn[i] or ""
        rel = _hn_relation(a, b)
        # contradictions: house numbers that are neither equal nor a digit-corruption of each other
        out[i, 8] = -1 if rel < 0 else (1.0 if rel == 5 else 0.0)
        out[i, 9] = _conflict(UNIT_RE.findall(t_comp[i] or ""), UNIT_RE.findall(s_comp[i] or ""))
        pt = [x for x in PC_RE.findall(t_num[i] or "") if x != a]
        ps = [x for x in PC_RE.findall(s_num[i] or "") if x != b]
        out[i, 10] = _conflict(pt, ps)
        if rel >= 0:
            x, y = int(a[:9]), int(b[:9])
            d = abs(x - y)
            out[i, 1] = np.log1p(d)
            out[i, 2] = d / max(x, y, 1)
        else:
            out[i, 1] = out[i, 2] = -1
        tn = (t_num[i] or "").split()
        sn = (s_num[i] or "").split()
        out[i, 11] = _conflict(tn, sn)
        out[i, 0] = rel
        out[i, 3] = (b in tn) if b and tn else -1
        out[i, 4] = (a in sn) if a and sn else -1
        out[i, 5] = len(a) - len(b) if a and b else -99
        tc = (t_core[i] or "").split()
        sc = (s_core[i] or "").split()
        out[i, 6] = _tok_unmatched(tc, sc)
        out[i, 7] = _tok_unmatched(sc, tc)
    return out


PY_FEATS = ["hn_rel", "hn_logdiff", "hn_reldiff", "hn_s_in_t", "hn_t_in_s", "hn_lendiff", "nm_xt", "nm_xs",
            "hn_conflict", "unit_conflict", "pc_conflict", "num_conflict"]


def python_features(df, workers=None):
    cols = [df[c].to_list() for c in ("t_a_hn", "s_a_hn", "t_a_num", "s_a_num", "t_n_core", "s_n_core", "t_a_comp", "s_a_comp")]
    n = df.height
    step = 50_000
    jobs = [tuple(c[a:a + step] for c in cols) for a in range(0, n, step)]
    res = _POOL.map(_py_features, jobs) if _POOL is not None else [_py_features(j) for j in jobs]
    X = np.vstack(res) if res else np.empty((0, len(PY_FEATS)), dtype=np.float32)
    out = pl.DataFrame({name: X[:, k] for k, name in enumerate(PY_FEATS)})
    g = lambda c: df[c].fill_null("").to_list()  # noqa: E731
    ts, ss = g("t_a_street"), g("s_a_street")
    out = out.with_columns(
        pl.Series("st_ratio", _sim(fuzz.ratio, ts, ss)),
        pl.Series("st_eq", [(1.0 if a and a == b else (0.0 if a and b else -1.0)) for a, b in zip(ts, ss)], dtype=pl.Float32),
    )
    lg = df.select(
        pl.col("t_n_legal").fill_null("").str.split(" ").list.eval(pl.element().filter(pl.element() != "")).alias("tl"),
        pl.col("s_n_legal").fill_null("").str.split(" ").list.eval(pl.element().filter(pl.element() != "")).alias("sl"),
    ).select(
        pl.col("tl").list.len().alias("lg_t"),
        pl.col("sl").list.len().alias("lg_s"),
        pl.col("tl").list.set_intersection("sl").list.len().alias("lg_common"),
    ).with_columns(
        ((pl.col("lg_t") > 0) & (pl.col("lg_s") > 0) & (pl.col("lg_common") == 0)).cast(pl.Int8).alias("lg_conflict"),
    )
    return pl.concat([out, lg], how="horizontal")


def name_idf(rec):
    """Source 1 document frequency of every core-name token -> (token, idf) table. Computed from the
    records of the split being processed (no labels): a token shared by two records is strong evidence
    when few Source 1 records carry it, and a token present on one side only is a strong contradiction
    when it is rare (a typo of a common token is rare too, which the typo-tolerant nm_xt/nm_xs cover)."""
    s1 = rec.filter(pl.col("src") == 1)
    n = max(s1.height, 1)
    df = (s1.lazy().select(pl.col("n_core").fill_null("").str.split(" ").alias("t")).explode("t")
          .filter(pl.col("t") != "").group_by("t").agg(pl.len().alias("df")).collect())
    return df.select("t", np.log1p(n / pl.col("df")).cast(pl.Float32).alias("idf")), float(np.log1p(n))


def idf_features(df, idf, idf_max):
    """xt_maxidf / xs_maxidf: rarest name token present only on the Source 2/3 / Source 1 side;
    common_maxidf: rarest shared token; *_sumidf: idf mass of the unmatched tokens. Unknown tokens
    (absent from Source 1) get the maximum idf; -1 when there is no such token."""
    tok = lambda c: pl.col(c).fill_null("").str.split(" ").list.eval(pl.element().filter(pl.element() != ""))  # noqa: E731
    x = df.select(tok("t_n_core").alias("tn"), tok("s_n_core").alias("sn")).with_row_index("i").with_columns(
        pl.col("tn").list.set_difference("sn").alias("xt"), pl.col("sn").list.set_difference("tn").alias("xs"),
        pl.col("tn").list.set_intersection("sn").alias("cm"))
    out = pl.DataFrame({"i": x["i"]})
    for col, name in (("xt", "xt"), ("xs", "xs"), ("cm", "common")):
        e = x.select("i", pl.col(col).alias("t")).explode("t").filter(pl.col("t").is_not_null()).join(idf, on="t", how="left").with_columns(
            pl.col("idf").fill_null(idf_max))
        agg = e.group_by("i").agg(pl.col("idf").max().alias(f"{name}_maxidf"), pl.col("idf").sum().alias(f"{name}_sumidf"))
        out = out.join(agg, on="i", how="left")
    return out.sort("i").drop("i").fill_null(-1.0)


def set_features(df):
    tok = lambda c: pl.col(c).fill_null("").str.split(" ").list.eval(pl.element().filter(pl.element() != ""))  # noqa: E731
    x = df.select(
        tok("t_n_core").alias("tn"), tok("s_n_core").alias("sn"),
        tok("t_a_norm").alias("ta"), tok("s_a_norm").alias("sa"),
        tok("t_a_num").alias("tu"), tok("s_a_num").alias("su"),
    )
    x = x.with_columns(
        pl.col("tn").list.len().alias("t_ntok"), pl.col("sn").list.len().alias("s_ntok"),
        pl.col("ta").list.len().alias("t_natok"), pl.col("sa").list.len().alias("s_natok"),
        pl.col("tu").list.len().alias("t_nnum"), pl.col("su").list.len().alias("s_nnum"),
        pl.col("tn").list.set_intersection("sn").list.len().alias("nm_common"),
        pl.col("ta").list.set_intersection("sa").list.len().alias("ad_common"),
        pl.col("tu").list.set_intersection("su").list.len().alias("num_common"),
        (pl.col("tu").list.first() == pl.col("su").list.first()).cast(pl.Int8).fill_null(-1).alias("num_first_eq"),
        pl.col("tn").list.first().alias("tn0"), pl.col("sn").list.first().alias("sn0"),
    )
    x = x.with_columns(
        (pl.col("nm_common") / (pl.col("t_ntok") + pl.col("s_ntok") - pl.col("nm_common")).clip(lower_bound=1)).alias("nm_jacc"),
        (pl.col("ad_common") / (pl.col("t_natok") + pl.col("s_natok") - pl.col("ad_common")).clip(lower_bound=1)).alias("ad_jacc"),
        (pl.col("num_common") / pl.max_horizontal("t_nnum", "s_nnum").clip(lower_bound=1)).alias("num_frac"),
        (pl.col("nm_common") / pl.col("s_ntok").clip(lower_bound=1)).alias("nm_cov_s"),
        (pl.col("nm_common") / pl.col("t_ntok").clip(lower_bound=1)).alias("nm_cov_t"),
    )
    first = _sim(fuzz.ratio, x["tn0"].fill_null("").to_list(), x["sn0"].fill_null("").to_list())
    return x.drop("tn", "sn", "ta", "sa", "tu", "su", "tn0", "sn0").with_columns(pl.Series("nm_first_ratio", first))


def record_features(df):
    return df.select(
        pl.col("t_src").cast(pl.Int8).alias("t_src"),
        pl.col("t_f_indic").cast(pl.Int8).alias("t_indic"),
        pl.col("t_f_alias").cast(pl.Int8).alias("t_alias"),
        (pl.col("t_n_domain").fill_null("") != "").cast(pl.Int8).alias("t_domain"),
        (pl.col("t_a_norm").fill_null("") == "").cast(pl.Int8).alias("t_noaddr"),
        (pl.col("s_a_norm").fill_null("") == "").cast(pl.Int8).alias("s_noaddr"),
        (pl.col("t_n_core").fill_null("") == "").cast(pl.Int8).alias("t_noname"),
        pl.col("t_n_full").fill_null("").str.len_chars().alias("t_nlen"),
        pl.col("s_n_full").fill_null("").str.len_chars().alias("s_nlen"),
        pl.col("s_a_mult").alias("s_addr_mult"),
        pl.col("s_a_smult").alias("s_street_mult"),
    )


def address_crowding(rec):
    """Adds a_mult / a_smult: how many Source 1 records of the same country share this record's
    exact street address (house number + street + locality) / its street (street + locality).
    Where many businesses share an address (common in some cities), an exact address match is
    weaker evidence of identity. Computed for Source 1 records (0 elsewhere)."""
    loc = pl.col("a_key").fill_null("").str.split(" ").list.set_difference(
        pl.col("a_street").fill_null("").str.split(" ")).list.sort().list.join(" ")
    k = rec.select(
        "rid", "src",
        pl.concat_str([pl.col("country").fill_null(""), pl.col("a_hn").fill_null(""), pl.col("a_street").fill_null(""), loc],
                      separator="|").alias("ka"),
        pl.concat_str([pl.col("country").fill_null(""), pl.col("a_street").fill_null(""), loc], separator="|").alias("ks"),
        ((pl.col("a_hn").fill_null("") != "") & (pl.col("a_street").fill_null("") != "")).alias("ok"),
    )
    s1 = pl.col("src") == 1
    k = k.with_columns(
        pl.when(s1 & pl.col("ok")).then(pl.len().over("ka", "src")).otherwise(0).cast(pl.Float32).alias("a_mult"),
        pl.when(s1 & pl.col("ok")).then(pl.len().over("ks", "src")).otherwise(0).cast(pl.Float32).alias("a_smult"),
    )
    return rec.with_columns(k["a_mult"], k["a_smult"])


def candidate_group_features(f):
    """Per Source 2/3 record, over its candidate list: how many candidates share its exact
    address, how many carry the same name (namesakes), and how many carry the same name at a
    contradicting address (t_n_namesake_contra). Plus the pair-level name/address contradiction:
    (near-)identical names whose house numbers disagree outright and whose streets differ."""
    exact = ((pl.col("hn_rel") == 0) & (pl.col("st_eq") == 1)).cast(pl.Float32)
    same = ((pl.col("nm_xt") == 0) & (pl.col("nm_xs") == 0)).cast(pl.Float32)
    contra = ((pl.col("nm_tset") >= 90) & (pl.col("hn_conflict") == 1) & (pl.col("st_eq") == 0)).cast(pl.Float32)
    return f.with_columns(
        exact.sum().over("t_rid").alias("t_n_addr_exact"),
        same.sum().over("t_rid").alias("t_n_namesake"),
        contra.alias("na_conflict"),
        (same * contra).sum().over("t_rid").alias("t_n_namesake_contra"),
    )


def chunk_bounds(t_rid, chunk):
    """Chunk boundaries that never split the candidate list of a Source 2/3 record."""
    n = len(t_rid)
    bounds, a = [], 0
    while a < n:
        b = min(a + chunk, n)
        while b < n and t_rid[b] == t_rid[b - 1]:
            b += 1
        bounds.append((a, b))
        a = b
    return bounds


def gather(rec, idx, prefix):
    return rec[idx].select(pl.all().name.prefix(prefix))


def build(cand, rec, chunk, out_dir, truth=None):
    """Computes features chunk by chunk and writes <out_dir>/part_XXXX.parquet (float32 features)."""
    os.makedirs(out_dir, exist_ok=True)
    for f in os.listdir(out_dir):
        os.remove(os.path.join(out_dir, f))
    cand = cand.with_row_index("pid")
    idf, idf_max = name_idf(rec)
    for i, (a, b) in enumerate(chunk_bounds(cand["t_rid"].to_numpy(), chunk)):
        c = cand.slice(a, b - a)
        t = gather(rec, c["t_rid"].to_numpy().astype(np.int64), "t_")
        s = gather(rec, c["s_rid"].to_numpy().astype(np.int64), "s_")
        df = pl.concat([t, s], how="horizontal")
        f = pl.concat([c, string_features(df), set_features(df), record_features(df), python_features(df), idf_features(df, idf, idf_max)],
                      how="horizontal")
        f = candidate_group_features(f)
        keep = {"pid", "t_rid", "s_rid"}
        f = f.with_columns(pl.col(x).cast(pl.Float32) for x in f.columns if x not in keep)
        if truth is not None:
            f = f.join(truth, on=["t_rid", "s_rid"], how="left").with_columns(pl.col("label").fill_null(0))
        f.write_parquet(os.path.join(out_dir, f"part_{i:04d}.parquet"))
        log(f"features {b}/{cand.height}")


def truth_pairs(rec, data_dir):
    gt = read_tsv(os.path.join(data_dir, "train", "train_ground_truth.tsv"))
    ids = rec.select("rid", "entity_id")
    return (
        gt.with_columns(pl.col("matched_entity_ids").str.split(","))
        .explode("matched_entity_ids")
        .drop_nulls("matched_entity_ids")
        .join(ids.rename({"rid": "s_rid", "entity_id": "source1_entity_id"}), on="source1_entity_id")
        .join(ids.rename({"rid": "t_rid", "entity_id": "matched_entity_ids"}), on="matched_entity_ids")
        .select(pl.col("t_rid").cast(pl.UInt32), pl.col("s_rid").cast(pl.UInt32), pl.lit(1, pl.Int8).alias("label"))
    )


def main():
    ap = base_args(__doc__)
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--max-rank", type=int, default=None, help="experiments only: keep rank < this")
    ap.add_argument("--min-score", type=float, default=None, help="experiments only: keep score >= this")
    ap.add_argument("--chunk", type=int, default=2_000_000)
    args = ap.parse_args()
    with Stage(args.work_dir, "features_" + args.split):
        run(args)


def run(args):
    d = split_dir(args.work_dir, args.split)
    rec = address_crowding(pl.read_parquet(os.path.join(d, "records.parquet"), columns=REC_COLS))
    cand = select_candidates(pl.read_parquet(os.path.join(d, "candidates_raw.parquet")), args.max_rank, args.min_score)
    cand = context_features(cand).sort("t_rid", "rank")
    log(f"{args.split}: {cand.height} candidate pairs")
    truth = truth_pairs(rec, args.data_dir) if args.split == "train" else None
    global _POOL
    with mp.get_context("spawn").Pool(n_workers()) as _POOL:
        build(cand, rec, args.chunk, os.path.join(d, "pairs"), truth)
    log("done")


if __name__ == "__main__":
    main()
