"""Step 2: multi-channel candidate generation (blocking).

For every country label (open set, whatever values appear in the data) TF-IDF vectors are built
over the blocking features of textnorm.blocking_features, one *block* per feature family:

  name  : n:<token>, p:<token pair> (order-free), k:<8-char compact-name prefix>
  nchar : g:<char 4-gram of the compact name>          (typo / transliteration tolerant)
  addr  : a:<token>, b:<adjacent bigram inside an address component>, h:<house number>_<key token>
  cross : c:<name token>_<key address token>, i.e. name x place

IDF is computed on Source 1 (the index side); every block is L2-normalised, so a block's
retrieval score is the cosine between the two records restricted to "index" features whose
Source 1 document frequency is <= --df-cap (very common features cost a lot and carry little
evidence, but they still count in the vector norms).

Retrieval channels (each one keeps its own top-k per Source 2/3 record; the candidate set is
the UNION of all channels):

  combined : cos_name + cos_addr + cos_cross               (the single channel of the previous version)
  name     : cos_name alone                                 (address-less / address-conflicting records)
  nchar    : cos over character 4-grams of the compact name (typos, concatenations, transliterations)
  addr     : cos_addr alone                                 (name in another script or heavily corrupted)
  cross    : cos_cross alone
  rare     : sum of idf over shared *rare* name features (Source 1 df <= --rare-df): a shared rare
             token proposes the pair whatever the rest of the name looks like
  hn       : shared h:<house number>_<key token> keys (numeric / address blocking)
  rev      : bidirectional retrieval: every Source 1 record retrieves its top-k Source 2/3 records
             on the combined score (a record crowded out of a namesake's top-k is still proposed
             by the entity that wants it)

--channels chooses the channels and their k, e.g. "combined=3,name=2,nchar=2,rare=2,hn=2,rev=2";
"combined=3" reproduces the previous single-channel candidate set exactly (up to the extra columns).

For every retrieved pair the exact per-block cosines (all shared features, no df cap) are computed;
`score` is the exact combined cosine and `rank` the pair's rank by that score within its Source 2/3
record's candidate list. The per-channel ranks (r_<channel>, -1 = not proposed by that channel) and
the number of channels that proposed the pair (n_ch, "retrieval agreement") are kept as features.

Candidate audit (training split, labels used for measurement only): <work>/train/blocking_recall.json
holds the true-pair recall of the union and of every channel, the share of Source 1 entities whose
complete record set was retrieved, the candidate *oracle* macro F0.5 (a perfect classifier on this
candidate set: the ceiling any matcher can reach) and the mean candidates per record.

Output: <work>/<split>/candidates_raw.parquet
  t_rid, s_rid, score, rank, cos_name, cos_addr, cos_cross, cos_nchar, n_ch, r_<channel>...
"""
import json
import multiprocessing as mp
import os
import shutil

import numpy as np
import polars as pl
import scipy.sparse as sp

from common import Stage, base_args, log, n_workers, split_dir

_G = {}


# ------------------------------------------------------------------ features
BLOCKS = ("feat_name", "feat_addr", "feat_cross", "feat_nchar")
BLOCK_NAMES = ("name", "addr", "cross", "nchar")
COMBINED = ("name", "addr", "cross")  # blocks whose cosines add up to the combined score
CHANNELS = ("combined", "name", "nchar", "addr", "cross", "rare", "hn", "rev")
DEFAULT_CHANNELS = "combined=3,name=2,nchar=2,addr=1,cross=1,rare=2,hn=2,rev=2"
# minimum retrieval score per channel (cosines for the tf-idf channels; any shared feature for rare / hn)
MIN_SCORE = {"combined": 0.1, "name": 0.2, "nchar": 0.3, "addr": 0.3, "cross": 0.2, "rare": 1e-6, "hn": 0.5, "rev": 0.1}


def parse_channels(spec):
    """'combined=3,name=2' -> {'combined': 3, 'name': 2} (validated, k > 0 only)."""
    out = {}
    for item in (spec or "").replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        name, _, k = item.partition("=")
        name = name.strip()
        if name not in CHANNELS:
            raise ValueError(f"unknown channel {name!r}; known: {CHANNELS}")
        k = int(k) if k else 3
        if k > 0:
            out[name] = k
    if "combined" not in out:
        raise ValueError("the 'combined' channel is required")
    return out


def explode(df, col, prefix=None):
    """(lrow, h) pairs of one feature block (optionally only the features starting with `prefix`)."""
    q = (
        df.lazy()
        .select(pl.col("lrow"), pl.col(col).str.split(" ").alias("f"))
        .explode("f")
        .filter(pl.col("f").is_not_null() & (pl.col("f") != ""))
    )
    if prefix:
        q = q.filter(pl.col("f").str.starts_with(prefix))
    return q.select("lrow", pl.col("f").hash(seed=17).alias("h")).collect()


def block_stats(trS, trT, nS):
    """IDF (from Source 1) of every feature of one block, and the features shared by both sides."""
    dfS = trS.group_by("h").agg(pl.len().cast(pl.Int64).alias("dfS"))
    dfT = trT.group_by("h").agg(pl.len().cast(pl.Int64).alias("dfT"))
    idf = dfS.with_columns((np.log1p(nS / pl.col("dfS"))).cast(pl.Float32).alias("idf")).select("h", "idf", "dfS")
    # features absent from S1 get the max idf (rare) for the norm computation
    idf_all = pl.concat([
        idf.select("h", "idf"),
        dfT.join(dfS, on="h", how="anti").select("h", pl.lit(float(np.log1p(nS)), pl.Float32).alias("idf")),
    ])
    shared = idf.join(dfT, on="h", how="inner")
    return idf_all, shared


def block_norms(trip, idf, n_rows):
    """Per-row L2 norm of the idf vector of one block."""
    w = trip.join(idf, on="h", how="left").with_columns(pl.col("idf").fill_null(pl.col("idf").max()))
    agg = w.group_by("lrow").agg((pl.col("idf") ** 2).sum().sqrt().alias("nrm"))
    norms = np.ones(n_rows, dtype=np.float32)
    norms[agg["lrow"].to_numpy()] = agg["nrm"].to_numpy()
    return norms


def build_matrix(trip, feat, norms, n_rows, n_cols, weight="idf"):
    """CSR matrix over the features in `feat` (h -> col, idf): idf/norm weights, or plain idf /
    binary weights when `weight` is 'idf_raw' / 'binary' (rare / hn channels: no normalisation)."""
    m = trip.join(feat, on="h", how="inner").sort("lrow")
    rows = m["lrow"].to_numpy()
    cols = m["col"].to_numpy().astype(np.int32)
    if weight == "idf":
        vals = (m["idf"].to_numpy() / norms[rows]).astype(np.float32)
    elif weight == "idf_raw":
        vals = m["idf"].to_numpy().astype(np.float32)
    else:
        vals = np.ones(len(rows), dtype=np.float32)
    del m
    indptr = np.zeros(n_rows + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(np.bincount(rows, minlength=n_rows))
    return sp.csr_matrix((vals, cols, indptr), shape=(n_rows, n_cols))


# --------------------------------------------------------------- retrieval
def _save_csr(mat, prefix):
    np.save(prefix + "_data.npy", mat.data.astype(np.float32))
    np.save(prefix + "_indices.npy", mat.indices.astype(np.int32))
    np.save(prefix + "_indptr.npy", mat.indptr.astype(np.int64))


def _load_csr(prefix, shape):
    d = np.load(prefix + "_data.npy", mmap_mode="r")
    i = np.load(prefix + "_indices.npy", mmap_mode="r")
    p = np.load(prefix + "_indptr.npy", mmap_mode="r")
    return sp.csr_matrix((d, i, p), shape=shape, copy=False)


def _init(tmp, mats, channels, reverse):
    """mats: {name: (T shape, ST shape)} per matrix family; loads the memory-mapped matrices."""
    _G.clear()
    for name, (t_shape, st_shape) in mats.items():
        _G["T_" + name] = _load_csr(os.path.join(tmp, "T_" + name), t_shape)
        _G["ST_" + name] = _load_csr(os.path.join(tmp, "ST_" + name), st_shape)
    _G["channels"] = channels
    _G["reverse"] = reverse


def topk_per_row(c, k, min_score, offset):
    """Top-k entries per row of a sparse product c (rows offset by `offset`)."""
    counts = np.diff(c.indptr)
    rows = np.repeat(np.arange(c.shape[0], dtype=np.int64), counts)
    data, cols = c.data, c.indices
    keep = data >= min_score
    rows, data, cols = rows[keep], data[keep], cols[keep]
    order = np.lexsort((-data, rows))
    rows, data, cols = rows[order], data[order], cols[order]
    counts = np.bincount(rows, minlength=c.shape[0])
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    rank = np.arange(len(rows)) - np.repeat(starts, counts)
    sel = rank < k
    return rows[sel] + offset, cols[sel].astype(np.int32), data[sel].astype(np.float32), rank[sel].astype(np.int16)


def _retrieve(bounds):
    """One chunk of Source 2/3 rows (or of Source 1 rows for the reverse channel): the per-channel
    top-k lists. Returns a list of (channel, rows, cols, score, rank) with rows/cols in (T, S) order."""
    a, b = bounds
    ch = _G["channels"]
    out = []
    if _G["reverse"]:
        # S1 rows against all Source 2/3 rows on the combined score
        c = None
        for name in COMBINED:
            if "T_" + name not in _G:
                continue
            prod = _G["ST_" + name][a:b] @ _G["T_" + name]  # here ST_<name> holds S (rows) and T_<name> holds T^T
            c = prod if c is None else c + prod
        rows, cols, data, rank = topk_per_row(c, ch["rev"], MIN_SCORE["rev"], a)
        out.append(("rev", cols.astype(np.int64), rows.astype(np.int32), data, rank))  # swap to (T, S)
        return out
    prods = {}
    for name in BLOCK_NAMES:
        if "T_" + name in _G:
            prods[name] = _G["T_" + name][a:b] @ _G["ST_" + name]
    comb = None
    for name in COMBINED:
        if name in prods:
            comb = prods[name] if comb is None else comb + prods[name]
    prods["combined"] = comb
    for extra in ("rare", "hn"):
        if "T_" + extra in _G:
            prods[extra] = _G["T_" + extra][a:b] @ _G["ST_" + extra]
    for name, k in ch.items():
        if name == "rev" or name not in prods or prods[name] is None:
            continue
        out.append((name, *topk_per_row(prods[name], k, MIN_SCORE[name], a)))
    return out


def rowwise_dot(A, B, ia, ib, chunk=1_000_000):
    out = np.empty(len(ia), dtype=np.float32)
    for s in range(0, len(ia), chunk):
        x = A[ia[s:s + chunk]].multiply(B[ib[s:s + chunk]])
        out[s:s + chunk] = np.asarray(x.sum(axis=1)).ravel()
    return out


def run_pool(tmp, mats, channels, reverse, bounds, tag):
    res = []
    with mp.get_context("spawn").Pool(n_workers(), initializer=_init, initargs=(tmp, mats, channels, reverse)) as pool:
        for k, r in enumerate(pool.imap_unordered(_retrieve, bounds)):
            res.extend(r)
            if k % 100 == 0:
                log(f"{tag} retrieval {k + 1}/{len(bounds)}")
    return res


def run_country(rec, country, args, tmp_root, channels):
    """Returns the union candidate DataFrame of one country (exact cosines, per-channel ranks)."""
    S = rec.filter((pl.col("src") == 1) & (pl.col("country") == country)).select("rid", *BLOCKS)
    T = rec.filter((pl.col("src") != 1) & (pl.col("country") == country)).select("rid", *BLOCKS)
    nS, nT = S.height, T.height
    log(f"[{country}] S1={nS} T={nT}")
    if nS == 0 or nT == 0:
        return None
    s_rid = S["rid"].to_numpy()
    t_rid = T["rid"].to_numpy()
    S = S.with_row_index("lrow")
    T = T.with_row_index("lrow")
    tmp = os.path.join(tmp_root, f"c_{abs(hash(country)) % 10**8}")
    rtmp = os.path.join(tmp, "rev")
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(rtmp)

    # ---- per block: idf, norms, df-capped index matrices (one block at a time to bound memory)
    mats, block_info = {}, {}
    need_rev = "rev" in channels
    for col, bname in zip(BLOCKS, BLOCK_NAMES):
        trS, trT = explode(S, col), explode(T, col)
        idf_all, shared = block_stats(trS, trT, nS)
        normS, normT = block_norms(trS, idf_all, nS), block_norms(trT, idf_all, nT)
        del idf_all
        capped = shared.filter(pl.col("dfS") <= args.df_cap)
        cost = int(capped.select((pl.col("dfS") * pl.col("dfT")).sum()).item() or 0)
        log(f"[{country}] block {bname}: shared feats={shared.height} capped multiply-adds={cost}")
        feat_ix = capped.with_row_index("col").select("h", pl.col("col").cast(pl.Int64), "idf")
        if bname in COMBINED or bname in channels:  # a block is a retrieval matrix only when some channel uses it
            XS = build_matrix(trS, feat_ix, normS, nS, feat_ix.height)
            XT = build_matrix(trT, feat_ix, normT, nT, feat_ix.height)
            _save_csr(XT, os.path.join(tmp, "T_" + bname))
            _save_csr(XS.T.tocsr(), os.path.join(tmp, "ST_" + bname))
            mats[bname] = ((nT, feat_ix.height), (feat_ix.height, nS))
            if need_rev and bname in COMBINED:
                # the reverse worker reads "ST_<b>" as S (nS x F) and "T_<b>" as T^T (F x nT)
                _save_csr(XS, os.path.join(rtmp, "ST_" + bname))
                _save_csr(XT.T.tocsr(), os.path.join(rtmp, "T_" + bname))
            del XS, XT
        if bname == "name":
            if "rare" in channels:
                rare = shared.filter(pl.col("dfS") <= args.rare_df).with_row_index("col").select("h", pl.col("col").cast(pl.Int64), "idf")
                _save_csr(build_matrix(trT, rare, normT, nT, rare.height, "idf_raw"), os.path.join(tmp, "T_rare"))
                _save_csr(build_matrix(trS, rare, normS, nS, rare.height, "binary").T.tocsr(), os.path.join(tmp, "ST_rare"))
                mats["rare"] = ((nT, rare.height), (rare.height, nS))
                log(f"[{country}] rare channel: {rare.height} features with df <= {args.rare_df}")
        if bname == "addr" and "hn" in channels:
            hS, hT = explode(S, col, "h:"), explode(T, col, "h:")
            hshared = hS.group_by("h").agg(pl.len().alias("dfS")).join(hT.select("h").unique(), on="h").filter(pl.col("dfS") <= args.df_cap)
            hfeat = hshared.with_row_index("col").select("h", pl.col("col").cast(pl.Int64), pl.lit(1.0, pl.Float32).alias("idf"))
            _save_csr(build_matrix(hT, hfeat, normT, nT, hfeat.height, "binary"), os.path.join(tmp, "T_hn"))
            _save_csr(build_matrix(hS, hfeat, normS, nS, hfeat.height, "binary").T.tocsr(), os.path.join(tmp, "ST_hn"))
            mats["hn"] = ((nT, hfeat.height), (hfeat.height, nS))
            log(f"[{country}] hn channel: {hfeat.height} house-number keys")
        block_info[bname] = (shared.select("h", "idf"), normS, normT)
        del trS, trT, feat_ix

    # ---- forward retrieval: every Source 2/3 record against Source 1, all channels
    bounds = [(a, min(a + args.chunk, nT)) for a in range(0, nT, args.chunk)]
    res = run_pool(tmp, mats, channels, False, bounds, f"[{country}]")
    if need_rev:
        # the reverse worker multiplies S (nS x F, read as "ST_<b>") by T^T (F x nT, read as "T_<b>")
        rmats = {b: ((mats[b][0][1], nT), (nS, mats[b][0][1])) for b in COMBINED if b in mats}
        rbounds = [(a, min(a + args.chunk, nS)) for a in range(0, nS, args.chunk)]
        res += run_pool(rtmp, rmats, channels, True, rbounds, f"[{country}] reverse")
    shutil.rmtree(tmp, ignore_errors=True)

    # ---- union of the channels: one row per (t, s) with the rank in every channel
    frames = []
    while res:
        name, rows, cols, score, rank = res.pop()
        if len(rows):
            frames.append(pl.DataFrame({"tl": rows.astype(np.int64), "sl": cols.astype(np.int64),
                                        "ch": np.full(len(rows), CHANNELS.index(name), dtype=np.int8), "rk": rank}))
    if not frames:
        return None
    pairs = pl.concat(frames)
    del frames
    per_ch = pairs.group_by("tl", "sl", "ch").agg(pl.col("rk").min())
    union = per_ch.group_by("tl", "sl").agg(pl.len().cast(pl.Int16).alias("n_ch"))
    for name, cid in ((n, CHANNELS.index(n)) for n in channels):
        r = per_ch.filter(pl.col("ch") == cid).select("tl", "sl", pl.col("rk").alias("r_" + name))
        union = union.join(r, on=["tl", "sl"], how="left").with_columns(pl.col("r_" + name).fill_null(-1).cast(pl.Int16))
    del pairs, per_ch
    union = union.sort("tl", "sl")
    tl, sl = union["tl"].to_numpy(), union["sl"].to_numpy()
    log(f"[{country}] union pairs {union.height} ({union.height / nT:.2f} per Source 2/3 record); "
        + ", ".join(f"{n}={int((union['r_' + n] >= 0).sum())}" for n in channels))

    # ---- exact per-block cosines with all shared features (no df cap)
    cos = {}
    for col, bname in zip(BLOCKS, BLOCK_NAMES):
        shared, normS, normT = block_info[bname]
        feat_all = shared.with_row_index("col").select("h", pl.col("col").cast(pl.Int64), "idf")
        XS = build_matrix(explode(S, col), feat_all, normS, nS, feat_all.height)
        XT = build_matrix(explode(T, col), feat_all, normT, nT, feat_all.height)
        cos[bname] = rowwise_dot(XT, XS, tl, sl)
        del XS, XT
    score = cos["name"] + cos["addr"] + cos["cross"]
    out = pl.DataFrame({
        "t_rid": t_rid[tl].astype(np.uint32), "s_rid": s_rid[sl].astype(np.uint32), "score": score,
        "cos_name": cos["name"], "cos_addr": cos["addr"], "cos_cross": cos["cross"], "cos_nchar": cos["nchar"],
        "n_ch": union["n_ch"], **{c: union[c] for c in union.columns if c.startswith("r_")},
    })
    return out.with_columns((pl.col("score").rank("ordinal", descending=True).over("t_rid").cast(pl.Int16) - 1).alias("rank"))


# ------------------------------------------------------------------ audit
def candidate_audit(cand, truth, s1_rids, channels, ambiguous=None):
    """Retrieval measured against the training labels (diagnostic only): true-pair recall of the
    union and per channel, complete-entity coverage, candidate oracle macro F0.5.

    ambiguous: optional bool per true pair marking pairs no retrieval can single out (an address-less
    Source 2/3 record whose Source 1 entity has exact-name namesakes): the misses are split into
    ambiguous / recoverable so that retrieval work is aimed at the recoverable ones."""
    hit = truth.join(cand.select("t_rid", "s_rid", "n_ch", *[c for c in cand.columns if c.startswith("r_")]), on=["t_rid", "s_rid"], how="left")
    found = hit["n_ch"].is_not_null().to_numpy()
    miss_split = None
    if ambiguous is not None:
        amb = np.asarray(ambiguous, dtype=bool)
        miss_split = {"ambiguous_true_pairs": int(amb.sum()), "missed_ambiguous": int((~found & amb).sum()),
                      "missed_recoverable": int((~found & ~amb).sum()),
                      "recall_on_recoverable": float(found[~amb].mean()) if (~amb).any() else None}
    per_channel = {n: float((hit["r_" + n].fill_null(-1) >= 0).mean()) for n in channels}
    only = {n: int(((hit["r_" + n].fill_null(-1) >= 0) & (hit["n_ch"].fill_null(0) == 1)).sum()) for n in channels}
    # entity level
    s_index = pl.DataFrame({"s_rid": s1_rids.astype(np.uint32)}).with_row_index("s_idx")
    ent = truth.join(s_index, on="s_rid").with_columns(pl.Series("found", found))
    e = ent.group_by("s_idx").agg(pl.len().alias("nt"), pl.col("found").sum().alias("tp"))
    nt = np.zeros(len(s1_rids), dtype=np.int64)
    tp = np.zeros(len(s1_rids), dtype=np.int64)
    nt[e["s_idx"].to_numpy()] = e["nt"].to_numpy()
    tp[e["s_idx"].to_numpy()] = e["tp"].to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        rec = np.where(nt > 0, tp / np.maximum(nt, 1), 0.0)
        f = np.where(rec > 0, 1.25 * 1.0 * rec / (0.25 * 1.0 + rec), 0.0)  # precision 1 (oracle)
    f = np.where(nt == 0, 1.0, f)
    non_single = nt > 0
    return {
        "n_true_pairs": int(truth.height),
        "pair_recall": float(found.mean()),
        "pair_recall_by_channel": per_channel,
        "true_pairs_found_by_one_channel_only": only,
        "entities": int(len(s1_rids)), "entities_with_matches": int(non_single.sum()),
        "entity_complete_coverage": float((tp[non_single] == nt[non_single]).mean()) if non_single.any() else None,
        "entity_zero_coverage": float((tp[non_single] == 0).mean()) if non_single.any() else None,
        "oracle_macro_f05": float(f.mean()),
        "oracle_macro_f05_non_singletons": float(f[non_single].mean()) if non_single.any() else None,
        "candidates_per_record": float(cand.height / max(cand["t_rid"].n_unique(), 1)),
        "candidates_total": int(cand.height),
        "channels": channels,
        "misses": miss_split,
    }


def main():
    ap = base_args(__doc__)
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--df-cap", type=int, default=1000)
    ap.add_argument("--channels", default=DEFAULT_CHANNELS, help="channel=k list; 'combined=3' = the previous single-channel retrieval")
    ap.add_argument("--rare-df", type=int, default=5, help="rare channel: name features with Source 1 df <= this")
    ap.add_argument("--chunk", type=int, default=4000)
    args = ap.parse_args()
    with Stage(args.work_dir, "blocking_" + args.split):
        run(args)


def run(args):
    d = split_dir(args.work_dir, args.split)
    channels = parse_channels(args.channels)
    log("channels", channels)
    rec = pl.read_parquet(os.path.join(d, "records.parquet"), columns=["rid", "src", "country", *BLOCKS])
    rec = rec.with_columns(pl.col("country").fill_null(""))
    countries = rec.filter(pl.col("src") == 1)["country"].unique().sort().to_list()
    tmp_root = os.path.join(d, "block_tmp")
    parts_dir = os.path.join(d, "cand_parts")
    shutil.rmtree(parts_dir, ignore_errors=True)
    os.makedirs(parts_dir)
    n_parts = 0
    for i, c in enumerate(countries):
        # each country's candidates go straight to disk: the full data has tens of millions of pairs
        r = run_country(rec, c, args, tmp_root, channels)
        if r is not None:
            r.write_parquet(os.path.join(parts_dir, f"part_{i:03d}.parquet"))
            n_parts += 1
        del r
    del rec
    out_path = os.path.join(d, "candidates_raw.parquet")
    if n_parts == 0:
        raise SystemExit("no candidate pairs retrieved")
    pl.scan_parquet(os.path.join(parts_dir, "part_*.parquet")).sink_parquet(out_path)  # streaming concat
    shutil.rmtree(parts_dir, ignore_errors=True)
    shutil.rmtree(tmp_root, ignore_errors=True)
    n_pairs = pl.scan_parquet(out_path).select(pl.len()).collect().item()
    log("candidates_raw", n_pairs, "pairs")
    if args.split == "train":
        from pair_features import truth_pairs  # noqa: E402  (label use is diagnostic only)
        ids = pl.read_parquet(os.path.join(d, "records.parquet"), columns=["rid", "entity_id", "src", "group", "a_norm"])
        truth = truth_pairs(ids, args.data_dir).select("t_rid", "s_rid")
        s1_rids = ids.filter(pl.col("src") == 1)["rid"].to_numpy().astype(np.uint32)
        cand = pl.read_parquet(out_path, columns=["t_rid", "s_rid", "n_ch"] + ["r_" + n for n in channels])
        # ambiguous true pair: address-less record + Source 1 entity with exact-name namesakes
        namesakes = ids.filter(pl.col("src") == 1).group_by("group").agg(pl.len().alias("n_ns"))
        amb = (truth.join(ids.select(pl.col("rid").cast(pl.UInt32).alias("t_rid"), (pl.col("a_norm").fill_null("") == "").alias("noaddr")), on="t_rid", how="left")
               .join(ids.select(pl.col("rid").cast(pl.UInt32).alias("s_rid"), "group"), on="s_rid", how="left")
               .join(namesakes, on="group", how="left")
               .select((pl.col("noaddr").fill_null(False) & (pl.col("n_ns").fill_null(1) > 1)).alias("amb"))["amb"].to_numpy())
        audit = candidate_audit(cand, truth, s1_rids, channels, amb)
        audit["recall_at_k"] = {}  # kept for older readers of this file
        with open(os.path.join(d, "blocking_recall.json"), "w") as f:
            json.dump(audit, f, indent=1)
        log("candidate audit:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in audit.items() if k not in ("pair_recall_by_channel", "true_pairs_found_by_one_channel_only")})
        log("  recall by channel:", {k: round(v, 4) for k, v in audit["pair_recall_by_channel"].items()})
        log("  true pairs found by one channel only:", audit["true_pairs_found_by_one_channel_only"])


if __name__ == "__main__":
    main()
