"""Step 2: candidate generation (blocking).

For every country label (open set, whatever values appear in the data) we build
TF-IDF vectors over the blocking features produced by textnorm.blocking_features:

  name block   : n:<token>, p:<token pair> (order-free), k:<8-char compact-name prefix>
  address block: a:<token>, b:<adjacent token bigram inside an address component>,
                 h:<house number>_<key address token>
  cross block  : c:<name token>_<key address token>, i.e. name x place (textnorm.blocking_features)

IDF is computed on Source 1 (the index side). Each block is L2-normalised, so the
retrieval score of a (Source 2/3 record t, Source 1 record s) pair is
cos_name(t, s) + cos_addr(t, s) + cos_cross(t, s), restricted to "index" features whose Source 1
document frequency is <= --df-cap (very common tokens cost a lot and carry
little evidence, but they still count in the vector norms).

Because every Source 2/3 record belongs to at most one Source 1 entity, retrieval
is done from the Source 2/3 side: a chunked sparse matrix product T x S1^T (run
in parallel over memory-mapped matrices) keeps the top --topk Source 1 records per
Source 2/3 record. For every retrieved pair we also compute exact TF-IDF cosine
similarities per block using *all* shared features (no df cap).

Output: <work>/<split>/candidates_raw.parquet
  t_rid, s_rid, score, rank, cos_name, cos_addr, cos_cross
"""
import multiprocessing as mp
import os
import shutil

import numpy as np
import polars as pl
import scipy.sparse as sp

from common import base_args, log, n_workers, split_dir

_G = {}


# ------------------------------------------------------------------ features
BLOCKS = ("feat_name", "feat_addr", "feat_cross")
BLOCK_NAMES = ("name", "addr", "cross")


def explode(df, col):
    """(lrow, h) pairs of one feature block."""
    return (
        df.lazy()
        .select(pl.col("lrow"), pl.col(col).str.split(" ").alias("f"))
        .explode("f")
        .filter(pl.col("f").is_not_null() & (pl.col("f") != ""))
        .select("lrow", pl.col("f").hash(seed=17).alias("h"))
        .collect()
    )


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


def build_matrix(trip, feat, norms, n_rows, n_cols):
    """CSR matrix of idf/norm weights over the features in `feat` (h -> col, idf)."""
    m = trip.join(feat, on="h", how="inner").sort("lrow")
    rows = m["lrow"].to_numpy()
    cols = m["col"].to_numpy().astype(np.int32)
    vals = (m["idf"].to_numpy() / norms[rows]).astype(np.float32)
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


def _init(tmp, t_shape, st_shape, topk, min_score):
    _G["T"] = _load_csr(os.path.join(tmp, "T"), t_shape)
    _G["ST"] = _load_csr(os.path.join(tmp, "ST"), st_shape)
    _G["topk"] = topk
    _G["min_score"] = min_score


def _retrieve(bounds):
    a, b = bounds
    c = _G["T"][a:b] @ _G["ST"]
    counts = np.diff(c.indptr)
    rows = np.repeat(np.arange(b - a, dtype=np.int64), counts)
    data, cols = c.data, c.indices
    keep = data >= _G["min_score"]
    rows, data, cols = rows[keep], data[keep], cols[keep]
    order = np.lexsort((-data, rows))
    rows, data, cols = rows[order], data[order], cols[order]
    counts = np.bincount(rows, minlength=b - a)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    rank = np.arange(len(rows)) - np.repeat(starts, counts)
    sel = rank < _G["topk"]
    return rows[sel] + a, cols[sel].astype(np.int32), data[sel].astype(np.float32), rank[sel].astype(np.int16)


def rowwise_dot(A, B, ia, ib, chunk=1_000_000):
    out = np.empty(len(ia), dtype=np.float32)
    for s in range(0, len(ia), chunk):
        x = A[ia[s:s + chunk]].multiply(B[ib[s:s + chunk]])
        out[s:s + chunk] = np.asarray(x.sum(axis=1)).ravel()
    return out


def run_country(rec, country, args, tmp_root):
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

    # ---- per block: idf, norms, df-capped index matrices (blocks are processed one at a time
    # to bound memory; the retrieval matrices of the blocks are stacked side by side)
    XS_parts, XT_parts, block_info = [], [], []
    for col, bname in zip(BLOCKS, BLOCK_NAMES):
        trS, trT = explode(S, col), explode(T, col)
        idf_all, shared = block_stats(trS, trT, nS)
        normS, normT = block_norms(trS, idf_all, nS), block_norms(trT, idf_all, nT)
        del idf_all
        cost = int(shared.filter(pl.col("dfS") <= args.df_cap).select((pl.col("dfS") * pl.col("dfT")).sum()).item() or 0)
        log(f"[{country}] block {bname}: shared feats={shared.height} capped multiply-adds={cost}")
        feat_ix = shared.filter(pl.col("dfS") <= args.df_cap).with_row_index("col").select(
            "h", pl.col("col").cast(pl.Int64), "idf")
        XS_parts.append(build_matrix(trS, feat_ix, normS, nS, feat_ix.height))
        XT_parts.append(build_matrix(trT, feat_ix, normT, nT, feat_ix.height))
        block_info.append((shared.select("h", "idf"), normS, normT))
        del trS, trT, feat_ix
    XS = sp.hstack(XS_parts, format="csr")
    XT = sp.hstack(XT_parts, format="csr")
    del XS_parts, XT_parts
    F = XS.shape[1]
    tmp = os.path.join(tmp_root, f"c_{abs(hash(country)) % 10**8}")
    os.makedirs(tmp, exist_ok=True)
    _save_csr(XT, os.path.join(tmp, "T"))
    _save_csr(XS.T.tocsr(), os.path.join(tmp, "ST"))
    del XS, XT
    bounds = [(a, min(a + args.chunk, nT)) for a in range(0, nT, args.chunk)]
    res = []
    with mp.get_context("spawn").Pool(
        n_workers(), initializer=_init, initargs=(tmp, (nT, F), (F, nS), args.topk, args.min_score)
    ) as pool:
        for k, r in enumerate(pool.imap_unordered(_retrieve, bounds)):
            res.append(r)
            if k % 100 == 0:
                log(f"[{country}] retrieval {k + 1}/{len(bounds)}")
    shutil.rmtree(tmp, ignore_errors=True)
    tl = np.concatenate([r[0] for r in res])
    sl = np.concatenate([r[1] for r in res])
    score = np.concatenate([r[2] for r in res])
    rank = np.concatenate([r[3] for r in res])
    del res
    log(f"[{country}] pairs retrieved {len(tl)}")

    # ---- exact per-block cosines with all shared features (no df cap)
    cos = []
    for col, (shared, normS, normT) in zip(BLOCKS, block_info):
        feat_all = shared.with_row_index("col").select("h", pl.col("col").cast(pl.Int64), "idf")
        XS = build_matrix(explode(S, col), feat_all, normS, nS, feat_all.height)
        XT = build_matrix(explode(T, col), feat_all, normT, nT, feat_all.height)
        cos.append(rowwise_dot(XT, XS, tl, sl))
        del XS, XT
    return pl.DataFrame({
        "t_rid": t_rid[tl].astype(np.uint32),
        "s_rid": s_rid[sl].astype(np.uint32),
        "score": score,
        "rank": rank,
        "cos_name": cos[0],
        "cos_addr": cos[1],
        "cos_cross": cos[2],
    })


def main():
    ap = base_args(__doc__)
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--df-cap", type=int, default=1000)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--min-score", type=float, default=0.05)
    ap.add_argument("--chunk", type=int, default=4000)
    args = ap.parse_args()
    d = split_dir(args.work_dir, args.split)
    rec = pl.read_parquet(os.path.join(d, "records.parquet"), columns=["rid", "src", "country", *BLOCKS])
    rec = rec.with_columns(pl.col("country").fill_null(""))
    countries = rec.filter(pl.col("src") == 1)["country"].unique().sort().to_list()
    tmp_root = os.path.join(d, "block_tmp")
    outs = []
    for c in countries:
        r = run_country(rec, c, args, tmp_root)
        if r is not None:
            outs.append(r)
    cand = pl.concat(outs)
    cand.write_parquet(os.path.join(d, "candidates_raw.parquet"))
    log("candidates_raw", cand.shape)


if __name__ == "__main__":
    main()
