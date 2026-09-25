"""Expected-F0.5 decoding.

Each Source 2/3 record t is offered only to its best Source 1 candidate s (one-to-one
structure), with probability q = p2(t, s). For every Source 1 entity the offered records are
sorted by q and we choose how many of them (k = 0..K) to link so as to maximise the expected
per-entity F0.5, treating the offers as independent Bernoulli(q) events:

    E[F_k] = sum_a sum_b P(TP_k = a) P(FN_k = b) * F(a, b, k)

where TP_k is the number of true records among the k linked ones and FN_k among the rest
(both Poisson-binomial, computed exactly by dynamic programming). k = 0 scores 1 only when
the entity has no true record at all (singleton). `miss` adds the expected number of true
records that blocking never proposed (they lower recall whatever we choose).
"""
import numpy as np
import polars as pl

BETA2 = 0.25


def _pb_dist(q):
    """Poisson-binomial pmf for each row of q (N, m) -> (N, m+1)."""
    N, m = q.shape
    d = np.zeros((N, m + 1))
    d[:, 0] = 1.0
    for j in range(m):
        p = q[:, j:j + 1]
        d[:, 1:] = d[:, 1:] * (1 - p) + d[:, :-1] * p
        d[:, 0] = d[:, 0] * (1 - p[:, 0])
    return d


def best_k(Q, miss=0.0):
    """Q: (N, K) offers sorted descending per row (0-padded). Returns chosen k per row."""
    N, K = Q.shape
    best = np.zeros(N, dtype=np.int64)
    best_val = np.full(N, -1.0)
    a = np.arange(K + 1)[:, None]
    b = np.arange(K + 1)[None, :]
    for k in range(K + 1):
        tp = _pb_dist(Q[:, :k]) if k > 0 else np.ones((N, 1))
        fn = _pb_dist(Q[:, k:]) if k < K else np.ones((N, 1))
        A = a[: tp.shape[1]]
        B = b[:, : fn.shape[1]]
        nt = A + B + miss
        if k == 0:
            # empty prediction scores 1 only for a singleton; with miss > 0 the entity may still
            # have unproposed true records, approximated as Poisson(miss): P(none) = exp(-miss)
            f = np.where(A + B == 0, np.exp(-miss), 0.0).astype(float)
        else:
            f = (1 + BETA2) * A / (BETA2 * nt + k)
        val = np.einsum("na,nb,ab->n", tp, fn, f)
        upd = val > best_val
        best[upd] = k
        best_val[upd] = val[upd]
    return best


def decode(meta, p, max_offers=12, miss=0.0, min_q=0.02):
    """meta (t_rid, s_rid) aligned with p -> links DataFrame (t_rid, s_rid, p)."""
    offers = (
        meta.select("t_rid", "s_rid").with_columns(pl.Series("p", p))
        .sort("p", descending=True)
        .unique("t_rid", keep="first")
        .filter(pl.col("p") >= min_q)
        .sort(["s_rid", "p"], descending=[False, True])
        .with_columns(pl.int_range(pl.len()).over("s_rid").alias("j"))
    )
    top = offers.filter(pl.col("j") < max_offers)
    s_ids, s_idx = np.unique(top["s_rid"].to_numpy(), return_inverse=True)
    Q = np.zeros((len(s_ids), max_offers))
    Q[s_idx, top["j"].to_numpy()] = top["p"].to_numpy()
    k = best_k(Q, miss=miss)
    kk = pl.DataFrame({"s_rid": s_ids, "k": k})
    return top.join(kk, on="s_rid").filter(pl.col("j") < pl.col("k")).select("t_rid", "s_rid", "p")
