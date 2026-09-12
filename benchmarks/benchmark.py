"""
Benchmark suite for VecDB.

Measures:
  - ExactIndex: ground-truth QPS
  - LSHIndex at n_tables ∈ {1,2,4,6,8,12,16,24,32,48}: recall@10 + QPS
  - Deletion correctness
  - Returns all results for plotting
"""

import sys, os, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.exact_index import ExactIndex
from core.lsh_index   import LSHIndex
from data.corpus      import build_corpus, build_query_set, DIM

N_BITS   = 6     # bits per table — calibrated for this corpus geometry
K        = 10    # nearest neighbours
N_TABLES = [1, 2, 4, 6, 8, 12, 16, 24, 32, 48]


def recall_at_k(pred, gt_set, k=K):
    return len(set(p[0] for p in pred[:k]) & gt_set) / len(gt_set)


def measure(index, qvecs, k=K, warmup=30):
    for q in qvecs[:warmup]:
        index.search(q, k=k)
    t0 = time.perf_counter()
    results = [index.search(q, k=k) for q in qvecs]
    elapsed = time.perf_counter() - t0
    qps = len(qvecs) / elapsed
    return results, qps, 1000.0 / qps


def test_deletion(dim=DIM, seed=7):
    print("\n── Deletion correctness ───────────────────────────────────────")
    rng = np.random.default_rng(seed)
    n   = 300
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10

    exact = ExactIndex(dim=dim)
    lsh   = LSHIndex(dim=dim, n_tables=8, n_bits=N_BITS, seed=seed)
    for i, v in enumerate(vecs):
        exact.insert(v, i)
        lsh.insert(v, i)

    to_del = list(range(0, n, 2))
    for vid in to_del:
        exact.delete(vid)
        lsh.delete(vid)

    assert len(exact) == n - len(to_del)
    assert len(lsh)   == n - len(to_del)

    q  = vecs[1]
    ds = set(to_del)
    ex_leak = [r[0] for r in exact.search(q, k=30) if r[0] in ds]
    lh_leak = [r[0] for r in lsh.search(q,   k=30) if r[0] in ds]
    print(f"  ExactIndex — deleted {len(to_del)}, leaked {len(ex_leak)} → {'PASS' if not ex_leak else 'FAIL'}")
    print(f"  LSHIndex   — deleted {len(to_del)}, leaked {len(lh_leak)} → {'PASS' if not lh_leak else 'FAIL'}")

    rid = to_del[5]
    exact.insert(vecs[rid], rid)
    lsh.insert(vecs[rid], rid)
    ef = any(r[0] == rid for r in exact.search(vecs[rid], k=5))
    lf = any(r[0] == rid for r in lsh.search(vecs[rid],   k=5))
    print(f"  Re-insert  — ExactIndex: {'PASS' if ef else 'FAIL'}, LSHIndex: {'PASS' if lf else 'FAIL'}")


def run_benchmark(n_corpus=5000, n_queries=500, seed=42):
    print("=" * 60)
    print("  VecDB Benchmark  |  numpy only, no vector DB libs")
    print("=" * 60)

    # ── corpus ──────────────────────────────────────────────────────
    print(f"\nBuilding corpus ({n_corpus} texts → R^{DIM})...")
    t0 = time.perf_counter()
    corpus_vecs, corpus_texts = build_corpus(n=n_corpus, dim=DIM, seed=seed)
    print(f"  done  {time.perf_counter()-t0:.2f}s  |  unique: {len(set(corpus_texts))}/{n_corpus}")

    # ── ground truth ────────────────────────────────────────────────
    query_vecs, query_texts, ground_truth = build_query_set(
        corpus_vecs, corpus_texts,
        n_queries=n_queries, k=K, seed=seed+1, dim=DIM,
    )

    # ── ExactIndex ──────────────────────────────────────────────────
    print(f"\nBuilding ExactIndex...")
    t0 = time.perf_counter()
    exact = ExactIndex(dim=DIM, metric="cosine")
    for i, v in enumerate(corpus_vecs):
        exact.insert(v, i)
    print(f"  insert: {time.perf_counter()-t0:.3f}s")

    _, exact_qps, exact_lat = measure(exact, query_vecs)
    print(f"  {exact_qps:.1f} QPS  |  {exact_lat:.3f} ms/query  |  recall: 1.000 (ground truth)")

    # ── LSH sweep ───────────────────────────────────────────────────
    print(f"\nLSH sweep  (n_bits={N_BITS}, k={K})")
    print(f"{'n_tables':>10}  {'recall@10':>10}  {'QPS':>10}  {'ms/q':>8}  {'speedup':>8}")
    print("─" * 55)

    lsh_results = []
    for L in N_TABLES:
        lsh = LSHIndex(dim=DIM, n_tables=L, n_bits=N_BITS, seed=seed)
        for i, v in enumerate(corpus_vecs):
            lsh.insert(v, i)

        all_res, qps, lat = measure(lsh, query_vecs)
        recalls = [recall_at_k(res, set(ground_truth[qi]))
                   for qi, res in enumerate(all_res)]
        r = float(np.mean(recalls))
        speedup = qps / exact_qps

        print(f"{L:>10}  {r:>10.3f}  {qps:>10.1f}  {lat:>8.3f}  {speedup:>7.1f}×")
        lsh_results.append(dict(n_tables=L, recall=r, qps=qps,
                                latency_ms=lat, speedup=speedup))

    print(f"{'Exact':>10}  {'1.000':>10}  {exact_qps:>10.1f}  {exact_lat:>8.3f}  {'1.0×':>8}")

    # ── deletion ────────────────────────────────────────────────────
    test_deletion()

    return dict(
        exact_qps=exact_qps, exact_latency_ms=exact_lat,
        lsh_results=lsh_results,
        corpus_vecs=corpus_vecs, corpus_texts=corpus_texts,
        query_vecs=query_vecs, query_texts=query_texts,
        ground_truth=ground_truth,
    )


if __name__ == "__main__":
    run_benchmark()
