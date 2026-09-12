# ITGEEKS_Vibe_Coding_Round
# VecDB — A Vector Database Written From Scratch

> numpy for arithmetic. Nothing above it.

Built to understand what happens inside a vector database — not just use one.

## What's inside

```
vecdb/
├── core/
│   ├── exact_index.py     Flat brute-force (ground truth, O(n·d) per query)
│   └── lsh_index.py       LSH with random projections (approximate, tunable)
├── data/
│   └── corpus.py          5 000 news-style texts, char-ngram + word BOW embedded
├── benchmarks/
│   ├── benchmark.py       Full measurement suite
│   └── plot_results.py    Tradeoff curve plotter
└── demo/
    └── search_demo.py     Interactive terminal demo
```

## The two indexes

### ExactIndex (`core/exact_index.py`)
Flat brute-force. Stores vectors in a `dict[id → ndarray]` and stacks them
into a matrix on first search (lazy, cached). One batched numpy matmul gives
cosine similarity against every vector simultaneously. Provably correct:
if your nearest neighbour exists, this finds it.

**Insert:** O(1). **Search:** O(n·d). **Delete:** O(1) — remove from dict,
invalidate matrix cache.

### LSHIndex (`core/lsh_index.py`)
Random-projection LSH for cosine similarity.

**The hash function:** sign(plane · vector) for B random unit hyperplanes
gives a B-bit fingerprint. Nearby vectors (high cosine) agree on most bits;
distant vectors agree on ~half (random).

**L tables:** L independent B-bit hash functions. At query time, retrieve
candidates from all L buckets, score them exactly, return top-k.

**The knob — `n_tables` (L):**
- L=1: fast, low recall (few candidates)
- L=32: slow, high recall (many candidates)
- The curve between them is the assignment deliverable.

**Insert:** O(L·B). **Search:** O(L·|candidates|·d). **Delete:** O(L) — each
ID's hash keys are stored at insert time, so deletion is O(1) per table.

## Benchmark results (5 000 corpus, 500 queries, k=10, n_bits=6)

| n_tables (L) | recall@10 | QPS     |
|-------------:|----------:|--------:|
| 1            | 0.328     | 2 003   |
| 2            | 0.504     | 1 225   |
| 4            | 0.755     | 727     |
| 6            | 0.860     | 548     |
| 8            | 0.916     | 402     |
| 12           | 0.969     | 295     |
| 16           | 0.988     | 238     |
| 24           | 0.998     | 177     |
| 32           | 1.000     | 144     |
| **Exact**    | **1.000** | **18 173** |

### Why is Exact faster than LSH here?

Because numpy's batched matmul on 5 000 vectors costs ~0.055 ms — a single
optimized BLAS call. LSH at this scale pays per-table Python overhead
(hash computation, dict lookup, candidate dedup) across L tables. The crossover
point where LSH wins is around n ≈ 100 000–1 000 000. At that scale, exact
search spends 6–60 ms and LSH stays at <1 ms regardless of corpus size.

**This is the real lesson:** approximate search is a *scalability* technique,
not a magic speedup at small n.

## Deletion design

**ExactIndex:** delete from the Python dict; set `_dirty=True` so the next
search rebuilds the matrix without the deleted vector. O(1).

**LSHIndex:** at insert time, we store `{id → (vector, [hash_key_per_table])}`.
Delete uses these stored keys to remove the ID from each bucket set directly —
no scan needed. O(L). No tombstones, no compaction, no iterator invalidation.

Both pass the leak test: deleted IDs never appear in search results.

## Running it

```bash
# Install deps (numpy + matplotlib only)
pip install numpy matplotlib

# Run benchmark + plot
python benchmarks/benchmark.py

# Interactive demo
python demo/search_demo.py
# > Enter query: scientists discover ancient fossil evolution
```

## Corpus

5 000 AG-News style headlines covering five topics: world affairs, sports,
business, technology, science. Each headline is filled from pools of 20–35
real entities (countries, cities, companies, players, teams).

**Embedding:** char-trigrams + word-unigrams → 4096-dim BOW →
random projection to R^64. Single fixed projection seed so corpus and query
vectors live in the same space. Same-topic texts reach cosine 0.4–0.9;
cross-topic pairs are near-zero. Lumpy geometry, not uniform — harder for LSH.

## The measurement

The speed–accuracy tradeoff exists because approximate search trades recall
for speed by looking at only a fraction of the corpus. The fraction is
controlled by `n_tables` (L): each table is one opportunity to retrieve the
true nearest neighbour; more tables = higher recall = more work = lower QPS.

A single accuracy number is a claim. A curve is a result.
