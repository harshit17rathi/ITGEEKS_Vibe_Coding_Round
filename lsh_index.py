"""
LSHIndex — Locality-Sensitive Hashing with random projections.

DESIGN
------
We hash each vector into B binary bits by comparing the dot product with
B random unit-normal hyperplanes (random projections). Vectors that are
close in cosine space collide in the same hash bucket with high probability.

At query time we look up the query's bucket (and optionally its Hamming
neighbours within radius `hamming_radius`) and score only that candidate
set against the query — skipping most of the corpus.

THE KNOB
--------
`n_tables` (L): number of independent hash tables.
More tables → more candidates retrieved → higher recall → slower queries.
This is the primary speed-accuracy lever: a curve over L is the assignment
deliverable.

`n_bits` (K): bits per table. More bits → smaller buckets → fewer
candidates per table → faster but lower recall on any individual table.
With L tables the two parameters trade off against each other.

DELETION
--------
Clean. We store per-bucket sets of IDs, so delete is O(1) per table:
look up the stored hash for that ID and remove it from the bucket set.
We keep a separate dict {id -> (vector, hashes_per_table)} for this.

INSERT
------
O(L·K) — compute L hashes, append to L buckets.

SEARCH
------
O(L·|candidates| · d) — retrieve candidates from L buckets, deduplicate,
score exact cosine against each.
"""

import numpy as np
from typing import List, Tuple, Dict, Set, Optional
from collections import defaultdict


class LSHIndex:
    """
    Random-projection LSH for cosine similarity.

    Parameters
    ----------
    dim : int
        Vector dimensionality.
    n_tables : int
        Number of hash tables (L). PRIMARY ACCURACY KNOB.
        Range: 1 → ~50. Higher = more recall, slower search.
    n_bits : int
        Hash bits per table (K). Tighter buckets with more bits.
    seed : int
        RNG seed for reproducibility.
    metric : str
        Only "cosine" supported (sign of dot product is the hash function).
    """

    def __init__(
        self,
        dim: int,
        n_tables: int = 10,
        n_bits: int = 16,
        seed: int = 42,
        metric: str = "cosine",
    ):
        self.dim = dim
        self.n_tables = n_tables
        self.n_bits = n_bits
        self.seed = seed
        self.metric = metric

        rng = np.random.default_rng(seed)

        # Random hyperplanes: shape (n_tables, n_bits, dim)
        # Each table has n_bits hyperplanes. The hash of a vector v under
        # table t is the n_bits-bit string: sign(planes[t] @ v) >= 0
        self._planes: np.ndarray = rng.standard_normal(
            (n_tables, n_bits, dim)
        ).astype(np.float32)

        # Hash tables: list of L dicts {hash_key -> set of ids}
        self._tables: List[Dict[int, Set[int]]] = [
            defaultdict(set) for _ in range(n_tables)
        ]

        # Per-ID storage: {id -> (vector, [hash_key_per_table])}
        self._store: Dict[int, Tuple[np.ndarray, List[int]]] = {}

    # ------------------------------------------------------------------ #
    #  Hashing                                                             #
    # ------------------------------------------------------------------ #

    def _hash_vector(self, vec: np.ndarray) -> List[int]:
        """
        Compute one integer hash key per table.
        Sign of (plane @ vec) gives a bit; pack n_bits into one int.
        """
        # planes: (n_tables, n_bits, dim), vec: (dim,)
        # projections: (n_tables, n_bits)
        projections = self._planes @ vec          # broadcasting: fine
        bits = (projections >= 0).astype(np.uint8)  # (n_tables, n_bits)
        keys = []
        for t in range(self.n_tables):
            # pack bits into a single Python int
            key = 0
            for b in range(self.n_bits):
                key = (key << 1) | int(bits[t, b])
            keys.append(key)
        return keys

    # ------------------------------------------------------------------ #
    #  Mutation                                                            #
    # ------------------------------------------------------------------ #

    def insert(self, vector: np.ndarray, vec_id: int) -> None:
        vec = np.array(vector, dtype=np.float32)
        assert vec.shape == (self.dim,)
        if vec_id in self._store:
            raise ValueError(f"ID {vec_id} already exists.")
        keys = self._hash_vector(vec)
        for t, key in enumerate(keys):
            self._tables[t][key].add(vec_id)
        self._store[vec_id] = (vec, keys)

    def delete(self, vec_id: int) -> None:
        """
        O(1) per table — we stored the hash keys at insert time,
        so we know exactly which bucket to remove from.
        """
        if vec_id not in self._store:
            raise KeyError(f"ID {vec_id} not found.")
        _, keys = self._store[vec_id]
        for t, key in enumerate(keys):
            self._tables[t][key].discard(vec_id)
            if not self._tables[t][key]:
                del self._tables[t][key]
        del self._store[vec_id]

    def update(self, vector: np.ndarray, vec_id: int) -> None:
        self.delete(vec_id)
        self.insert(vector, vec_id)

    # ------------------------------------------------------------------ #
    #  Search                                                              #
    # ------------------------------------------------------------------ #

    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        hamming_radius: int = 0,
    ) -> List[Tuple[int, float]]:
        """
        Return [(id, cosine_score), ...] best-first.

        hamming_radius: also probe buckets within this Hamming distance
        (secondary accuracy knob, default 0 for speed).
        """
        if not self._store:
            return []
        q = np.array(query, dtype=np.float32)
        q_keys = self._hash_vector(q)

        # --- collect candidates ---
        candidates: Set[int] = set()
        for t, key in enumerate(q_keys):
            if hamming_radius == 0:
                candidates.update(self._tables[t].get(key, set()))
            else:
                for probe_key in self._hamming_neighbours(key, hamming_radius):
                    candidates.update(self._tables[t].get(probe_key, set()))

        if not candidates:
            return []

        # --- exact re-rank candidates ---
        q_norm = q / (np.linalg.norm(q) + 1e-10)
        scored = []
        for cid in candidates:
            vec, _ = self._store[cid]
            v_norm = vec / (np.linalg.norm(vec) + 1e-10)
            score = float(q_norm @ v_norm)
            scored.append((cid, score))
        scored.sort(key=lambda x: -x[1])
        return scored[:k]

    def _hamming_neighbours(self, key: int, radius: int) -> List[int]:
        """All keys within Hamming distance `radius` of `key`."""
        if radius == 0:
            return [key]
        results = [key]
        # flip each bit combination up to radius flips
        for bit in range(self.n_bits):
            neighbour = key ^ (1 << bit)
            results.append(neighbour)
            if radius >= 2:
                for bit2 in range(bit + 1, self.n_bits):
                    results.append(neighbour ^ (1 << bit2))
        return results

    # ------------------------------------------------------------------ #
    #  Introspection                                                       #
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self._store)

    def stats(self) -> dict:
        bucket_sizes = []
        for table in self._tables:
            bucket_sizes.extend(len(v) for v in table.values())
        arr = np.array(bucket_sizes) if bucket_sizes else np.array([0])
        return {
            "n_vectors": len(self._store),
            "n_tables": self.n_tables,
            "n_bits": self.n_bits,
            "total_buckets": sum(len(t) for t in self._tables),
            "mean_bucket_size": float(arr.mean()),
            "max_bucket_size": int(arr.max()),
        }
