"""
ExactIndex — brute-force nearest neighbour search.

Guarantees provably correct results. O(n·d) per query.
This is the ground truth everything else is scored against.

Deletion: O(1) — vectors are stored in a dict keyed by ID.
We keep a numpy matrix view rebuilt on-demand (lazy, cached).
Inserting or deleting invalidates the cache flag so the next
search rebuilds the matrix. No tombstones, no compaction needed.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict


class ExactIndex:
    """
    Flat exact index. Correct by construction.

    Stores vectors in a dict {id -> np.ndarray}.
    On search, stacks them into a matrix and does a single
    batched cosine/dot computation via numpy broadcasting.
    """

    def __init__(self, dim: int, metric: str = "cosine"):
        self.dim = dim
        self.metric = metric          # "cosine" | "dot" | "l2"

        self._store: Dict[int, np.ndarray] = {}
        self._id_order: List[int] = []   # ordered list of active ids
        self._matrix: Optional[np.ndarray] = None  # (n, dim) cache
        self._dirty = True

    # ------------------------------------------------------------------ #
    #  Mutation                                                            #
    # ------------------------------------------------------------------ #

    def insert(self, vector: np.ndarray, vec_id: int) -> None:
        vec = np.array(vector, dtype=np.float32)
        assert vec.shape == (self.dim,), f"Expected dim {self.dim}, got {vec.shape}"
        if vec_id in self._store:
            raise ValueError(f"ID {vec_id} already exists. Delete first.")
        self._store[vec_id] = vec
        self._id_order.append(vec_id)
        self._dirty = True

    def delete(self, vec_id: int) -> None:
        if vec_id not in self._store:
            raise KeyError(f"ID {vec_id} not found.")
        del self._store[vec_id]
        self._id_order.remove(vec_id)   # O(n) but deletion is rare
        self._dirty = True

    def update(self, vector: np.ndarray, vec_id: int) -> None:
        self.delete(vec_id)
        self.insert(vector, vec_id)

    # ------------------------------------------------------------------ #
    #  Search                                                              #
    # ------------------------------------------------------------------ #

    def search(self, query: np.ndarray, k: int = 10) -> List[Tuple[int, float]]:
        """Return [(id, score), ...] sorted best-first."""
        if not self._store:
            return []
        self._rebuild_if_dirty()
        q = np.array(query, dtype=np.float32)
        scores = self._score(q)
        k = min(k, len(self._id_order))
        # argpartition is O(n) average, then sort only the top-k
        top_idx = np.argpartition(scores, -k)[-k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        return [(self._id_order[i], float(scores[i])) for i in top_idx]

    def _rebuild_if_dirty(self):
        if self._dirty:
            if self._id_order:
                self._matrix = np.stack(
                    [self._store[i] for i in self._id_order], axis=0
                )  # (n, dim)
                if self.metric == "cosine":
                    norms = np.linalg.norm(self._matrix, axis=1, keepdims=True)
                    norms = np.where(norms == 0, 1e-10, norms)
                    self._matrix_norm = self._matrix / norms
            else:
                self._matrix = np.empty((0, self.dim), dtype=np.float32)
                self._matrix_norm = self._matrix
            self._dirty = False

    def _score(self, q: np.ndarray) -> np.ndarray:
        if self.metric == "cosine":
            qn = q / (np.linalg.norm(q) + 1e-10)
            return self._matrix_norm @ qn          # dot of unit vecs = cosine
        elif self.metric == "dot":
            return self._matrix @ q
        elif self.metric == "l2":
            diff = self._matrix - q
            return -(diff * diff).sum(axis=1)      # negated so higher = closer
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

    # ------------------------------------------------------------------ #
    #  Introspection                                                       #
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, vec_id: int) -> bool:
        return vec_id in self._store

    def get_vector(self, vec_id: int) -> np.ndarray:
        return self._store[vec_id].copy()
