"""
Interactive search demo.

Usage:
    python demo/search_demo.py

Type any statement. Shows results from both ExactIndex and LSHIndex,
reports agreement, and marks which LSH results were correct (✓) or missed (✗).
"""

import sys, os, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.exact_index import ExactIndex
from core.lsh_index   import LSHIndex
from data.corpus       import build_corpus, text_to_bow, bow_to_embedding, DIM, VOCAB

N_CORPUS = 5000
SEED     = 42


def embed_query(text: str) -> np.ndarray:
    bow = text_to_bow(text, vocab_size=VOCAB)
    return bow_to_embedding(bow, dim=DIM)


def build_indexes():
    print("\nLoading corpus (5 000 texts)...")
    t0 = time.perf_counter()
    corpus_vecs, corpus_texts = build_corpus(n=N_CORPUS, dim=DIM, seed=SEED)
    print(f"  Embeddings ready  {time.perf_counter()-t0:.2f}s")

    print("Building ExactIndex...")
    exact = ExactIndex(dim=DIM, metric="cosine")
    for i, v in enumerate(corpus_vecs):
        exact.insert(v, i)

    print("Building LSHIndex (L=16, K=6)...")
    lsh = LSHIndex(dim=DIM, n_tables=16, n_bits=6, seed=SEED)
    for i, v in enumerate(corpus_vecs):
        lsh.insert(v, i)

    return exact, lsh, corpus_texts


def search_both(query_text: str, exact, lsh, corpus_texts, k: int = 5):
    q = embed_query(query_text)

    t0 = time.perf_counter()
    exact_results = exact.search(q, k=k)
    exact_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    lsh_results = lsh.search(q, k=k)
    lsh_ms = (time.perf_counter() - t0) * 1000

    exact_ids = {r[0] for r in exact_results}
    lsh_ids   = [r[0] for r in lsh_results]
    agreement = len(set(lsh_ids[:k]) & exact_ids) / k

    print(f"\n{'─'*65}")
    print(f"  Query: \"{query_text}\"")
    print(f"{'─'*65}")

    print(f"\n  EXACT  ({exact_ms:.2f} ms) — ground truth")
    for rank, (vid, score) in enumerate(exact_results, 1):
        print(f"  {rank}. [{score:.4f}]  {corpus_texts[vid]}")

    print(f"\n  LSH    ({lsh_ms:.2f} ms) — approximate  (L=16, K=6)")
    for rank, (vid, score) in enumerate(lsh_results, 1):
        mark = "✓" if vid in exact_ids else "✗"
        print(f"  {rank}. [{score:.4f}] {mark}  {corpus_texts[vid]}")

    speedup_str = f"{exact_ms/max(lsh_ms,0.001):.1f}×" if lsh_ms > 0 else "N/A"
    print(f"\n  Agreement: {agreement*100:.0f}%   Latency — exact: {exact_ms:.2f}ms  lsh: {lsh_ms:.2f}ms")


PRESET_QUERIES = [
    "scientists discover ancient fossil rewriting human evolution",
    "central bank raises interest rates to fight inflation",
    "tech company acquires AI startup for billions",
    "football team wins championship after dramatic comeback",
    "United Nations crisis meeting over border conflict",
    "quantum computing breakthrough achieved by research team",
    "cybersecurity breach exposes millions of user records",
    "climate change study warns of accelerating Arctic ice loss",
]


def main():
    exact, lsh, corpus_texts = build_indexes()

    print("\n" + "═"*65)
    print("  VecDB Interactive Search Demo")
    print("  Commands: 'demo' for preset queries | 'quit' to exit")
    print("═"*65)

    while True:
        try:
            user_input = input("\n  Enter query: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Bye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("  Bye!")
            break
        if user_input.lower() == "demo":
            for q in PRESET_QUERIES:
                search_both(q, exact, lsh, corpus_texts)
            continue

        search_both(user_input, exact, lsh, corpus_texts)


if __name__ == "__main__":
    main()
