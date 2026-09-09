"""The two indexes: dense vectors and BM25.

On the vector side this is a brute-force cosine scan over a numpy matrix rather
than FAISS. For a corpus of a few thousand chunks a full scan is a single
matrix-vector product and takes well under a millisecond; FAISS would add a
dependency, a build step and an approximation error to save time that is not
being spent. The `VectorIndex` interface is narrow enough that swapping in an
ANN backend later is a one-file change -- worth doing somewhere north of a
million chunks.

BM25 is implemented directly (it is about forty lines) so the scoring is
inspectable when a retrieval goes wrong, which is most of what debugging a RAG
system consists of.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from .chunking import Chunk
from .embeddings import tokenize


class VectorIndex:
    def __init__(self, vectors: np.ndarray) -> None:
        self.vectors = vectors

    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[int, float]]:
        if len(self.vectors) == 0:
            return []
        scores = self.vectors @ query_vector  # both sides are L2-normalised
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(int(i), float(scores[i])) for i in top]


class BM25Index:
    """Okapi BM25 over the same chunk set."""

    def __init__(self, documents: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_len = [len(d) for d in documents]
        self.avg_len = (sum(self.doc_len) / len(self.doc_len)) if documents else 0.0
        self.term_freqs = [Counter(d) for d in documents]

        doc_freq: Counter[str] = Counter()
        for tf in self.term_freqs:
            doc_freq.update(tf.keys())

        n = len(documents)
        self.idf = {
            term: math.log(1 + (n - df + 0.5) / (df + 0.5))
            for term, df in doc_freq.items()
        }
        self.postings: dict[str, list[int]] = {}
        for i, tf in enumerate(self.term_freqs):
            for term in tf:
                self.postings.setdefault(term, []).append(i)

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        terms = tokenize(query)
        candidates: set[int] = set()
        for term in terms:
            candidates.update(self.postings.get(term, ()))

        scored: list[tuple[int, float]] = []
        for i in candidates:
            tf = self.term_freqs[i]
            length = self.doc_len[i] or 1
            score = 0.0
            for term in terms:
                f = tf.get(term)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * length / (self.avg_len or 1))
                score += self.idf.get(term, 0.0) * f * (self.k1 + 1) / denom
            if score > 0:
                scored.append((i, score))

        scored.sort(key=lambda p: -p[1])
        return scored[:k]


class Store:
    """Chunks plus both indexes, persisted together so they cannot drift apart."""

    def __init__(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunk/vector count mismatch")
        self.chunks = chunks
        self.vector_index = VectorIndex(vectors)
        self.bm25 = BM25Index([tokenize(c.embed_text) for c in chunks])

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, vectors=self.vector_index.vectors)
        meta = [
            {
                "id": c.id,
                "text": c.text,
                "embed_text": c.embed_text,
                "source": c.source,
                "heading": c.heading,
                "position": c.position,
            }
            for c in self.chunks
        ]
        path.with_suffix(".chunks.json").write_text(
            json.dumps(meta, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> "Store":
        path = Path(path)
        meta_path = path.with_suffix(".chunks.json")
        if not path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"No index at {path}. Build one first: python -m scripts.ingest"
            )
        vectors = np.load(path)["vectors"]
        raw = json.loads(meta_path.read_text(encoding="utf-8"))
        chunks = [Chunk(**item) for item in raw]
        return cls(chunks, vectors)
