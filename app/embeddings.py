"""Embedding backends behind one interface.

Two implementations:

`MiniLMEmbedder` is the real one -- sentence-transformers, 384 dimensions,
what the deployed system uses.

`HashEmbedder` is a deterministic bag-of-words projection with no model
download and no network. It exists so the whole retrieval path is testable
offline and in CI: the tests below run against it in about a second, which is
the difference between a test suite that gets run and one that does not.

It is not a substitute for real embeddings. Hashing has no notion of
similarity beyond exact token overlap, so paraphrase retrieval -- the entire
reason for having a dense retriever -- does not work under it, and the cosine
values it produces are meaningless as magnitudes. Calibrating
`ABSTAIN_MIN_SIMILARITY` against hash embeddings would set the gate to noise;
build the index with the real embedder before tuning it.

Both return L2-normalised float32 rows, so `VectorIndex` can treat a dot
product as cosine similarity.
"""

from __future__ import annotations

import re
from typing import Protocol

import numpy as np

TOKEN = re.compile(r"[a-z0-9_]+")


def tokenize(text: str) -> list[str]:
    """Shared by BM25 and the hash embedder so the two see the same terms.

    Deliberately simple: lowercase, alphanumeric-and-underscore runs. No
    stemming, because identifiers and error strings are exactly what BM25 is
    here to catch and stemming mangles them.
    """
    return TOKEN.findall(text.lower())


def normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # an empty chunk would otherwise divide by zero
    return (matrix / norms).astype(np.float32)


class Embedder(Protocol):
    dim: int

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return one L2-normalised row per input text."""
        ...


class HashEmbedder:
    """Deterministic hashed bag-of-words. Offline, no model, no network.

    Counts are hashed into non-negative buckets, so every vector sits in the
    positive orthant and cosine similarity stays in [0, 1]. That keeps the
    abstention gate's comparisons meaningful under test even though the values
    themselves carry no semantic weight.
    """

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def encode(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for term in tokenize(text):
                # Seeded rather than Python's salted hash(), which varies per
                # process and would make a saved index unreadable by the next run.
                bucket = _stable_hash(term) % self.dim
                matrix[row, bucket] += 1.0
        return normalise(matrix)


class MiniLMEmbedder:
    """sentence-transformers. Loaded once; loading costs a couple of seconds."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer  # lazy: heavy import

        self.model = SentenceTransformer(model_name)
        self.dim = int(self.model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            batch_size=32,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return normalise(np.asarray(vectors, dtype=np.float32))


def _stable_hash(term: str) -> int:
    """FNV-1a. Small, dependency-free, and stable across processes."""
    h = 0x811C9DC5
    for byte in term.encode("utf-8"):
        h ^= byte
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


def build_embedder(kind: str, model_name: str, dim: int) -> Embedder:
    kind = (kind or "").strip().lower()
    if kind == "hash":
        return HashEmbedder(dim=dim)
    if kind in {"minilm", "sentence-transformers", "st"}:
        return MiniLMEmbedder(model_name)
    raise ValueError(f"Unknown EMBEDDER={kind!r}. Use 'minilm' or 'hash'.")
