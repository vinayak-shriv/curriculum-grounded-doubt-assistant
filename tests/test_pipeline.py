"""Tests for the parts that break silently.

All of these run offline with the hash embedder -- no API key, no model
download -- so they are usable as a pre-commit check and in CI.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.chunking import chunk_document, load_corpus
from app.config import Config
from app.embeddings import HashEmbedder
from app.generate import build_context, clean_citations
from app.index import BM25Index, Store
from app.retrieval import reciprocal_rank_fusion, retrieve, should_abstain

DOC = """# Sorting

Sorting puts records into order.

## Merge sort

Merge sort divides the array in half, sorts each half, and merges them.
It runs in O(n log n) time in all cases and needs O(n) auxiliary space.

## Quicksort

Quicksort partitions around a pivot. Average case is O(n log n) but a bad
pivot choice on already-sorted input degrades it to O(n^2).
"""

OTHER = """# Indexing

A clustered index determines the physical order of rows in the table, so a
table can have only one. A non-clustered index is a separate structure holding
pointers back to the rows.
"""


@pytest.fixture
def store() -> Store:
    chunks = chunk_document(DOC, "sorting.md") + chunk_document(OTHER, "indexing.md")
    embedder = HashEmbedder(dim=128)
    vectors = embedder.encode([c.embed_text for c in chunks])
    return Store(chunks, vectors)


@pytest.fixture
def config() -> Config:
    return Config(embedder="hash", embed_dim=128, top_k_final=3,
                  abstain_min_similarity=0.0, abstain_min_bm25=0.0)


# --- chunking ---

def test_heading_path_travels_with_chunk():
    chunks = chunk_document(DOC, "sorting.md")
    merge = [c for c in chunks if "merges them" in c.text][0]
    assert merge.heading == "Sorting > Merge sort"
    # The heading path is prepended to what gets embedded, so a chunk whose body
    # only says "it" is still reachable by a question that names the section.
    assert merge.embed_text.startswith("sorting.md :: Sorting > Merge sort\n")
    assert "Sorting > Merge sort" not in merge.text


def test_chunks_respect_size_budget():
    long_doc = "# T\n\n" + "\n\n".join("word " * 80 for _ in range(10))
    chunks = chunk_document(long_doc, "long.md", chunk_tokens=100, chunk_overlap=20)
    assert len(chunks) > 1
    # Allow the overlap carry-over on top of the budget.
    assert all(len(c.text.split()) <= 100 + 80 for c in chunks)


def test_oversized_paragraph_is_split_not_dropped():
    doc = "# T\n\n" + "word " * 900
    chunks = chunk_document(doc, "big.md", chunk_tokens=100, chunk_overlap=0)
    assert len(chunks) >= 9
    assert sum(len(c.text.split()) for c in chunks) >= 900


def test_missing_corpus_directory_is_explicit():
    with pytest.raises(FileNotFoundError):
        load_corpus("no/such/dir")


# --- indexes ---

def test_bm25_matches_rare_exact_token(store: Store):
    hits = store.bm25.search("clustered", k=3)
    assert hits
    assert store.chunks[hits[0][0]].source == "indexing.md"


def test_bm25_ignores_unseen_terms():
    idx = BM25Index([["alpha", "beta"], ["gamma"]])
    assert idx.search("delta", k=5) == []


def test_vector_search_returns_sorted_scores(store: Store):
    q = HashEmbedder(dim=128).encode(["quicksort pivot partition"])[0]
    hits = store.vector_index.search(q, k=4)
    scores = [s for _, s in hits]
    assert scores == sorted(scores, reverse=True)


def test_store_roundtrip(tmp_path, store: Store):
    path = tmp_path / "index.npz"
    store.save(path)
    loaded = Store.load(path)
    assert [c.id for c in loaded.chunks] == [c.id for c in store.chunks]
    assert np.allclose(loaded.vector_index.vectors, store.vector_index.vectors)


def test_store_rejects_mismatched_vectors(store: Store):
    with pytest.raises(ValueError):
        Store(store.chunks, store.vector_index.vectors[:-1])


# --- fusion ---

def test_rrf_rewards_agreement_between_retrievers():
    # Doc 7 is second on both lists; doc 1 is first on one and absent from the
    # other. Agreement should win -- that is the whole point of fusing.
    fused = reciprocal_rank_fusion([[1, 7, 3], [9, 7, 4]], k=60)
    assert fused[7] > fused[1]


def test_rrf_score_is_rank_only():
    a = reciprocal_rank_fusion([[5]], k=60)
    b = reciprocal_rank_fusion([[99]], k=60)
    assert a[5] == b[99]


def test_hybrid_retrieval_finds_the_right_document(store: Store, config: Config):
    results = retrieve("clustered index physical order", store, HashEmbedder(128), config)
    assert results[0].chunk.source == "indexing.md"


def test_retrieved_carries_both_ranks(store: Store, config: Config):
    results = retrieve("merge sort", store, HashEmbedder(128), config)
    assert any(r.vector_rank is not None for r in results)
    assert any(r.lexical_rank is not None for r in results)


# --- abstention ---

def test_abstains_on_empty_results(config: Config):
    assert should_abstain([], config)


def test_gate_fires_when_both_signals_are_weak(store: Store):
    strict = Config(embedder="hash", embed_dim=128, top_k_final=3,
                    abstain_min_similarity=2.0, abstain_min_bm25=1e6)
    results = retrieve("merge sort", store, HashEmbedder(128), strict)
    assert should_abstain(results, strict)


def test_gate_holds_when_only_one_signal_is_weak(store: Store):
    # Cosine is unreachable, BM25 is trivially satisfied. One live signal is
    # enough to answer -- this is the case a single-signal gate gets wrong.
    lexical_only = Config(embedder="hash", embed_dim=128, top_k_final=3,
                          abstain_min_similarity=2.0, abstain_min_bm25=0.0)
    results = retrieve("merge sort", store, HashEmbedder(128), lexical_only)
    assert not should_abstain(results, lexical_only)


def test_raw_scores_survive_fusion(store: Store, config: Config):
    # The regression this guards: RRF scores are rank-only and identical across
    # queries, so the gate must read the raw scores, which are not.
    results = retrieve("quicksort pivot", store, HashEmbedder(128), config)
    assert any(r.vector_score > 0 for r in results)
    assert any(r.lexical_score > 0 for r in results)


# --- citations ---

def test_invalid_citation_markers_are_stripped():
    text, cited, dropped = clean_citations("Merge sort is O(n log n) [1], and quicksort [7].", {1, 2})
    assert "[7]" not in text
    assert cited == [1]
    assert dropped == [7]


def test_valid_citations_survive_untouched():
    text, cited, _ = clean_citations("Claim one [1]. Claim two [2].", {1, 2})
    assert text == "Claim one [1]. Claim two [2]."
    assert cited == [1, 2]


def test_context_numbering_is_one_based(store: Store, config: Config):
    results = retrieve("merge sort", store, HashEmbedder(128), config)
    context = build_context(results)
    assert context.startswith("[1] ")
    assert "[0]" not in context


def test_ordering_is_deterministic(store: Store, config: Config):
    # Ties in RRF are common; without an explicit tie-break the order depends on
    # dict insertion and eval runs stop being comparable.
    runs = [
        [r.chunk.id for r in retrieve("sorting index deadlock", store, HashEmbedder(128), config)]
        for _ in range(5)
    ]
    assert all(run == runs[0] for run in runs)


def test_ties_break_toward_higher_similarity(store: Store, config: Config):
    results = retrieve("quicksort pivot partition", store, HashEmbedder(128), config)
    for a, b in zip(results, results[1:]):
        if a.score == b.score:
            assert a.vector_score >= b.vector_score
