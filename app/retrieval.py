"""Hybrid retrieval: BM25 and vector search fused by reciprocal rank.

Why both. Dense retrieval handles paraphrase -- "why is my loop slow" finding a
passage about time complexity -- but it is unreliable on rare exact tokens:
identifiers, error strings, API names. BM25 is the reverse. On the eval set the
two miss on different questions, which is exactly the case where fusing helps.

Why RRF rather than a weighted sum of scores. Cosine similarity and BM25 are on
incomparable scales, so a weighted sum needs per-corpus normalisation that has
to be re-tuned whenever the corpus changes. RRF only uses rank position, so it
has one parameter and no calibration step:

    score(d) = sum over retrievers of 1 / (k + rank(d))

The cost is that RRF discards score magnitude, and that cost is not academic.
The first version of this file gated abstention on the fused score, which is
broken: whatever document ranks first on both retrievers scores exactly
2/(k+1), whether it is a perfect match or the least-bad chunk in a corpus that
does not contain the answer. An out-of-corpus question scored identically to a
well-covered one and the gate never fired.

So the two signals are kept separate by purpose. RRF decides the *order* of
results, where rank agreement is exactly the right signal. The raw cosine
similarity and BM25 scores are carried through untouched and decide *whether to
answer at all*, since only they carry magnitude.
"""

from __future__ import annotations

from dataclasses import dataclass

from .chunking import Chunk
from .config import Config
from .embeddings import Embedder
from .index import Store


@dataclass
class Retrieved:
    chunk: Chunk
    score: float                # fused RRF score -- ordering only
    vector_score: float         # raw cosine similarity, 0 if not in vector hits
    lexical_score: float        # raw BM25 score, 0 if not in lexical hits
    vector_rank: int | None
    lexical_rank: int | None

    def citation(self) -> str:
        return f"{self.chunk.source} :: {self.chunk.heading}" if self.chunk.heading else self.chunk.source


def reciprocal_rank_fusion(ranked_lists: list[list[int]], k: int) -> dict[int, float]:
    fused: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return fused


def retrieve(query: str, store: Store, embedder: Embedder, config: Config) -> list[Retrieved]:
    query_vector = embedder.encode([query])[0]
    vector_hits = store.vector_index.search(query_vector, config.top_k_vector)
    lexical_hits = store.bm25.search(query, config.top_k_lexical)

    vector_score = {doc: s for doc, s in vector_hits}
    lexical_score = {doc: s for doc, s in lexical_hits}
    vector_ids = [i for i, _ in vector_hits]
    lexical_ids = [i for i, _ in lexical_hits]

    fused = reciprocal_rank_fusion([vector_ids, lexical_ids], config.rrf_k)
    vector_rank = {doc: r for r, doc in enumerate(vector_ids, start=1)}
    lexical_rank = {doc: r for r, doc in enumerate(lexical_ids, start=1)}

    # RRF produces ties readily -- rank 1 on one retriever and rank 3 on the
    # other scores exactly the same as the reverse. Left alone, ties resolve by
    # dict insertion order, which makes results non-reproducible across runs and
    # quietly poisons any eval comparison. Raw similarity breaks the tie, with
    # the chunk id as a final deterministic fallback.
    ordered = sorted(
        fused.items(),
        key=lambda p: (-p[1], -vector_score.get(p[0], 0.0), -lexical_score.get(p[0], 0.0), p[0]),
    )[: config.top_k_final]
    return [
        Retrieved(
            chunk=store.chunks[doc_id],
            score=score,
            vector_score=vector_score.get(doc_id, 0.0),
            lexical_score=lexical_score.get(doc_id, 0.0),
            vector_rank=vector_rank.get(doc_id),
            lexical_rank=lexical_rank.get(doc_id),
        )
        for doc_id, score in ordered
    ]


def should_abstain(results: list[Retrieved], config: Config) -> bool:
    """True when no retriever found anything strong enough to ground an answer.

    Both signals must fail. Either one alone is too brittle: a question that
    uses none of the corpus vocabulary can still be a semantic match, and a
    question naming an exact identifier can score well lexically while the
    embedding misses it entirely.
    """
    if not results:
        return True
    best_similarity = max(r.vector_score for r in results)
    best_lexical = max(r.lexical_score for r in results)
    return (
        best_similarity < config.abstain_min_similarity
        and best_lexical < config.abstain_min_bm25
    )
