"""Build the index from the corpus.

    python -m scripts.ingest --corpus corpus

Chunk every document, embed the chunks, and write vectors and chunk metadata
to a single file pair so the two cannot drift apart -- an index whose vectors
outlive the chunks they were built from returns confident citations pointing at
the wrong text, which is the worst failure mode this system has.

Re-run after changing the corpus, the chunk size, or the embedder. Switching
embedders without re-ingesting silently compares vectors from two different
spaces; the check below refuses that rather than returning nonsense.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from app.chunking import load_corpus
from app.config import config
from app.embeddings import build_embedder
from app.index import Store


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunk and index the corpus.")
    parser.add_argument("--corpus", default=config.corpus_dir)
    parser.add_argument("--out", default=config.index_path)
    parser.add_argument("--chunk-tokens", type=int, default=config.chunk_tokens)
    parser.add_argument("--chunk-overlap", type=int, default=config.chunk_overlap)
    args = parser.parse_args()

    started = time.perf_counter()

    chunks = load_corpus(args.corpus, args.chunk_tokens, args.chunk_overlap)
    print(f"chunked {len(chunks)} chunks from {args.corpus}")

    embedder = build_embedder(config.embedder, config.embed_model, config.embed_dim)
    vectors = embedder.encode([c.embed_text for c in chunks])
    print(f"embedded with {config.embedder} -> {vectors.shape[1]} dimensions")

    if config.embedder == "hash":
        print(
            "\n  note: hash embeddings are for offline testing. Cosine values "
            "from them\n  are meaningless as magnitudes -- do not calibrate "
            "ABSTAIN_MIN_SIMILARITY\n  against this index.\n"
        )

    store = Store(chunks, vectors)
    store.save(args.out)

    sources = sorted({c.source for c in chunks})
    print(f"wrote {Path(args.out).resolve()} in {time.perf_counter() - started:.1f}s")
    print(f"sources: {', '.join(sources)}")


if __name__ == "__main__":
    main()
