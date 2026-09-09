"""Configuration, read once from the environment.

Everything tunable lives here rather than as literals scattered through the
retrieval code, because the abstention thresholds have to be recalibrated per
corpus and a threshold you cannot find is a threshold nobody re-tunes.

`Config` is a plain dataclass with defaults on every field so tests can build
one directly -- `Config(embedder="hash", abstain_min_similarity=0.0)` -- without
touching the process environment. The module-level `config` is the one the
application uses, built from `.env`.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, fields

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    # --- embedding ---
    embedder: str = "minilm"          # "minilm" | "hash"
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embed_dim: int = 384              # only consulted by the hash backend

    # --- chunking ---
    chunk_tokens: int = 300
    chunk_overlap: int = 60

    # --- retrieval ---
    top_k_vector: int = 20
    top_k_lexical: int = 20
    top_k_final: int = 5
    rrf_k: int = 60

    # --- abstention gate ---
    # Read against the RAW cosine and BM25 scores, never the fused RRF score,
    # which is rank-only and identical across queries. See retrieval.py.
    # Calibrated against eval/questions.example.jsonl with the MiniLM embedder:
    # the out-of-corpus questions top out at 0.214 cosine and the weakest
    # answerable one sits at 0.407, so 0.32 falls in the gap. BM25 does NOT
    # separate the two classes on a corpus this small -- common English words
    # give an out-of-corpus question a respectable lexical score -- so its
    # threshold is set above the unanswerable range rather than in a gap that
    # does not exist. Recalibrate both against your own corpus; see README.
    abstain_min_similarity: float = 0.32
    abstain_min_bm25: float = 5.0

    # --- generation ---
    answer_model: str = "claude-sonnet-5"
    judge_model: str = "claude-opus-5"
    max_tokens: int = 1024

    # --- storage ---
    index_path: str = "data/index.npz"
    database_url: str = "sqlite:///./app.db"
    corpus_dir: str = "corpus"

    def as_dict(self) -> dict:
        """Config as reported by /health and stamped into eval results.

        Every eval number is only meaningful next to the settings that produced
        it, so the summary carries them rather than relying on whoever ran it
        to remember.
        """
        return asdict(self)


def _env(name: str, default, cast):
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return cast(raw)
    except ValueError as exc:
        raise ValueError(f"{name}={raw!r} is not a valid {cast.__name__}") from exc


def _bool(raw: str) -> bool:
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def from_env() -> Config:
    defaults = {f.name: f.default for f in fields(Config)}
    return Config(
        embedder=_env("EMBEDDER", defaults["embedder"], str),
        embed_model=_env("EMBED_MODEL", defaults["embed_model"], str),
        embed_dim=_env("EMBED_DIM", defaults["embed_dim"], int),
        chunk_tokens=_env("CHUNK_TOKENS", defaults["chunk_tokens"], int),
        chunk_overlap=_env("CHUNK_OVERLAP", defaults["chunk_overlap"], int),
        top_k_vector=_env("TOP_K_VECTOR", defaults["top_k_vector"], int),
        top_k_lexical=_env("TOP_K_LEXICAL", defaults["top_k_lexical"], int),
        top_k_final=_env("TOP_K_FINAL", defaults["top_k_final"], int),
        rrf_k=_env("RRF_K", defaults["rrf_k"], int),
        abstain_min_similarity=_env("ABSTAIN_MIN_SIMILARITY", defaults["abstain_min_similarity"], float),
        abstain_min_bm25=_env("ABSTAIN_MIN_BM25", defaults["abstain_min_bm25"], float),
        answer_model=_env("ANSWER_MODEL", defaults["answer_model"], str),
        judge_model=_env("JUDGE_MODEL", defaults["judge_model"], str),
        max_tokens=_env("MAX_TOKENS", defaults["max_tokens"], int),
        index_path=_env("INDEX_PATH", defaults["index_path"], str),
        database_url=_env("DATABASE_URL", defaults["database_url"], str),
        corpus_dir=_env("CORPUS_DIR", defaults["corpus_dir"], str),
    )


config = from_env()
