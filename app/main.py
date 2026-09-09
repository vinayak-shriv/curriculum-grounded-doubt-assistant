"""FastAPI service.

The index and the embedding model are loaded once at startup and held in
module state. Loading MiniLM takes a couple of seconds; doing it per request
would dominate the latency budget.
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import config
from .db import Feedback, Session as SessionRow, make_session_factory, log_query
from .embeddings import build_embedder
from .generate import answer_question
from .index import Store
from .retrieval import retrieve, should_abstain

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["store"] = Store.load(config.index_path)
    state["embedder"] = build_embedder(config.embedder, config.embed_model, config.embed_dim)
    state["sessions"] = make_session_factory(config.database_url)
    yield
    state.clear()


app = FastAPI(title="Curriculum doubt-resolution assistant", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    session_id: str | None = None


class FeedbackRequest(BaseModel):
    query_id: int
    helpful: bool
    note: str = ""


@app.get("/health")
def health() -> dict:
    store = state.get("store")
    return {
        "status": "ok" if store else "loading",
        "chunks": len(store.chunks) if store else 0,
        "config": config.as_dict(),
    }


@app.post("/ask")
def ask(request: AskRequest) -> dict:
    started = time.perf_counter()

    session_id = request.session_id or str(uuid.uuid4())
    factory = state["sessions"]
    with factory() as db:
        if not db.get(SessionRow, session_id):
            db.add(SessionRow(id=session_id))
            db.commit()

    results = retrieve(request.question, state["store"], state["embedder"], config)
    abstain = should_abstain(results, config)

    try:
        answer = answer_question(
            request.question,
            results,
            abstain,
            model=config.answer_model,
            max_tokens=config.max_tokens,
        )
    except KeyError:
        raise HTTPException(500, "ANTHROPIC_API_KEY is not set on the server.")

    latency_ms = int((time.perf_counter() - started) * 1000)
    query_id = log_query(
        factory,
        session_id=session_id,
        question=request.question,
        answer=answer.text,
        abstained=answer.abstained,
        top_score=max((r.vector_score for r in results), default=0.0),
        retrieved_ids=[r.chunk.id for r in results],
        cited_indices=answer.cited_indices,
        latency_ms=latency_ms,
    )

    return {
        "query_id": query_id,
        "session_id": session_id,
        "answer": answer.text,
        "abstained": answer.abstained,
        "sources": answer.sources,
        "cited": answer.cited_indices,
        "latency_ms": latency_ms,
    }


@app.post("/feedback")
def feedback(request: FeedbackRequest) -> dict:
    with state["sessions"]() as db:
        db.add(Feedback(query_id=request.query_id, helpful=request.helpful, note=request.note))
        db.commit()
    return {"recorded": True}
