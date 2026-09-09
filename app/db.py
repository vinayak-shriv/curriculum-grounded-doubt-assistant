"""Persistence: sessions, query logs, feedback.

SQLAlchemy against SQLite by default and MySQL by setting DATABASE_URL, with
the same schema either way.

`queries` is the table that earns its keep. Every question is stored with the
chunks that were retrieved, the raw top score, whether the gate fired, and the
latency. That log is what turns "it felt worse this week" into a specific list
of queries you can pull into the eval set -- the alternative is re-running
questions from memory and guessing.

`retrieved_ids` and `cited_indices` are stored as JSON text rather than a join
table. They are read as a unit for a single query row and never joined against,
so a child table would buy nothing and cost a migration.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Session(Base):
    """One browser session. Groups queries so a thread can be replayed."""

    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)


class Query(Base):
    __tablename__ = "queries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(64), ForeignKey("sessions.id"), index=True)
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    abstained: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    # The raw cosine of the best hit -- the number the gate actually reads.
    # Logged so thresholds can be re-tuned against real traffic rather than
    # only against the eval set.
    top_score: Mapped[float] = mapped_column(Float, default=0.0)
    retrieved_ids: Mapped[str] = mapped_column(Text, default="[]")
    cited_indices: Mapped[str] = mapped_column(Text, default="[]")
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now, index=True)


class Feedback(Base):
    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    query_id: Mapped[int] = mapped_column(Integer, ForeignKey("queries.id"), index=True)
    helpful: Mapped[bool] = mapped_column(Boolean)
    note: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)


def make_session_factory(database_url: str) -> sessionmaker:
    """Create the engine, ensure the schema exists, return a session factory."""
    kwargs: dict = {"future": True, "pool_pre_ping": True}
    if database_url.startswith("sqlite"):
        # FastAPI serves requests from a thread pool; the default SQLite check
        # would reject a connection reused across threads.
        kwargs["connect_args"] = {"check_same_thread": False}
        kwargs.pop("pool_pre_ping")

    engine = create_engine(database_url, **kwargs)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)


def log_query(
    factory: sessionmaker,
    *,
    session_id: str,
    question: str,
    answer: str,
    abstained: bool,
    top_score: float,
    retrieved_ids: list[str],
    cited_indices: list[int],
    latency_ms: int,
) -> int:
    with factory() as db:
        row = Query(
            session_id=session_id,
            question=question,
            answer=answer,
            abstained=abstained,
            top_score=float(top_score),
            retrieved_ids=json.dumps(retrieved_ids),
            cited_indices=json.dumps(cited_indices),
            latency_ms=latency_ms,
        )
        db.add(row)
        db.commit()
        return int(row.id)
