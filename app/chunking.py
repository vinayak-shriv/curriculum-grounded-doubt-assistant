"""Turning source documents into retrievable chunks.

Chunking is the decision that most affects retrieval quality, so it is kept
separate from everything else and is testable on its own.

The strategy is structure-first: split on markdown headings, then pack
paragraphs into windows of roughly `chunk_tokens` words with an overlap. The
heading path travels with the chunk and is prepended to its embedded text, so a
chunk that reads "It runs in O(n log n)" still carries "Sorting > Merge sort"
and can be retrieved by a question that names the algorithm.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass
class Chunk:
    id: str
    text: str          # raw chunk text, shown to the user as a citation
    embed_text: str    # heading path + text, what actually gets embedded
    source: str        # file name, for citation display
    heading: str       # e.g. "Sorting > Merge sort"
    position: int      # ordinal within the source file
    meta: dict = field(default_factory=dict)


def _words(text: str) -> list[str]:
    return text.split()


def _sections(markdown: str) -> list[tuple[str, str]]:
    """Split a markdown document into (heading_path, body) sections."""
    sections: list[tuple[str, str]] = []
    stack: list[str] = []
    buf: list[str] = []

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body:
            sections.append((" > ".join(stack), body))
        buf.clear()

    for line in markdown.splitlines():
        m = HEADING.match(line)
        if m:
            flush()
            level = len(m.group(1))
            stack[:] = stack[: level - 1]
            stack.append(m.group(2).strip())
        else:
            buf.append(line)
    flush()
    return sections


def chunk_document(
    markdown: str,
    source: str,
    chunk_tokens: int = 300,
    chunk_overlap: int = 60,
) -> list[Chunk]:
    """Split one document into overlapping, heading-aware chunks."""
    chunks: list[Chunk] = []
    position = 0

    for heading, body in _sections(markdown):
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
        window: list[str] = []
        length = 0

        def emit() -> None:
            nonlocal window, length, position
            if not window:
                return
            text = "\n\n".join(window).strip()
            prefix = f"{source} :: {heading}\n" if heading else f"{source}\n"
            chunks.append(
                Chunk(
                    id=f"{source}#{position}",
                    text=text,
                    embed_text=prefix + text,
                    source=source,
                    heading=heading,
                    position=position,
                )
            )
            position += 1
            # Carry the tail of this window into the next one so an answer that
            # straddles a boundary is not cut in half.
            tail: list[str] = []
            kept = 0
            for para in reversed(window):
                n = len(_words(para))
                if kept + n > chunk_overlap:
                    break
                tail.insert(0, para)
                kept += n
            window = tail
            length = kept

        for para in paragraphs:
            n = len(_words(para))
            # A single oversized paragraph gets hard-split rather than dropped.
            if n > chunk_tokens:
                emit()
                words = _words(para)
                for i in range(0, len(words), chunk_tokens):
                    window = [" ".join(words[i : i + chunk_tokens])]
                    length = chunk_tokens
                    emit()
                continue
            if length + n > chunk_tokens:
                emit()
            window.append(para)
            length += n
        emit()

    return chunks


def load_corpus(
    corpus_dir: str | Path,
    chunk_tokens: int = 300,
    chunk_overlap: int = 60,
) -> list[Chunk]:
    """Read every .md/.txt file under `corpus_dir` and chunk it."""
    root = Path(corpus_dir)
    if not root.exists():
        raise FileNotFoundError(f"No corpus directory at {root.resolve()}")

    chunks: list[Chunk] = []
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in {".md", ".txt"})
    if not files:
        raise ValueError(f"No .md or .txt files found under {root.resolve()}")

    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        chunks.extend(
            chunk_document(
                text,
                source=str(path.relative_to(root)),
                chunk_tokens=chunk_tokens,
                chunk_overlap=chunk_overlap,
            )
        )
    return chunks
