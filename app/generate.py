"""Answer generation, grounded and citable.

Two things are enforced here rather than merely requested in the prompt:

1. Abstention. If retrieval is weak the model is never called at all -- the
   assistant says the material does not cover it. A prompt instruction to "say
   you don't know" is a suggestion; not making the call is a guarantee.

2. Citation validity. The model cites sources as [1], [2] against a numbered
   context block. After generation, every marker is checked against the numbers
   actually supplied, and unknown markers are stripped rather than shown. A
   citation that points at nothing is worse than no citation, because it looks
   verified.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

from .retrieval import Retrieved

CITATION = re.compile(r"\[(\d+)\]")

SYSTEM_PROMPT = """You answer questions using only the numbered source material provided.

Rules:
- Use only the sources given. Do not add facts from your own knowledge, even if you are confident they are correct.
- Cite the source for each claim with its bracketed number, like [2]. Cite the specific source the claim came from, not every source you were given.
- If the sources only partially cover the question, answer the part they cover and say plainly which part they do not.
- If the sources do not answer the question at all, say so in one sentence. Do not guess.
- Be direct. No preamble, no restating the question."""


ABSTENTION_MESSAGE = (
    "That isn't covered in the indexed material. Try rephrasing it, or check "
    "whether the relevant notes have been added to the corpus."
)


@dataclass
class Answer:
    text: str
    sources: list[dict]
    abstained: bool
    cited_indices: list[int]
    dropped_citations: list[int]


def build_context(results: list[Retrieved]) -> str:
    blocks = []
    for i, r in enumerate(results, start=1):
        blocks.append(f"[{i}] {r.citation()}\n{r.chunk.text}")
    return "\n\n---\n\n".join(blocks)


def clean_citations(text: str, valid: set[int]) -> tuple[str, list[int], list[int]]:
    """Strip citation markers that do not correspond to a supplied source."""
    cited: list[int] = []
    dropped: list[int] = []

    def replace(match: re.Match) -> str:
        n = int(match.group(1))
        if n in valid:
            if n not in cited:
                cited.append(n)
            return match.group(0)
        if n not in dropped:
            dropped.append(n)
        return ""

    cleaned = CITATION.sub(replace, text)
    cleaned = re.sub(r" +([.,;:])", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip(), cited, dropped


def _source_payload(results: list[Retrieved]) -> list[dict]:
    return [
        {
            "n": i,
            "source": r.chunk.source,
            "heading": r.chunk.heading,
            "score": round(r.score, 5),
            "vector_rank": r.vector_rank,
            "lexical_rank": r.lexical_rank,
            "excerpt": r.chunk.text[:400],
        }
        for i, r in enumerate(results, start=1)
    ]


def answer_question(
    question: str,
    results: list[Retrieved],
    abstain: bool,
    model: str,
    max_tokens: int,
) -> Answer:
    if abstain:
        return Answer(
            text=ABSTENTION_MESSAGE,
            sources=_source_payload(results),
            abstained=True,
            cited_indices=[],
            dropped_citations=[],
        )

    import anthropic  # imported lazily so the retrieval path needs no API key

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    context = build_context(results)

    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Sources:\n\n{context}\n\n---\n\nQuestion: {question}",
            }
        ],
    )
    raw = "".join(block.text for block in message.content if block.type == "text")
    valid = set(range(1, len(results) + 1))
    cleaned, cited, dropped = clean_citations(raw, valid)

    return Answer(
        text=cleaned,
        sources=_source_payload(results),
        abstained=False,
        cited_indices=cited,
        dropped_citations=dropped,
    )
