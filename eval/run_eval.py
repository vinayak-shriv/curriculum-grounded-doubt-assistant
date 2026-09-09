"""Offline evaluation.

    python -m eval.run_eval --questions eval/questions.jsonl

Four metrics, chosen because each one catches a failure the others miss:

  recall@k         Did the right chunk get retrieved at all? Everything
                   downstream is capped by this, so it is measured separately
                   from answer quality. When the answer is wrong, this number
                   says whether it is a retrieval bug or a generation bug.

  mrr              Where in the list did it land? Recall alone cannot tell you
                   that a fix pushed the right chunk from rank 5 to rank 1,
                   which matters when the context window is trimmed.

  faithfulness     Is every claim in the answer supported by the retrieved
                   text? Graded by a model that sees only the sources and the
                   answer, never the reference answer, so it is judging
                   groundedness rather than agreement.

  citation_validity  Of the claims that carry a citation, does the cited source
                   actually support them? A faithful answer with shuffled
                   citation numbers still misleads a learner who clicks through.

Plus abstention, measured on questions marked answerable: false, which are
deliberately outside the corpus. Without them the threshold optimises to zero
and the system confidently answers everything.

The judge is a model, so these numbers are noisy in the third digit. They are
useful as a regression gate on a change, not as an absolute score to quote.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
from pathlib import Path

from app.config import config
from app.embeddings import build_embedder
from app.generate import answer_question, build_context
from app.index import Store
from app.retrieval import retrieve, should_abstain

JUDGE_PROMPT = """You are grading a retrieval-augmented answer. You see the sources and the answer only.

Return strict JSON, no other text:
{"faithfulness": <0-1>, "citation_validity": <0-1>, "unsupported": ["..."]}

faithfulness: the fraction of factual claims in the answer that are supported by the sources. An answer that correctly says the sources do not cover something scores 1.
citation_validity: of the claims carrying a [n] citation, the fraction where source [n] actually supports that claim. If no claims carry citations, return 1.
unsupported: the claims you judged unsupported, quoted briefly."""


def judge(question: str, answer: str, context: str, model: str) -> dict:
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    message = client.messages.create(
        model=model,
        max_tokens=600,
        system=JUDGE_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Sources:\n{context}\n\nQuestion: {question}\n\nAnswer:\n{answer}",
            }
        ],
    )
    raw = "".join(b.text for b in message.content if b.type == "text")
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"faithfulness": 0.0, "citation_validity": 0.0, "unsupported": ["judge returned non-JSON"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default="eval/questions.jsonl")
    parser.add_argument("--out", default="eval/results.json")
    parser.add_argument("--no-judge", action="store_true", help="retrieval metrics only, no API calls")
    args = parser.parse_args()

    questions = [
        json.loads(line)
        for line in Path(args.questions).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    store = Store.load(config.index_path)
    embedder = build_embedder(config.embedder, config.embed_model, config.embed_dim)

    rows = []
    for item in questions:
        results = retrieve(item["question"], store, embedder, config)
        abstain = should_abstain(results, config)
        retrieved = [r.chunk.source for r in results]
        expected = item.get("expected_sources", [])
        answerable = item.get("answerable", True)

        hit = any(source in retrieved for source in expected) if expected else None
        rank = next(
            (i for i, s in enumerate(retrieved, start=1) if s in expected),
            None,
        )

        row = {
            "question": item["question"],
            "answerable": answerable,
            "abstained": abstain,
            "top_score": max((r.vector_score for r in results), default=0.0),
            "hit": hit,
            "rank": rank,
            "retrieved": retrieved,
        }

        if not args.no_judge and answerable and not abstain:
            answer = answer_question(
                item["question"], results, abstain,
                model=config.answer_model, max_tokens=config.max_tokens,
            )
            grade = judge(item["question"], answer.text, build_context(results), config.judge_model)
            row["answer"] = answer.text
            row["faithfulness"] = float(grade.get("faithfulness", 0.0))
            row["citation_validity"] = float(grade.get("citation_validity", 0.0))
            row["unsupported"] = grade.get("unsupported", [])
            row["dropped_citations"] = answer.dropped_citations

        rows.append(row)
        print(f"  {'ABSTAIN' if abstain else 'answer '}  rank={rank}  {item['question'][:60]}")

    answerable_rows = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    judged = [r for r in rows if "faithfulness" in r]

    def mean(values: list[float]) -> float:
        return round(statistics.fmean(values), 3) if values else 0.0

    summary = {
        "n": len(rows),
        f"recall@{config.top_k_final}": mean([1.0 if r["hit"] else 0.0 for r in answerable_rows if r["hit"] is not None]),
        "mrr": mean([1.0 / r["rank"] if r["rank"] else 0.0 for r in answerable_rows]),
        "faithfulness": mean([r["faithfulness"] for r in judged]),
        "citation_validity": mean([r["citation_validity"] for r in judged]),
        "correct_abstentions": mean([1.0 if r["abstained"] else 0.0 for r in unanswerable]),
        "false_abstentions": mean([1.0 if r["abstained"] else 0.0 for r in answerable_rows]),
        "config": config.as_dict(),
    }

    Path(args.out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    print("\n" + json.dumps({k: v for k, v in summary.items() if k != "config"}, indent=2))
    print(f"\nfull results -> {args.out}")


if __name__ == "__main__":
    main()
