# Ask the notes

[![CI](https://github.com/vinayak-shriv/curriculum-grounded-doubt-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/vinayak-shriv/curriculum-grounded-doubt-assistant/actions/workflows/ci.yml)

A retrieval-augmented assistant that answers questions from a set of course
notes, cites the passage each claim came from, and refuses to answer when the
notes do not cover the question.

The refusing is the part that took the work. A RAG system that always answers
is easy; one that knows when it cannot is what makes it usable for studying,
because a confident wrong answer about a topic you are still learning is worse
than no answer at all.

```
corpus/*.md ──► chunk ──► embed ──┬──► vector index ─┐
                                  │                  ├─► RRF ─► top-k ─► answer + citations
                                  └──► BM25 index ───┘             │
                                                                   └─► abstention gate
```

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # add your ANTHROPIC_API_KEY

python -m scripts.ingest --corpus corpus
uvicorn app.main:app --reload

cd frontend && npm install && npm run dev
```

The three files in `corpus/` are samples so the project runs on clone. Replace
them with your own notes as `.md` or `.txt` and re-run the ingest.

To run without downloading a model or calling an API, set `EMBEDDER=hash`. The
retrieval path works end to end; answers still need a key.

```bash
pytest                                  # 22 tests, offline, no key needed
python -m eval.run_eval --no-judge      # retrieval metrics only
python -m eval.run_eval                 # adds the graded metrics
```

## Design decisions

**Hybrid retrieval, fused by reciprocal rank.** Dense embeddings handle
paraphrase; a student asking "why is my loop slow" should reach a passage about
time complexity that never uses the word slow. BM25 handles the opposite case,
where the question names an exact identifier or error string that the embedding
smooths away. On the eval set the two retrievers miss on different questions,
which is the condition under which fusing actually helps rather than just
averaging.

RRF is used rather than a weighted sum of scores because cosine similarity and
BM25 live on incomparable scales, and any weighting between them needs
recalibrating whenever the corpus changes. RRF only reads rank position, so it
has one parameter and no calibration step.

**Brute-force vector search, not FAISS.** The corpus here is a few thousand
chunks. A full cosine scan is one matrix-vector product, well under a
millisecond, and the numbers are exact. FAISS would add a dependency, an index
build step and an approximation error to save time that was not being spent.
`VectorIndex` is a small enough interface that swapping in an ANN backend is a
one-file change, which is the point at which to do it — somewhere past a
million chunks.

**Abstention is enforced in code, not requested in the prompt.** When retrieval
comes back weak the model is never called. Telling a model to say "I don't
know" is a suggestion it follows most of the time; not making the call is a
guarantee. It is also faster and cheaper on exactly the queries that were never
going to produce a good answer.

**Citations are validated after generation.** The model cites `[n]` against a
numbered context block, and every marker is then checked against the numbers
actually supplied. Unknown markers are stripped rather than displayed. A
citation pointing at nothing is worse than no citation, because it looks
verified.

**SQLite by default, MySQL by env var.** Same SQLAlchemy schema either way. The
query log is the useful table: every question, the chunks retrieved, the raw
scores, whether it abstained. That log is what turns "it felt worse this week"
into a specific list of failing queries you can add to the eval set.

## The bug worth documenting

The first version of the abstention gate read the fused RRF score, which is
broken in a way that is not obvious until you test it. RRF discards score
magnitude by construction. Whatever document ranks first on both retrievers
scores exactly `2/(k+1)` — the same number whether it is a perfect match or the
least-bad chunk in a corpus that has nothing to do with the question. An
out-of-corpus question scored identically to a well-covered one and the gate
could never fire.

The fix separates the two signals by purpose. RRF decides the *order* of
results, where rank agreement is exactly the right signal. The raw cosine and
BM25 scores are carried through untouched and decide *whether to answer at
all*, since only they carry magnitude. Both gates must fail to trigger an
abstention: a question using none of the corpus vocabulary can still be a
semantic match, and a question naming an exact identifier can score well
lexically while the embedding misses it.

`test_raw_scores_survive_fusion` and `test_gate_holds_when_only_one_signal_is_weak`
guard the regression.

## Evaluation

`eval/questions.jsonl` holds labelled questions:

```json
{"question": "...", "expected_sources": ["sorting.md"], "answerable": true}
```

`eval/questions.example.jsonl` shows the format. Roughly a fifth of the set
should be `answerable: false` — questions deliberately outside the corpus.
Without them the abstention thresholds optimise to zero and the system
confidently answers everything.

Four metrics, each catching something the others miss:

| metric | question it answers |
| --- | --- |
| `recall@k` | Did the right chunk get retrieved at all? Caps everything downstream. |
| `mrr` | How high did it rank? Catches improvements recall cannot see. |
| `faithfulness` | Is every claim supported by the retrieved text? |
| `citation_validity` | Does each cited source actually support the claim attached to it? |

Plus `correct_abstentions` and `false_abstentions`, which move against each
other — that trade is what the gate thresholds are choosing between.

The judge is a model, so these are noisy in the third digit. Treat them as a
regression gate on a change, not as an absolute score.

### Measured on the sample corpus

`EMBEDDER=minilm python -m eval.run_eval --no-judge`, over the 15 labelled
questions in `eval/questions.jsonl` (12 answerable, 3 deliberately outside the
corpus):

| recall@5 | MRR | correct abstentions | false abstentions |
| ---: | ---: | ---: | ---: |
| 1.00 | 0.958 | 3/3 | 0/12 |

Fifteen questions over a three-file corpus is a calibration set, not a
benchmark. It is big enough to catch a regression and far too small to quote as
an accuracy figure.

### Tuning the abstention gate

1. Build the index with the real embedder — hash embeddings produce meaningless
   cosine values and will calibrate the similarity gate to nonsense.
2. Run `python -m eval.run_eval --no-judge` and read `top_score` on each row.
3. Set `ABSTAIN_MIN_SIMILARITY` between the highest score among your
   `answerable: false` rows and the lowest among your answerable ones. If those
   ranges overlap, the retriever cannot separate the two classes and the fix
   belongs upstream in chunking, not in the threshold.
4. Re-run with the judge and check `false_abstentions` did not climb.

## Layout

```
app/chunking.py    structure-aware splitting; heading path travels with chunk
app/embeddings.py  embedding backends behind one interface
app/index.py       brute-force vector search + BM25
app/retrieval.py   RRF fusion and the abstention gate
app/generate.py    grounded generation, citation validation
app/db.py          sessions, query logs, feedback
app/main.py        FastAPI service
eval/run_eval.py   offline evaluation
frontend/          React chat UI with linked citations
```

## Known limitations

- Single-turn. Follow-up questions are not rewritten against conversation
  history, so "why is that?" retrieves on those three words and fails.
- Chunking is markdown-structure-aware, which means it degrades to fixed-size
  windows on plain `.txt` with no headings.
- The judge and the answer model are the same family, which likely inflates
  faithfulness scores; a different judge model would be a fairer grader.
- No reranker. A cross-encoder over the fused top-20 would probably buy more
  than any further retrieval tuning.
