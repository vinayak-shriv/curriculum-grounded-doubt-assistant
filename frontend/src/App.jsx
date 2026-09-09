import { useState, useRef, useEffect } from "react";

const API = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

// Splits answer text on [n] markers so each one can be rendered as a control
// that highlights its source. The citation is the point of the interface --
// an ungrounded answer here is a bug, so the link back has to be one click.
function withCitations(text, onFocus) {
  return text.split(/(\[\d+\])/g).map((part, i) => {
    const match = part.match(/^\[(\d+)\]$/);
    if (!match) return <span key={i}>{part}</span>;
    const n = Number(match[1]);
    return (
      <button
        key={i}
        className="cite"
        onClick={() => onFocus(n)}
        aria-label={`Show source ${n}`}
      >
        {n}
      </button>
    );
  });
}

export default function App() {
  const [question, setQuestion] = useState("");
  const [turns, setTurns] = useState([]);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState(null);
  const [focused, setFocused] = useState(null);
  const sessionId = useRef(null);
  const endRef = useRef(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [turns, pending]);

  async function ask() {
    const q = question.trim();
    if (!q || pending) return;

    setPending(true);
    setError(null);
    setQuestion("");
    setFocused(null);

    try {
      const res = await fetch(`${API}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q, session_id: sessionId.current }),
      });
      if (!res.ok) {
        // The server sends a specific reason (a missing key, an unbuilt
        // index). Surface it rather than collapsing every failure into
        // "can't reach the server", which sends you debugging the wrong end.
        const detail = await res.json().then((d) => d.detail).catch(() => null);
        throw new Error(detail || `Server returned ${res.status}`, { cause: "server" });
      }
      const data = await res.json();
      sessionId.current = data.session_id;
      setTurns((t) => [...t, { question: q, ...data }]);
    } catch (e) {
      setError(
        e.cause === "server"
          ? e.message
          : `Couldn't reach the assistant at ${API}. Check that the server is running.`
      );
      setQuestion(q);
    } finally {
      setPending(false);
    }
  }

  async function rate(queryId, helpful) {
    setTurns((t) =>
      t.map((turn) => (turn.query_id === queryId ? { ...turn, rated: helpful } : turn))
    );
    await fetch(`${API}/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query_id: queryId, helpful }),
    }).catch(() => {});
  }

  return (
    <main>
      <header>
        <h1>Ask the notes</h1>
        <p>
          Answers come only from the indexed course material. Every claim links
          to the passage it came from.
        </p>
      </header>

      {turns.length === 0 && !pending && (
        <p className="empty">
          Ask something the material covers &mdash; a definition, a trade-off, a
          why. If it isn&rsquo;t in the notes, you&rsquo;ll be told so rather
          than guessed at.
        </p>
      )}

      <ol className="thread">
        {turns.map((turn) => (
          <li key={turn.query_id}>
            <p className="asked">{turn.question}</p>

            <div className={turn.abstained ? "answer abstained" : "answer"}>
              {withCitations(turn.answer, setFocused)}
            </div>

            {turn.sources.length > 0 && !turn.abstained && (
              <ol className="sources">
                {turn.sources.map((s) => (
                  <li
                    key={s.n}
                    className={focused === s.n ? "source open" : "source"}
                    hidden={!turn.cited.includes(s.n) && focused !== s.n}
                  >
                    <span className="mark">{s.n}</span>
                    <div>
                      <p className="where">
                        {s.source}
                        {s.heading && <span className="head"> {s.heading}</span>}
                      </p>
                      {focused === s.n && <p className="excerpt">{s.excerpt}</p>}
                    </div>
                  </li>
                ))}
              </ol>
            )}

            <p className="meta">
              <span>{turn.latency_ms} ms</span>
              {turn.rated === undefined ? (
                <>
                  <button onClick={() => rate(turn.query_id, true)}>
                    This helped
                  </button>
                  <button onClick={() => rate(turn.query_id, false)}>
                    This missed
                  </button>
                </>
              ) : (
                <span className="thanks">
                  {turn.rated ? "Marked helpful" : "Marked as a miss"}
                </span>
              )}
            </p>
          </li>
        ))}
      </ol>

      {pending && <p className="working">Searching the notes&hellip;</p>}
      {error && <p className="error">{error}</p>}

      <div className="composer">
        <textarea
          rows={2}
          value={question}
          placeholder="Why does quicksort degrade to O(n²)?"
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              ask();
            }
          }}
        />
        <button className="send" onClick={ask} disabled={pending || !question.trim()}>
          Ask
        </button>
      </div>
    </main>
  );
}
