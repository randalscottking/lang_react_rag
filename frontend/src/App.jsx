import { useState } from "react";

export default function App() {
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function submit() {
    const q = question.trim();
    if (!q || loading) return;
    setLoading(true);
    setError("");
    setResult(null);
    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail || `Request failed (${res.status})`);
      }
      setResult(await res.json());
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  function onKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  }

  return (
    <main className="page">
      <textarea
        className="prompt"
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        onKeyDown={onKeyDown}
        placeholder="Ask anything…"
        rows={2}
        autoFocus
        disabled={loading}
      />
      <div className="status" aria-live="polite">
        {loading && <span className="muted">Thinking…</span>}
        {error && <span className="error">{error}</span>}
      </div>
      {result && (
        <section className="answer">
          <p>{result.answer}</p>
          {result.sources.length > 0 && (
            <p className="sources">{result.sources.join(" · ")}</p>
          )}
        </section>
      )}
    </main>
  );
}
