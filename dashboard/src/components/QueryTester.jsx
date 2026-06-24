import { useState, useRef } from 'react'

const API = '/api'

function ScoreBar({ score, type = 'rerank' }) {
  const pct = Math.max(0, Math.min(100, score * 100))
  return (
    <div className="score-bar-wrap">
      <span className="score-value">{score.toFixed(3)}</span>
      <div className="score-bar">
        <div
          className={`score-fill ${type === 'vector' ? 'vector' : ''}`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  )
}

export default function QueryTester({ addToast }) {
  const [query, setQuery] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [showAll, setShowAll] = useState(false)
  const textareaRef = useRef(null)

  const runQuery = async () => {
    if (!query.trim()) return
    setLoading(true)
    setResult(null)
    try {
      const res = await fetch(`${API}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: query.trim() }),
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || 'Query failed')
      setResult(data)
    } catch (err) {
      addToast?.(err.message || 'Query failed', 'error')
    } finally {
      setLoading(false)
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) runQuery()
  }

  const EXAMPLE_QUERIES = [
    'What is the main topic of the uploaded documents?',
    'Summarize the key findings',
    'What are the important dates mentioned?',
  ]

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      {/* Query input card */}
      <div className="card">
        <div className="card-header">
          <h2 className="card-title"><span>🔍</span> Query Tester</h2>
          <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>⌘ + Enter to submit</span>
        </div>

        <div className="query-form">
          <div className="query-input-wrap">
            <textarea
              ref={textareaRef}
              className="query-textarea"
              placeholder="Ask a question about your indexed documents…"
              value={query}
              onChange={e => setQuery(e.target.value)}
              onKeyDown={handleKeyDown}
              id="query-input"
            />
          </div>

          {/* Example queries */}
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
            {EXAMPLE_QUERIES.map(q => (
              <button
                key={q}
                className="btn btn-secondary btn-sm"
                onClick={() => setQuery(q)}
                id={`example-query-${q.slice(0, 10).replace(/\s/g, '-')}`}
              >
                {q.length > 40 ? q.slice(0, 40) + '…' : q}
              </button>
            ))}
          </div>

          <div className="query-actions">
            <button
              className="btn btn-primary btn-lg"
              onClick={runQuery}
              disabled={loading || !query.trim()}
              id="btn-run-query"
              style={{ flex: 1 }}
            >
              {loading
                ? <><div className="spinner" /> Running query…</>
                : '🚀 Run Query'}
            </button>
            {result && (
              <button className="btn btn-secondary" onClick={() => { setResult(null); setQuery('') }} id="btn-clear-query">
                Clear
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Results */}
      {loading && (
        <div className="card" style={{ textAlign: 'center', padding: '3rem' }}>
          <div className="spinner" style={{ width: 40, height: 40, margin: '0 auto 1rem' }} />
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
            Embedding → searching → reranking → generating answer…
          </p>
        </div>
      )}

      {result && !loading && (
        <>
          {/* Answer */}
          <div className="answer-box" id="answer-box">
            <div className="answer-label">🤖 AI Answer</div>
            <p className="answer-text">{result.answer}</p>
            <div className="answer-meta">
              <div className="answer-meta-item">🔁 Reranker: <strong style={{ color: 'var(--text-primary)' }}>{result.reranker_mode}</strong></div>
              <div className="answer-meta-item">🤖 Model: <strong style={{ color: 'var(--text-primary)' }}>{result.model}</strong></div>
              <div className="answer-meta-item">🪙 Tokens: <strong style={{ color: 'var(--text-primary)' }}>{result.tokens_used}</strong></div>
              <div className="answer-meta-item">📦 Sources: <strong style={{ color: 'var(--text-primary)' }}>{result.reranked_results.length} chunks</strong></div>
            </div>
          </div>

          {/* Side-by-side comparison */}
          <div className="card">
            <div className="card-header">
              <h3 className="card-title">
                <span>📊</span> Retrieval Comparison
                <span className="tag tag-accent">Vector Search vs Reranked</span>
              </h3>
              <button
                className="btn btn-secondary btn-sm"
                onClick={() => setShowAll(v => !v)}
                id="btn-toggle-all"
              >
                {showAll ? `Show top ${result.reranked_results.length}` : `Show all ${result.vector_results.length} vector results`}
              </button>
            </div>

            <div className="results-grid">
              {/* Vector results */}
              <div className="results-panel">
                <div className="results-panel-title">
                  <span>🔵</span> Vector Search
                  <span className="tag">{showAll ? result.vector_results.length : result.reranked_results.length} of {result.vector_results.length}</span>
                </div>
                {(showAll ? result.vector_results : result.vector_results.slice(0, result.reranked_results.length))
                  .map((r, idx) => (
                  <div
                    className={`result-item ${idx === 0 ? 'top-result' : ''}`}
                    key={r.id}
                    id={`vector-result-${idx}`}
                  >
                    <div className="result-item-header">
                      <div className="rank-badge">{idx + 1}</div>
                      <ScoreBar score={r.score} type="vector" />
                    </div>
                    <p className="result-text">{r.text}</p>
                    <div className="chunk-meta" style={{ marginTop: '0.5rem', gap: '0.4rem' }}>
                      <span className="tag">{r.metadata?.source || 'unknown'}</span>
                    </div>
                  </div>
                ))}
              </div>

              {/* Reranked results */}
              <div className="results-panel">
                <div className="results-panel-title">
                  <span>⚡</span> After Re-ranking
                  <span className="tag tag-success">{result.reranked_results.length} top chunks</span>
                </div>
                {result.reranked_results.map((r, idx) => (
                  <div
                    className={`result-item ${idx === 0 ? 'top-result' : ''}`}
                    key={r.id}
                    id={`reranked-result-${idx}`}
                  >
                    <div className="result-item-header">
                      <div className="rank-badge">{r.rank}</div>
                      <ScoreBar score={r.rerank_score} type="rerank" />
                    </div>
                    <p className="result-text">{r.text}</p>
                    <div className="chunk-meta" style={{ marginTop: '0.5rem', gap: '0.4rem' }}>
                      <span className="tag">{r.metadata?.source || 'unknown'}</span>
                      <span className="tag" style={{ color: 'var(--text-muted)', fontSize: '0.68rem' }}>
                        vec: {r.vector_score.toFixed(3)}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </>
      )}

      {/* How it works explainer */}
      {!result && !loading && (
        <div className="card">
          <div className="card-header">
            <h3 className="card-title"><span>⚙️</span> How the Pipeline Works</h3>
          </div>
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
            gap: '1rem',
          }}>
            {[
              { icon: '🔢', step: '1', title: 'Embed Query', desc: 'Your question is converted to a vector using OpenAI embeddings.' },
              { icon: '🗄️', step: '2', title: 'Vector Search', desc: `Top-K semantically similar chunks are fetched from the vector store.` },
              { icon: '⚡', step: '3', title: 'Re-rank', desc: 'A cross-encoder scores each chunk against your query. Top-N are selected.' },
              { icon: '🤖', step: '4', title: 'LLM Answer', desc: 'Reranked chunks are used as context to generate a grounded answer.' },
            ].map(({ icon, step, title, desc }) => (
              <div
                key={step}
                style={{
                  background: 'var(--bg-elevated)',
                  border: '1px solid var(--border)',
                  borderRadius: 'var(--radius-md)',
                  padding: '1.25rem',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '0.5rem',
                }}
              >
                <div style={{ fontSize: '1.5rem' }}>{icon}</div>
                <div style={{ fontSize: '0.7rem', fontWeight: 700, color: 'var(--accent-light)', textTransform: 'uppercase', letterSpacing: '0.08em' }}>Step {step}</div>
                <div style={{ fontWeight: 600, fontSize: '0.9rem' }}>{title}</div>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', lineHeight: 1.6 }}>{desc}</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
