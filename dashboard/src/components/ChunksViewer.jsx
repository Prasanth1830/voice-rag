import { useState, useEffect, useCallback } from 'react'

const API = '/api'
const PAGE_SIZE = 30

export default function ChunksViewer({ onDelete, addToast }) {
  const [chunks, setChunks] = useState([])
  const [loading, setLoading] = useState(false)
  const [search, setSearch] = useState('')
  const [offset, setOffset] = useState(0)
  const [hasMore, setHasMore] = useState(true)
  const [expanded, setExpanded] = useState(new Set())
  const [deleting, setDeleting] = useState(new Set())

  const fetchChunks = useCallback(async (reset = false) => {
    setLoading(true)
    const currentOffset = reset ? 0 : offset
    try {
      const res = await fetch(`${API}/chunks?limit=${PAGE_SIZE}&offset=${currentOffset}`)
      if (!res.ok) throw new Error('Failed to fetch chunks')
      const data = await res.json()
      if (reset) {
        setChunks(data)
        setOffset(PAGE_SIZE)
      } else {
        setChunks(prev => [...prev, ...data])
        setOffset(prev => prev + PAGE_SIZE)
      }
      setHasMore(data.length === PAGE_SIZE)
    } catch (err) {
      addToast?.('Failed to load chunks', 'error')
    } finally {
      setLoading(false)
    }
  }, [offset, addToast])

  useEffect(() => { fetchChunks(true) }, [])

  const deleteChunk = async (id) => {
    setDeleting(prev => new Set([...prev, id]))
    try {
      const res = await fetch(`${API}/chunks/${id}`, { method: 'DELETE' })
      if (!res.ok) throw new Error()
      setChunks(prev => prev.filter(c => c.id !== id))
      onDelete?.()
    } catch {
      addToast?.('Failed to delete chunk', 'error')
    } finally {
      setDeleting(prev => { const s = new Set(prev); s.delete(id); return s })
    }
  }

  const toggleExpand = (id) => {
    setExpanded(prev => {
      const s = new Set(prev)
      if (s.has(id)) s.delete(id)
      else s.add(id)
      return s
    })
  }

  const filtered = chunks.filter(c =>
    !search ||
    c.text.toLowerCase().includes(search.toLowerCase()) ||
    c.metadata?.source?.toLowerCase().includes(search.toLowerCase())
  )

  // Group by source
  const bySource = filtered.reduce((acc, chunk) => {
    const src = chunk.metadata?.source || 'Unknown'
    if (!acc[src]) acc[src] = []
    acc[src].push(chunk)
    return acc
  }, {})

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      <div className="card">
        <div className="card-header">
          <h2 className="card-title"><span>📦</span> Indexed Chunks
            <span className="tag tag-accent">{chunks.length} loaded</span>
          </h2>
          <div className="chunks-controls">
            <input
              className="search-input"
              placeholder="Search chunks or filenames…"
              value={search}
              onChange={e => setSearch(e.target.value)}
              id="search-chunks"
            />
            <button className="btn btn-secondary" onClick={() => fetchChunks(true)} disabled={loading} id="btn-refresh-chunks">
              {loading ? <div className="spinner" /> : '🔄'} Refresh
            </button>
          </div>
        </div>

        {loading && chunks.length === 0 ? (
          <div className="empty-state">
            <div className="spinner" style={{ width: 40, height: 40, margin: '0 auto 1rem' }} />
            <p className="empty-state-subtitle">Loading chunks…</p>
          </div>
        ) : filtered.length === 0 ? (
          <div className="empty-state">
            <span className="empty-state-icon">{search ? '🔍' : '📭'}</span>
            <p className="empty-state-title">{search ? 'No chunks match your search' : 'No chunks indexed yet'}</p>
            <p className="empty-state-subtitle">{search ? 'Try a different query' : 'Upload documents in the Upload tab to get started'}</p>
          </div>
        ) : (
          Object.entries(bySource).map(([source, sourceChunks]) => (
            <div key={source} style={{ marginBottom: '1.5rem' }}>
              {/* Source header */}
              <div style={{
                display: 'flex', alignItems: 'center', gap: '0.75rem',
                padding: '0.6rem 1rem', background: 'var(--bg-elevated)',
                borderRadius: 'var(--radius-sm)', marginBottom: '0.75rem',
                border: '1px solid var(--border)'
              }}>
                <span style={{ fontSize: '1.1rem' }}>📄</span>
                <span style={{ fontWeight: 600, fontSize: '0.9rem', flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{source}</span>
                <span className="tag">{sourceChunks.length} chunks</span>
              </div>

              <div className="chunks-list">
                {sourceChunks.map(chunk => (
                  <div className="chunk-item" key={chunk.id} id={`chunk-${chunk.id.slice(0, 8)}`}>
                    <div className="chunk-header">
                      <div className="chunk-meta">
                        <span className="tag tag-accent">#{chunk.metadata?.chunk_index ?? '?'}</span>
                        <span className="tag">{chunk.metadata?.file_type?.toUpperCase() || 'TXT'}</span>
                        {chunk.metadata?.total_chunks && (
                          <span className="tag">of {chunk.metadata.total_chunks}</span>
                        )}
                      </div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <button
                          className="btn btn-secondary btn-sm"
                          onClick={() => toggleExpand(chunk.id)}
                          id={`btn-expand-${chunk.id.slice(0, 8)}`}
                        >
                          {expanded.has(chunk.id) ? '▲ Less' : '▼ More'}
                        </button>
                        <button
                          className="btn btn-danger btn-sm"
                          onClick={() => deleteChunk(chunk.id)}
                          disabled={deleting.has(chunk.id)}
                          id={`btn-delete-${chunk.id.slice(0, 8)}`}
                        >
                          {deleting.has(chunk.id) ? <div className="spinner" style={{ width: 12, height: 12 }} /> : '🗑️'}
                        </button>
                      </div>
                    </div>
                    <p className={`chunk-text ${expanded.has(chunk.id) ? 'expanded' : ''}`}>
                      {chunk.text}
                    </p>
                    <div className="chunk-id">ID: {chunk.id}</div>
                  </div>
                ))}
              </div>
            </div>
          ))
        )}

        {hasMore && !search && (
          <div style={{ display: 'flex', justifyContent: 'center', marginTop: '1rem' }}>
            <button
              className="btn btn-secondary"
              onClick={() => fetchChunks(false)}
              disabled={loading}
              id="btn-load-more"
            >
              {loading ? <><div className="spinner" /> Loading…</> : 'Load more'}
            </button>
          </div>
        )}
      </div>
    </div>
  )
}
