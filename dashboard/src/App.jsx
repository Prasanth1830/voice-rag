import { useState, useEffect, useCallback } from 'react'
import UploadDocs from './components/UploadDocs.jsx'
import ChunksViewer from './components/ChunksViewer.jsx'
import QueryTester from './components/QueryTester.jsx'

const TABS = [
  { id: 'upload',  label: 'Upload Docs',    icon: '📤' },
  { id: 'chunks',  label: 'Indexed Chunks', icon: '📦' },
  { id: 'query',   label: 'Query Tester',   icon: '🔍' },
]

const API = '/api'

export default function App() {
  const [activeTab, setActiveTab] = useState('upload')
  const [stats, setStats] = useState(null)
  const [apiOnline, setApiOnline] = useState(null)
  const [toasts, setToasts] = useState([])

  // ── Toast system ──────────────────────────────────────────────────────────
  const addToast = useCallback((message, type = 'info') => {
    const id = Date.now()
    setToasts(prev => [...prev, { id, message, type }])
    setTimeout(() => setToasts(prev => prev.filter(t => t.id !== id)), 4000)
  }, [])

  // ── Fetch stats ───────────────────────────────────────────────────────────
  const fetchStats = useCallback(async () => {
    try {
      const res = await fetch(`${API}/stats`)
      if (!res.ok) throw new Error()
      const data = await res.json()
      setStats(data)
      setApiOnline(true)
    } catch {
      setApiOnline(false)
    }
  }, [])

  useEffect(() => {
    fetchStats()
    const interval = setInterval(fetchStats, 15000)
    return () => clearInterval(interval)
  }, [fetchStats])

  return (
    <div className="app-layout">
      {/* ── Header ─────────────────────────────────────────────────────── */}
      <header className="header">
        <div className="header-inner">
          <div className="logo">
            <div className="logo-icon">🧠</div>
            <span className="logo-text">RAG Dashboard</span>
            <span className="logo-badge">Admin</span>
          </div>

          <nav>
            <ul className="nav-tabs">
              {TABS.map(tab => (
                <li key={tab.id}>
                  <button
                    className={`nav-tab ${activeTab === tab.id ? 'active' : ''}`}
                    onClick={() => setActiveTab(tab.id)}
                    id={`tab-${tab.id}`}
                  >
                    <span className="tab-icon">{tab.icon}</span>
                    <span>{tab.label}</span>
                  </button>
                </li>
              ))}
            </ul>
          </nav>

          <div className={`status-badge`}>
            <div className={`status-dot ${apiOnline === false ? 'offline' : ''}`} />
            <span>{apiOnline === null ? 'Connecting…' : apiOnline ? 'API Online' : 'API Offline'}</span>
          </div>
        </div>
      </header>

      {/* ── Main ───────────────────────────────────────────────────────── */}
      <main className="main-content">
        {/* Stats bar */}
        {stats && (
          <div className="stats-grid">
            <div className="stat-card">
              <div className="stat-icon">📦</div>
              <div className="stat-info">
                <div className="stat-value">{stats.total_chunks.toLocaleString()}</div>
                <div className="stat-label">Indexed Chunks</div>
              </div>
            </div>
            <div className="stat-card">
              <div className="stat-icon">🗄️</div>
              <div className="stat-info">
                <div className="stat-value" style={{fontSize:'1.1rem', marginTop:'4px'}}>{stats.vector_store}</div>
                <div className="stat-label">Vector Store</div>
              </div>
            </div>
            <div className="stat-card">
              <div className="stat-icon">⚡</div>
              <div className="stat-info">
                <div className="stat-value" style={{fontSize:'1.1rem', marginTop:'4px'}}>{stats.reranker_mode}</div>
                <div className="stat-label">Reranker Mode</div>
              </div>
            </div>
            <div className="stat-card">
              <div className="stat-icon">🤖</div>
              <div className="stat-info">
                <div className="stat-value" style={{fontSize:'0.95rem', marginTop:'4px'}}>{stats.chat_model}</div>
                <div className="stat-label">LLM Model</div>
              </div>
            </div>
          </div>
        )}

        {/* Tab content */}
        {activeTab === 'upload' && (
          <UploadDocs onUploadSuccess={() => { fetchStats(); addToast('Document indexed successfully!', 'success') }} addToast={addToast} />
        )}
        {activeTab === 'chunks' && (
          <ChunksViewer onDelete={() => { fetchStats(); addToast('Chunk deleted', 'info') }} addToast={addToast} />
        )}
        {activeTab === 'query' && (
          <QueryTester addToast={addToast} />
        )}
      </main>

      {/* ── Toasts ─────────────────────────────────────────────────────── */}
      <div className="toast-container">
        {toasts.map(t => (
          <div key={t.id} className={`toast toast-${t.type}`}>
            <span>{t.type === 'success' ? '✅' : t.type === 'error' ? '❌' : 'ℹ️'}</span>
            <span>{t.message}</span>
          </div>
        ))}
      </div>
    </div>
  )
}
