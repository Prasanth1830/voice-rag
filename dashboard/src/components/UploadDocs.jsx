import { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'

const API = '/api'

const ACCEPTED = {
  'application/pdf': ['.pdf'],
  'text/plain': ['.txt'],
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`
}

function fileIcon(name) {
  const ext = name.split('.').pop().toLowerCase()
  if (ext === 'pdf') return '📄'
  if (ext === 'docx') return '📝'
  return '📃'
}

export default function UploadDocs({ onUploadSuccess, addToast }) {
  const [queue, setQueue] = useState([])    // { file, status, result, error }
  const [uploading, setUploading] = useState(false)

  const onDrop = useCallback((accepted) => {
    const newItems = accepted.map(file => ({ file, status: 'pending', result: null, error: null }))
    setQueue(prev => [...prev, ...newItems])
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: ACCEPTED,
    multiple: true,
  })

  const uploadAll = async () => {
    setUploading(true)
    for (let i = 0; i < queue.length; i++) {
      if (queue[i].status !== 'pending') continue

      setQueue(prev => prev.map((item, idx) =>
        idx === i ? { ...item, status: 'loading' } : item
      ))

      const formData = new FormData()
      formData.append('file', queue[i].file)

      try {
        let res
        try {
          res = await fetch(`${API}/upload`, { method: 'POST', body: formData })
        } catch {
          throw new Error('Cannot reach backend — is the API server running on port 8000?')
        }

        let data
        try {
          data = await res.json()
        } catch {
          throw new Error(`Server error (status ${res.status}) — check that the API is running`)
        }

        if (!res.ok) throw new Error(data.detail || 'Upload failed')

        setQueue(prev => prev.map((item, idx) =>
          idx === i ? { ...item, status: 'success', result: data } : item
        ))
        onUploadSuccess?.()
      } catch (err) {
        setQueue(prev => prev.map((item, idx) =>
          idx === i ? { ...item, status: 'error', error: err.message } : item
        ))
        addToast?.(`Failed: ${queue[i].file.name} — ${err.message}`, 'error')
      }
    }
    setUploading(false)
  }

  const clearCompleted = () => setQueue(prev => prev.filter(i => i.status === 'pending' || i.status === 'loading'))
  const clearAll = () => setQueue([])

  const pendingCount = queue.filter(i => i.status === 'pending').length

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      <div className="card">
        <div className="card-header">
          <h2 className="card-title"><span>📤</span> Upload Documents</h2>
          {queue.length > 0 && (
            <div style={{ display: 'flex', gap: '0.5rem' }}>
              <button className="btn btn-secondary btn-sm" onClick={clearCompleted} id="btn-clear-completed">
                Clear done
              </button>
              <button className="btn btn-secondary btn-sm" onClick={clearAll} id="btn-clear-all">
                Clear all
              </button>
            </div>
          )}
        </div>

        {/* Drop zone */}
        <div {...getRootProps()} className={`dropzone ${isDragActive ? 'active' : ''}`} id="dropzone">
          <input {...getInputProps()} id="file-input" />
          <span className="dropzone-icon">☁️</span>
          <p className="dropzone-title">
            {isDragActive ? 'Drop files here…' : 'Drag & drop your documents here'}
          </p>
          <p className="dropzone-subtitle">or <strong style={{ color: 'var(--accent-light)' }}>click to browse</strong></p>
          <p className="dropzone-types">PDF · TXT · DOCX · Up to 50 MB each</p>
        </div>

        {/* Queue */}
        {queue.length > 0 && (
          <>
            <div className="upload-queue">
              {queue.map((item, idx) => (
                <div className="upload-item" key={idx} id={`upload-item-${idx}`}>
                  <span className="upload-item-icon">{fileIcon(item.file.name)}</span>
                  <div className="upload-item-info">
                    <div className="upload-item-name">{item.file.name}</div>
                    <div className="upload-item-size">{formatBytes(item.file.size)}</div>
                    {item.status === 'loading' && (
                      <div className="progress-bar">
                        <div className="progress-fill indeterminate" />
                      </div>
                    )}
                    {item.status === 'success' && item.result && (
                      <div className="upload-item-size" style={{ color: 'var(--success)' }}>
                        ✓ {item.result.total_chunks} chunks indexed
                      </div>
                    )}
                    {item.status === 'error' && (
                      <div className="upload-item-size" style={{ color: 'var(--danger)' }}>
                        ✗ {item.error}
                      </div>
                    )}
                  </div>
                  <div className={`upload-item-status status-${item.status}`}>
                    {item.status === 'loading' && <><div className="spinner" style={{ width: 14, height: 14 }} /> Processing</>}
                    {item.status === 'pending' && '⏳ Pending'}
                    {item.status === 'success' && '✅ Done'}
                    {item.status === 'error' && '❌ Failed'}
                  </div>
                </div>
              ))}
            </div>

            <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: '1.25rem' }}>
              <button
                className="btn btn-primary btn-lg"
                onClick={uploadAll}
                disabled={uploading || pendingCount === 0}
                id="btn-upload-all"
              >
                {uploading
                  ? <><div className="spinner" /> Processing…</>
                  : `⬆️ Upload ${pendingCount > 0 ? pendingCount : ''} File${pendingCount !== 1 ? 's' : ''}`}
              </button>
            </div>
          </>
        )}
      </div>

      {/* Tips card */}
      <div className="card">
        <div className="card-header">
          <h3 className="card-title"><span>💡</span> Tips</h3>
        </div>
        <ul style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem', paddingLeft: '1.25rem', color: 'var(--text-secondary)', fontSize: '0.875rem' }}>
          <li>Upload multiple files at once — they'll be processed sequentially.</li>
          <li>Large documents are automatically split into overlapping chunks for better retrieval.</li>
          <li>Duplicate documents are detected via SHA-256 hash — re-uploading is safe.</li>
          <li>After uploading, switch to <strong style={{ color: 'var(--text-primary)' }}>Indexed Chunks</strong> to verify storage.</li>
          <li>Test retrieval quality in the <strong style={{ color: 'var(--text-primary)' }}>Query Tester</strong> tab.</li>
        </ul>
      </div>
    </div>
  )
}
