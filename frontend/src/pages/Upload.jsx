// frontend/src/pages/Upload.jsx
// Admin-only drag-and-drop multi-file PDF upload.
// Shows list of already uploaded GRs below the drop zone.

import { useState, useEffect, useRef, useCallback } from 'react'
import { FileText, Download, Trash2 } from 'lucide-react'
import client from '../api/client'

export default function Upload() {
  const [files,       setFiles]       = useState([])   // uploaded GR list from backend
  const [stats,       setStats]       = useState(null)
  const [loading,     setLoading]     = useState(true)
  const [error,       setError]       = useState('')
  const [dragActive,  setDragActive]  = useState(false)
  const [uploadQueue, setUploadQueue] = useState([])    // [{name, status, message}]

  const fileInputRef = useRef(null)

  useEffect(() => {
    loadUploads()
  }, [])

  const loadUploads = async () => {
    setLoading(true)
    setError('')
    try {
      const [listRes, statsRes] = await Promise.all([
        client.get('/upload/list'),
        client.get('/upload/stats'),
      ])
      setFiles(listRes.data.files || listRes.data.uploads || [])
      setStats(statsRes.data || null)
    } catch (err) {
      setError(err.response?.data?.detail || 'Could not load uploaded documents.')
    } finally {
      setLoading(false)
    }
  }

  const uploadOne = async (file) => {
    setUploadQueue(prev => [...prev, { name: file.name, status: 'uploading', message: 'Uploading…' }])

    const formData = new FormData()
    formData.append('file', file)

    try {
      const res = await client.post('/upload/', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })
      setUploadQueue(prev => prev.map(q =>
        q.name === file.name
          ? { ...q, status: 'done', message: `${res.data.page_count || '?'} pages, ${res.data.size_kb || '?'} KB` }
          : q
      ))
      return true
    } catch (err) {
      setUploadQueue(prev => prev.map(q =>
        q.name === file.name
          ? { ...q, status: 'failed', message: err.response?.data?.detail || 'Upload failed' }
          : q
      ))
      return false
    }
  }

  const handleFiles = async (fileList) => {
    const pdfFiles = Array.from(fileList).filter(f => f.type === 'application/pdf' || f.name.toLowerCase().endsWith('.pdf'))
    if (pdfFiles.length === 0) {
      setError('Only PDF files are supported.')
      return
    }
    setError('')

    // Upload sequentially — avoids hammering the backend with parallel writes
    for (const file of pdfFiles) {
      await uploadOne(file)
    }

    await loadUploads()
  }

  const onDrop = useCallback((e) => {
    e.preventDefault()
    setDragActive(false)
    if (e.dataTransfer.files?.length) handleFiles(e.dataTransfer.files)
  }, [])

  const onDragOver = (e) => { e.preventDefault(); setDragActive(true) }
  const onDragLeave = (e) => { e.preventDefault(); setDragActive(false) }

  const onFileInputChange = (e) => {
    if (e.target.files?.length) handleFiles(e.target.files)
    e.target.value = '' // allow re-uploading same filename later
  }

  const deleteFile = async (filename) => {
    if (!window.confirm(`Delete "${filename}"? This cannot be undone.`)) return
    try {
      await client.delete(`/upload/${encodeURIComponent(filename)}`)
      setFiles(prev => prev.filter(f => f.filename !== filename))
    } catch (err) {
      setError(err.response?.data?.detail || 'Could not delete file.')
    }
  }

  const formatDate = (iso) => {
    if (!iso) return 'Unknown'
    try {
      return new Date(iso).toLocaleDateString('en-IN', { day: 'numeric', month: 'short', year: 'numeric' })
    } catch { return iso }
  }

  const clearFinishedUploads = () => {
    setUploadQueue(prev => prev.filter(q => q.status === 'uploading'))
  }

  return (
    <div style={{
      minHeight: '100%', background: '#0F1117',
      padding: '28px 32px', fontFamily: 'Inter, sans-serif',
    }}>

      {/* Header */}
      <div style={{ marginBottom: '20px' }}>
        <h1 style={{ color: '#E8EAF0', fontSize: '1.4rem', margin: '0 0 4px', fontWeight: '700' }}>
          Upload GR Documents
        </h1>
        <p style={{ color: '#555', fontSize: '0.85rem', margin: 0 }}>
          Drag and drop PDF files, or click to browse. Multiple files supported.
        </p>
      </div>

      {/* Stats bar */}
      {stats && (
        <div style={{ display: 'flex', gap: '12px', marginBottom: '20px', flexWrap: 'wrap' }}>
          <StatChip label="Total Uploaded" value={stats.total_uploaded ?? files.length} color="#6FB3FF" />
          <StatChip label="Embedded" value={stats.total_embedded ?? '—'} color="#4ADE80" />
          <StatChip label="Total Pages" value={stats.total_pages ?? '—'} color="#FACC15" />
        </div>
      )}

      {/* Error */}
      {error && (
        <div style={{
          background: '#2A0F0F', border: '1px solid #EF4444', borderRadius: '8px',
          padding: '12px 16px', color: '#EF4444', fontSize: '0.85rem', marginBottom: '20px',
        }}>
          {error}
        </div>
      )}

      {/* Drop zone */}
      <div
        onDrop={onDrop}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onClick={() => fileInputRef.current?.click()}
        style={{
          border: `2px dashed ${dragActive ? '#FF6B00' : '#2A2D3E'}`,
          borderRadius: '14px',
          background: dragActive ? '#1A1D2E' : '#13151F',
          padding: '48px 24px',
          textAlign: 'center',
          cursor: 'pointer',
          transition: 'border-color 0.15s, background 0.15s',
          marginBottom: '20px',
        }}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf,application/pdf"
          multiple
          onChange={onFileInputChange}
          style={{ display: 'none' }}
        />
        <div style={{ fontSize: '2.4rem', marginBottom: '10px' }}>
          {dragActive ? <Download size={38} /> : <FileText size={38} />}
        </div>
        <div style={{ color: '#E8EAF0', fontSize: '0.95rem', fontWeight: '600', marginBottom: '4px' }}>
          {dragActive ? 'Drop PDFs here' : 'Drag & drop GR PDFs here'}
        </div>
        <div style={{ color: '#555', fontSize: '0.82rem' }}>
          or click to browse — .pdf files only
        </div>
      </div>

      {/* Upload progress queue */}
      {uploadQueue.length > 0 && (
        <div style={{
          background: '#1A1D2E', border: '1px solid #2A2D3E', borderRadius: '12px',
          padding: '14px 16px', marginBottom: '24px',
        }}>
          <div style={{
            display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px',
          }}>
            <span style={{ color: '#B0B4C8', fontSize: '0.82rem', fontWeight: '600' }}>Upload Progress</span>
            {uploadQueue.every(q => q.status !== 'uploading') && (
              <button
                onClick={clearFinishedUploads}
                style={{
                  background: 'none', border: 'none', color: '#555',
                  fontSize: '0.75rem', cursor: 'pointer', fontFamily: 'Inter, sans-serif',
                }}
              >
                Clear
              </button>
            )}
          </div>
          {uploadQueue.map((q, i) => (
            <div key={i} style={{
              display: 'flex', justifyContent: 'space-between', alignItems: 'center',
              padding: '6px 0', borderBottom: i < uploadQueue.length - 1 ? '1px solid #2A2D3E' : 'none',
            }}>
              <span style={{ color: '#E8EAF0', fontSize: '0.82rem' }}>{q.name}</span>
              <span style={{
                fontSize: '0.75rem',
                color: q.status === 'done' ? '#4ADE80' : q.status === 'failed' ? '#EF4444' : '#FACC15',
                fontFamily: 'monospace',
              }}>
                {/* No prefix while in flight — q.message already reads
                    'Uploading…', so the row is never blank. */}
                {q.status === 'done' ? '✓ ' : q.status === 'failed' ? '✕ ' : ''}
                {q.message}
              </span>
            </div>
          ))}
        </div>
      )}

      {/* Uploaded files list */}
      <div style={{ color: '#B0B4C8', fontSize: '0.9rem', fontWeight: '600', marginBottom: '10px' }}>
        Uploaded Documents ({files.length})
      </div>

      {loading && (
        <div style={{ color: '#555', fontSize: '0.85rem', padding: '20px 0' }}>Loading…</div>
      )}

      {!loading && files.length === 0 && (
        <div style={{ color: '#555', fontSize: '0.85rem', padding: '20px 0', fontStyle: 'italic' }}>
          No GR documents uploaded yet.
        </div>
      )}

      {!loading && files.length > 0 && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          {files.map((f, i) => (
            <div key={i} style={{
              background: '#1A1D2E', border: '1px solid #2A2D3E', borderRadius: '10px',
              padding: '12px 16px', display: 'flex', alignItems: 'center', justifyContent: 'space-between',
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px', overflow: 'hidden' }}>
                <span style={{ fontSize: '1.2rem', flexShrink: 0 }}><FileText size={18} /></span>
                <div style={{ overflow: 'hidden' }}>
                  <div style={{
                    color: '#E8EAF0', fontSize: '0.85rem', fontWeight: '600',
                    whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
                  }}>
                    {f.filename}
                  </div>
                  <div style={{ color: '#555', fontSize: '0.72rem', fontFamily: 'monospace', marginTop: '2px' }}>
                    {f.page_count ?? '?'} pages · {f.file_size_kb ?? '?'} KB · {formatDate(f.uploaded_at)}
                  </div>
                </div>
              </div>

              <div style={{ display: 'flex', alignItems: 'center', gap: '10px', flexShrink: 0 }}>
                <span style={{
                  fontSize: '0.7rem', padding: '3px 10px', borderRadius: '12px',
                  fontFamily: 'monospace',
                  color: f.embedded ? '#4ADE80' : '#888',
                  background: f.embedded ? '#0F2A1A' : '#1E2135',
                  border: `1px solid ${f.embedded ? '#4ADE8033' : '#2A2D3E'}`,
                }}>
                  {f.embedded ? '✓ Embedded' : 'Not embedded'}
                </span>
                <button
                  onClick={() => deleteFile(f.filename)}
                  style={{
                    background: 'none', border: 'none', color: '#EF4444',
                    cursor: 'pointer', fontSize: '0.9rem', padding: '4px',
                  }}
                  title="Delete file"
                >
                  <Trash2 size={15} />
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

function StatChip({ label, value, color }) {
  return (
    <div style={{
      background: '#1A1D2E', border: '1px solid #2A2D3E', borderRadius: '10px',
      padding: '10px 16px', minWidth: '120px',
    }}>
      <div style={{ color, fontSize: '1.3rem', fontWeight: '700', fontFamily: 'monospace' }}>{value}</div>
      <div style={{ color: '#555', fontSize: '0.72rem', textTransform: 'uppercase', letterSpacing: '0.5px' }}>{label}</div>
    </div>
  )
}