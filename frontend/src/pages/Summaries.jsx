// frontend/src/pages/Summaries.jsx
// Google Drive style list of generated GR summaries.
// Available to all logged-in users (admin generates, everyone reads).
// Search bar filters by filename/subject/department/gr_number.
// Click a card → modal with full summary text + download button.

import { useState, useEffect, useMemo, useCallback } from 'react'
import { FileText, Inbox, Download } from 'lucide-react'
import client from '../api/client'

export default function Summaries() {
  const [summaries, setSummaries] = useState([])
  const [loading,   setLoading]   = useState(true)
  const [error,     setError]     = useState('')
  const [search,    setSearch]    = useState('')
  const [selected,  setSelected]  = useState(null) // full summary object for modal

  const loadSummaries = useCallback(async () => {
    setLoading(true)
    setError('')
    try {
      const res = await client.get('/summary/list')
      setSummaries(res.data.summaries || [])
    } catch (err) {
      setError(err.response?.data?.detail || 'Could not load summaries.')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    loadSummaries()
  }, [loadSummaries])

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase()
    if (!q) return summaries
    return summaries.filter(s =>
      (s.filename   || '').toLowerCase().includes(q) ||
      (s.subject    || '').toLowerCase().includes(q) ||
      (s.department || '').toLowerCase().includes(q) ||
      (s.gr_number  || '').toLowerCase().includes(q)
    )
  }, [search, summaries])

  const openSummary = async (item) => {
    // We already have metadata from /list, but full summary text
    // lives in the saved JSON, not returned by /list.
    // Fetch it via download endpoint isn't JSON-friendly, so
    // re-fetch list detail isn't available — use txt download link
    // for full text, and show what we have (metadata) in modal
    // immediately for a fast open, then lazy-load full text.
    setSelected({ ...item, fullText: null, loadingText: true })
    try {
      const res = await client.get(
        `/summary/download/${encodeURIComponent(item.filename_txt || getTxtName(item))}`,
        { responseType: 'text' }
      )
      setSelected(prev => prev && prev.filename === item.filename
        ? { ...prev, fullText: res.data, loadingText: false }
        : prev)
    } catch {
      setSelected(prev => prev && prev.filename === item.filename
        ? { ...prev, fullText: null, loadingText: false, textError: true }
        : prev)
    }
  }

  const getTxtName = (item) => {
    // txt_path is a full filesystem path — extract just the filename
    if (!item.txt_path) return ''
    const parts = item.txt_path.replace(/\\/g, '/').split('/')
    return parts[parts.length - 1]
  }

  const downloadSummary = (item) => {
    const txtName = getTxtName(item)
    if (!txtName) return
    window.open(`/api/summary/download/${encodeURIComponent(txtName)}`, '_blank')
  }

  const formatDate = (iso) => {
    if (!iso) return 'Unknown date'
    try {
      const d = new Date(iso)
      return d.toLocaleDateString('en-IN', { day: 'numeric', month: 'short', year: 'numeric' })
    } catch { return iso }
  }

  return (
    <div style={{
      minHeight: '100%', background: '#0F1117',
      padding: '28px 32px', fontFamily: 'Inter, sans-serif',
    }}>

      {/* Header */}
      <div style={{ marginBottom: '24px' }}>
        <h1 style={{ color: '#E8EAF0', fontSize: '1.4rem', margin: '0 0 4px', fontWeight: '700' }}>
          GR Summaries
        </h1>
        <p style={{ color: '#555', fontSize: '0.85rem', margin: 0 }}>
          Browse AI-generated summaries of embedded Government Resolutions.
        </p>
      </div>

      {/* Search bar */}
      <div style={{ marginBottom: '24px', maxWidth: '480px' }}>
        <input
          value={search}
          onChange={e => setSearch(e.target.value)}
          placeholder="Search by filename, subject, department, GR number…"
          style={{
            width: '100%', background: '#1A1D2E', border: '1px solid #2A2D3E',
            borderRadius: '10px', padding: '11px 16px', color: '#E8EAF0',
            fontSize: '0.9rem', outline: 'none', boxSizing: 'border-box',
            fontFamily: 'Inter, sans-serif',
          }}
          onFocus={e => e.target.style.borderColor = '#FF6B00'}
          onBlur={e  => e.target.style.borderColor = '#2A2D3E'}
        />
      </div>

      {/* Error */}
      {error && (
        <div style={{
          background: '#2A0F0F', border: '1px solid #EF4444', borderRadius: '8px',
          padding: '12px 16px', color: '#EF4444', fontSize: '0.85rem', marginBottom: '20px',
        }}>
          {error}
        </div>
      )}

      {/* Loading state */}
      {loading && (
        <div style={{ color: '#555', fontSize: '0.9rem', padding: '40px 0', textAlign: 'center' }}>
          Loading summaries…
        </div>
      )}

      {/* Empty state */}
      {!loading && !error && filtered.length === 0 && (
        <div style={{ textAlign: 'center', padding: '60px 24px' }}>
          <div style={{ fontSize: '2.2rem', marginBottom: '10px' }}><Inbox size={34} /></div>
          <p style={{ color: '#555', fontSize: '0.9rem' }}>
            {summaries.length === 0
              ? 'No summaries generated yet. Ask an admin to generate one.'
              : 'No summaries match your search.'}
          </p>
        </div>
      )}

      {/* Card grid — Google Drive style */}
      {!loading && filtered.length > 0 && (
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fill, minmax(260px, 1fr))',
          gap: '14px',
        }}>
          {filtered.map((item, i) => (
            <div
              key={i}
              onClick={() => openSummary(item)}
              style={{
                background: '#1A1D2E', border: '1px solid #2A2D3E',
                borderRadius: '12px', padding: '16px 18px',
                cursor: 'pointer', transition: 'border-color 0.15s, transform 0.1s',
              }}
              onMouseEnter={e => { e.currentTarget.style.borderColor = '#FF6B00' }}
              onMouseLeave={e => { e.currentTarget.style.borderColor = '#2A2D3E' }}
            >
              <div style={{ display: 'flex', alignItems: 'flex-start', gap: '10px', marginBottom: '10px' }}>
                <div style={{ fontSize: '1.4rem', flexShrink: 0 }}><FileText size={20} /></div>
                <div style={{ overflow: 'hidden' }}>
                  <div style={{
                    color: '#E8EAF0', fontSize: '0.88rem', fontWeight: '600',
                    whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
                  }}>
                    {item.filename}
                  </div>
                  <div style={{
                    color: '#7A7F94', fontSize: '0.78rem', marginTop: '2px',
                    whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
                  }}>
                    {item.subject !== 'N/A' ? item.subject : 'No subject extracted'}
                  </div>
                </div>
              </div>

              <div style={{
                display: 'flex', flexWrap: 'wrap', gap: '6px', marginTop: '10px',
              }}>
                {item.department !== 'N/A' && (
                  <Tag color="#6FB3FF" bg="#0D2137">{item.department}</Tag>
                )}
                {item.gr_number !== 'N/A' && (
                  <Tag color="#FACC15" bg="#2A2410">GR: {item.gr_number}</Tag>
                )}
              </div>

              <div style={{
                color: '#444', fontSize: '0.72rem', marginTop: '12px', fontFamily: 'monospace',
              }}>
                {formatDate(item.processed_at)}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* ── Modal ──────────────────────────────────── */}
      {selected && (
        <div
          onClick={() => setSelected(null)}
          style={{
            position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.6)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            zIndex: 500, padding: '24px',
          }}
        >
          <div
            onClick={e => e.stopPropagation()}
            style={{
              background: '#1A1D2E', border: '1px solid #2A2D3E', borderRadius: '14px',
              width: '100%', maxWidth: '640px', maxHeight: '82vh',
              display: 'flex', flexDirection: 'column', overflow: 'hidden',
            }}
          >
            {/* Modal header */}
            <div style={{
              padding: '18px 22px', borderBottom: '1px solid #2A2D3E',
              display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start',
            }}>
              <div style={{ overflow: 'hidden' }}>
                <div style={{ color: '#E8EAF0', fontSize: '1rem', fontWeight: '700' }}>
                  {selected.filename}
                </div>
                <div style={{ color: '#555', fontSize: '0.78rem', marginTop: '4px' }}>
                  {formatDate(selected.processed_at)}
                </div>
              </div>
              <button
                onClick={() => setSelected(null)}
                style={{
                  background: 'none', border: 'none', color: '#555',
                  fontSize: '1.2rem', cursor: 'pointer', flexShrink: 0, marginLeft: '12px',
                }}
              >✕</button>
            </div>

            {/* Modal body */}
            <div style={{ padding: '20px 22px', overflowY: 'auto', flex: 1 }}>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', marginBottom: '16px' }}>
                {selected.department !== 'N/A' && <Tag color="#6FB3FF" bg="#0D2137">{selected.department}</Tag>}
                {selected.gr_number !== 'N/A' && <Tag color="#FACC15" bg="#2A2410">GR: {selected.gr_number}</Tag>}
              </div>

              {selected.loadingText && (
                <div style={{ color: '#555', fontSize: '0.85rem' }}>Loading full summary…</div>
              )}

              {selected.textError && (
                <div style={{ color: '#EF4444', fontSize: '0.85rem' }}>
                  Could not load full summary text. Try downloading instead.
                </div>
              )}

              {selected.fullText && (
                <pre style={{
                  color: '#C4C8D8', fontSize: '0.84rem', lineHeight: '1.6',
                  whiteSpace: 'pre-wrap', fontFamily: 'Inter, sans-serif', margin: 0,
                }}>
                  {selected.fullText}
                </pre>
              )}
            </div>

            {/* Modal footer */}
            <div style={{
              padding: '14px 22px', borderTop: '1px solid #2A2D3E',
              display: 'flex', justifyContent: 'flex-end',
            }}>
              <button
                onClick={() => downloadSummary(selected)}
                style={{
                  background: '#FF6B00', color: 'white', border: 'none',
                  borderRadius: '8px', padding: '9px 18px', fontSize: '0.85rem',
                  fontWeight: '600', cursor: 'pointer', fontFamily: 'Inter, sans-serif',
                }}
              >
                <Download size={14} /> Download .txt
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

function Tag({ children, color, bg }) {
  return (
    <span style={{
      background: bg, color, border: `1px solid ${color}33`,
      padding: '3px 10px', borderRadius: '14px',
      fontSize: '0.72rem', fontFamily: 'monospace', whiteSpace: 'nowrap',
    }}>
      {children}
    </span>
  )
}