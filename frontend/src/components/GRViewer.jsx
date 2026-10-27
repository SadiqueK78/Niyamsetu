// frontend/src/components/GRViewer.jsx
// ─────────────────────────────────────────────────────────
// Modal PDF viewer for one uploaded GR, opened at a given page.
//
// Why the PDF is fetched rather than linked:
//   /api/files/{filename} requires the Authorization header, and a plain
//   <iframe src="/api/files/GR1.pdf"> cannot send one — the browser makes
//   that request itself, outside axios, with no interceptor to attach the
//   token. So the bytes are fetched through the app's authenticated axios
//   instance as a blob, turned into an object URL, and the iframe is
//   pointed at that. The URL is revoked on close; a blob URL that is
//   never revoked holds the whole file in memory for the life of the tab.
//
// The "#page=N" fragment is the PDF Open Parameters convention, honoured
// by Chrome's and Firefox's built-in viewers. N is 1-based, which is the
// same number the citation pill shows — see CitationPill for why that
// conversion happens at display time only.
// ─────────────────────────────────────────────────────────

import { useState, useEffect, useRef } from 'react'
import { X, ExternalLink, FileText } from 'lucide-react'
import client from '../api/client'

export default function GRViewer({ file, page = 1, onClose }) {
  const [blobUrl, setBlobUrl] = useState('')
  const [loading, setLoading] = useState(true)
  const [error,   setError]   = useState('')

  // Held in a ref as well as state so cleanup can revoke the current URL
  // without the effect depending on the state it sets.
  const urlRef = useRef('')

  const stem = (file || '').replace(/\.pdf$/i, '')
  const safePage = Number.isFinite(Number(page)) && Number(page) > 0
    ? Math.floor(Number(page))
    : 1

  // Fetch on mount and whenever the file changes. setState here is the
  // point of the effect — it is synchronising React with an external
  // system (the network and the object-URL registry), which is exactly
  // what effects are for; nothing is set synchronously in the body.
  useEffect(() => {
    if (!file) return

    let cancelled = false

    const revoke = () => {
      if (urlRef.current) {
        URL.revokeObjectURL(urlRef.current)
        urlRef.current = ''
      }
    }

    client
      .get(`/files/${encodeURIComponent(file)}`, { responseType: 'blob' })
      .then(res => {
        if (cancelled) return
        revoke()
        const url = URL.createObjectURL(
          new Blob([res.data], { type: 'application/pdf' }),
        )
        urlRef.current = url
        setBlobUrl(url)
        setLoading(false)
      })
      .catch(err => {
        if (cancelled) return
        // The endpoint answers every refusal with the same 404, so there
        // is nothing more specific to report than this.
        setError(
          err.response?.status === 404
            ? 'This document is no longer available on the server.'
            : 'Could not load the document.',
        )
        setLoading(false)
      })

    return () => {
      cancelled = true
      revoke()
    }
  }, [file])

  // Escape closes, and the listener is removed on unmount so a closed
  // viewer cannot keep swallowing key presses.
  useEffect(() => {
    const onKey = (e) => { if (e.key === 'Escape') onClose?.() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  const headerButton = {
    display: 'flex', alignItems: 'center', gap: '6px',
    background: 'transparent', border: '1px solid #2A2D3E',
    borderRadius: '6px', color: '#8A93A6', padding: '5px 11px',
    fontSize: '0.78rem', cursor: 'pointer', fontFamily: 'Inter, sans-serif',
  }

  return (
    <div
      onClick={onClose}
      style={{
        position: 'fixed', inset: 0, zIndex: 200,
        background: 'rgba(8, 9, 13, 0.82)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        padding: '28px', fontFamily: 'Inter, sans-serif',
      }}
    >
      {/* Stops a click inside the panel from reaching the backdrop. */}
      <div
        onClick={e => e.stopPropagation()}
        style={{
          width: '100%', maxWidth: '960px', height: '100%',
          background: '#141726', border: '1px solid #2A2D3E',
          borderRadius: '12px', display: 'flex', flexDirection: 'column',
          overflow: 'hidden',
        }}
      >
        {/* Header */}
        <div style={{
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          gap: '12px', padding: '12px 16px', borderBottom: '1px solid #2A2D3E',
          flexShrink: 0,
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', minWidth: 0 }}>
            <FileText size={17} color="#FF6B00" />
            <span style={{
              color: '#E8EAF0', fontWeight: '700', fontSize: '0.92rem',
              whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
            }}>
              {stem || 'Document'}
            </span>
            <span style={{
              color: '#8A93A6', fontSize: '0.78rem', fontFamily: 'monospace',
              flexShrink: 0,
            }}>
              page {safePage}
            </span>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexShrink: 0 }}>
            <button
              onClick={() => { if (blobUrl) window.open(blobUrl, '_blank', 'noopener') }}
              disabled={!blobUrl}
              title="Open in new tab"
              style={{ ...headerButton, opacity: blobUrl ? 1 : 0.45,
                       cursor: blobUrl ? 'pointer' : 'default' }}
            >
              <ExternalLink size={13} />
              Open in new tab
            </button>
            <button
              onClick={onClose}
              title="Close"
              style={{ ...headerButton, padding: '5px 8px' }}
            >
              <X size={15} />
            </button>
          </div>
        </div>

        {/* Body */}
        <div style={{ flex: 1, minHeight: 0, background: '#0D1016' }}>
          {loading ? (
            <div style={{
              height: '100%', display: 'flex', alignItems: 'center',
              justifyContent: 'center', color: '#6B7280', fontSize: '0.88rem',
            }}>
              Loading document...
            </div>
          ) : error ? (
            <div style={{
              height: '100%', display: 'flex', flexDirection: 'column',
              alignItems: 'center', justifyContent: 'center', gap: '8px',
              color: '#FF8A8A', fontSize: '0.88rem', textAlign: 'center',
              padding: '24px',
            }}>
              <div style={{ fontWeight: '700' }}>{error}</div>
              <div style={{ color: '#6B7280', fontSize: '0.82rem' }}>
                The citation still shows what the answer was based on.
              </div>
            </div>
          ) : (
            <iframe
              src={`${blobUrl}#page=${safePage}`}
              title={`${stem} page ${safePage}`}
              style={{ width: '100%', height: '100%', border: 'none' }}
            />
          )}
        </div>
      </div>
    </div>
  )
}
