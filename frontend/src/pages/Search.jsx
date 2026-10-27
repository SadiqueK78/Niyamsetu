// frontend/src/pages/Search.jsx
// ─────────────────────────────────────────────────────────
// Retrieval inspector. The same hybrid search the chat answers from,
// shown without a model in the way.
//
// Why the per-retriever badges matter:
//   When an answer cites the wrong document, the question is which half
//   of retrieval put it there. A chunk BM25 ranked first and the dense
//   side never saw is a keyword coincidence. One the dense side alone
//   found shares no wording with the question. One both ranked highly is
//   a real match. The fused list cannot be read that way, so the ranks
//   are shown separately.
//
// Reads POST /api/query/search through src/api/client.js. No LLM is
// involved on either side of this page.
// ─────────────────────────────────────────────────────────

import { useState, useCallback } from 'react'
import { Search as SearchIcon, AlertTriangle } from 'lucide-react'
import client from '../api/client'

const TOP_K = 12

// Badge colours per retriever, so the three are distinguishable without
// reading the labels.
const BADGE = {
  dense:   { text: '#C792EA', border: '#4A3A5F' },
  lexical: { text: '#58D6A8', border: '#2A5247' },
  fused:   { text: '#FFD166', border: '#5A4A28' },
}

export default function Search() {
  const [input,    setInput]    = useState('')
  const [query,    setQuery]    = useState('')
  const [results,  setResults]  = useState([])
  const [denseOk,  setDenseOk]  = useState(true)
  const [loading,  setLoading]  = useState(false)
  const [error,    setError]    = useState('')
  const [searched, setSearched] = useState(false)

  const runSearch = useCallback(async (text) => {
    const trimmed = (text || '').trim()
    if (!trimmed) return

    setLoading(true)
    setError('')
    try {
      const res = await client.post('/query/search', {
        query: trimmed,
        top_k: TOP_K,
      })
      setResults(res.data.results || [])
      // Absent on an older backend, where dense retrieval was the only
      // path and could not be reported as unavailable.
      setDenseOk(res.data.dense_ok !== false)
      setQuery(trimmed)
      setSearched(true)
    } catch (err) {
      setError(err.response?.data?.detail || 'Search failed.')
      setResults([])
    } finally {
      setLoading(false)
    }
  }, [])

  const badge = (label, value, tone) => (
    <span style={{
      background: '#0D1016',
      border: `1px solid ${BADGE[tone].border}`,
      color: value === null || value === undefined ? '#5A6478' : BADGE[tone].text,
      padding: '2px 9px',
      borderRadius: '20px',
      fontSize: '0.7rem',
      fontFamily: 'monospace',
      whiteSpace: 'nowrap',
    }}>
      {label} {value === null || value === undefined ? '—' : `#${value}`}
    </span>
  )

  return (
    <div style={{
      minHeight: '100%', background: '#0F1117',
      padding: '28px 32px', fontFamily: 'Inter, sans-serif',
    }}>

      <div style={{ marginBottom: '18px' }}>
        <h1 style={{ color: '#E8EAF0', fontSize: '1.4rem', margin: '0 0 4px',
                     fontWeight: '700', display: 'flex', alignItems: 'center', gap: '10px' }}>
          <SearchIcon size={20} color="#FF6B00" />
          Retrieval Search
        </h1>
        <p style={{ color: '#6B7280', fontSize: '0.85rem', margin: 0 }}>
          The same chunks the chat answers from, with no model in the way.
          Each result shows where the dense and keyword retrievers ranked it.
        </p>
      </div>

      {/* Query box */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: '10px',
        background: '#1A1D2E', border: '1px solid #2A2D3E',
        borderRadius: '10px', padding: '10px 14px', marginBottom: '14px',
      }}>
        <SearchIcon size={16} color="#6B7280" />
        <input
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => { if (e.key === 'Enter') runSearch(input) }}
          placeholder="Search the embedded documents, then press Enter"
          style={{
            background: 'transparent', border: 'none', outline: 'none',
            color: '#E8EAF0', fontSize: '0.9rem', flex: 1, minWidth: 0,
            fontFamily: 'Inter, sans-serif',
          }}
        />
        <button
          onClick={() => runSearch(input)}
          disabled={loading || !input.trim()}
          style={{
            background: loading || !input.trim() ? '#2A2D3E' : '#FF6B00',
            border: 'none', borderRadius: '7px',
            color: loading || !input.trim() ? '#6B7280' : '#FFFFFF',
            padding: '7px 16px', fontSize: '0.84rem', fontWeight: '600',
            cursor: loading || !input.trim() ? 'default' : 'pointer',
            fontFamily: 'Inter, sans-serif', flexShrink: 0,
          }}
        >
          Search
        </button>
      </div>

      {error && (
        <div style={{
          background: '#2A1215', border: '1px solid #5A2228', color: '#FF8A8A',
          padding: '10px 14px', borderRadius: '8px', marginBottom: '16px', fontSize: '0.85rem',
        }}>
          {error}
        </div>
      )}

      {/* Dense retrieval unavailable */}
      {searched && !denseOk && (
        <div style={{
          display: 'flex', alignItems: 'center', gap: '9px',
          background: '#2A2112', border: '1px solid #5A4A22', color: '#FFD166',
          padding: '10px 14px', borderRadius: '8px', marginBottom: '16px', fontSize: '0.85rem',
        }}>
          <AlertTriangle size={16} />
          Dense retrieval unavailable, showing lexical results only
        </div>
      )}

      {loading ? (
        <div style={{ color: '#6B7280', fontSize: '0.9rem', padding: '40px 0' }}>
          Searching...
        </div>
      ) : !searched ? (
        <div style={{
          border: '1px dashed #2A2D3E', borderRadius: '10px',
          padding: '48px 24px', textAlign: 'center', color: '#6B7280',
        }}>
          <SearchIcon size={30} color="#2A2D3E" />
          <div style={{ color: '#8A93A6', fontSize: '0.95rem', marginTop: '12px', fontWeight: '600' }}>
            Nothing searched yet
          </div>
          <div style={{ fontSize: '0.84rem', marginTop: '6px', lineHeight: '1.5' }}>
            Type a question above and press Enter to see which chunks
            retrieval would hand to the model.
          </div>
        </div>
      ) : results.length === 0 ? (
        <div style={{
          border: '1px dashed #2A2D3E', borderRadius: '10px',
          padding: '40px 24px', textAlign: 'center', color: '#6B7280',
        }}>
          <div style={{ color: '#8A93A6', fontSize: '0.92rem', fontWeight: '600' }}>
            No matches
          </div>
          <div style={{ fontSize: '0.84rem', marginTop: '6px' }}>
            Nothing in the embedded documents matched {`"${query}"`}.
          </div>
        </div>
      ) : (
        <>
          <div style={{ color: '#8A93A6', fontSize: '0.8rem',
                        fontFamily: 'monospace', marginBottom: '12px' }}>
            <span style={{ color: '#E8EAF0', fontWeight: '700' }}>{results.length}</span> result(s)
          </div>

          {results.map((r, i) => {
            const relevance = typeof r.relevance === 'number'
              ? r.relevance
              : (typeof r.score === 'number' ? r.score : 0)

            return (
              <div key={`${r.file}-${r.page}-${i}`} style={{
                background: '#1A1D2E', border: '1px solid #2A2D3E',
                borderRadius: '10px', padding: '14px 16px', marginBottom: '12px',
              }}>
                <div style={{
                  display: 'flex', alignItems: 'center', gap: '10px',
                  flexWrap: 'wrap', marginBottom: '9px',
                }}>
                  <span style={{ color: '#E8EAF0', fontSize: '0.88rem', fontWeight: '700' }}>
                    {r.file}
                  </span>
                  <span style={{ color: '#6B7280', fontSize: '0.78rem', fontFamily: 'monospace' }}>
                    page {r.page}
                  </span>
                  {r.section_type && (
                    <span style={{
                      background: '#0D2137', border: '1px solid #1E3A5F', color: '#6FB3FF',
                      padding: '2px 9px', borderRadius: '20px',
                      fontSize: '0.7rem', fontFamily: 'monospace',
                    }}>
                      {r.section_type}
                    </span>
                  )}
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: '7px',
                              flexWrap: 'wrap', marginBottom: '10px' }}>
                  {badge('Dense', r.dense_rank, 'dense')}
                  {badge('BM25', r.lexical_rank, 'lexical')}
                  {badge('Fused', r.fused_rank, 'fused')}
                </div>

                {/* Relevance bar */}
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px',
                              marginBottom: '10px' }}>
                  <div style={{
                    flex: 1, height: '5px', background: '#0D1016',
                    borderRadius: '3px', overflow: 'hidden', minWidth: 0,
                  }}>
                    <div style={{
                      width: `${Math.round(Math.max(0, Math.min(1, relevance)) * 100)}%`,
                      height: '100%', background: '#FF6B00',
                    }} />
                  </div>
                  <span style={{ color: '#8A93A6', fontSize: '0.74rem',
                                 fontFamily: 'monospace', flexShrink: 0, width: '42px',
                                 textAlign: 'right' }}>
                    {Math.round(Math.max(0, Math.min(1, relevance)) * 100)}%
                  </span>
                </div>

                <div style={{
                  color: '#99A2B5', fontSize: '0.82rem', lineHeight: '1.55',
                  whiteSpace: 'pre-wrap', wordBreak: 'break-word',
                }}>
                  {r.preview}
                </div>
              </div>
            )
          })}
        </>
      )}
    </div>
  )
}
