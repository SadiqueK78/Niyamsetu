// frontend/src/pages/Registry.jsx
// ─────────────────────────────────────────────────────────
// The document registry — every embedded GR as one row.
//
// Why a table and not the graph:
//   The graph answers "what is connected to what". It cannot answer
//   "which orders did the finance department issue in 2025", or "is the
//   GR I am holding already in the system", because a force layout has no
//   order and no searchable text. Those are table questions, and this is
//   the table.
//
// Reads GET /api/meta/ for the per-document fields and GET /api/graph/ for
// the citation counts, which only the edge list knows. Both are read-only
// and neither involves a model.
// ─────────────────────────────────────────────────────────

import { useState, useEffect, useCallback, useMemo } from 'react'
import { Table, ArrowUp, ArrowDown, Search as SearchIcon } from 'lucide-react'
import client from '../api/client'

// Characters of the subject shown in the cell; the full text is in the
// title attribute, so a long subject is readable without widening the
// column for every other row.
const SUBJECT_CHARS = 70

// Columns the header can sort by. The rest are counts or truncated text
// where a sort would be more confusing than useful.
const SORTABLE = ['source_file', 'department', 'doc_date']

const COLUMNS = [
  { key: 'source_file', label: 'File' },
  { key: 'gr_number',   label: 'GR number' },
  { key: 'department',  label: 'Department' },
  { key: 'doc_date',    label: 'Date' },
  { key: 'subject',     label: 'Subject' },
  { key: 'cites',       label: 'Cites' },
  { key: 'cited_by',    label: 'Cited by' },
]

// react-force-graph is not involved here, but the graph endpoint returns
// edges whose endpoints may already be node objects if something else
// mutated them. Reading an id back has to cope with both forms.
function endpointId(end) {
  return (end && typeof end === 'object') ? end.id : end
}

export default function Registry() {
  const [files,   setFiles]   = useState([])
  const [edges,   setEdges]   = useState([])
  const [loading, setLoading] = useState(true)
  const [error,   setError]   = useState('')

  const [search,     setSearch]     = useState('')
  const [department, setDepartment] = useState('')
  const [sortKey,    setSortKey]    = useState('source_file')
  const [sortAsc,    setSortAsc]    = useState(true)

  // No setLoading(true)/setError('') prologue: this runs once on mount and
  // both states already start that way.
  const load = useCallback(async () => {
    try {
      // The graph is secondary — a registry with no citation counts is
      // still a useful registry, so its failure does not fail the page.
      const [metaRes, graphRes] = await Promise.all([
        client.get('/meta/'),
        client.get('/graph/').catch(() => ({ data: { edges: [] } })),
      ])
      setFiles(metaRes.data.files || [])
      setEdges(graphRes.data.edges || [])
    } catch (err) {
      setError(err.response?.data?.detail || 'Could not load the document registry.')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    load()
  }, [load])

  // ── Derived data ────────────────────────────────────────

  // Counted once over the edge list rather than filtered per row, so the
  // table stays O(files) instead of O(files x edges).
  const counts = useMemo(() => {
    const out = new Map()
    const bump = (id, field) => {
      if (!id) return
      const row = out.get(id) || { cites: 0, cited_by: 0 }
      row[field] += 1
      out.set(id, row)
    }
    edges.forEach(e => {
      bump(endpointId(e.source), 'cites')
      bump(endpointId(e.target), 'cited_by')
    })
    return out
  }, [edges])

  const rows = useMemo(() => files.map(f => {
    const c = counts.get(f.source_file) || { cites: 0, cited_by: 0 }
    return { ...f, cites: c.cites, cited_by: c.cited_by }
  }), [files, counts])

  const departments = useMemo(() => {
    const seen = new Set()
    rows.forEach(r => { if (r.department) seen.add(r.department) })
    return Array.from(seen).sort()
  }, [rows])

  const visible = useMemo(() => {
    const needle = search.trim().toLowerCase()

    const matches = rows.filter(r => {
      if (department && r.department !== department) return false
      if (!needle) return true
      // Every text column, so a search for a GR number, a department or a
      // word from the subject all work without the user choosing a field.
      return ['source_file', 'stem', 'gr_number', 'department', 'doc_date',
              'doc_date_raw', 'subject']
        .some(key => String(r[key] || '').toLowerCase().includes(needle))
    })

    // Direction is applied inside the comparator rather than by reversing
    // the result, so missing values stay LAST in both directions. A
    // document with no detected date is not the earliest one, it is
    // unknown, and sorting it to the top of a descending list would read
    // as "most recent".
    return [...matches].sort((a, b) => {
      const av = a[sortKey]
      const bv = b[sortKey]
      if (!av && !bv) return 0
      if (!av) return 1
      if (!bv) return -1
      const compared = String(av).localeCompare(String(bv), undefined, { numeric: true })
      return sortAsc ? compared : -compared
    })
  }, [rows, search, department, sortKey, sortAsc])

  const toggleSort = (key) => {
    if (!SORTABLE.includes(key)) return
    if (key === sortKey) {
      setSortAsc(!sortAsc)
    } else {
      setSortKey(key)
      setSortAsc(true)
    }
  }

  // ── Render ──────────────────────────────────────────────

  const cell = {
    padding: '9px 12px',
    borderBottom: '1px solid #20232F',
    color: '#C8CEDA',
    fontSize: '0.82rem',
    verticalAlign: 'top',
  }

  const numericCell = {
    ...cell,
    fontFamily: 'monospace',
    textAlign: 'right',
    width: '70px',
  }

  return (
    <div style={{
      minHeight: '100%', background: '#0F1117',
      padding: '28px 32px', fontFamily: 'Inter, sans-serif',
    }}>

      <div style={{ marginBottom: '18px' }}>
        <h1 style={{ color: '#E8EAF0', fontSize: '1.4rem', margin: '0 0 4px',
                     fontWeight: '700', display: 'flex', alignItems: 'center', gap: '10px' }}>
          <Table size={20} color="#FF6B00" />
          Document Registry
        </h1>
        <p style={{ color: '#6B7280', fontSize: '0.85rem', margin: 0 }}>
          Every embedded document, with the number, department and date read
          off its letterhead, and how often it cites or is cited by another.
        </p>
      </div>

      {error && (
        <div style={{
          background: '#2A1215', border: '1px solid #5A2228', color: '#FF8A8A',
          padding: '10px 14px', borderRadius: '8px', marginBottom: '16px', fontSize: '0.85rem',
        }}>
          {error}
        </div>
      )}

      {/* Controls */}
      {!loading && rows.length > 0 && (
        <div style={{
          display: 'flex', gap: '12px', flexWrap: 'wrap',
          alignItems: 'center', marginBottom: '14px',
        }}>
          <div style={{
            display: 'flex', alignItems: 'center', gap: '8px',
            background: '#1A1D2E', border: '1px solid #2A2D3E',
            borderRadius: '8px', padding: '7px 12px', minWidth: '260px',
          }}>
            <SearchIcon size={15} color="#6B7280" />
            <input
              value={search}
              onChange={e => setSearch(e.target.value)}
              placeholder="Search file, number, department, subject"
              style={{
                background: 'transparent', border: 'none', outline: 'none',
                color: '#E8EAF0', fontSize: '0.84rem', flex: 1, minWidth: 0,
                fontFamily: 'Inter, sans-serif',
              }}
            />
          </div>

          <select
            value={department}
            onChange={e => setDepartment(e.target.value)}
            style={{
              background: '#1A1D2E', border: '1px solid #2A2D3E', borderRadius: '8px',
              color: '#C8CEDA', fontSize: '0.84rem', padding: '8px 12px',
              fontFamily: 'Inter, sans-serif', cursor: 'pointer', maxWidth: '320px',
            }}
          >
            <option value="">All departments</option>
            {departments.map(name => (
              <option key={name} value={name}>{name}</option>
            ))}
          </select>

          <div style={{ color: '#8A93A6', fontSize: '0.8rem', fontFamily: 'monospace' }}>
            <span style={{ color: '#E8EAF0', fontWeight: '700' }}>{visible.length}</span>
            {visible.length === rows.length ? ' documents' : ` of ${rows.length} documents`}
          </div>
        </div>
      )}

      {loading ? (
        <div style={{ color: '#6B7280', fontSize: '0.9rem', padding: '40px 0' }}>
          Loading registry...
        </div>
      ) : rows.length === 0 ? (
        <div style={{
          border: '1px dashed #2A2D3E', borderRadius: '10px',
          padding: '48px 24px', textAlign: 'center', color: '#6B7280',
        }}>
          <Table size={30} color="#2A2D3E" />
          <div style={{ color: '#8A93A6', fontSize: '0.95rem', marginTop: '12px', fontWeight: '600' }}>
            No documents yet
          </div>
          <div style={{ fontSize: '0.84rem', marginTop: '6px', lineHeight: '1.5' }}>
            The registry is built from the embedded documents.
            Ask an admin to run embedding, and every uploaded GR
            will appear here.
          </div>
        </div>
      ) : visible.length === 0 ? (
        <div style={{
          border: '1px dashed #2A2D3E', borderRadius: '10px',
          padding: '40px 24px', textAlign: 'center', color: '#6B7280',
        }}>
          <div style={{ color: '#8A93A6', fontSize: '0.92rem', fontWeight: '600' }}>
            Nothing matches
          </div>
          <div style={{ fontSize: '0.84rem', marginTop: '6px' }}>
            Clear the search box or pick a different department.
          </div>
        </div>
      ) : (
        <div style={{
          background: '#1A1D2E', border: '1px solid #2A2D3E',
          borderRadius: '10px', overflowX: 'auto',
        }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', minWidth: '900px' }}>
            <thead>
              <tr>
                {COLUMNS.map(col => {
                  const sortable = SORTABLE.includes(col.key)
                  const isActive = sortKey === col.key
                  return (
                    <th
                      key={col.key}
                      onClick={() => toggleSort(col.key)}
                      title={sortable ? `Sort by ${col.label.toLowerCase()}` : undefined}
                      style={{
                        textAlign: ['cites', 'cited_by'].includes(col.key) ? 'right' : 'left',
                        padding: '10px 12px',
                        borderBottom: '1px solid #2A2D3E',
                        color: isActive ? '#E8EAF0' : '#8A93A6',
                        fontSize: '0.74rem',
                        fontWeight: '700',
                        textTransform: 'uppercase',
                        letterSpacing: '0.5px',
                        whiteSpace: 'nowrap',
                        cursor: sortable ? 'pointer' : 'default',
                        userSelect: 'none',
                      }}
                    >
                      <span style={{ display: 'inline-flex', alignItems: 'center', gap: '5px' }}>
                        {col.label}
                        {isActive && (sortAsc
                          ? <ArrowUp size={12} />
                          : <ArrowDown size={12} />)}
                      </span>
                    </th>
                  )
                })}
              </tr>
            </thead>
            <tbody>
              {visible.map(row => (
                <tr key={row.source_file}>
                  <td style={{ ...cell, whiteSpace: 'nowrap' }}>{row.source_file}</td>
                  <td style={{ ...cell, fontFamily: 'monospace', fontSize: '0.76rem',
                               wordBreak: 'break-all', maxWidth: '220px' }}>
                    {row.gr_number || '—'}
                  </td>
                  <td style={{ ...cell, maxWidth: '200px' }}>{row.department || '—'}</td>
                  <td style={{ ...cell, fontFamily: 'monospace', whiteSpace: 'nowrap' }}>
                    {row.doc_date || '—'}
                  </td>
                  <td style={{ ...cell, maxWidth: '360px' }} title={row.subject || ''}>
                    {row.subject
                      ? (row.subject.length > SUBJECT_CHARS
                          ? `${row.subject.slice(0, SUBJECT_CHARS)}…`
                          : row.subject)
                      : '—'}
                  </td>
                  <td style={numericCell}>{row.cites}</td>
                  <td style={numericCell}>{row.cited_by}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
