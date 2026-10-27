// frontend/src/components/CitationPill.jsx
// ─────────────────────────────────────────────────────────
// Small pill tag shown below every AI answer.
// Displays which file and page the answer came from, and opens that page
// of the actual PDF when clicked.
// Example: GR_2024_transfer.pdf — p.3
//
// Page numbering:
//   The API already reports 1-based pages — core/rag.py build_citations()
//   stores str(int(raw_page) + 1), converting the index's 0-based page at
//   response time. So the number shown here is passed to the viewer
//   unchanged, and the pill, the answer text and the page the PDF opens
//   at are all the same number. Nothing is converted twice.
//
// A chunk can also reach an answer through the GR relationship graph
// rather than through search — the superseding order, say, which shares
// no wording with the question. Those get a "linked via graph" tag, so a
// reader can tell a cited source from a source that was pulled in because
// of what it does to a cited source.
//
// Accepts either the citation object (preferred, since the extra fields
// then travel with it) or the individual props it has always taken.
// ─────────────────────────────────────────────────────────

import { useState } from 'react'
import GRViewer from './GRViewer'

export default function CitationPill({
  citation,
  file,
  page,
  relevance,
  via_graph,
  graph_relation,
  onOpen,
}) {
  // Individual props win where given, so existing callers are unaffected.
  const source = citation || {}
  const fileName   = file      !== undefined ? file      : source.file
  const pageNumber = page      !== undefined ? page      : source.page
  const score      = relevance !== undefined ? relevance : source.relevance_score
  const viaGraph   = via_graph !== undefined ? via_graph : source.via_graph
  const relation   = graph_relation !== undefined ? graph_relation : source.graph_relation

  // Whole-number percentage, and only when there is a real number to show.
  // The API sends relevance_score as null on retrieval paths that do not
  // score their results, and Number.isFinite also keeps a malformed value
  // from rendering as "NaN%".
  const relevanceText = Number.isFinite(Number(score)) && score !== null && score !== ''
    ? ` (${Math.round(Number(score) * 100)}%)`
    : '';

  // Only used when the caller supplies no onOpen, so the pill still works
  // on its own wherever it is dropped in.
  const [viewer, setViewer] = useState(false)

  // The page handed to the viewer is the SAME number printed on the pill.
  const viewerPage = Number.isFinite(Number(pageNumber)) && Number(pageNumber) > 0
    ? Math.floor(Number(pageNumber))
    : 1

  const openDocument = () => {
    if (!fileName) return
    if (onOpen) onOpen(fileName, viewerPage)
    else setViewer(true)
  }

  return (
    <span style={{ display: 'inline-flex', alignItems: 'center', gap: '6px' }}>
      <button
        onClick={openDocument}
        disabled={!fileName}
        title={fileName ? `Open ${fileName} at page ${viewerPage}` : undefined}
        style={{
          background: '#0D2137',
          border: '1px solid #1E3A5F',
          color: '#6FB3FF',
          padding: '3px 12px',
          borderRadius: '20px',
          fontSize: '0.75rem',
          fontFamily: 'monospace',
          whiteSpace: 'nowrap',
          cursor: fileName ? 'pointer' : 'default',
        }}
      >
        {fileName} — p.{pageNumber}{relevanceText}
      </button>

      {viaGraph && (
        <span
          title={relation ? `graph relation: ${relation}` : 'pulled in via the GR relationship graph'}
          style={{
            color: '#8A93A6',
            fontSize: '0.7rem',
            fontFamily: 'monospace',
            whiteSpace: 'nowrap',
          }}
        >
          linked via graph{relation ? ` (${relation})` : ''}
        </span>
      )}

      {viewer && (
        <GRViewer
          file={fileName}
          page={viewerPage}
          onClose={() => setViewer(false)}
        />
      )}
    </span>
  )
}
