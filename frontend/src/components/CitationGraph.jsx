// frontend/src/components/CitationGraph.jsx
// ─────────────────────────────────────────────────────────
// The citation graph for one answer: the files it cited, plus the files
// directly linked to those in the GR relationship graph.
//
// Why it sits next to the answer:
//   A citation list says where the answer came from. It cannot say that
//   one of those documents was amended by another, or that two of them
//   are the same order at different dates. That is the context a reader
//   needs to judge the answer, and it is one hop away in the graph.
//
// The graph snapshot is fetched ONCE per page load and held in a
// module-level variable. It is a whole-corpus structure that changes only
// when documents are re-embedded, so refetching it per message would be
// one request per answer for data that never moved.
// ─────────────────────────────────────────────────────────

import { useState, useEffect, useRef, useMemo } from 'react'
import ForceGraph2D from 'react-force-graph-2d'
import { Network, ChevronRight } from 'lucide-react'
import client from '../api/client'
import GRViewer from './GRViewer'

// Shared across every mount for the life of the page. The promise itself
// is cached, not just the result, so two panels mounting at once make one
// request rather than two.
let graphPromise = null

function loadGraphOnce() {
  if (!graphPromise) {
    graphPromise = client
      .get('/graph/')
      .then(res => ({
        nodes: res.data.nodes || [],
        edges: res.data.edges || [],
      }))
      // A failed graph must not be cached as a permanent failure, but it
      // must also not break the panel: the caller falls back to showing
      // the cited files alone, with no error banner.
      .catch(() => {
        graphPromise = null
        return { nodes: [], edges: [] }
      })
  }
  return graphPromise
}

// Same relation colours as the full graph page, so a relation means the
// same thing in both places.
const RELATION_COLOR = {
  supersedes: '#FF4D4D',
  amends:     '#FF6B00',
  references: '#4A6FA5',
  refers_to:  '#4A6FA5',
}

const CITED_COLOR  = '#FF6B00'
const LINKED_COLOR = '#6B7280'

// Node radii in graph units, so a line can be trimmed to a node's edge.
const NODE_RADIUS_CITED  = 6
const NODE_RADIUS_LINKED = 4

// Link drawing, all in SCREEN pixels and divided by globalScale at paint
// time, so a line is the same thickness at any zoom. Graph units looked
// right at zoom 1 and vanished when the panel auto-fitted a wide graph.
const LINK_WIDTH_PX   = 2.5
const ARROW_LENGTH_PX = 9
const LINK_ALPHA      = 0.95

// At most this many nodes, cited files first. A panel 320px wide stops
// being readable well before this.
const MAX_NODES = 8

// Labels are stems only, truncated — see Graph.jsx for the same rule.
const LABEL_CHARS = 18

function relationColor(relation) {
  return RELATION_COLOR[relation] || '#4A6FA5'
}

function stemOf(filename) {
  return String(filename || '').replace(/\.pdf$/i, '')
}

function nodeRadius(node) {
  return node?.cited ? NODE_RADIUS_CITED : NODE_RADIUS_LINKED
}

function shortLabel(filename) {
  const stem = stemOf(filename)
  return stem.length > LABEL_CHARS ? `${stem.slice(0, LABEL_CHARS - 1)}…` : stem
}

// react-force-graph replaces each link's endpoints with the node objects
// once the simulation runs, so an id has to be read back from either form.
function endpointId(end) {
  return (end && typeof end === 'object') ? end.id : end
}

export default function CitationGraph({ citations, onOpenFile }) {
  const [snapshot, setSnapshot] = useState(null)
  const [viewer, setViewer] = useState(null)

  const wrapRef = useRef(null)
  const [size, setSize] = useState({ width: 290, height: 240 })

  // Fetch (or reuse) the snapshot. Synchronising with an external system,
  // and nothing is set synchronously in the effect body.
  useEffect(() => {
    let cancelled = false
    loadGraphOnce().then(data => { if (!cancelled) setSnapshot(data) })
    return () => { cancelled = true }
  }, [])

  useEffect(() => {
    const el = wrapRef.current
    if (!el) return

    const measure = () => setSize({
      width:  el.clientWidth || 290,
      height: el.clientHeight || 240,
    })
    measure()

    if (typeof ResizeObserver === 'undefined') {
      window.addEventListener('resize', measure)
      return () => window.removeEventListener('resize', measure)
    }
    const observer = new ResizeObserver(measure)
    observer.observe(el)
    return () => observer.disconnect()
  }, [])

  // Distinct cited files, in citation order, with the via_graph flag
  // carried along so the panel can label those nodes.
  const cited = useMemo(() => {
    const out = []
    const seen = new Set()
    for (const c of citations || []) {
      const file = c?.file
      if (!file || seen.has(file)) continue
      seen.add(file)
      out.push({
        file,
        via_graph: Boolean(c.via_graph),
        graph_relation: c.graph_relation || null,
      })
    }
    return out
  }, [citations])

  const { nodes, links } = useMemo(() => {
    const citedFiles = cited.map(c => c.file)
    const citedSet = new Set(citedFiles)

    const resolvedIds = new Set(
      (snapshot?.nodes || [])
        .filter(n => n.resolved && n.id)
        .map(n => n.id),
    )
    const byId = new Map((snapshot?.nodes || []).map(n => [n.id, n]))

    // One hop out from the cited files, resolved nodes only. External
    // "ext:" nodes are excluded by the resolved check and never reach
    // here, which is deliberate: there is no document behind them to open.
    const neighbours = []
    const seenNeighbour = new Set()
    for (const edge of snapshot?.edges || []) {
      const source = endpointId(edge.source)
      const target = endpointId(edge.target)

      for (const [near, far] of [[source, target], [target, source]]) {
        if (!citedSet.has(near)) continue
        if (citedSet.has(far) || seenNeighbour.has(far)) continue
        if (!resolvedIds.has(far)) continue
        seenNeighbour.add(far)
        neighbours.push(far)
      }
    }

    // Cited files first, so the cap drops linked-only files before it
    // drops anything the answer actually used.
    const keptFiles = [...citedFiles, ...neighbours].slice(0, MAX_NODES)
    const keptSet = new Set(keptFiles)

    const nodeList = keptFiles.map(file => {
      const meta = byId.get(file) || {}
      const citation = cited.find(c => c.file === file)
      return {
        id: file,
        label: shortLabel(file),
        stem: stemOf(file),
        cited: citedSet.has(file),
        via_graph: Boolean(citation?.via_graph),
        graph_relation: citation?.graph_relation || null,
        gr_number: meta.gr_number || '',
        department: meta.department || '',
        doc_date: meta.doc_date || '',
      }
    })

    const linkList = (snapshot?.edges || [])
      .filter(e => keptSet.has(endpointId(e.source)) && keptSet.has(endpointId(e.target)))
      .map(e => ({
        source: endpointId(e.source),
        target: endpointId(e.target),
        relation: e.relation || 'references',
      }))

    return { nodes: nodeList, links: linkList }
  }, [snapshot, cited])

  const graphData = useMemo(() => ({ nodes, links }), [nodes, links])

  const paintNode = (node, ctx, scale) => {
    const radius = nodeRadius(node)
    ctx.beginPath()
    ctx.arc(node.x, node.y, radius, 0, 2 * Math.PI)
    ctx.fillStyle = node.cited ? CITED_COLOR : LINKED_COLOR
    ctx.fill()

    const fontSize = Math.max(9 / scale, 2.5)
    ctx.font = `${fontSize}px Inter, sans-serif`
    ctx.textAlign = 'center'
    ctx.textBaseline = 'top'
    ctx.fillStyle = node.cited ? '#E8EAF0' : '#8A93A6'
    ctx.fillText(node.label, node.x, node.y + radius + 1.5)
  }

  // Same rules as the graph page: the line and its arrowhead are drawn by
  // one painter, in screen-pixel widths, trimmed to the node edges. The
  // library's linkDirectionalArrow pass is off below, because it draws
  // even under a custom link painter and left arrowheads with no line.
  const paintLink = (link, ctx, globalScale) => {
    const a = link.source
    const b = link.target
    if (!a || !b || typeof a !== 'object' || typeof b !== 'object') return
    if (!Number.isFinite(a.x) || !Number.isFinite(a.y)) return
    if (!Number.isFinite(b.x) || !Number.isFinite(b.y)) return

    const scale = globalScale || 1
    const dx = b.x - a.x
    const dy = b.y - a.y
    const length = Math.hypot(dx, dy)
    if (length < 0.5) return

    const ux = dx / length
    const uy = dy / length

    const startPad = nodeRadius(a) + 1
    const endPad   = nodeRadius(b) + 1.5
    if (length <= startPad + endPad) return

    const x1 = a.x + ux * startPad
    const y1 = a.y + uy * startPad
    const x2 = b.x - ux * endPad
    const y2 = b.y - uy * endPad

    const colour = relationColor(link.relation)
    ctx.globalAlpha = LINK_ALPHA
    ctx.strokeStyle = colour
    ctx.fillStyle = colour
    ctx.lineWidth = LINK_WIDTH_PX / scale
    ctx.lineCap = 'round'

    ctx.beginPath()
    ctx.moveTo(x1, y1)
    ctx.lineTo(x2, y2)
    ctx.stroke()

    const arrow = ARROW_LENGTH_PX / scale
    if (length > startPad + endPad + arrow * 0.6) {
      const half = arrow * 0.42
      const px = -uy
      const py = ux

      ctx.beginPath()
      ctx.moveTo(x2, y2)
      ctx.lineTo(x2 - ux * arrow + px * half, y2 - uy * arrow + py * half)
      ctx.lineTo(x2 - ux * arrow - px * half, y2 - uy * arrow - py * half)
      ctx.closePath()
      ctx.fill()
    }

    ctx.globalAlpha = 1
  }

  const open = (file) => {
    if (onOpenFile) onOpenFile(file, 1)
    else setViewer({ file, page: 1 })
  }

  if (!cited.length) return null

  const viaGraphFiles = cited.filter(c => c.via_graph)

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', minHeight: 0 }}>

      <div
        ref={wrapRef}
        style={{
          height: '240px', background: '#0D1016',
          border: '1px solid #2A2D3E', borderRadius: '8px',
          overflow: 'hidden', flexShrink: 0,
        }}
      >
        {nodes.length > 0 && (
          <ForceGraph2D
            graphData={graphData}
            width={size.width}
            height={size.height}
            backgroundColor="#0D1016"
            nodeId="id"
            nodeLabel={n => `${n.stem}${n.gr_number ? ` — ${n.gr_number}` : ''}`}
            nodeRelSize={5}
            nodeCanvasObject={paintNode}
            nodePointerAreaPaint={(node, color, ctx) => {
              ctx.fillStyle = color
              ctx.beginPath()
              ctx.arc(node.x, node.y, 9, 0, 2 * Math.PI)
              ctx.fill()
            }}
            linkLabel={l => l.relation}
            linkColor={l => relationColor(l.relation)}
            linkCanvasObject={paintLink}
            linkCanvasObjectMode={() => 'replace'}
            onNodeClick={node => open(node.id)}
            cooldownTicks={80}
            d3VelocityDecay={0.3}
          />
        )}
      </div>

      {/* Legend — the same three relation colours as the graph page. */}
      {links.length > 0 && (
        <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
          {[['supersedes', '#FF4D4D'], ['amends', '#FF6B00'], ['references', '#4A6FA5']]
            .map(([name, colour]) => (
              <span key={name} style={{
                display: 'flex', alignItems: 'center', gap: '5px',
                color: '#6B7280', fontSize: '0.68rem', fontFamily: 'monospace',
              }}>
                <span style={{ width: '12px', height: '2px', background: colour }} />
                {name}
              </span>
            ))}
        </div>
      )}

      {/* Clickable list, so the panel works when the layout is too dense
          to click a node accurately at 320px wide. */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '5px' }}>
        {nodes.map(node => (
          <button
            key={node.id}
            onClick={() => open(node.id)}
            title={`${node.stem}${node.department ? ` · ${node.department}` : ''}`}
            style={{
              display: 'flex', alignItems: 'center', gap: '7px',
              background: 'transparent', border: '1px solid #2A2D3E',
              borderRadius: '6px', padding: '6px 9px', cursor: 'pointer',
              textAlign: 'left', fontFamily: 'Inter, sans-serif', width: '100%',
            }}
          >
            <span style={{
              width: '7px', height: '7px', borderRadius: '50%', flexShrink: 0,
              background: node.cited ? CITED_COLOR : LINKED_COLOR,
            }} />
            <span style={{
              color: node.cited ? '#E8EAF0' : '#8A93A6', fontSize: '0.78rem',
              fontFamily: 'monospace', whiteSpace: 'nowrap',
              overflow: 'hidden', textOverflow: 'ellipsis', flex: 1, minWidth: 0,
            }}>
              {node.stem}
            </span>
            {node.via_graph && (
              <span style={{ color: '#6B7280', fontSize: '0.64rem', flexShrink: 0 }}>
                via graph
              </span>
            )}
            <ChevronRight size={12} color="#3A3F52" style={{ flexShrink: 0 }} />
          </button>
        ))}
      </div>

      {viaGraphFiles.length > 0 && (
        <div style={{
          display: 'flex', alignItems: 'center', gap: '6px',
          color: '#6B7280', fontSize: '0.7rem', lineHeight: '1.4',
        }}>
          <Network size={12} />
          {viaGraphFiles.length === 1
            ? '1 source was linked via graph'
            : `${viaGraphFiles.length} sources were linked via graph`}
        </div>
      )}

      {viewer && (
        <GRViewer
          file={viewer.file}
          page={viewer.page}
          onClose={() => setViewer(null)}
        />
      )}
    </div>
  )
}
