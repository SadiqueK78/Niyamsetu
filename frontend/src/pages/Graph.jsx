// frontend/src/pages/Graph.jsx
// ─────────────────────────────────────────────────────────
// GR relationship graph — which documents supersede, amend or merely
// cite which others.
//
// Why this page matters:
//   A citation pill tells you an answer came from GR5. It cannot tell you
//   that GR5 was cancelled by a later order. That relationship is the
//   difference between a correct answer and an obsolete one, and this is
//   the only place it is visible.
//
// ── WHY LABELS ARE STEMS ─────────────────────────────────
// The on-canvas label is the filename stem and nothing else. GR numbers
// in this corpus run to forty characters of Devanagari and punctuation
// ("मराग्रं-2025/प्र.क्र.58/ई-1093601"), and drawing them turned the graph
// into overlapping text with no readable structure. The number, the
// department and the subject are all in the side panel and the hover
// tooltip, where there is room for them.
//
// ── WHY THERE ARE THREE LAYOUTS ──────────────────────────
// A force layout only works on nodes that have edges. Resolved documents
// with no relationships were being flung to the rim or collapsing into
// the middle, so they are parked in a fixed row along the bottom instead
// — present and readable, visibly separate from the connected graph.
// External nodes sit further out because their link distance is longer,
// so they orbit whichever document cites them rather than competing with
// it for the centre.
//
// Reads GET /api/graph/. Nodes carry label, gr_number, subject, doc_date,
// department, header_page and resolved; edges carry relation, ref_index,
// match_reason, ambiguous and evidence.
// ─────────────────────────────────────────────────────────

import { useState, useEffect, useRef, useCallback, useMemo } from 'react'
import ForceGraph2D from 'react-force-graph-2d'
import { Network, X, Maximize2 } from 'lucide-react'
import client from '../api/client'

// Relation colours. Supersession and amendment change what a document
// MEANS, so they are warm and loud; a plain citation is cool and quiet.
const RELATION_COLOR = {
  supersedes: '#FF4D4D',
  amends:     '#FF6B00',
  references: '#4A6FA5',
  refers_to:  '#4A6FA5',
}

// The three relations the legend names, in the order it names them.
const RELATION_LEGEND = [
  ['supersedes', RELATION_COLOR.supersedes],
  ['amends',     RELATION_COLOR.amends],
  ['references', RELATION_COLOR.references],
]

// Ten department colours, picked to stay distinguishable on the dark
// canvas and from the relation colours above.
const DEPARTMENT_PALETTE = [
  '#6FB3FF', '#58D6A8', '#C792EA', '#FFD166',
  '#F78FB3', '#7FD4E8', '#A3D977', '#FFA26B',
  '#B5A8FF', '#E8A1C8',
]

const NODE_UNRESOLVED = '#6B6B6B'
const NODE_SELECTED   = '#FF6B00'
const NO_DEPARTMENT   = '#8A93A6'

// Labels longer than this are truncated. Eighteen characters is about as
// wide as a node can be before neighbours start colliding.
const LABEL_CHARS = 18

// Force parameters.
//
// CHARGE vs LINK was the whole layout bug: a charge of -420 with no
// distanceMax is a global repulsion that every node applies to every
// other, and across nineteen nodes it simply overpowered a 90-unit link,
// settling connected pairs about 250px apart. Capping the repulsion's
// reach and giving the link force an explicit strength lets the links win
// at short range, which is what makes components cluster.
const LINK_DISTANCE          = 90
const LINK_DISTANCE_EXTERNAL = 170
const LINK_STRENGTH          = 0.7
const CHARGE_STRENGTH        = -170
const CHARGE_MAX_DISTANCE    = 320

// Node radii in graph units. Lines are trimmed to these so an arrowhead
// lands on the node's edge rather than under its centre.
const NODE_RADIUS          = 5.5
const NODE_RADIUS_EXTERNAL = 3

// Link drawing, all in SCREEN pixels — every one of these is divided by
// globalScale at paint time, so a line is the same thickness zoomed in or
// out. The previous painter used graph units, which is why the lines
// vanished at fit zoom.
const LINK_WIDTH_PX          = 2.5
const LINK_WIDTH_FOCUS_PX    = 3.5
const ARROW_LENGTH_PX        = 9
const LINK_ALPHA             = 0.95
const LINK_ALPHA_DIMMED      = 0.12

// Isolated resolved nodes are parked on a fixed grid below the cluster.
// 130 units between columns is set against the 14-character label cap
// below: at that width two neighbouring labels cannot touch.
const PARKED_COLUMNS   = 4
const PARKED_SPACING   = 130
const PARKED_ROW_GAP   = 36
const PARKED_GAP       = 80
const PARKED_LABEL_MAX = 120

// Labels on parked nodes are capped harder than the rest: 18 characters
// does not fit inside a 130-unit column.
const LABEL_CHARS_PARKED = 14

// Label font size in screen pixels, divided by globalScale when painted.
const LABEL_SIZE_PX = 11

function relationColor(relation) {
  return RELATION_COLOR[relation] || RELATION_COLOR.references
}

// Stable across reloads and independent of how many departments exist, so
// a department keeps its colour when documents are added. Plain
// multiply-and-add accumulation — this only has to be deterministic, not
// cryptographic.
function departmentColor(department) {
  if (!department) return NO_DEPARTMENT
  let hash = 0
  for (let i = 0; i < department.length; i += 1) {
    hash = (hash * 31 + department.charCodeAt(i)) % 1000000007
  }
  return DEPARTMENT_PALETTE[hash % DEPARTMENT_PALETTE.length]
}

function isExternal(node) {
  return String(node?.id || '').startsWith('ext:')
}

function nodeRadius(node) {
  return node?.external || isExternal(node) ? NODE_RADIUS_EXTERNAL : NODE_RADIUS
}

// The filename stem, which is what every label shows.
function stemOf(node) {
  if (!node) return ''
  if (node.filename) return String(node.filename).replace(/\.pdf$/i, '')
  if (node.label && !isExternal(node)) return String(node.label)
  return String(node.id || '').replace(/\.pdf$/i, '')
}

function shortLabel(text, cap = LABEL_CHARS) {
  const value = String(text || '')
  return value.length > cap ? `${value.slice(0, cap - 1)}…` : value
}

// react-force-graph mutates the objects it is given, replacing each link's
// source/target id with the node object itself. Reading an id back has to
// cope with both forms.
function endpointId(end) {
  return (end && typeof end === 'object') ? end.id : end
}

export default function Graph() {
  const [nodes,    setNodes]    = useState([])
  const [edges,    setEdges]    = useState([])
  const [stats,    setStats]    = useState(null)
  const [loading,  setLoading]  = useState(true)
  const [error,    setError]    = useState('')
  const [selected, setSelected] = useState(null)
  const [hovered,  setHovered]  = useState(null)

  // Off by default. Most external nodes are documents cited once in
  // passing, and they outnumber the uploaded documents two to one — which
  // buries the relationships between the documents that are actually here.
  const [showExternal, setShowExternal] = useState(false)

  // null means "not chosen yet", resolved to the newest date once the data
  // arrives. Kept as an ISO string so it compares directly against
  // doc_date.
  const [cutoff, setCutoff] = useState(null)

  const wrapRef  = useRef(null)
  const graphRef = useRef(null)
  const [size, setSize] = useState({ width: 800, height: 620 })

  // No setLoading(true)/setError('') prologue: this runs once on mount and
  // both states already start that way.
  const loadGraph = useCallback(async () => {
    try {
      const res = await client.get('/graph/')
      // Copied before handing them to the force graph, which writes
      // simulation state (x, y, vx, vy) onto whatever it is given.
      setNodes((res.data.nodes || []).map(n => ({ ...n })))
      setEdges((res.data.edges || []).map(e => ({ ...e })))
      setStats(res.data.stats || null)
    } catch (err) {
      setError(err.response?.data?.detail || 'Could not load the relationship graph.')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    loadGraph()
  }, [loadGraph])

  // The canvas needs explicit pixel dimensions, so the container is
  // measured rather than sized in CSS.
  useEffect(() => {
    const el = wrapRef.current
    if (!el) return

    const measure = () => setSize({
      width:  el.clientWidth  || 800,
      height: el.clientHeight || 620,
    })
    measure()

    if (typeof ResizeObserver === 'undefined') {
      window.addEventListener('resize', measure)
      return () => window.removeEventListener('resize', measure)
    }
    const observer = new ResizeObserver(measure)
    observer.observe(el)
    return () => observer.disconnect()
  }, [loading])

  // ── Derived data ────────────────────────────────────────

  const resolvedCount = nodes.filter(n => n.resolved).length
  const externalCount = nodes.length - resolvedCount

  const dates = useMemo(
    () => nodes.map(n => n.doc_date).filter(Boolean).sort(),
    [nodes],
  )
  const minDate = dates.length ? dates[0] : null
  const maxDate = dates.length ? dates[dates.length - 1] : null

  // The distinct dates, in order, so the slider steps between real
  // document dates rather than over empty calendar days.
  const dateSteps = useMemo(() => Array.from(new Set(dates)), [dates])

  // Defaults to the newest date, i.e. showing everything.
  const activeCutoff = cutoff ?? maxDate

  const departments = useMemo(() => {
    const seen = new Map()
    nodes.forEach(n => {
      if (n.resolved && n.department && !seen.has(n.department)) {
        seen.set(n.department, departmentColor(n.department))
      }
    })
    return Array.from(seen.entries())
  }, [nodes])

  // Which nodes survive the timeline and the external checkbox. Computed
  // before layout, because layout depends on who has edges to whom among
  // the survivors.
  const keptIds = useMemo(() => {
    const out = new Set()
    nodes.forEach(n => {
      if (!n.resolved && !showExternal) return
      // A node with no date is always shown: the timeline can only speak
      // about documents whose date is known, and hiding the rest would
      // silently drop them from the graph.
      if (activeCutoff && n.doc_date && n.doc_date > activeCutoff) return
      out.add(n.id)
    })
    return out
  }, [nodes, showExternal, activeCutoff])

  const visibleEdges = useMemo(
    () => edges.filter(e =>
      keptIds.has(endpointId(e.source)) && keptIds.has(endpointId(e.target))),
    [edges, keptIds],
  )

  // Degree among RESOLVED nodes only, which is what decides whether a
  // document belongs in the force layout. An external-only neighbour does
  // not rescue a document from the parked row, because that row is about
  // documents with no relationships to other real documents.
  const resolvedDegree = useMemo(() => {
    const byId = new Map(nodes.map(n => [n.id, n]))
    const degree = new Map()
    edges.forEach(e => {
      const a = byId.get(endpointId(e.source))
      const b = byId.get(endpointId(e.target))
      if (!a || !b || !a.resolved || !b.resolved) return
      if (!keptIds.has(a.id) || !keptIds.has(b.id)) return
      degree.set(a.id, (degree.get(a.id) || 0) + 1)
      degree.set(b.id, (degree.get(b.id) || 0) + 1)
    })
    return degree
  }, [nodes, edges, keptIds])

  // Any edge at all, including to an external node — used so an external
  // node's own placement is left to the force layout.
  const anyDegree = useMemo(() => {
    const degree = new Map()
    visibleEdges.forEach(e => {
      const a = endpointId(e.source)
      const b = endpointId(e.target)
      degree.set(a, (degree.get(a) || 0) + 1)
      degree.set(b, (degree.get(b) || 0) + 1)
    })
    return degree
  }, [visibleEdges])

  const visibleNodes = useMemo(() => {
    const kept = nodes.filter(n => keptIds.has(n.id))

    // Resolved, connected to another resolved document: free in the force
    // layout. Resolved and isolated: parked. External: free, but pushed
    // out by a longer link distance.
    const parked = kept.filter(n =>
      n.resolved && !(resolvedDegree.get(n.id) > 0) && !(anyDegree.get(n.id) > 0))

    const parkedIndex = new Map(parked.map((n, i) => [n.id, i]))
    const columns = Math.min(PARKED_COLUMNS, Math.max(1, parked.length))
    const rowWidth = (columns - 1) * PARKED_SPACING

    // The grid starts below the connected cluster. The cluster's real
    // extent is not known until the simulation runs, so it is estimated
    // from how many nodes are in it — a force layout's radius grows with
    // roughly the square root of its node count — and PARKED_GAP is added
    // on top so the row cannot sit inside the cluster.
    const connectedCount = kept.length - parked.length
    const clusterRadius = 60 + 26 * Math.sqrt(Math.max(0, connectedCount))
    const parkedTop = clusterRadius + PARKED_GAP

    return kept.map(n => {
      const node = {
        ...n,
        stem: stemOf(n),
        external: isExternal(n),
        connected: (resolvedDegree.get(n.id) || 0) > 0,
      }

      const index = parkedIndex.get(n.id)
      if (index !== undefined) {
        const row = Math.floor(index / columns)
        const col = index % columns
        // fx/fy pin a node in place: the simulation reads them and skips
        // its own positioning, which is what keeps this row tidy.
        node.fx = col * PARKED_SPACING - rowWidth / 2
        node.fy = parkedTop + row * PARKED_ROW_GAP
        node.parked = true
      }

      return node
    })
  }, [nodes, keptIds, resolvedDegree, anyDegree])

  const graphData = useMemo(
    () => ({ nodes: visibleNodes, links: visibleEdges }),
    [visibleNodes, visibleEdges],
  )

  // A node hidden by a control cannot stay selected, and the panel lists
  // only edges that are currently drawn, so panel and canvas never
  // disagree about what exists.
  const active = useMemo(() => {
    if (!selected) return null
    return keptIds.has(selected.id) ? selected : null
  }, [selected, keptIds])

  const nodeById = useMemo(
    () => new Map(visibleNodes.map(n => [n.id, n])),
    [visibleNodes],
  )

  // The node whose neighbourhood is highlighted: whatever is hovered,
  // falling back to the selection so a click keeps the highlight.
  const focusId = hovered?.id || active?.id || null

  const neighbourIds = useMemo(() => {
    if (!focusId) return null
    const out = new Set([focusId])
    visibleEdges.forEach(e => {
      const a = endpointId(e.source)
      const b = endpointId(e.target)
      if (a === focusId) out.add(b)
      if (b === focusId) out.add(a)
    })
    return out
  }, [focusId, visibleEdges])

  const incoming = active
    ? visibleEdges.filter(e => endpointId(e.target) === active.id)
    : []
  const outgoing = active
    ? visibleEdges.filter(e => endpointId(e.source) === active.id)
    : []

  // ── View control ────────────────────────────────────────

  const fitToView = useCallback(() => {
    graphRef.current?.zoomToFit(400, 60)
  }, [])

  // Force tuning. Imperative calls on a ref, so no state is involved.
  //
  // Re-applied whenever the data changes, because the component rebuilds
  // its forces when it receives new graphData and a distance set once at
  // mount would be discarded.
  useEffect(() => {
    const fg = graphRef.current
    if (!fg) return

    const link = fg.d3Force('link')
    if (link) {
      link.distance(l => {
        const a = nodeById.get(endpointId(l.source))
        const b = nodeById.get(endpointId(l.target))
        return (a?.external || b?.external) ? LINK_DISTANCE_EXTERNAL : LINK_DISTANCE
      })
      // Explicit strength. d3's default is 1/min(degree), which on these
      // small components is weak enough that the charge below wins and
      // connected nodes drift apart instead of clustering.
      link.strength(LINK_STRENGTH)
    }

    const charge = fg.d3Force('charge')
    if (charge) {
      charge.strength(CHARGE_STRENGTH)
      // Local repulsion only. Beyond this it contributes nothing, so a
      // link at 90 units is no longer fighting every node on the canvas.
      charge.distanceMax(CHARGE_MAX_DISTANCE)
    }
  }, [nodeById, graphData, loading])

  // Fit once per data change, after the simulation has settled — see
  // onEngineStop on the graph below. The flag is what makes it once:
  // onEngineStop also fires after a drag, and re-fitting then would yank
  // the view out from under the user.
  const fittedRef = useRef(false)

  useEffect(() => {
    fittedRef.current = false
  }, [graphData])

  const fitOnceWhenSettled = useCallback(() => {
    if (fittedRef.current) return
    fittedRef.current = true
    graphRef.current?.zoomToFit(400, 60)
  }, [])

  const centreOn = (id) => {
    const node = nodeById.get(id)
    if (!node) return
    setSelected(node)
    if (typeof node.x === 'number' && typeof node.y === 'number') {
      graphRef.current?.centerAt(node.x, node.y, 500)
      graphRef.current?.zoom(2.2, 500)
    }
  }

  // ── Canvas painters ─────────────────────────────────────

  const dimmed = (id) => Boolean(neighbourIds) && !neighbourIds.has(id)

  const paintNode = (node, ctx, globalScale) => {
    const scale = globalScale || 1
    const isSelected = active && node.id === active.id
    const radius = nodeRadius(node)
    const faded = dimmed(node.id)

    ctx.globalAlpha = faded ? 0.18 : 1

    ctx.beginPath()
    ctx.arc(node.x, node.y, radius, 0, 2 * Math.PI)
    ctx.fillStyle = isSelected
      ? NODE_SELECTED
      : (node.external ? NODE_UNRESOLVED : departmentColor(node.department))
    ctx.fill()

    if (isSelected) {
      ctx.lineWidth = 2 / scale
      ctx.strokeStyle = '#FFFFFF'
      ctx.stroke()
    }

    // External nodes carry no canvas label at all — there are dozens of
    // them and their labels are whole sentences. The side panel and the
    // hover tooltip show the full text.
    if (!node.external) {
      // Parked nodes sit in a grid, so their labels are capped to the
      // column width: the font shrinks rather than letting two labels
      // touch. Connected nodes are spread by the force layout and keep a
      // constant screen size.
      const text = node.parked
        ? shortLabel(node.stem, LABEL_CHARS_PARKED)
        : shortLabel(node.stem)

      let fontSize = LABEL_SIZE_PX / scale
      ctx.font = `${fontSize}px Inter, sans-serif`

      if (node.parked) {
        const width = ctx.measureText(text).width
        if (width > PARKED_LABEL_MAX) {
          // measureText returns graph units because the context is
          // already scaled, so this is a true "does it fit the column"
          // test at whatever zoom is current.
          fontSize *= PARKED_LABEL_MAX / width
          ctx.font = `${fontSize}px Inter, sans-serif`
        }
      }

      ctx.textAlign = 'center'
      ctx.textBaseline = 'top'
      ctx.fillStyle = faded ? '#5A6478' : '#D6DCE8'
      ctx.fillText(text, node.x, node.y + radius + 2)
    }

    ctx.globalAlpha = 1
  }

  // Draws the line AND its arrowhead, so the two can never disagree.
  // The library's own linkDirectionalArrow pass is switched off for
  // exactly that reason: with linkCanvasObjectMode 'replace' it kept
  // drawing arrows while this painter drew the lines, which is how an
  // arrowhead ended up floating with no line under it.
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

    // Trimmed to each node's edge, so the line starts outside the source
    // circle and the arrow tip lands on the target's rim.
    const startPad = nodeRadius(a) + 1
    const endPad   = nodeRadius(b) + 1.5
    if (length <= startPad + endPad) return

    const x1 = a.x + ux * startPad
    const y1 = a.y + uy * startPad
    const x2 = b.x - ux * endPad
    const y2 = b.y - uy * endPad

    const focused = Boolean(neighbourIds)
    const isNeighbour = !focused || (neighbourIds.has(a.id) && neighbourIds.has(b.id))

    const colour = relationColor(link.relation)
    ctx.globalAlpha = isNeighbour ? LINK_ALPHA : LINK_ALPHA_DIMMED
    ctx.strokeStyle = colour
    ctx.fillStyle = colour
    ctx.lineWidth = ((focused && isNeighbour) ? LINK_WIDTH_FOCUS_PX : LINK_WIDTH_PX) / scale
    ctx.lineCap = 'round'

    ctx.beginPath()
    ctx.moveTo(x1, y1)
    ctx.lineTo(x2, y2)
    ctx.stroke()

    // Arrowhead: a fixed screen size, pointing from the citing document
    // to the cited one, with its tip exactly at the end of the line it
    // was just drawn on.
    const arrow = ARROW_LENGTH_PX / scale
    if (length > startPad + endPad + arrow * 0.6) {
      const half = arrow * 0.42
      // Perpendicular to the line, for the two base corners.
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

  // ── Small render helpers ────────────────────────────────

  const panelLabel = (text) => (
    <div style={{ color: '#6B7280', fontSize: '0.7rem', textTransform: 'uppercase',
                  letterSpacing: '0.5px', marginBottom: '3px' }}>
      {text}
    </div>
  )

  const panelField = (label, value, mono) => (
    <div style={{ marginBottom: '12px' }}>
      {panelLabel(label)}
      <div style={{
        color: '#C8CEDA', fontSize: '0.82rem', lineHeight: '1.5',
        fontFamily: mono ? 'monospace' : 'Inter, sans-serif',
        wordBreak: 'break-word',
      }}>
        {value || '—'}
      </div>
    </div>
  )

  const edgeRow = (edge, key, direction) => {
    const otherId = direction === 'out'
      ? endpointId(edge.target)
      : endpointId(edge.source)
    const other = nodeById.get(otherId)
    const otherStem = other ? (other.stem || otherId) : otherId

    return (
      <div key={key} style={{
        border: '1px solid #2A2D3E', borderRadius: '6px',
        padding: '8px 10px', marginBottom: '8px', background: '#141726',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
          <span style={{
            background: '#0D2137', border: `1px solid ${relationColor(edge.relation)}`,
            color: relationColor(edge.relation), padding: '2px 9px', borderRadius: '20px',
            fontSize: '0.7rem', fontFamily: 'monospace', whiteSpace: 'nowrap',
          }}>
            {edge.relation}
          </span>
          <button
            onClick={() => centreOn(otherId)}
            title={`Centre the view on ${otherStem}`}
            style={{
              background: 'transparent', border: 'none', padding: 0,
              color: '#6FB3FF', fontSize: '0.78rem', fontFamily: 'monospace',
              cursor: 'pointer', textAlign: 'left', wordBreak: 'break-all',
              textDecoration: 'underline', textDecorationColor: '#2A4766',
            }}
          >
            {direction === 'out' ? '→ ' : '← '}{shortLabel(otherStem)}
          </button>
        </div>

        <div style={{ color: '#5A6478', fontSize: '0.7rem', fontFamily: 'monospace', marginTop: '5px' }}>
          ref {edge.ref_index ?? '-'} · {edge.match_reason || 'unresolved'}
          {edge.ambiguous ? ' · ambiguous' : ''}
        </div>

        {edge.evidence && (
          <div style={{
            color: '#99A2B5', fontSize: '0.74rem', marginTop: '6px',
            borderLeft: '2px solid #2A2D3E', paddingLeft: '8px', lineHeight: '1.45',
            whiteSpace: 'pre-wrap', wordBreak: 'break-word',
          }}>
            {edge.evidence}
          </div>
        )}
      </div>
    )
  }

  const controlButton = {
    display: 'flex', alignItems: 'center', gap: '6px',
    background: 'transparent', border: '1px solid #2A2D3E',
    borderRadius: '6px', color: '#8A93A6', padding: '5px 11px',
    fontSize: '0.78rem', cursor: 'pointer', fontFamily: 'Inter, sans-serif',
  }

  // ── Render ──────────────────────────────────────────────

  return (
    <div style={{
      minHeight: '100%', background: '#0F1117',
      padding: '28px 32px', fontFamily: 'Inter, sans-serif',
    }}>

      <div style={{ marginBottom: '18px' }}>
        <h1 style={{ color: '#E8EAF0', fontSize: '1.4rem', margin: '0 0 4px',
                     fontWeight: '700', display: 'flex', alignItems: 'center', gap: '10px' }}>
          <Network size={20} color="#FF6B00" />
          GR Relationship Graph
        </h1>
        <p style={{ color: '#6B7280', fontSize: '0.85rem', margin: 0 }}>
          Which government resolutions supersede, amend or cite which others.
          Node colour is the issuing department; hover a node to isolate its
          neighbours. Documents with no relationships sit in the row below.
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

      {/* Controls + counts */}
      {!loading && nodes.length > 0 && (
        <div style={{ marginBottom: '14px' }}>
          <div style={{
            display: 'flex', gap: '18px', flexWrap: 'wrap',
            alignItems: 'center', marginBottom: '10px',
          }}>
            <label style={{
              display: 'flex', alignItems: 'center', gap: '8px',
              color: '#C8CEDA', fontSize: '0.85rem', cursor: 'pointer',
              userSelect: 'none',
            }}>
              <input
                type="checkbox"
                checked={showExternal}
                onChange={e => setShowExternal(e.target.checked)}
                style={{ width: '15px', height: '15px', cursor: 'pointer', accentColor: '#FF6B00' }}
              />
              Show external references
            </label>

            <button onClick={fitToView} style={controlButton} title="Fit the whole graph in view">
              <Maximize2 size={13} />
              Fit to view
            </button>

            <div style={{ color: '#8A93A6', fontSize: '0.8rem', fontFamily: 'monospace' }}>
              <span style={{ color: '#E8EAF0', fontWeight: '700' }}>{resolvedCount}</span> resolved
              <span style={{ margin: '0 8px', color: '#3A3F52' }}>|</span>
              <span style={{ color: '#E8EAF0', fontWeight: '700' }}>{externalCount}</span> external
            </div>

            <div style={{ color: '#8A93A6', fontSize: '0.8rem', fontFamily: 'monospace' }}>
              <span style={{ color: '#E8EAF0', fontWeight: '700' }}>{visibleNodes.length}</span> shown
              <span style={{ margin: '0 8px', color: '#3A3F52' }}>|</span>
              <span style={{ color: '#E8EAF0', fontWeight: '700' }}>{visibleEdges.length}</span> edges
              {stats && (
                <>
                  <span style={{ margin: '0 8px', color: '#3A3F52' }}>|</span>
                  <span style={{ color: '#E8EAF0', fontWeight: '700' }}>{stats.supersedes_count ?? 0}</span> supersedes
                  <span style={{ margin: '0 8px', color: '#3A3F52' }}>|</span>
                  <span style={{ color: '#E8EAF0', fontWeight: '700' }}>{stats.amends_count ?? 0}</span> amends
                </>
              )}
            </div>
          </div>

          {/* Timeline */}
          {dateSteps.length > 1 && (
            <div style={{
              display: 'flex', alignItems: 'center', gap: '12px',
              flexWrap: 'wrap', marginBottom: '10px',
            }}>
              <span style={{ color: '#6B7280', fontSize: '0.78rem',
                             textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                As of
              </span>
              <input
                type="range"
                min={0}
                max={dateSteps.length - 1}
                step={1}
                value={Math.max(0, dateSteps.indexOf(activeCutoff))}
                onChange={e => setCutoff(dateSteps[Number(e.target.value)])}
                style={{ width: '280px', cursor: 'pointer', accentColor: '#FF6B00' }}
              />
              <span style={{ color: '#E8EAF0', fontSize: '0.82rem',
                             fontFamily: 'monospace', fontWeight: '700' }}>
                {activeCutoff || '—'}
              </span>
              <span style={{ color: '#5A6478', fontSize: '0.74rem', fontFamily: 'monospace' }}>
                ({minDate} to {maxDate})
              </span>
              {activeCutoff !== maxDate && (
                <button onClick={() => setCutoff(maxDate)} style={controlButton}>
                  Reset
                </button>
              )}
            </div>
          )}

          {/* Legends: relations, then departments. */}
          <div style={{ display: 'flex', gap: '14px', flexWrap: 'wrap',
                        alignItems: 'center', marginBottom: '8px' }}>
            {RELATION_LEGEND.map(([name, colour]) => (
              <span key={name} style={{
                display: 'flex', alignItems: 'center', gap: '6px',
                color: '#8A93A6', fontSize: '0.76rem', fontFamily: 'monospace',
              }}>
                <span style={{ width: '16px', height: '2.5px', background: colour, flexShrink: 0 }} />
                {name}
              </span>
            ))}
          </div>

          {departments.length > 0 && (
            <div style={{ display: 'flex', gap: '14px', flexWrap: 'wrap', alignItems: 'center' }}>
              {departments.map(([name, colour]) => (
                <span key={name} style={{
                  display: 'flex', alignItems: 'center', gap: '6px',
                  color: '#8A93A6', fontSize: '0.76rem',
                }}>
                  <span style={{
                    width: '9px', height: '9px', borderRadius: '50%',
                    background: colour, flexShrink: 0,
                  }} />
                  {name}
                </span>
              ))}
              {showExternal && externalCount > 0 && (
                <span style={{ display: 'flex', alignItems: 'center', gap: '6px',
                               color: '#8A93A6', fontSize: '0.76rem' }}>
                  <span style={{ width: '7px', height: '7px', borderRadius: '50%',
                                 background: NODE_UNRESOLVED, flexShrink: 0 }} />
                  not uploaded
                </span>
              )}
            </div>
          )}
        </div>
      )}

      {loading ? (
        <div style={{ color: '#6B7280', fontSize: '0.9rem', padding: '40px 0' }}>
          Loading graph...
        </div>
      ) : nodes.length === 0 ? (
        <div style={{
          border: '1px dashed #2A2D3E', borderRadius: '10px',
          padding: '48px 24px', textAlign: 'center', color: '#6B7280',
        }}>
          <Network size={30} color="#2A2D3E" />
          <div style={{ color: '#8A93A6', fontSize: '0.95rem', marginTop: '12px', fontWeight: '600' }}>
            No relationships yet
          </div>
          <div style={{ fontSize: '0.84rem', marginTop: '6px', lineHeight: '1.5' }}>
            The graph is built from the embedded documents.
            Ask an admin to run embedding, and the relationships
            between the uploaded GRs will appear here.
          </div>
        </div>
      ) : visibleNodes.length === 0 ? (
        <div style={{
          border: '1px dashed #2A2D3E', borderRadius: '10px',
          padding: '40px 24px', textAlign: 'center', color: '#6B7280',
        }}>
          <div style={{ color: '#8A93A6', fontSize: '0.92rem', fontWeight: '600' }}>
            Nothing to show with the current filters
          </div>
          <div style={{ fontSize: '0.84rem', marginTop: '6px', lineHeight: '1.5' }}>
            Move the timeline later, or tick Show external references.
          </div>
        </div>
      ) : (
        <div style={{ display: 'flex', gap: '16px', alignItems: 'flex-start' }}>

          <div
            ref={wrapRef}
            style={{
              flex: 1, minWidth: 0, height: '620px',
              background: '#141726', border: '1px solid #2A2D3E',
              borderRadius: '10px', overflow: 'hidden',
            }}
          >
            <ForceGraph2D
              ref={graphRef}
              graphData={graphData}
              width={size.width}
              height={size.height}
              backgroundColor="#141726"
              nodeId="id"
              nodeLabel={n => {
                const rows = [
                  `<b>${n.stem || n.id}</b>`,
                  n.gr_number ? n.gr_number : '',
                  n.department ? n.department : '',
                  n.doc_date ? n.doc_date : '',
                  n.external ? (n.label || '') : '',
                ].filter(Boolean)
                return `<div style="font-family:Inter,sans-serif;font-size:12px;
                        max-width:280px;line-height:1.45">${rows.join('<br/>')}</div>`
              }}
              nodeRelSize={5}
              nodeCanvasObject={paintNode}
              nodePointerAreaPaint={(node, color, ctx) => {
                ctx.fillStyle = color
                ctx.beginPath()
                ctx.arc(node.x, node.y, node.external ? 6 : 10, 0, 2 * Math.PI)
                ctx.fill()
              }}
              linkLabel={l => {
                const evidence = String(l.evidence || '').slice(0, 200)
                return `<div style="font-family:Inter,sans-serif;font-size:12px;
                        max-width:320px;line-height:1.45"><b>${l.relation}</b>${
                  evidence ? `<br/>${evidence}` : ''}</div>`
              }}
              linkCanvasObject={paintLink}
              linkCanvasObjectMode={() => 'replace'}
              linkColor={l => relationColor(l.relation)}
              // No linkDirectionalArrow* here on purpose: paintLink draws
              // the arrowhead itself. The library's arrow pass runs even
              // under 'replace' mode, so leaving it on produced a second
              // arrow with no line beneath it.
              onEngineStop={fitOnceWhenSettled}
              onNodeHover={node => setHovered(node || null)}
              onNodeClick={node => setSelected(node)}
              onBackgroundClick={() => setSelected(null)}
              enableNodeDrag
              enableZoomInteraction
              enablePanInteraction
              cooldownTicks={150}
            />
          </div>

          {/* Side panel */}
          {active && (
            <div style={{
              width: '360px', flexShrink: 0, maxHeight: '620px', overflowY: 'auto',
              background: '#1A1D2E', border: '1px solid #2A2D3E',
              borderRadius: '10px', padding: '16px',
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between',
                            alignItems: 'flex-start', gap: '10px', marginBottom: '14px' }}>
                <div style={{ color: '#E8EAF0', fontSize: '1rem', fontWeight: '700',
                              wordBreak: 'break-word' }}>
                  {active.stem || active.id}
                </div>
                <button
                  onClick={() => setSelected(null)}
                  title="Close"
                  style={{
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    width: '26px', height: '26px', flexShrink: 0, background: 'transparent',
                    border: 'none', borderRadius: '6px', color: '#6B7280', cursor: 'pointer',
                  }}
                >
                  <X size={15} />
                </button>
              </div>

              <div style={{ marginBottom: '12px' }}>
                <span style={{
                  background: active.external ? '#232323' : '#0D2137',
                  border: `1px solid ${active.external ? '#3A3A3A' : '#1E3A5F'}`,
                  color: active.external ? NODE_UNRESOLVED : departmentColor(active.department),
                  padding: '3px 12px', borderRadius: '20px',
                  fontSize: '0.72rem', fontFamily: 'monospace',
                }}>
                  {active.external ? 'not uploaded' : 'in corpus'}
                </span>
              </div>

              {/* External nodes have no stem worth showing, but they do
                  have the reference text they were built from. */}
              {active.external
                ? panelField('Referenced as', active.label, false)
                : panelField('Stem', active.stem, false)}
              {panelField('GR number', active.gr_number, true)}
              {panelField('Department', active.department, false)}
              {panelField('Date', active.doc_date, true)}
              {panelField('Subject', active.subject, false)}

              <div style={{ borderTop: '1px solid #2A2D3E', paddingTop: '14px', marginBottom: '14px' }}>
                {panelLabel(`Cites (${outgoing.length})`)}
                {outgoing.length === 0
                  ? <div style={{ color: '#5A6478', fontSize: '0.78rem' }}>
                      This document cites nothing in the graph.
                    </div>
                  : outgoing.map((e, i) => edgeRow(e, `out-${i}`, 'out'))}
              </div>

              <div style={{ borderTop: '1px solid #2A2D3E', paddingTop: '14px' }}>
                {panelLabel(`Cited by (${incoming.length})`)}
                {incoming.length === 0
                  ? <div style={{ color: '#5A6478', fontSize: '0.78rem' }}>
                      Nothing in the graph cites this document.
                    </div>
                  : incoming.map((e, i) => edgeRow(e, `in-${i}`, 'in'))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
