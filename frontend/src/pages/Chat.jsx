// frontend/src/pages/Chat.jsx
// Session management handled by ChatContext + Sidebar.
// This page only handles sending messages and displaying them.

import { useRef, useEffect, useState, useMemo } from 'react'
import { useChat } from '../context/ChatContext'
import { Landmark, Network, X } from 'lucide-react'
import client from '../api/client'
import CitationPill from '../components/CitationPill'
import CitationGraph from '../components/CitationGraph'
import GRViewer from '../components/GRViewer'

export default function Chat() {
  const {
    messages, history, activeSessionId,
    startExchange, completeExchange, failExchange,
    updateSessionTitle, createSession,
    setActiveSessionId,
  } = useChat()

  const [input,          setInput]          = useState('')
  const [loading,        setLoading]        = useState(false)
  const [error,          setError]          = useState('')
  // Collapsed state only. Whether the panel has anything to show is
  // derived from the messages below, not tracked separately, so it can
  // never disagree with what is on screen.
  const [panelOpen, setPanelOpen] = useState(true)

  // The document a citation or a graph node asked to open. One viewer for
  // the whole page rather than one per pill.
  const [viewer, setViewer] = useState(null)

  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  // The panel is hidden below this width rather than squeezed: at 320px
  // it would take most of a phone screen. Subscribed to, not measured in
  // a render, so a resize updates it without a layout read on every pass.
  const [isNarrow, setIsNarrow] = useState(
    () => typeof window !== 'undefined'
      && typeof window.matchMedia === 'function'
      && window.matchMedia('(max-width: 1100px)').matches,
  )

  useEffect(() => {
    if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') return
    const query = window.matchMedia('(max-width: 1100px)')
    const onChange = (e) => setIsNarrow(e.matches)
    query.addEventListener('change', onChange)
    return () => query.removeEventListener('change', onChange)
  }, [])

  // The citations of the most recent assistant message that has any.
  // Derived, so it follows the conversation with no extra state and no
  // request per message — CitationGraph fetches the graph snapshot once
  // per page load and caches it in a module-level variable.
  const latestCitations = useMemo(() => {
    for (let i = messages.length - 1; i >= 0; i -= 1) {
      const msg = messages[i]
      if (msg.role === 'assistant' && !msg.pending && msg.citations?.length) {
        return msg.citations
      }
    }
    return []
  }, [messages])

  const showPanel = latestCitations.length > 0 && panelOpen && !isNarrow

  const openDocument = (file, page) => setViewer({ file, page })

  const sendMessage = async (e) => {
    e.preventDefault()
    const query = input.trim()
    if (!query || loading) return

    // Auto-create session if none active
    let sessionId = activeSessionId
    if (!sessionId) {
      sessionId = await createSession()
      if (!sessionId) { setError('Could not create session.'); return }
    }

    setInput('')
    setError('')

    const userMsg = { role: 'user', content: query }

    // Writes the question plus ONE pending assistant bubble. Every exit
    // path below replaces that same bubble — none of them append a second.
    startExchange(userMsg)

    setLoading(true)

    try {
      const res = await client.post('/query/chat', { query, history })
      const { answer, citations, elapsed_sec, language } = res.data

      const assistantMsg = {
        role: 'assistant', content: answer,
        citations:   citations   || [],
        elapsed_sec: elapsed_sec || 0,
        language:    language    || 'english',
      }

      // Overwrites the pending bubble — it does not add one.
      completeExchange(userMsg, assistantMsg)

      // Save to MongoDB session
      try {
        await client.post(`/sessions/${sessionId}/messages`, {
          user_msg:      { role: 'user', content: query },
          assistant_msg: assistantMsg,
        })
        // Update title in sidebar if it was auto-titled
        if (query.length > 0) updateSessionTitle(sessionId, query.slice(0, 50))
      } catch { /* non-critical */ }

    } catch (err) {
      // Turn the pending bubble into the error itself. Leaving it pending
      // is what used to strand a placeholder on screen after a failure.
      failExchange(err.response?.data?.detail || 'Something went wrong.')
    } finally {
      setLoading(false)
    }
  }

  const suggestions = [
    'What is this Government Resolution about?',
    'या शासन निर्णयाचा विषय काय आहे?',
    'Who signed this resolution?',
    'What are the key provisions?',
    'What is the GR number and date?',
    'Does this GR supersede any previous order?',
  ]

  return (
    <div style={{
      display: 'flex', height: '100%',
      fontFamily: 'Inter, sans-serif', position: 'relative', overflow: 'hidden',
    }}>

      {/* ── MAIN CHAT ──────────────────────────────── */}
      <div style={{
        flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden',
        marginRight: showPanel ? '320px' : '0',
        transition: 'margin-right 0.25s ease',
      }}>

        {/* Messages */}
        <div style={{
          flex: 1, overflowY: 'auto',
          padding: '24px 32px',
          display: 'flex', flexDirection: 'column', gap: '20px',
        }}>

          {/* Empty state */}
          {messages.length === 0 && (
            <div style={{ textAlign: 'center', padding: '48px 24px' }}>
              <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '12px' }}><Landmark size={40} color="#FF6B00" /></div>
              <h2 style={{ color: '#FF6B00', fontSize: '1.4rem', margin: '0 0 8px' }}>
                नमस्कार! Hello!
              </h2>
              <p style={{ color: '#555', fontSize: '0.9rem', margin: '0 0 28px' }}>
                Ask anything about uploaded Government Resolutions.<br />
                English and Marathi both supported.
              </p>
              <div style={{
                display: 'flex', flexWrap: 'wrap', gap: '8px',
                justifyContent: 'center', maxWidth: '580px', margin: '0 auto',
              }}>
                {suggestions.map((s, i) => (
                  <button key={i} onClick={() => setInput(s)} style={{
                    background: '#1A1D2E', border: '1px solid #2A2D3E',
                    borderRadius: '20px', color: '#B0B4C8',
                    padding: '7px 14px', fontSize: '0.82rem',
                    cursor: 'pointer', fontFamily: 'Inter, sans-serif',
                  }}
                    onMouseEnter={e => e.target.style.borderColor = '#FF6B00'}
                    onMouseLeave={e => e.target.style.borderColor = '#2A2D3E'}
                  >
                    {s}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Message list */}
          {/*
            Rendered straight from `messages` — no filtering. There is
            exactly one entry per turn now, and the pending one renders as
            the loading state below, so there is nothing to hide.
          */}
          {messages.map((msg, i) => (
            <div key={i}>
              {msg.role === 'user' ? (
                <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
                  <div style={{
                    background: '#FF6B00', color: 'white',
                    padding: '12px 18px', borderRadius: '18px 18px 4px 18px',
                    maxWidth: '70%', fontSize: '0.95rem', lineHeight: '1.5',
                  }}>
                    {msg.content}
                  </div>
                </div>
              ) : (
                <div style={{ display: 'flex', gap: '12px', alignItems: 'flex-start' }}>
                  <div style={{
                    width: '36px', height: '36px', background: '#1E3A5F',
                    borderRadius: '50%', display: 'flex', alignItems: 'center',
                    justifyContent: 'center', flexShrink: 0, marginTop: '2px',
                  }}><Landmark size={18} color="#FF6B00" /></div>
                  <div style={{ maxWidth: '75%' }}>
                    <div style={{
                      background: '#1A1D2E',
                      border: `1px solid ${msg.error ? '#EF4444' : '#2A2D3E'}`,
                      borderRadius: '4px 18px 18px 18px',
                      padding: '14px 18px',
                      color: msg.pending ? '#555' : (msg.error ? '#EF4444' : '#E8EAF0'),
                      fontSize: msg.pending ? '0.88rem' : '0.95rem',
                      lineHeight: '1.65', whiteSpace: 'pre-wrap',
                    }}>
                      {/* Same bubble in all three states — the content is
                          swapped, so the answer never stacks under a
                          leftover loading bubble. */}
                      {msg.error && (
                        <div style={{
                          fontSize: '0.72rem', fontWeight: '700',
                          letterSpacing: '0.5px', textTransform: 'uppercase',
                          marginBottom: '6px',
                        }}>
                          Error
                        </div>
                      )}
                      {msg.pending
                        ? 'Searching GR documents…'
                        : msg.content}
                    </div>
                    {msg.citations?.length > 0 && (
                      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px', marginTop: '8px' }}>
                        {msg.citations.map((c, ci) => (
                          <CitationPill key={ci} citation={c} onOpen={openDocument} />
                        ))}
                      </div>
                    )}
                    {/* Timing only means something once an answer exists —
                        it used to render "undefineds" on the placeholder. */}
                    {!msg.pending && !msg.error && (
                      <div style={{ color: '#444', fontSize: '0.72rem', marginTop: '6px', fontFamily: 'monospace' }}>
                        {msg.elapsed_sec}s
                        {msg.language && (
                          <span style={{ marginLeft: '8px' }}>
                            · {msg.language === 'marathi' ? 'Marathi' : 'English'}
                          </span>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          ))}

          {/* No separate loading bubble. It used to render alongside the
              placeholder message, so a slow request showed two assistant
              bubbles at once. The pending message IS the loading state. */}

          {error && (
            <div style={{
              background: '#2A0F0F', border: '1px solid #EF4444',
              borderRadius: '8px', padding: '12px 16px',
              color: '#EF4444', fontSize: '0.85rem',
            }}>
              {error}
            </div>
          )}

          <div ref={bottomRef} />
        </div>

        {/* Input */}
        <div style={{
          borderTop: '1px solid #2A2D3E', padding: '14px 24px', background: '#0F1117',
        }}>
          <form onSubmit={sendMessage} style={{ display: 'flex', gap: '10px' }}>
            <input
              value={input}
              onChange={e => setInput(e.target.value)}
              placeholder="Type in English or Marathi… / मराठीत प्रश्न विचारा…"
              disabled={loading}
              style={{
                flex: 1, background: '#1A1D2E', border: '1px solid #2A2D3E',
                borderRadius: '12px', padding: '12px 18px', color: '#E8EAF0',
                fontSize: '0.95rem', outline: 'none', fontFamily: 'Inter, sans-serif',
              }}
              onFocus={e => e.target.style.borderColor = '#FF6B00'}
              onBlur={e  => e.target.style.borderColor = '#2A2D3E'}
            />
            <button
              type="submit" disabled={loading || !input.trim()}
              style={{
                background: loading || !input.trim() ? '#3A1800' : '#FF6B00',
                color: 'white', border: 'none', borderRadius: '12px',
                padding: '12px 20px', fontSize: '1rem',
                cursor: loading || !input.trim() ? 'not-allowed' : 'pointer',
                fontWeight: '600', fontFamily: 'Inter, sans-serif',
              }}
            >➤</button>
          </form>
        </div>
      </div>

      {/* ── CITATION GRAPH PANEL ───────────────────── */}
      {/* Shown only when the latest answer cited something. Collapsing it
          leaves the toggle below, so it can be brought back without
          sending another message. */}
      {showPanel && (
        <div style={{
          position: 'fixed', right: 0, top: '56px',
          width: '320px', height: 'calc(100vh - 56px)',
          background: '#1A1D2E', borderLeft: '1px solid #2A2D3E',
          zIndex: 50, display: 'flex', flexDirection: 'column',
        }}>
          <div style={{
            display: 'flex', alignItems: 'center', justifyContent: 'space-between',
            padding: '14px 16px', borderBottom: '1px solid #2A2D3E', flexShrink: 0,
          }}>
            <div style={{ minWidth: 0 }}>
              <div style={{
                color: '#E8EAF0', fontWeight: '600', fontSize: '0.92rem',
                display: 'flex', alignItems: 'center', gap: '8px',
              }}>
                <Network size={15} color="#FF6B00" />
                Citation graph
              </div>
              <div style={{ color: '#6B7280', fontSize: '0.72rem', fontFamily: 'monospace' }}>
                {latestCitations.length} citation{latestCitations.length !== 1 ? 's' : ''} in this answer
              </div>
            </div>
            <button
              onClick={() => setPanelOpen(false)}
              title="Hide the citation graph"
              style={{
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                width: '26px', height: '26px', flexShrink: 0,
                background: 'transparent', border: 'none', borderRadius: '6px',
                color: '#6B7280', cursor: 'pointer',
              }}
            >
              <X size={15} />
            </button>
          </div>

          <div style={{ flex: 1, overflowY: 'auto', padding: '12px' }}>
            <CitationGraph citations={latestCitations} onOpenFile={openDocument} />
            <div style={{
              color: '#5A6478', fontSize: '0.7rem', lineHeight: '1.45',
              marginTop: '12px', borderTop: '1px solid #2A2D3E', paddingTop: '10px',
            }}>
              Orange nodes are the documents this answer cited. Grey nodes are
              linked to them in the GR graph. Click any node to open the PDF.
            </div>
          </div>
        </div>
      )}

      {/* Re-open affordance, so hiding the panel is not one-way. */}
      {latestCitations.length > 0 && !panelOpen && !isNarrow && (
        <button
          onClick={() => setPanelOpen(true)}
          title="Show the citation graph"
          style={{
            position: 'fixed', right: '16px', top: '72px', zIndex: 50,
            display: 'flex', alignItems: 'center', gap: '7px',
            background: '#1A1D2E', border: '1px solid #2A2D3E',
            borderRadius: '8px', color: '#8A93A6', padding: '7px 12px',
            fontSize: '0.78rem', cursor: 'pointer', fontFamily: 'Inter, sans-serif',
          }}
        >
          <Network size={14} />
          Citation graph
        </button>
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