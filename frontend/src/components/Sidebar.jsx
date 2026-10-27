// frontend/src/components/Sidebar.jsx
// Sidebar navigation with inline chat session dropdown.
// Admin items on top, GR Summaries + Chat dropdown below.

import { useState, useRef, useEffect } from 'react'
import { useNavigate, useLocation } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import { useChat } from '../context/ChatContext'
import {
  Upload, Database, BarChart3, Users, ClipboardList,
  MessageSquare, Pin, PinOff, Pencil, Trash2, LogOut,
} from 'lucide-react'
import { useAdmin } from '../context/AdminContext'

const ADMIN_ITEMS = [
  { icon: <Upload size={16} />,         label: 'Upload GR',      path: '/admin/upload'    },
  { icon: <Database size={16} />,       label: 'Knowledge Base', path: '/admin/knowledge' },
  { icon: <BarChart3 size={16} />,      label: 'Analytics',      path: '/admin/analytics' },
  { icon: <Users size={16} />,          label: 'Users',          path: '/admin/users'     },
]

export default function Sidebar({ collapsed }) {
  const { isAdmin, logout } = useAuth()
  const navigate  = useNavigate()
  const location  = useLocation()

  const {
    sessions, activeSessionId,
    createSession, loadSession, deleteSession, renameSession, pinSession,
  } = useChat()

  const [chatExpanded, setChatExpanded] = useState(true)
  const [openMenuId,   setOpenMenuId]   = useState(null)
  const [renamingId,   setRenamingId]   = useState(null)
  const [renameValue,  setRenameValue]  = useState('')

  const menuRef   = useRef(null)
  const renameRef = useRef(null)

  // Close menu on outside click
  useEffect(() => {
    const h = (e) => {
      if (menuRef.current && !menuRef.current.contains(e.target))
        setOpenMenuId(null)
    }
    document.addEventListener('mousedown', h)
    return () => document.removeEventListener('mousedown', h)
  }, [])

  useEffect(() => {
    if (renamingId) renameRef.current?.focus()
  }, [renamingId])

  const handleNewChat = async () => {
    await createSession()
    navigate('/chat')
  }

  const handleSessionClick = async (sessionId) => {
    await loadSession(sessionId)
    navigate('/chat')
    setOpenMenuId(null)
  }

  const handleLogout = async () => {
    await logout()
    navigate('/login', { replace: true })
  }

  const startRename = (session) => {
    setOpenMenuId(null)
    setRenamingId(session._id)
    setRenameValue(session.title)
  }

  const submitRename = async (sessionId) => {
    const t = renameValue.trim()
    if (t) await renameSession(sessionId, t)
    setRenamingId(null)
  }

  const isChatActive = location.pathname === '/chat'

  return (
    <>
      <div style={{
        width: collapsed ? '0px' : '220px',
        minWidth: collapsed ? '0px' : '220px',
        height: 'calc(100vh - 56px)',
        background: '#13151F',
        borderRight: collapsed ? 'none' : '1px solid #2A2D3E',
        overflow: 'hidden',
        transition: 'width 0.2s ease, min-width 0.2s ease',
        display: 'flex',
        flexDirection: 'column',
      }}>
        <div style={{
          opacity: collapsed ? 0 : 1,
          transition: 'opacity 0.15s',
          display: 'flex',
          flexDirection: 'column',
          height: '100%',
          padding: '10px 8px',
          overflow: 'hidden',
        }}>

          {/* ── ADMIN SECTION ──────────────────────── */}
          {isAdmin && (
            <>
              <SectionLabel>Admin</SectionLabel>
              {ADMIN_ITEMS.map(item => (
                <div key={item.path}>
                  <NavBtn
                    icon={item.icon}
                    label={item.label}
                    active={location.pathname === item.path}
                    onClick={() => navigate(item.path)}
                  />
                  {item.path === '/admin/knowledge' && <SummaryBadge />}
                </div>
              ))}
              <Divider />
            </>
          )}

          {/* ── GR SUMMARIES ───────────────────────── */}
          <NavBtn
            icon={<ClipboardList size={16} />}
            label="GR Summaries"
            active={location.pathname === '/summaries'}
            onClick={() => navigate('/summaries')}
          />

          <Divider />

          {/* ── CHAT SECTION with inline dropdown ──── */}
          {/* Chat header row — click to expand/collapse */}
          <button
            onClick={() => { setChatExpanded(!chatExpanded); navigate('/chat') }}
            style={{
              width: '100%',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              padding: '9px 10px',
              borderRadius: '8px',
              border: 'none',
              background: isChatActive ? '#1E2135' : 'transparent',
              color: isChatActive ? '#E8EAF0' : '#B0B4C8',
              fontSize: '0.88rem',
              fontWeight: isChatActive ? '600' : '400',
              cursor: 'pointer',
              fontFamily: 'Inter, sans-serif',
            }}
            onMouseEnter={e => { if (!isChatActive) e.currentTarget.style.background = '#161825' }}
            onMouseLeave={e => { if (!isChatActive) e.currentTarget.style.background = 'transparent' }}
          >
            <span style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span><MessageSquare size={16} /></span> Chat
            </span>
            <span style={{
              fontSize: '0.7rem', color: '#555',
              transform: chatExpanded ? 'rotate(90deg)' : 'rotate(0deg)',
              transition: 'transform 0.15s',
              display: 'inline-block',
            }}>▶</span>
          </button>

          {/* Inline session dropdown */}
          {chatExpanded && (
            <div style={{
              marginLeft: '8px',
              borderLeft: '1px solid #2A2D3E',
              paddingLeft: '8px',
              overflow: 'hidden',
            }}>

              {/* New Chat button */}
              <button
                onClick={handleNewChat}
                style={{
                  width: '100%',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px',
                  padding: '7px 8px',
                  borderRadius: '6px',
                  border: 'none',
                  background: 'transparent',
                  color: '#FF6B00',
                  fontSize: '0.8rem',
                  fontWeight: '600',
                  cursor: 'pointer',
                  fontFamily: 'Inter, sans-serif',
                  marginTop: '2px',
                }}
                onMouseEnter={e => e.currentTarget.style.background = '#1A1D2E'}
                onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
              >
                <span>＋</span> New Chat
              </button>

              {/* Session list */}
              <div style={{ maxHeight: '280px', overflowY: 'auto' }}>
                {sessions.length === 0 && (
                  <div style={{
                    color: '#3A3D4E', fontSize: '0.75rem',
                    padding: '8px 8px', fontStyle: 'italic',
                  }}>
                    No chats yet
                  </div>
                )}

                {sessions.map(session => (
                  <div
                    key={session._id}
                    style={{
                      position: 'relative',
                      display: 'flex',
                      alignItems: 'center',
                      padding: '6px 8px',
                      borderRadius: '6px',
                      cursor: 'pointer',
                      background: activeSessionId === session._id
                        ? '#1E2135' : 'transparent',
                      marginBottom: '1px',
                      gap: '4px',
                    }}
                    onClick={() => handleSessionClick(session._id)}
                    onMouseEnter={e => {
                      if (activeSessionId !== session._id)
                        e.currentTarget.style.background = '#161825'
                      // show 3-dot button
                      const btn = e.currentTarget.querySelector('.dot-btn')
                      if (btn) btn.style.opacity = '1'
                    }}
                    onMouseLeave={e => {
                      if (activeSessionId !== session._id)
                        e.currentTarget.style.background = 'transparent'
                      // hide 3-dot button unless menu is open
                      if (openMenuId !== session._id) {
                        const btn = e.currentTarget.querySelector('.dot-btn')
                        if (btn) btn.style.opacity = '0'
                      }
                    }}
                  >
                    {/* Title or rename input */}
                    <div style={{ flex: 1, overflow: 'hidden' }}>
                      {renamingId === session._id ? (
                        <input
                          ref={renameRef}
                          value={renameValue}
                          onChange={e => setRenameValue(e.target.value)}
                          onBlur={() => submitRename(session._id)}
                          onKeyDown={e => {
                            if (e.key === 'Enter') submitRename(session._id)
                            if (e.key === 'Escape') setRenamingId(null)
                          }}
                          onClick={e => e.stopPropagation()}
                          style={{
                            width: '100%',
                            background: '#0F1117',
                            border: '1px solid #FF6B00',
                            borderRadius: '4px',
                            color: '#E8EAF0',
                            fontSize: '0.78rem',
                            padding: '2px 5px',
                            outline: 'none',
                            fontFamily: 'Inter, sans-serif',
                          }}
                        />
                      ) : (
                        <div style={{
                          color: activeSessionId === session._id ? '#E8EAF0' : '#7A7F94',
                          fontSize: '0.8rem',
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          whiteSpace: 'nowrap',
                          display: 'flex',
                          alignItems: 'center',
                          gap: '4px',
                        }}>
                          {session.pinned && (
                            <span style={{ color: '#FF6B00', fontSize: '0.65rem', flexShrink: 0 }}><Pin size={11} /></span>
                          )}
                          {session.title}
                        </div>
                      )}
                    </div>

                    {/* 3-dot menu trigger */}
                    {renamingId !== session._id && (
                      <button
                        className="dot-btn"
                        onClick={e => {
                          e.stopPropagation()
                          setOpenMenuId(openMenuId === session._id ? null : session._id)
                        }}
                        style={{
                          background: 'none', border: 'none',
                          color: '#666', cursor: 'pointer',
                          padding: '1px 4px', fontSize: '0.85rem',
                          borderRadius: '4px', flexShrink: 0,
                          opacity: activeSessionId === session._id || openMenuId === session._id ? '1' : '0',
                          transition: 'opacity 0.1s',
                          lineHeight: 1,
                        }}
                      >
                        ···
                      </button>
                    )}

                    {/* Dropdown menu */}
                    {openMenuId === session._id && (
                      <div
                        ref={menuRef}
                        onClick={e => e.stopPropagation()}
                        style={{
                          position: 'fixed',
                          background: '#1A1D2E',
                          border: '1px solid #2A2D3E',
                          borderRadius: '8px',
                          zIndex: 300,
                          minWidth: '148px',
                          boxShadow: '0 8px 24px rgba(0,0,0,0.5)',
                          overflow: 'hidden',
                          marginLeft: '160px',
                        }}
                      >
                        {[
                          { icon: session.pinned ? <PinOff size={14} /> : <Pin size={14} />, label: session.pinned ? 'Unpin' : 'Pin to top', action: () => { pinSession(session._id); setOpenMenuId(null) }, danger: false },
                          { icon: <Pencil size={14} />, label: 'Rename',     action: () => startRename(session),                            danger: false },
                          { icon: <Trash2 size={14} />, label: 'Delete',     action: () => { deleteSession(session._id); setOpenMenuId(null) }, danger: true  },
                        ].map(item => (
                          <button
                            key={item.label}
                            onClick={item.action}
                            style={{
                              width: '100%',
                              display: 'flex', alignItems: 'center', gap: '8px',
                              padding: '9px 14px',
                              background: 'none', border: 'none',
                              color: item.danger ? '#EF4444' : '#B0B4C8',
                              fontSize: '0.83rem', cursor: 'pointer',
                              fontFamily: 'Inter, sans-serif', textAlign: 'left',
                            }}
                            onMouseEnter={e => e.currentTarget.style.background = '#13151F'}
                            onMouseLeave={e => e.currentTarget.style.background = 'none'}
                          >
                            <span>{item.icon}</span> {item.label}
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Spacer */}
          <div style={{ flex: 1 }} />

          {/* Sign out */}
          <NavBtn icon={<LogOut size={16} />} label="Sign Out" active={false} onClick={handleLogout} />
        </div>
      </div>
    </>
  )
}

// ── Small reusable components ─────────────────────────────

function NavBtn({ icon, label, active, onClick }) {
  const [hovered, setHovered] = useState(false)
  return (
    <button
      onClick={onClick}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        width: '100%', display: 'flex', alignItems: 'center',
        gap: '9px', padding: '9px 10px', borderRadius: '8px',
        border: 'none',
        background: active ? '#FF6B00' : hovered ? '#161825' : 'transparent',
        color: active ? 'white' : '#B0B4C8',
        fontSize: '0.88rem', fontWeight: active ? '600' : '400',
        cursor: 'pointer', textAlign: 'left',
        fontFamily: 'Inter, sans-serif', transition: 'background 0.12s',
        marginBottom: '1px',
      }}
    >
      <span>{icon}</span><span>{label}</span>
    </button>
  )
}

function SectionLabel({ children }) {
  return (
    <div style={{
      color: '#3A3D4E', fontSize: '0.68rem', fontWeight: '600',
      textTransform: 'uppercase', letterSpacing: '0.8px',
      padding: '8px 10px 4px', fontFamily: 'monospace',
    }}>
      {children}
    </div>
  )
}

function Divider() {
  return <div style={{ height: '1px', background: '#1E2135', margin: '8px 4px' }} />
}

function SummaryBadge() {
  const { pendingCount, batchState } = useAdmin()
  const isRunning = batchState?.running

  if (!isRunning && (!pendingCount || pendingCount === 0)) return null

  const completed = batchState?.completed ?? 0
  const total     = batchState?.total_files ?? 0

  return (
    <div style={{
      margin: '2px 6px 6px 34px',
      padding: '4px 10px',
      background: isRunning ? '#2A2410' : '#1E2135',
      border: `1px solid ${isRunning ? '#FACC1533' : '#2A2D3E'}`,
      borderRadius: '8px',
      fontSize: '0.7rem',
      fontFamily: 'monospace',
      color: isRunning ? '#FACC15' : '#7A7F94',
      display: 'inline-flex',
      alignItems: 'center',
      gap: '5px',
      whiteSpace: 'nowrap',
    }}>
      {isRunning ? `${completed}/${total} summarized` : `${pendingCount} pending`}
    </div>
  )
}