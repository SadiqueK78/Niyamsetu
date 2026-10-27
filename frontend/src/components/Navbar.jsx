// frontend/src/components/Navbar.jsx
// ─────────────────────────────────────────────────────────
// Top navigation bar — visible on all pages after login.
// Shows: logo left, username + role badge right.
// ─────────────────────────────────────────────────────────

import { Landmark, Network, Table, Search } from 'lucide-react'
import { NavLink } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'

export default function Navbar({ collapsed, onToggleSidebar }) {
  const { user, isAdmin } = useAuth()

  return (
    <div style={{
      height: '56px',
      background: '#1A1D2E',
      borderBottom: '1px solid #2A2D3E',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      padding: '0 24px',
      position: 'fixed',
      top: 0,
      left: 0,
      right: 0,
      zIndex: 100,
      fontFamily: 'Inter, sans-serif',
    }}>

      {/* Left — sidebar toggle + brand */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: '14px',
      }}>
        <button
          onClick={onToggleSidebar}
          title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          style={{
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            width: '32px', height: '32px', flexShrink: 0,
            background: 'transparent', border: 'none', borderRadius: '6px',
            color: '#888', cursor: 'pointer', fontSize: '1rem',
          }}
          onMouseEnter={e => e.currentTarget.style.background = '#242840'}
          onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
        >
          ☰
        </button>
        <span style={{ display: 'flex', alignItems: 'center', flexShrink: 0 }}><Landmark size={20} color="#FF6B00" /></span>
        <span style={{
          color: '#FF6B00',
          fontWeight: '700',
          fontSize: '1.05rem',
          letterSpacing: '0.3px',
        }}>
          Niyamsetu
        </span>
      </div>

      {/* Middle — graph link. The relationship graph is not a document
          list, so it has no home in the per-document sidebar. */}
      {user && (
        <NavLink
          to="/graph"
          title="GR relationship graph"
          style={({ isActive }) => ({
            display: 'flex',
            alignItems: 'center',
            gap: '7px',
            padding: '5px 12px',
            borderRadius: '6px',
            textDecoration: 'none',
            fontSize: '0.85rem',
            fontWeight: '600',
            color: isActive ? '#FF6B00' : '#8A93A6',
            background: isActive ? '#242840' : 'transparent',
          })}
        >
          <Network size={16} />
          <span>Graph</span>
        </NavLink>
      )}

      {user && (
        <NavLink
          to="/registry"
          title="Document registry"
          style={({ isActive }) => ({
            display: 'flex',
            alignItems: 'center',
            gap: '7px',
            padding: '5px 12px',
            borderRadius: '6px',
            textDecoration: 'none',
            fontSize: '0.85rem',
            fontWeight: '600',
            color: isActive ? '#FF6B00' : '#8A93A6',
            background: isActive ? '#242840' : 'transparent',
          })}
        >
          <Table size={16} />
          <span>Registry</span>
        </NavLink>
      )}

      {user && (
        <NavLink
          to="/search"
          title="Retrieval search"
          style={({ isActive }) => ({
            display: 'flex',
            alignItems: 'center',
            gap: '7px',
            padding: '5px 12px',
            borderRadius: '6px',
            textDecoration: 'none',
            fontSize: '0.85rem',
            fontWeight: '600',
            color: isActive ? '#FF6B00' : '#8A93A6',
            background: isActive ? '#242840' : 'transparent',
          })}
        >
          <Search size={16} />
          <span>Search</span>
        </NavLink>
      )}

      {/* Right — user info */}
      {user && (
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '10px',
        }}>
          <span style={{ color: '#888', fontSize: '0.85rem' }}>
            {user.name}
          </span>
          <span style={{
            background: isAdmin ? '#FF6B00' : '#1E3A5F',
            color: isAdmin ? 'white' : '#6FB3FF',
            padding: '2px 10px',
            borderRadius: '20px',
            fontSize: '0.72rem',
            fontWeight: '600',
            fontFamily: 'monospace',
            letterSpacing: '0.5px',
          }}>
            {isAdmin ? 'ADMIN' : 'USER'}
          </span>
        </div>
      )}

    </div>
  )
}