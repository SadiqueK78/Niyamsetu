// frontend/src/pages/Analytics.jsx
// Admin dashboard — query stats + recent query log table.
// Doubles as benchmark data source for the copyright application.

import { useState, useEffect, useCallback } from 'react'
import { MessageSquare, Users, Timer } from 'lucide-react'
import client from '../api/client'

export default function Analytics() {
  const [stats,   setStats]   = useState(null)
  const [logs,    setLogs]    = useState([])
  const [loading, setLoading] = useState(true)
  const [error,   setError]   = useState('')

  const loadData = useCallback(async () => {
    setLoading(true)
    setError('')
    try {
      const [statsRes, logsRes] = await Promise.all([
        client.get('/logs/stats'),
        client.get('/logs/all?limit=100'),
      ])
      setStats(statsRes.data)
      setLogs(logsRes.data.logs || [])
    } catch (err) {
      setError(err.response?.data?.detail || 'Could not load analytics.')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    loadData()
  }, [loadData])

  const formatDate = (iso) => {
    if (!iso) return '—'
    try {
      return new Date(iso).toLocaleString('en-IN', {
        day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit',
      })
    } catch { return iso }
  }

  const languagePct = (lang) => {
    if (!stats?.languages || !stats.total_queries) return 0
    const count = stats.languages[lang] || 0
    return Math.round((count / stats.total_queries) * 100)
  }

  return (
    <div style={{
      minHeight: '100%', background: '#0F1117',
      padding: '28px 32px', fontFamily: 'Inter, sans-serif',
    }}>

      {/* Header */}
      <div style={{ marginBottom: '20px' }}>
        <h1 style={{ color: '#E8EAF0', fontSize: '1.4rem', margin: '0 0 4px', fontWeight: '700' }}>
          Analytics
        </h1>
        <p style={{ color: '#555', fontSize: '0.85rem', margin: 0 }}>
          Query volume, performance, and language breakdown across all users.
        </p>
      </div>

      {error && (
        <div style={{
          background: '#2A0F0F', border: '1px solid #EF4444', borderRadius: '8px',
          padding: '12px 16px', color: '#EF4444', fontSize: '0.85rem', marginBottom: '20px',
        }}>
          {error}
        </div>
      )}

      {loading && (
        <div style={{ color: '#555', fontSize: '0.85rem', padding: '20px 0' }}>Loading…</div>
      )}

      {!loading && stats && (
        <>
          {/* Stat cards */}
          <div style={{
            display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
            gap: '14px', marginBottom: '24px',
          }}>
            <StatCard label="Total Queries" value={stats.total_queries ?? 0} color="#FF6B00" icon={<MessageSquare size={18} />} />
            <StatCard label="Unique Users" value={stats.unique_users ?? 0} color="#6FB3FF" icon={<Users size={18} />} />
            <StatCard label="Avg Response Time" value={`${(stats.avg_elapsed ?? 0).toFixed(2)}s`} color="#FACC15" icon={<Timer size={18} />} />
            <StatCard label="Success Rate" value={`${Math.round(stats.success_rate ?? 0)}%`} color="#4ADE80" icon="✓" />
          </div>

          {/* Language breakdown */}
          <div style={{
            background: '#1A1D2E', border: '1px solid #2A2D3E', borderRadius: '14px',
            padding: '18px 20px', marginBottom: '24px',
          }}>
            <div style={{ color: '#B0B4C8', fontSize: '0.85rem', fontWeight: '600', marginBottom: '12px' }}>
              Language Breakdown
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
              <LangBar label="English" count={stats.languages?.english ?? 0} pct={languagePct('english')} color="#6FB3FF" />
              <LangBar label="Marathi" count={stats.languages?.marathi ?? 0} pct={languagePct('marathi')} color="#FF6B00" />
            </div>
          </div>
        </>
      )}

      {/* Recent queries table */}
      <div style={{ color: '#B0B4C8', fontSize: '0.9rem', fontWeight: '600', marginBottom: '10px' }}>
        Recent Queries ({logs.length})
      </div>

      {!loading && logs.length === 0 && (
        <div style={{ color: '#555', fontSize: '0.85rem', padding: '20px 0', fontStyle: 'italic' }}>
          No queries logged yet.
        </div>
      )}

      {!loading && logs.length > 0 && (
        <div style={{
          background: '#1A1D2E', border: '1px solid #2A2D3E', borderRadius: '12px',
          overflow: 'hidden',
        }}>
          {/* Table header */}
          <div style={{
            display: 'grid',
            gridTemplateColumns: '110px 1fr 90px 90px 70px 120px',
            gap: '10px', padding: '10px 16px',
            borderBottom: '1px solid #2A2D3E',
            color: '#555', fontSize: '0.72rem', fontFamily: 'monospace',
            textTransform: 'uppercase', letterSpacing: '0.5px',
          }}>
            <div>User</div>
            <div>Query</div>
            <div>Language</div>
            <div>Time</div>
            <div>Status</div>
            <div>When</div>
          </div>

          {/* Table rows — scrollable */}
          <div style={{ maxHeight: '460px', overflowY: 'auto' }}>
            {logs.map((log, i) => (
              <div key={i} style={{
                display: 'grid',
                gridTemplateColumns: '110px 1fr 90px 90px 70px 120px',
                gap: '10px', padding: '10px 16px', alignItems: 'center',
                borderBottom: i < logs.length - 1 ? '1px solid #1E2135' : 'none',
              }}>
                <div style={{
                  color: '#B0B4C8', fontSize: '0.78rem',
                  whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
                }}>
                  {log.username}
                </div>
                <div style={{
                  color: '#E8EAF0', fontSize: '0.8rem',
                  whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
                }} title={log.query}>
                  {log.query}
                </div>
                <div style={{ color: '#7A7F94', fontSize: '0.75rem' }}>
                  {log.language === 'marathi' ? 'MR' : 'EN'}
                </div>
                <div style={{ color: '#7A7F94', fontSize: '0.75rem', fontFamily: 'monospace' }}>
                  {log.elapsed_sec != null ? `${log.elapsed_sec.toFixed(1)}s` : '—'}
                </div>
                <div style={{
                  fontSize: '0.8rem',
                  color: log.was_successful ? '#4ADE80' : '#EF4444',
                }}>
                  {log.was_successful ? '✓' : '✕'}
                </div>
                <div style={{ color: '#555', fontSize: '0.72rem', fontFamily: 'monospace' }}>
                  {formatDate(log.created_at)}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

function StatCard({ label, value, color, icon }) {
  return (
    <div style={{
      background: '#1A1D2E', border: '1px solid #2A2D3E', borderRadius: '12px',
      padding: '16px 18px',
    }}>
      <div style={{ fontSize: '1.1rem', marginBottom: '6px' }}>{icon}</div>
      <div style={{ color, fontSize: '1.5rem', fontWeight: '700', fontFamily: 'monospace' }}>{value}</div>
      <div style={{ color: '#555', fontSize: '0.75rem', marginTop: '2px' }}>{label}</div>
    </div>
  )
}

function LangBar({ label, count, pct, color }) {
  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
        <span style={{ color: '#B0B4C8', fontSize: '0.8rem' }}>{label}</span>
        <span style={{ color: '#555', fontSize: '0.78rem', fontFamily: 'monospace' }}>{count} ({pct}%)</span>
      </div>
      <div style={{ background: '#0F1117', borderRadius: '6px', height: '6px', overflow: 'hidden' }}>
        <div style={{ width: `${pct}%`, height: '100%', background: color, transition: 'width 0.4s ease' }} />
      </div>
    </div>
  )
}