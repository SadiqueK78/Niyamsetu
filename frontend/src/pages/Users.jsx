// frontend/src/pages/Users.jsx
// Admin-only user management — list, create, and delete accounts.
// Real-world use case: GAD officer creates accounts for department officials.

import { useState, useEffect, useCallback } from 'react'
import { Trash2 } from 'lucide-react'
import client from '../api/client'
import { useAuth } from '../context/AuthContext'

export default function Users() {
  const { user: currentUser } = useAuth()

  const [users,   setUsers]   = useState([])
  const [loading, setLoading] = useState(true)
  const [error,   setError]   = useState('')

  const [showForm, setShowForm] = useState(false)
  const [form, setForm] = useState({ username: '', password: '', name: '', role: 'user' })
  const [formError, setFormError] = useState('')
  const [creating, setCreating] = useState(false)

  const loadUsers = useCallback(async () => {
    setLoading(true)
    setError('')
    try {
      const res = await client.get('/auth/users')
      setUsers(res.data || [])
    } catch (err) {
      setError(err.response?.data?.detail || 'Could not load users.')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    loadUsers()
  }, [loadUsers])

  const handleCreate = async (e) => {
    e.preventDefault()
    setFormError('')

    if (!form.username.trim() || !form.password.trim() || !form.name.trim()) {
      setFormError('All fields are required.')
      return
    }

    setCreating(true)
    try {
      await client.post('/auth/users', {
        username: form.username.trim().toLowerCase(),
        password: form.password,
        name:     form.name.trim(),
        role:     form.role,
      })
      setForm({ username: '', password: '', name: '', role: 'user' })
      setShowForm(false)
      await loadUsers()
    } catch (err) {
      setFormError(err.response?.data?.detail || 'Could not create user.')
    } finally {
      setCreating(false)
    }
  }

  const handleDelete = async (username) => {
    if (username === currentUser?.username) return // safety, though button is hidden anyway
    if (!window.confirm(`Delete user "${username}"? This cannot be undone.`)) return

    try {
      await client.delete(`/auth/users/${encodeURIComponent(username)}`)
      setUsers(prev => prev.filter(u => u.username !== username))
    } catch (err) {
      setError(err.response?.data?.detail || 'Could not delete user.')
    }
  }

  return (
    <div style={{
      minHeight: '100%', background: '#0F1117',
      padding: '28px 32px', fontFamily: 'Inter, sans-serif',
    }}>

      {/* Header */}
      <div style={{
        display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start',
        marginBottom: '20px', flexWrap: 'wrap', gap: '12px',
      }}>
        <div>
          <h1 style={{ color: '#E8EAF0', fontSize: '1.4rem', margin: '0 0 4px', fontWeight: '700' }}>
            Users
          </h1>
          <p style={{ color: '#555', fontSize: '0.85rem', margin: 0 }}>
            Manage accounts for department officials.
          </p>
        </div>

        <button
          onClick={() => { setShowForm(!showForm); setFormError('') }}
          style={{
            background: showForm ? '#1A1D2E' : '#FF6B00',
            border: showForm ? '1px solid #2A2D3E' : 'none',
            color: 'white', borderRadius: '10px',
            padding: '10px 18px', fontSize: '0.85rem', fontWeight: '600',
            cursor: 'pointer', fontFamily: 'Inter, sans-serif',
          }}
        >
          {showForm ? '✕ Cancel' : '＋ New User'}
        </button>
      </div>

      {error && (
        <div style={{
          background: '#2A0F0F', border: '1px solid #EF4444', borderRadius: '8px',
          padding: '12px 16px', color: '#EF4444', fontSize: '0.85rem', marginBottom: '20px',
        }}>
          {error}
        </div>
      )}

      {/* Create user form */}
      {showForm && (
        <form onSubmit={handleCreate} style={{
          background: '#1A1D2E', border: '1px solid #2A2D3E', borderRadius: '14px',
          padding: '20px 22px', marginBottom: '24px',
        }}>
          <div style={{
            display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '14px', marginBottom: '14px',
          }}>
            <FormField label="Username">
              <input
                value={form.username}
                onChange={e => setForm({ ...form, username: e.target.value })}
                placeholder="e.g. rmeshram"
                style={inputStyle}
              />
            </FormField>
            <FormField label="Full Name">
              <input
                value={form.name}
                onChange={e => setForm({ ...form, name: e.target.value })}
                placeholder="e.g. Rajesh Meshram"
                style={inputStyle}
              />
            </FormField>
            <FormField label="Password">
              <input
                type="password"
                value={form.password}
                onChange={e => setForm({ ...form, password: e.target.value })}
                placeholder="Temporary password"
                style={inputStyle}
              />
            </FormField>
            <FormField label="Role">
              <select
                value={form.role}
                onChange={e => setForm({ ...form, role: e.target.value })}
                style={{ ...inputStyle, cursor: 'pointer' }}
              >
                <option value="user">User</option>
                <option value="admin">Admin</option>
              </select>
            </FormField>
          </div>

          {formError && (
            <div style={{ color: '#EF4444', fontSize: '0.8rem', marginBottom: '12px' }}>
              {formError}
            </div>
          )}

          <button
            type="submit"
            disabled={creating}
            style={{
              background: creating ? '#7A3300' : '#FF6B00',
              color: 'white', border: 'none', borderRadius: '10px',
              padding: '10px 20px', fontSize: '0.85rem', fontWeight: '600',
              cursor: creating ? 'not-allowed' : 'pointer', fontFamily: 'Inter, sans-serif',
            }}
          >
            {creating ? 'Creating…' : 'Create User'}
          </button>
        </form>
      )}

      {/* User list */}
      <div style={{ color: '#B0B4C8', fontSize: '0.9rem', fontWeight: '600', marginBottom: '10px' }}>
        All Users ({users.length})
      </div>

      {loading && (
        <div style={{ color: '#555', fontSize: '0.85rem', padding: '20px 0' }}>Loading…</div>
      )}

      {!loading && users.length > 0 && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          {users.map((u, i) => (
            <div key={i} style={{
              background: '#1A1D2E', border: '1px solid #2A2D3E', borderRadius: '10px',
              padding: '12px 16px', display: 'flex', alignItems: 'center', justifyContent: 'space-between',
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                <div style={{
                  width: '34px', height: '34px', borderRadius: '50%',
                  background: u.role === 'admin' ? '#FF6B00' : '#1E3A5F',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  color: 'white', fontWeight: '700', fontSize: '0.85rem', flexShrink: 0,
                }}>
                  {u.name?.charAt(0).toUpperCase() || u.username.charAt(0).toUpperCase()}
                </div>
                <div>
                  <div style={{ color: '#E8EAF0', fontSize: '0.85rem', fontWeight: '600' }}>
                    {u.name}
                    {u.username === currentUser?.username && (
                      <span style={{ color: '#555', fontWeight: '400' }}> (you)</span>
                    )}
                  </div>
                  <div style={{ color: '#555', fontSize: '0.75rem', fontFamily: 'monospace' }}>
                    @{u.username}
                  </div>
                </div>
              </div>

              <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                <span style={{
                  fontSize: '0.7rem', padding: '3px 10px', borderRadius: '12px',
                  fontFamily: 'monospace',
                  color: u.role === 'admin' ? 'white' : '#6FB3FF',
                  background: u.role === 'admin' ? '#FF6B00' : '#0D2137',
                }}>
                  {u.role.toUpperCase()}
                </span>

                {u.username !== currentUser?.username && (
                  <button
                    onClick={() => handleDelete(u.username)}
                    style={{
                      background: 'none', border: 'none', color: '#EF4444',
                      cursor: 'pointer', fontSize: '0.9rem', padding: '4px',
                    }}
                    title="Delete user"
                  >
                    <Trash2 size={14} />
                  </button>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

const inputStyle = {
  width: '100%', background: '#13151F', border: '1px solid #2A2D3E',
  borderRadius: '8px', padding: '10px 12px', color: '#E8EAF0',
  fontSize: '0.85rem', outline: 'none', boxSizing: 'border-box',
  fontFamily: 'Inter, sans-serif',
}

function FormField({ label, children }) {
  return (
    <div>
      <label style={{
        display: 'block', color: '#888', fontSize: '0.72rem',
        marginBottom: '5px', fontFamily: 'monospace',
        textTransform: 'uppercase', letterSpacing: '0.5px',
      }}>
        {label}
      </label>
      {children}
    </div>
  )
}