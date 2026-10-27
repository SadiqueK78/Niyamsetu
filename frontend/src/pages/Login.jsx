// frontend/src/pages/Login.jsx
// ─────────────────────────────────────────────────────────
// Login page — first thing every user sees.
// Calls AuthContext.login() which hits POST /api/auth/login.
// On success → admin goes to /admin/dashboard, user goes to /chat
//
// Local dev accounts are seeded by backend/db/users.py :: seed_default_users()
// and only when the users collection is empty — see that function's docstring
// for the usernames. They are deliberately NOT shown on this screen: printing
// working credentials on a login form hands them to anyone who opens the page,
// including in a demo or a screenshot.
// ─────────────────────────────────────────────────────────

import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Landmark } from 'lucide-react'
import { useAuth } from '../context/AuthContext'

export default function Login() {
  const { login } = useAuth()
  const navigate  = useNavigate()

  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error,    setError]    = useState('')
  const [loading,  setLoading]  = useState(false)

  const handleLogin = async (e) => {
    e.preventDefault()
    setError('')
    setLoading(true)

    try {
      const user = await login(username, password)
      // Admin → dashboard, User → chat
      navigate(user.role === 'admin' ? '/admin/upload' : '/chat', { replace: true })
    } catch (err) {
      setError(err.response?.data?.detail || 'Login failed. Check credentials.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{
      minHeight: '100vh',
      background: '#0F1117',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      fontFamily: 'Inter, sans-serif',
      padding: '24px',
    }}>
      <div style={{
        background: '#1A1D2E',
        border: '1px solid #2A2D3E',
        borderRadius: '16px',
        padding: '48px 40px',
        width: '100%',
        maxWidth: '420px',
        boxShadow: '0 24px 64px rgba(0,0,0,0.5)',
      }}>

        {/* Logo and title */}
        <div style={{ textAlign: 'center', marginBottom: '36px' }}>
          <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '8px' }}><Landmark size={44} color="#FF6B00" /></div>
          <h1 style={{
            color: '#FF6B00',
            fontSize: '1.5rem',
            fontWeight: '700',
            margin: '0 0 6px',
          }}>
            Niyamsetu
          </h1>
          <p style={{ color: '#555', fontSize: '0.85rem', margin: 0 }}>
            Maharashtra GR Intelligence System
          </p>
          {/* Marathi subtitle */}
          <p style={{
            color: '#444',
            fontSize: '0.8rem',
            margin: '4px 0 0',
            fontFamily: 'serif',
          }}>
            महाराष्ट्र शासन निर्णय AI सहाय्यक
          </p>
        </div>

        {/* Login form */}
        <form onSubmit={handleLogin}>

          {/* Username field */}
          <div style={{ marginBottom: '16px' }}>
            <label style={{
              display: 'block',
              color: '#888',
              fontSize: '0.82rem',
              marginBottom: '6px',
              fontFamily: 'monospace',
              textTransform: 'uppercase',
              letterSpacing: '0.5px',
            }}>
              Username
            </label>
            <input
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="Enter username"
              required
              style={{
                width: '100%',
                background: '#13151F',
                border: '1px solid #2A2D3E',
                borderRadius: '10px',
                padding: '12px 16px',
                color: '#E8EAF0',
                fontSize: '0.95rem',
                outline: 'none',
                boxSizing: 'border-box',
              }}
              onFocus={e => e.target.style.borderColor = '#FF6B00'}
              onBlur={e  => e.target.style.borderColor = '#2A2D3E'}
            />
          </div>

          {/* Password field */}
          <div style={{ marginBottom: '24px' }}>
            <label style={{
              display: 'block',
              color: '#888',
              fontSize: '0.82rem',
              marginBottom: '6px',
              fontFamily: 'monospace',
              textTransform: 'uppercase',
              letterSpacing: '0.5px',
            }}>
              Password
            </label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="Enter password"
              required
              style={{
                width: '100%',
                background: '#13151F',
                border: '1px solid #2A2D3E',
                borderRadius: '10px',
                padding: '12px 16px',
                color: '#E8EAF0',
                fontSize: '0.95rem',
                outline: 'none',
                boxSizing: 'border-box',
              }}
              onFocus={e => e.target.style.borderColor = '#FF6B00'}
              onBlur={e  => e.target.style.borderColor = '#2A2D3E'}
            />
          </div>

          {/* Error message */}
          {error && (
            <div style={{
              background: '#2A0F0F',
              border: '1px solid #EF4444',
              borderRadius: '8px',
              padding: '10px 14px',
              color: '#EF4444',
              fontSize: '0.85rem',
              marginBottom: '16px',
            }}>
              {error}
            </div>
          )}

          {/* Submit button */}
          <button
            type="submit"
            disabled={loading}
            style={{
              width: '100%',
              background: loading ? '#7A3300' : '#FF6B00',
              color: 'white',
              border: 'none',
              borderRadius: '10px',
              padding: '13px',
              fontSize: '1rem',
              fontWeight: '600',
              cursor: loading ? 'not-allowed' : 'pointer',
              transition: 'background 0.15s',
            }}
          >
            {loading ? 'Signing in...' : 'Sign In →'}
          </button>

        </form>

      </div>
    </div>
  )
}