import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { signIn, signUp } from '@/auth/client';

export function LoginPage() {
  const navigate = useNavigate();
  const [isSignUp, setIsSignUp] = useState(false);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [name, setName] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [serverDown, setServerDown] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    try {
      if (isSignUp) {
        const res = await signUp({ email, password, name: name || email.split('@')[0]! });
        if (res.error) {
          setError(res.error.message ?? 'Sign up failed');
          setLoading(false);
          return;
        }
      } else {
        const res = await signIn({ email, password });
        if (res.error) {
          setError(res.error.message ?? 'Sign in failed');
          setLoading(false);
          return;
        }
      }
      navigate('/dashboard');
    } catch {
      setError('Cannot reach server.');
      setServerDown(true);
      setLoading(false);
    }
  };

  const handleSocial = async (provider: 'github' | 'google') => {
    setLoading(true);
    try {
      await signIn({ provider, callbackURL: '/dashboard' });
    } catch {
      setError('Cannot reach server for social login.');
      setServerDown(true);
      setLoading(false);
    }
  };

  return (
    <div
      style={{
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        minHeight: '100vh',
        background: '#11111b',
        color: '#cdd6f4',
      }}
    >
      <div
        style={{
          width: 400,
          padding: 32,
          background: '#1e1e2e',
          borderRadius: 12,
          border: '1px solid #313244',
        }}
      >
        <h1 style={{ fontSize: 24, marginBottom: 8, textAlign: 'center' }}>Forge3D</h1>
        <p style={{ fontSize: 14, color: '#888', textAlign: 'center', marginBottom: 24 }}>
          {isSignUp ? 'Create your account' : 'Sign in to continue'}
        </p>

        {error && (
          <div
            role="alert"
            style={{
              padding: '8px 12px',
              background: '#45243780',
              color: '#f38ba8',
              borderRadius: 6,
              fontSize: 13,
              marginBottom: 16,
            }}
          >
            {error}
          </div>
        )}

        <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          {isSignUp && (
            <input
              type="text"
              placeholder="Name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              style={inputStyle}
            />
          )}
          <input
            type="email"
            placeholder="Email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
            style={inputStyle}
          />
          <input
            type="password"
            placeholder="Password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
            minLength={8}
            style={inputStyle}
          />
          <button type="submit" disabled={loading} style={buttonStyle}>
            {loading ? 'Loading...' : isSignUp ? 'Sign Up' : 'Sign In'}
          </button>
        </form>

        <div style={{ display: 'flex', gap: 8, marginTop: 16 }}>
          <button onClick={() => handleSocial('github')} disabled={loading} style={socialButton}>
            GitHub
          </button>
          <button onClick={() => handleSocial('google')} disabled={loading} style={socialButton}>
            Google
          </button>
        </div>

        {serverDown && (
          <button
            onClick={() => navigate('/editor')}
            style={{
              ...socialButton,
              width: '100%',
              marginTop: 12,
              background: '#45475a',
              textAlign: 'center' as const,
            }}
          >
            Continue Offline
          </button>
        )}

        <p style={{ fontSize: 13, textAlign: 'center', marginTop: 20, color: '#666' }}>
          {isSignUp ? 'Already have an account?' : "Don't have an account?"}{' '}
          <button
            onClick={() => setIsSignUp(!isSignUp)}
            style={{
              background: 'none',
              border: 'none',
              color: '#89b4fa',
              cursor: 'pointer',
              fontSize: 13,
            }}
          >
            {isSignUp ? 'Sign In' : 'Sign Up'}
          </button>
        </p>
      </div>
    </div>
  );
}

const inputStyle: React.CSSProperties = {
  padding: '10px 12px',
  background: '#313244',
  border: '1px solid #45475a',
  borderRadius: 6,
  color: '#cdd6f4',
  fontSize: 14,
  outline: 'none',
};

const buttonStyle: React.CSSProperties = {
  padding: '10px 12px',
  background: '#2563eb',
  border: 'none',
  borderRadius: 6,
  color: '#fff',
  fontSize: 14,
  cursor: 'pointer',
  fontWeight: 600,
};

const socialButton: React.CSSProperties = {
  flex: 1,
  padding: '10px 12px',
  background: '#313244',
  border: '1px solid #45475a',
  borderRadius: 6,
  color: '#cdd6f4',
  fontSize: 13,
  cursor: 'pointer',
};
