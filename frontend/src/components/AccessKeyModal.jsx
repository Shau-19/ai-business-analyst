import React, { useState } from 'react';
import { KeyRound, AlertCircle, RefreshCw } from 'lucide-react';
import { api, setApiKey } from '../api';

// Shown when the server requires an access key and this browser doesn't
// have a valid one. The key is checked against /auth/check before saving.
export default function AccessKeyModal({ onAuthorized }) {
  const [key, setKey] = useState('');
  const [error, setError] = useState('');
  const [checking, setChecking] = useState(false);

  const submit = async (e) => {
    e.preventDefault();
    const trimmed = key.trim();
    if (!trimmed) return;
    setChecking(true);
    setError('');
    setApiKey(trimmed);
    try {
      const res = await api.checkAuth();
      if (res.valid) {
        onAuthorized();
      } else {
        setApiKey('');
        setError('That access key is not valid.');
      }
    } catch {
      setApiKey('');
      setError('Could not reach the server. Check your connection and try again.');
    } finally {
      setChecking(false);
    }
  };

  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center p-4" style={{ background: 'rgba(10,10,14,.55)', backdropFilter: 'blur(2px)' }}>
      <div className="card w-full max-w-sm p-6 space-y-4" style={{ boxShadow: 'var(--shadow-lg)' }}>
        <h3 className="text-[14.5px] font-semibold text-[var(--text)] flex items-center gap-2">
          <KeyRound className="w-4 h-4" style={{ color: 'var(--accent)' }} />
          Access key required
        </h3>
        <p className="text-[13px] text-[var(--text-muted)]">
          This deployment is private. Enter the access key you were given. It is stored in this browser only.
        </p>
        <form onSubmit={submit} className="space-y-3">
          <input
            type="password"
            autoFocus
            autoComplete="off"
            value={key}
            onChange={(e) => { setKey(e.target.value); setError(''); }}
            placeholder="Access key"
            className="input"
          />
          {error && (
            <p className="text-[12px] flex items-start gap-1.5" style={{ color: 'var(--red)' }}>
              <AlertCircle className="w-3.5 h-3.5 mt-0.5 shrink-0" />
              <span>{error}</span>
            </p>
          )}
          <div className="flex justify-end">
            <button type="submit" disabled={checking || !key.trim()} className="btn btn-primary">
              {checking && <RefreshCw className="w-3.5 h-3.5 animate-spin" />}
              <span>{checking ? 'Checking…' : 'Continue'}</span>
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
