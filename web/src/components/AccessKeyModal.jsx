import { useState } from 'react';
import { KeyRound, LoaderCircle } from 'lucide-react';
import { api, setApiKey } from '../lib/api';

// Shown only when the server requires a key and this browser has no valid one.
export default function AccessKeyModal({ onAuthorized }) {
  const [key, setKey] = useState('');
  const [error, setError] = useState('');
  const [checking, setChecking] = useState(false);

  const submit = async (e) => {
    e.preventDefault();
    const k = key.trim();
    if (!k) return;
    setChecking(true);
    setError('');
    setApiKey(k);
    try {
      const res = await api.checkAuth();
      if (res.valid) onAuthorized();
      else { setApiKey(''); setError('That access key isn’t valid. Check it and try again.'); }
    } catch {
      setApiKey('');
      setError('Couldn’t reach the server. Make sure the backend is running.');
    } finally {
      setChecking(false);
    }
  };

  return (
    <div className="modal-scrim" role="dialog" aria-modal="true" aria-labelledby="key-title">
      <form className="modal" onSubmit={submit}>
        <span className="modal-icon"><KeyRound /></span>
        <h2 id="key-title" className="display">Access key</h2>
        <p className="modal-text">This workspace is private. Enter the key you were given; it stays in this browser.</p>
        <label className="sr-only" htmlFor="access-key">Access key</label>
        <input
          id="access-key" className="input" type="password" autoFocus autoComplete="off"
          value={key} onChange={(e) => { setKey(e.target.value); setError(''); }} placeholder="Paste your key"
        />
        {error && <p className="form-error" role="alert">{error}</p>}
        <button className="btn btn-primary btn-lg" type="submit" disabled={checking || !key.trim()}>
          {checking ? <LoaderCircle className="spin" /> : null}
          {checking ? 'Checking…' : 'Continue'}
        </button>
      </form>
    </div>
  );
}
