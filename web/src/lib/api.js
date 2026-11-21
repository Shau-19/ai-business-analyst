// Thin fetch client for the FastAPI backend.
// The access key is never bundled: the user enters it once (only if the
// server requires one) and it lives in this browser's localStorage.

const BASE = import.meta.env.VITE_API_BASE_URL || '';
const KEY_STORAGE = 'analyst_api_key';
export const AUTH_EVENT = 'analyst:auth-required';

export function getApiKey() {
  try { return localStorage.getItem(KEY_STORAGE) || ''; } catch { return ''; }
}
export function setApiKey(key) {
  try { key ? localStorage.setItem(KEY_STORAGE, key) : localStorage.removeItem(KEY_STORAGE); } catch { /* storage blocked */ }
}

export class ApiError extends Error {
  constructor(message, status) { super(message); this.status = status; }
}

function headers(extra = {}) {
  const key = getApiKey();
  return { ...(key ? { 'X-Api-Key': key } : {}), ...extra };
}

async function request(path, { method = 'GET', body, headers: h = {}, signal } = {}) {
  const isForm = body instanceof FormData;
  const res = await fetch(BASE + path, {
    method,
    signal,
    headers: headers(isForm || body === undefined ? h : { 'Content-Type': 'application/json', ...h }),
    body: body === undefined ? undefined : isForm ? body : JSON.stringify(body),
  });
  if (res.status === 401) window.dispatchEvent(new Event(AUTH_EVENT));
  let data = null;
  try { data = await res.json(); } catch { /* empty or non-JSON */ }
  if (!res.ok) {
    const detail = data?.detail || data?.error || res.statusText || `Request failed (${res.status})`;
    throw new ApiError(typeof detail === 'string' ? detail : JSON.stringify(detail), res.status);
  }
  return data;
}

export const api = {
  checkAuth: () => request('/auth/check'),

  listConversations: (userId) => request('/conversations?limit=50', { headers: { 'X-User-ID': userId } }),
  createConversation: (userId, title = 'New analysis') =>
    request('/conversations', { method: 'POST', body: { user_id: userId, title }, headers: { 'X-User-ID': userId } }),
  getConversation: (cid) => request(`/conversations/${cid}`),
  deleteConversation: (cid) => request(`/conversations/${cid}`, { method: 'DELETE' }),
  renameConversation: (cid, title) => request(`/conversations/${cid}`, { method: 'PATCH', body: { title } }),

  upload: (cid, files) => {
    const fd = new FormData();
    Array.from(files).forEach((f) => fd.append('files', f));
    return request('/upload', { method: 'POST', body: fd, headers: { 'X-Conversation-ID': cid } });
  },
  processingStatus: (cid) => request(`/conversations/${cid}/processing-status`),
  profile: (cid) => request(`/conversations/${cid}/profile`),
  numericColumns: (cid) => request(`/conversations/${cid}/numeric-columns`),

  getDashboard: (cid) => request(`/conversations/${cid}/dashboard`),
  saveDashboard: (cid, state) => request(`/conversations/${cid}/dashboard`, { method: 'POST', body: { state } }),

  // Runs the pipeline without writing chat history (dashboard / forecast).
  silentQuery: (cid, question, { valueCol = null, wantChart = false } = {}) =>
    request('/query/silent', {
      method: 'POST',
      body: { question, conversation_id: cid, value_col: valueCol, want_chart: wantChart },
    }),

  // Server-sent events: meta → token… → done (or error).
  async streamQuery(cid, question, { onMeta, onToken, onDone, onError, signal }) {
    try {
      const res = await fetch(BASE + '/query/stream', {
        method: 'POST',
        signal,
        headers: headers({ 'Content-Type': 'application/json' }),
        body: JSON.stringify({ question, conversation_id: cid }),
      });
      if (res.status === 401) window.dispatchEvent(new Event(AUTH_EVENT));
      if (!res.ok) {
        let detail = res.statusText;
        try { detail = (await res.json()).detail || detail; } catch { /* not JSON */ }
        throw new ApiError(detail || `Request failed (${res.status})`, res.status);
      }
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const blocks = buffer.split('\n\n');
        buffer = blocks.pop() || '';
        for (const block of blocks) {
          let event = 'message', data = '';
          for (const line of block.split('\n')) {
            if (line.startsWith('event: ')) event = line.slice(7).trim();
            else if (line.startsWith('data: ')) data += line.slice(6);
          }
          if (!data) continue;
          let parsed;
          try { parsed = JSON.parse(data); } catch { continue; }
          if (event === 'meta') onMeta?.(parsed);
          else if (event === 'token') onToken?.(parsed);
          else if (event === 'done') onDone?.(parsed);
          else if (event === 'error') onError?.(new ApiError(parsed.error || 'Something went wrong'));
        }
      }
    } catch (err) {
      if (err.name === 'AbortError') return;
      onError?.(err);
    }
  },
};
