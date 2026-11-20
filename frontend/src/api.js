import axios from 'axios';

const API_BASE = import.meta.env.VITE_API_BASE_URL || '';

// The access key is never bundled into the app (anything in frontend code is
// public). The user enters it once when the server says one is required, and
// it is kept in this browser only.
const KEY_STORAGE = 'analyst_api_key';

export function getApiKey() {
  try { return localStorage.getItem(KEY_STORAGE) || ''; } catch { return ''; }
}
export function setApiKey(key) {
  try {
    if (key) localStorage.setItem(KEY_STORAGE, key);
    else localStorage.removeItem(KEY_STORAGE);
  } catch { /* storage unavailable — key lasts for this page only */ }
}

// Tell the app to (re)ask for a key. App.jsx listens for this event.
export const AUTH_EVENT = 'analyst:auth-required';
function signalAuthRequired() {
  window.dispatchEvent(new Event(AUTH_EVENT));
}

function authHeaders() {
  const key = getApiKey();
  return key ? { 'X-Api-Key': key } : {};
}

const client = axios.create({
  baseURL: API_BASE,
  headers: { 'Content-Type': 'application/json' },
});
client.interceptors.request.use((config) => {
  Object.assign(config.headers, authHeaders());
  return config;
});
client.interceptors.response.use(
  (res) => res,
  (err) => {
    if (err?.response?.status === 401) signalAuthRequired();
    return Promise.reject(err);
  },
);

export const api = {
  // { required, valid, quota } — whether this server needs a key, and
  // whether the stored one works.
  checkAuth: async () => {
    const res = await client.get('/auth/check');
    return res.data;
  },

  // Session & Conversation
  createConversation: async (userId, title = 'New Session') => {
    const res = await client.post('/conversations', { user_id: userId, title }, {
      headers: { 'X-User-ID': userId },
    });
    return res.data;
  },

  getConversations: async (userId) => {
    const res = await client.get('/conversations', {
      headers: { 'X-User-ID': userId },
    });
    return res.data;
  },

  getConversation: async (conversationId) => {
    const res = await client.get(`/conversations/${conversationId}`);
    return res.data;
  },

  // Document Management
  uploadFiles: async (conversationId, files) => {
    const formData = new FormData();
    Array.from(files).forEach((f) => formData.append('files', f));
    const res = await client.post('/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
        'X-Conversation-ID': conversationId,
      },
    });
    return res.data;
  },

  getProcessingStatus: async (conversationId) => {
    const res = await client.get(`/conversations/${conversationId}/processing-status`);
    return res.data;
  },

  getDocuments: async (conversationId) => {
    const res = await client.get(`/conversations/${conversationId}/documents`);
    return res.data;
  },

  getNumericColumns: async (conversationId) => {
    const res = await client.get(`/conversations/${conversationId}/numeric-columns`);
    return res.data;
  },

  // Dashboards (Session Level)
  getDashboard: async (conversationId) => {
    const res = await client.get(`/conversations/${conversationId}/dashboard`);
    return res.data;
  },

  saveDashboard: async (conversationId, state) => {
    const res = await client.post(`/conversations/${conversationId}/dashboard`, { state });
    return res.data;
  },

  // Query Execution
  runQuery: async (conversationId, question, valueCol = null) => {
    const res = await client.post('/query', {
      question,
      conversation_id: conversationId,
      value_col: valueCol,
    });
    return res.data;
  },

  // Runs the full pipeline without writing to chat history — for the
  // Dashboard and Forecast tabs, so their queries don't leak into the chat.
  // wantChart: the Dashboard needs a chart back, so the backend routes to SQL
  // and builds a chart whenever the result is chartable.
  runSilentQuery: async (conversationId, question, valueCol = null, wantChart = false) => {
    const res = await client.post('/query/silent', {
      question,
      conversation_id: conversationId,
      value_col: valueCol,
      want_chart: wantChart,
    });
    return res.data;
  },

  // SSE Stream helper. Pass `signal` (from an AbortController) so callers can
  // cancel an in-flight stream, e.g. when the user switches sessions.
  streamQuery: async (conversationId, question, valueCol, { onMeta, onToken, onDone, onError, signal }) => {
    try {
      const response = await fetch(`${API_BASE}/query/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...authHeaders(),
        },
        body: JSON.stringify({
          question,
          conversation_id: conversationId,
          value_col: valueCol,
        }),
        signal,
      });

      if (response.status === 401) signalAuthRequired();
      if (!response.ok) {
        let detail = response.statusText;
        try { detail = (await response.json()).detail || detail; } catch { /* not JSON */ }
        throw new Error(detail || `Request failed (${response.status})`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split('\n\n');
        buffer = parts.pop() || '';

        for (const block of parts) {
          let eventName = 'message';
          let dataStr = '';

          for (const line of block.split('\n')) {
            if (line.startsWith('event: ')) eventName = line.slice(7).trim();
            if (line.startsWith('data: ')) dataStr = line.slice(6).trim();
          }

          if (!dataStr) continue;

          try {
            const parsed = JSON.parse(dataStr);
            if (eventName === 'meta' && onMeta) onMeta(parsed);
            else if (eventName === 'token' && onToken) onToken(parsed);
            else if (eventName === 'done' && onDone) onDone(parsed);
            else if (eventName === 'error' && onError) onError(parsed);
          } catch (e) {
            console.error('SSE JSON parse error', e);
          }
        }
      }
    } catch (err) {
      if (err.name === 'AbortError') return;
      if (onError) onError(err);
    }
  },
};
