import { useCallback, useEffect, useRef, useState } from 'react';
import { api } from '../lib/api';
import Sidebar from './Sidebar';
import Topbar from './Topbar';
import Overview from './Overview';
import Ask from './Ask';
import Dashboards from './Dashboards';
import Forecast from './Forecast';
import DataView from './DataView';
import Toast from './Toast';

const DEFAULT_TITLES = new Set(['New Chat', 'New Session', 'New analysis', '']);
const SAMPLE_FILES = ['Sales_Report_H1_2025.csv', 'Strategic_Planning_Jan2025.docx'];

function useUserId() {
  const [id] = useState(() => {
    try {
      const saved = localStorage.getItem('analyst_user_id');
      if (saved) return saved;
      const fresh = `user_${Math.random().toString(36).slice(2, 10)}`;
      localStorage.setItem('analyst_user_id', fresh);
      return fresh;
    } catch {
      return 'default_user';
    }
  });
  return id;
}

export default function Workspace({ theme, authReady }) {
  const userId = useUserId();
  const [conversations, setConversations] = useState([]);
  const [activeId, setActiveId] = useState(null);
  const [tab, setTab] = useState('overview');
  const [messages, setMessages] = useState([]);
  const [profile, setProfile] = useState(null);           // null = loading
  const [processing, setProcessing] = useState(null);     // {status, files, error}
  const [streaming, setStreaming] = useState(false);
  const [toast, setToast] = useState(null);
  const [mobileNav, setMobileNav] = useState(false);

  // Async results must never land in a session the user has left.
  const activeRef = useRef(activeId);
  useEffect(() => { activeRef.current = activeId; }, [activeId]);
  const abortRef = useRef(null);
  const pollRef = useRef(null);

  const notify = useCallback((text, kind = 'info', action = null) => setToast({ text, kind, action, id: Date.now() }), []);

  const loadConversations = useCallback(async ({ selectFirst = false } = {}) => {
    try {
      const res = await api.listConversations(userId);
      const list = res.conversations || [];
      setConversations(list);
      if (selectFirst && !activeRef.current && list.length) setActiveId(list[0].conversation_id);
      return list;
    } catch (e) {
      if (e.status !== 401) notify(`Couldn’t load your analyses: ${e.message}`, 'error');
      return [];
    }
  }, [userId, notify]);

  const loadProfile = useCallback(async (cid) => {
    try {
      const res = await api.profile(cid);
      if (activeRef.current === cid) setProfile(res.success ? res : { datasets: [], documents: [], files: [], error: res.error });
    } catch (e) {
      if (activeRef.current === cid) setProfile({ datasets: [], documents: [], files: [], error: e.message });
    }
  }, []);

  // ── session switch ───────────────────────────────────────────────────
  useEffect(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    clearInterval(pollRef.current);
    setStreaming(false);
    setMessages([]);
    setProcessing(null);
    if (!activeId) { setProfile({ datasets: [], documents: [], files: [] }); return; }
    setProfile(null);
    const cid = activeId;
    api.getConversation(cid)
      .then((r) => { if (activeRef.current === cid) setMessages(r.conversation?.messages || []); })
      .catch(() => {});
    api.processingStatus(cid)
      .then((s) => {
        if (activeRef.current !== cid) return;
        if (s.status === 'processing') { setProcessing(s); startPolling(cid); }
      })
      .catch(() => {});
    loadProfile(cid);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeId]);

  // ── initial load (after the access key, if one is needed) ───────────
  useEffect(() => {
    if (!authReady) return;
    (async () => {
      let wantSample = false;
      try { wantSample = sessionStorage.getItem('analyst_load_sample') === '1'; sessionStorage.removeItem('analyst_load_sample'); } catch { /* ignore */ }
      await loadConversations({ selectFirst: !wantSample });
      if (wantSample) loadSample({ fresh: true });
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [authReady]);

  useEffect(() => () => { abortRef.current?.abort(); clearInterval(pollRef.current); }, []);

  // ── sessions ─────────────────────────────────────────────────────────
  const createSession = useCallback(async (title = 'New Chat') => {
    const res = await api.createConversation(userId, title);
    activeRef.current = res.conversation_id;
    setActiveId(res.conversation_id);
    setTab('overview');
    loadConversations();
    return res.conversation_id;
  }, [userId, loadConversations]);

  const newSession = async () => {
    try { await createSession(); setMobileNav(false); } catch (e) { notify(`Couldn’t start a new analysis: ${e.message}`, 'error'); }
  };

  const deleteSession = async (cid) => {
    try {
      await api.deleteConversation(cid);
      if (activeRef.current === cid) setActiveId(null);
      const list = await loadConversations();
      if (activeRef.current === null && list.length) setActiveId(list[0].conversation_id);
      notify('Analysis deleted');
    } catch (e) {
      notify(`Couldn’t delete: ${e.message}`, 'error');
    }
  };

  // ── uploads ──────────────────────────────────────────────────────────
  function startPolling(cid) {
    clearInterval(pollRef.current);
    let tries = 0;
    pollRef.current = setInterval(async () => {
      tries += 1;
      if (tries > 150) { clearInterval(pollRef.current); return; }
      try {
        const s = await api.processingStatus(cid);
        if (activeRef.current !== cid) { clearInterval(pollRef.current); return; }
        if (s.status === 'processing') return;
        clearInterval(pollRef.current);
        setProcessing(s.status === 'error' ? s : null);
        if (s.status === 'error') notify(`Some files couldn’t be read: ${s.error}`, 'error');
        else if (s.error) notify(`Indexed with warnings: ${s.error}`, 'warn');
        else notify(`Ready: ${(s.files || []).join(', ')}`, 'good');
        loadProfile(cid);
        loadConversations();
      } catch { /* transient; keep polling */ }
    }, 1200);
  }

  const uploadFiles = async (files, { fresh = false } = {}) => {
    const list = Array.from(files || []);
    if (!list.length) return;
    try {
      let cid = fresh ? null : activeRef.current;
      if (!cid) cid = await createSession();
      setProcessing({ status: 'processing', files: list.map((f) => f.name), error: null });
      setTab('overview');
      await api.upload(cid, list);
      // Name the analysis after its first data file, if it still has a default title.
      const conv = conversations.find((c) => c.conversation_id === cid);
      if (!conv || DEFAULT_TITLES.has(conv.title || '')) {
        const main = list.find((f) => /\.(csv|xlsx|xls)$/i.test(f.name)) || list[0];
        api.renameConversation(cid, main.name.replace(/\.[^.]+$/, '').replace(/[_-]+/g, ' ')).then(() => loadConversations()).catch(() => {});
      }
      startPolling(cid);
    } catch (e) {
      setProcessing(null);
      notify(`Upload failed: ${e.message}`, 'error');
    }
  };

  async function loadSample({ fresh = false } = {}) {
    try {
      const files = await Promise.all(SAMPLE_FILES.map(async (name) => {
        const r = await fetch(`samples/${name}`);
        if (!r.ok) throw new Error(`couldn’t fetch ${name}`);
        return new File([await r.blob()], name);
      }));
      await uploadFiles(files, { fresh });
    } catch (e) {
      notify(`Couldn’t load the sample: ${e.message}`, 'error');
    }
  }

  // ── asking ───────────────────────────────────────────────────────────
  const ask = async (question) => {
    const q = question.trim();
    if (!q || streaming) return;
    let cid = activeRef.current;
    try { if (!cid) cid = await createSession(); } catch (e) { notify(e.message, 'error'); return; }
    setTab('ask');
    setMessages((m) => [
      ...m,
      { role: 'user', content: q },
      { role: 'assistant', content: '', routing: '', metadata: {}, pending: true },
    ]);
    setStreaming(true);
    const controller = new AbortController();
    abortRef.current = controller;
    const stale = () => activeRef.current !== cid;
    const patchLast = (fn) => setMessages((m) => {
      const next = m.slice();
      next[next.length - 1] = fn({ ...next[next.length - 1] });
      return next;
    });

    await api.streamQuery(cid, q, {
      signal: controller.signal,
      onMeta: (meta) => {
        if (stale()) return;
        patchLast((msg) => ({
          ...msg, routing: meta.routing,
          metadata: { plot: meta.plot, sql_query: meta.sql_query, sources: meta.sources, data: meta.data },
          failed: meta.success === false,
        }));
      },
      onToken: (t) => { if (!stale()) patchLast((msg) => ({ ...msg, content: msg.content + t })); },
      onDone: () => {
        if (stale()) return;
        patchLast((msg) => ({ ...msg, pending: false }));
        setStreaming(false);
        loadConversations();
      },
      onError: (err) => {
        if (stale()) return;
        patchLast((msg) => ({ ...msg, pending: false, failed: true, content: msg.content || err.message }));
        setStreaming(false);
      },
    });
    if (abortRef.current === controller) abortRef.current = null;
    if (!stale()) { setStreaming(false); patchLast((msg) => ({ ...msg, pending: false })); }
  };

  const stopAsking = () => { abortRef.current?.abort(); setStreaming(false); };

  // ── pinning ──────────────────────────────────────────────────────────
  const pinChart = async (plot, question) => {
    const cid = activeRef.current;
    if (!cid || !plot) return;
    try {
      const res = await api.getDashboard(cid);
      const state = res.state?.dashboards ? res.state : { dashboards: {}, activeId: null };
      let target = state.activeId && state.dashboards[state.activeId] ? state.activeId : Object.keys(state.dashboards)[0];
      if (!target) {
        target = `dash_${Date.now()}`;
        state.dashboards[target] = { id: target, title: 'Main dashboard', panels: [] };
      }
      state.dashboards[target].panels.push({ title: plot.title || question || 'Pinned chart', plot, ...(question ? { query: question } : {}) });
      state.activeId = target;
      await api.saveDashboard(cid, state);
      if (activeRef.current === cid) {
        notify(`Pinned to “${state.dashboards[target].title}”`, 'good', { label: 'View', run: () => setTab('dashboards') });
      }
    } catch (e) {
      notify(`Couldn’t pin: ${e.message}`, 'error');
    }
  };

  const activeConv = conversations.find((c) => c.conversation_id === activeId);
  const hasData = !!profile?.datasets?.length;
  const shared = { activeId, profile, processing, onUpload: uploadFiles, onSample: () => loadSample(), onAsk: ask, onPin: pinChart, setTab, notify };

  return (
    <div className={`ws ${mobileNav ? 'nav-open' : ''}`}>
      <Sidebar
        conversations={conversations}
        activeId={activeId}
        onSelect={(id) => { setActiveId(id); setMobileNav(false); }}
        onNew={newSession}
        onDelete={deleteSession}
        onUpload={uploadFiles}
        uploading={processing?.status === 'processing'}
        theme={theme}
        onClose={() => setMobileNav(false)}
      />
      {mobileNav && <button className="ws-scrim" aria-label="Close menu" onClick={() => setMobileNav(false)} />}
      <div className="ws-main">
        <Topbar
          title={activeConv?.title}
          profile={profile}
          tab={tab}
          setTab={setTab}
          onUpload={uploadFiles}
          onMenu={() => setMobileNav(true)}
          hasData={hasData}
        />
        <main className="ws-view" key={`${activeId}-${tab}`}>
          {tab === 'overview' && <Overview {...shared} />}
          {tab === 'ask' && (
            <Ask {...shared} messages={messages} streaming={streaming} onStop={stopAsking} />
          )}
          {tab === 'dashboards' && <Dashboards {...shared} />}
          {tab === 'forecast' && <Forecast {...shared} />}
          {tab === 'data' && <DataView {...shared} />}
        </main>
      </div>
      {toast && <Toast key={toast.id} toast={toast} onClose={() => setToast(null)} />}
    </div>
  );
}
