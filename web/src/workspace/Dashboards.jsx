import { useEffect, useRef, useState } from 'react';
import { Plus, RefreshCw, Trash2, X, Maximize2, Minimize2, GripVertical, LayoutGrid, CircleAlert, LoaderCircle } from 'lucide-react';
import Chart from '../components/Chart';
import { api } from '../lib/api';

// Build a dashboard panel from a question; returns {panel} or {error}.
async function buildPanel(cid, prompt) {
  try {
    const res = await api.silentQuery(cid, prompt, { wantChart: true });
    if (!res.success) return { error: res.error || 'The query failed.' };
    if (!res.plot) {
      const answer = (res.explanation || res.answer || '').trim();
      return {
        error: answer
          ? `That returns a single answer (“${answer.slice(0, 140)}”), not a chart. Try a breakdown, like “… by region” or “… per month”.`
          : 'That question didn’t return chartable data. Charts need a label column and a number, over 2+ rows.',
      };
    }
    return { panel: { title: res.plot.title || prompt, plot: res.plot, query: prompt } };
  } catch (e) {
    return { error: e.message };
  }
}

export default function Dashboards({ activeId, profile, notify }) {
  const [dashes, setDashes] = useState({});
  const [current, setCurrent] = useState(null);
  const [loaded, setLoaded] = useState(false);
  const [prompt, setPrompt] = useState('');
  const [adding, setAdding] = useState(false);
  const [error, setError] = useState('');
  const [refreshing, setRefreshing] = useState({});
  const [naming, setNaming] = useState(false);
  const [newName, setNewName] = useState('');
  const [dragFrom, setDragFrom] = useState(null);

  // Latest state for async work, and the session it belongs to.
  const sessionRef = useRef(activeId);
  const dashRef = useRef(dashes);
  const currentRef = useRef(current);
  useEffect(() => { sessionRef.current = activeId; }, [activeId]);
  useEffect(() => { dashRef.current = dashes; }, [dashes]);
  useEffect(() => { currentRef.current = current; }, [current]);

  useEffect(() => {
    if (!activeId) { setLoaded(true); return; }
    const cid = activeId;
    api.getDashboard(cid).then((res) => {
      if (sessionRef.current !== cid) return;
      const st = res.state;
      if (st?.dashboards) {
        setDashes(st.dashboards);
        setCurrent(st.activeId && st.dashboards[st.activeId] ? st.activeId : Object.keys(st.dashboards)[0] || null);
      } else if (st?.panels) {
        setDashes({ default: { id: 'default', title: st.title || 'Main dashboard', panels: st.panels } });
        setCurrent('default');
      }
    }).catch(() => {}).finally(() => { if (sessionRef.current === cid) setLoaded(true); });
  }, [activeId]);

  const persist = async (cid, next, nextCurrent) => {
    if (sessionRef.current === cid) {
      dashRef.current = next; currentRef.current = nextCurrent;
      setDashes(next); setCurrent(nextCurrent);
    }
    try {
      await api.saveDashboard(cid, { dashboards: next, activeId: nextCurrent, built: Object.keys(next).length > 0 });
    } catch (e) {
      if (sessionRef.current === cid) notify(`Couldn’t save the dashboard: ${e.message}`, 'error');
    }
  };

  const createDashboard = (e) => {
    e.preventDefault();
    const title = newName.trim() || 'Untitled dashboard';
    const id = `dash_${Date.now()}`;
    persist(activeId, { ...dashRef.current, [id]: { id, title, panels: [] } }, id);
    setNaming(false); setNewName('');
  };

  const deleteDashboard = (id) => {
    const d = dashRef.current[id];
    if (d?.panels?.length && !window.confirm(`Delete “${d.title}” and its ${d.panels.length} chart(s)?`)) return;
    const next = { ...dashRef.current };
    delete next[id];
    persist(activeId, next, currentRef.current === id ? Object.keys(next)[0] || null : currentRef.current);
  };

  const addChart = async (e) => {
    e.preventDefault();
    const cid = activeId, q = prompt.trim();
    if (!q || !cid) return;
    let dashId = currentRef.current;
    setAdding(true); setError('');
    const { panel, error: err } = await buildPanel(cid, q);
    if (sessionRef.current !== cid) return;
    setAdding(false);
    if (err) { setError(err); return; }
    let next = dashRef.current;
    if (!dashId || !next[dashId]) {
      dashId = `dash_${Date.now()}`;
      next = { ...next, [dashId]: { id: dashId, title: 'Main dashboard', panels: [] } };
    }
    const d = next[dashId];
    setPrompt('');
    persist(cid, { ...next, [dashId]: { ...d, panels: [...d.panels, panel] } }, dashId);
  };

  const updatePanels = (fn) => {
    const id = currentRef.current, d = dashRef.current[id];
    if (!d) return;
    persist(activeId, { ...dashRef.current, [id]: { ...d, panels: fn([...d.panels]) } }, id);
  };

  const refreshPanel = async (idx) => {
    const cid = activeId, id = currentRef.current;
    const p = dashRef.current[id]?.panels?.[idx];
    if (!p?.query) return;
    setRefreshing((r) => ({ ...r, [idx]: true }));
    const { panel, error: err } = await buildPanel(cid, p.query);
    if (sessionRef.current !== cid) return;
    setRefreshing((r) => ({ ...r, [idx]: false }));
    if (err) { notify(`Couldn’t refresh “${p.title}”: ${err}`, 'error'); return; }
    const d = dashRef.current[id];
    if (!d?.panels[idx]) return;
    const panels = [...d.panels];
    panels[idx] = { ...panel, title: p.title, wide: p.wide };
    await persist(cid, { ...dashRef.current, [id]: { ...d, panels } }, id);
  };

  const refreshAll = async () => {
    const d = dashRef.current[currentRef.current];
    for (let i = 0; i < (d?.panels?.length || 0); i++) if (d.panels[i].query) await refreshPanel(i);
    notify('Dashboard refreshed', 'good');
  };

  const onDrop = (to) => {
    if (dragFrom === null || dragFrom === to) return;
    updatePanels((ps) => { const [m] = ps.splice(dragFrom, 1); ps.splice(to, 0, m); return ps; });
    setDragFrom(null);
  };

  if (!loaded) return <div className="dash"><div className="skeleton" style={{ height: 300, borderRadius: 18 }} /></div>;

  const list = Object.values(dashes);
  const dash = current ? dashes[current] : null;
  const hasData = !!profile?.datasets?.length;
  const anyRefreshing = Object.values(refreshing).some(Boolean);

  return (
    <div className="dash">
      <div className="dash-tabs" role="tablist" aria-label="Dashboards">
        {list.map((d) => (
          <div key={d.id} className={`dash-tab ${d.id === current ? 'is-on' : ''}`}>
            <button role="tab" aria-selected={d.id === current} onClick={() => setCurrent(d.id)}>
              {d.title} <span className="num">{d.panels?.length || 0}</span>
            </button>
            <button className="dash-tab-x" onClick={() => deleteDashboard(d.id)} aria-label={`Delete ${d.title}`}><X /></button>
          </div>
        ))}
        {naming ? (
          <form className="dash-name" onSubmit={createDashboard}>
            <label className="sr-only" htmlFor="dash-name">Dashboard name</label>
            <input id="dash-name" className="input" autoFocus value={newName} onChange={(e) => setNewName(e.target.value)} placeholder="e.g. Regional sales" />
            <button className="btn btn-sm btn-primary" type="submit">Create</button>
            <button className="btn btn-sm btn-quiet" type="button" onClick={() => setNaming(false)}>Cancel</button>
          </form>
        ) : (
          <button className="btn btn-sm btn-quiet" onClick={() => setNaming(true)}><Plus /> New dashboard</button>
        )}
      </div>

      <form className="dash-add" onSubmit={addChart}>
        <label className="sr-only" htmlFor="dash-prompt">Describe a chart to add</label>
        <input id="dash-prompt" className="input" value={prompt} disabled={adding || !hasData}
          onChange={(e) => { setPrompt(e.target.value); setError(''); }}
          placeholder={hasData ? 'Describe a chart: “revenue by month”, “units per region”…' : 'Upload a CSV or Excel file to build charts'} />
        <button className="btn btn-primary" type="submit" disabled={adding || !prompt.trim() || !hasData}>
          {adding ? <LoaderCircle className="spin" /> : <Plus />} {adding ? 'Building…' : 'Add chart'}
        </button>
        {dash?.panels?.some((p) => p.query) && (
          <button className="btn btn-ghost" type="button" onClick={refreshAll} disabled={anyRefreshing}>
            <RefreshCw className={anyRefreshing ? 'spin' : ''} /> Refresh all
          </button>
        )}
      </form>
      {error && <div className="banner banner-bad"><CircleAlert aria-hidden="true" /> {error}</div>}

      {!dash || dash.panels.length === 0 ? (
        <div className="dash-empty">
          <LayoutGrid aria-hidden="true" />
          <h2 className="display">{list.length ? 'This dashboard is empty.' : 'No dashboards yet.'}</h2>
          <p>Describe a chart above, or pin one from Overview or any answer in Ask.</p>
        </div>
      ) : (
        <div className="dash-grid">
          {dash.panels.map((p, i) => (
            <section
              key={`${current}-${i}-${p.title}`}
              className={`tile panel ${p.wide ? 'is-wide' : ''} ${dragFrom === i ? 'is-dragging' : ''}`}
              draggable
              onDragStart={() => setDragFrom(i)}
              onDragEnd={() => setDragFrom(null)}
              onDragOver={(e) => e.preventDefault()}
              onDrop={() => onDrop(i)}
              aria-label={p.title}
            >
              <div className="tile-head">
                <span className="panel-grip" aria-hidden="true"><GripVertical /></span>
                <div className="panel-title">
                  <h3 className="tile-title">{p.title}</h3>
                  {p.query && <span className="tile-sub">“{p.query}”</span>}
                </div>
                <div className="tile-actions">
                  {p.query && (
                    <button className="icon-btn" onClick={() => refreshPanel(i)} disabled={!!refreshing[i]} aria-label="Refresh with latest data" title="Refresh">
                      <RefreshCw className={refreshing[i] ? 'spin' : ''} />
                    </button>
                  )}
                  <button className="icon-btn" onClick={() => updatePanels((ps) => { ps[i] = { ...ps[i], wide: !ps[i].wide }; return ps; })}
                    aria-label={p.wide ? 'Make narrow' : 'Make wide'} title={p.wide ? 'Narrow' : 'Wide'}>
                    {p.wide ? <Minimize2 /> : <Maximize2 />}
                  </button>
                  <button className="icon-btn" onClick={() => updatePanels((ps) => { ps.splice(i, 1); return ps; })} aria-label="Remove chart" title="Remove"><Trash2 /></button>
                </div>
              </div>
              <Chart spec={p.plot} height={p.wide ? 300 : 250} />
            </section>
          ))}
        </div>
      )}
    </div>
  );
}
