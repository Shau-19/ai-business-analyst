import React, { useState, useEffect, useRef } from 'react';
import { Plus, Sparkles, Trash2, RefreshCw, BarChart2, X, AlertCircle } from 'lucide-react';
import ChartCanvas from './ChartCanvas';
import { api } from '../api';

// Ask the backend for a chart built from this session's data.
// Returns { panel } on success or { error } with a user-facing reason.
async function buildPanel(sessionId, prompt) {
  try {
    const res = await api.runSilentQuery(sessionId, prompt, null, true);
    if (!res.success) {
      return { error: res.error || 'The query failed.' };
    }
    if (!res.plot) {
      const answer = (res.explanation || res.answer || '').trim();
      return {
        error: answer
          ? `No chart for this — the answer is: "${answer.slice(0, 160)}". Try asking for a breakdown, e.g. "… by region" or "… per month".`
          : 'This question did not return chartable data (need a label column and a numeric column with 2+ rows).',
      };
    }
    return { panel: { title: res.plot.title || prompt, plot: res.plot, query: prompt } };
  } catch (e) {
    return { error: e?.response?.data?.detail || e.message || 'Request failed.' };
  }
}

export default function DashboardView({ activeSessionId }) {
  // dashboards: { [dashId]: { id, title, prompt, panels: [{ title, plot, query? }] } }
  const [dashboards, setDashboards] = useState({});
  const [activeDashId, setActiveDashId] = useState(null);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [newDashTitle, setNewDashTitle] = useState('');
  const [newDashPrompt, setNewDashPrompt] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [modalError, setModalError] = useState('');
  const [addPrompt, setAddPrompt] = useState('');
  const [isAdding, setIsAdding] = useState(false);
  const [notice, setNotice] = useState('');
  const [refreshing, setRefreshing] = useState({}); // { [panelIndex]: true }

  // Async work (chart generation can take seconds) must never write one
  // session's dashboards into another: every async path re-checks the
  // active session, and reads the latest state through refs rather than
  // the render-time closure.
  const activeSessionRef = useRef(activeSessionId);
  const dashboardsRef = useRef(dashboards);
  const activeDashIdRef = useRef(activeDashId);
  useEffect(() => { activeSessionRef.current = activeSessionId; }, [activeSessionId]);
  useEffect(() => { dashboardsRef.current = dashboards; }, [dashboards]);
  useEffect(() => { activeDashIdRef.current = activeDashId; }, [activeDashId]);

  // Load dashboards when session changes
  useEffect(() => {
    setDashboards({});
    setActiveDashId(null);
    setNotice('');
    setRefreshing({});
    setIsGenerating(false);   // in-flight work for the old session is dropped
    setIsAdding(false);
    setModalError('');
    if (!activeSessionId) return;
    loadDashboardState(activeSessionId);
  }, [activeSessionId]);

  const loadDashboardState = async (sessionId) => {
    try {
      const res = await api.getDashboard(sessionId);
      if (activeSessionRef.current !== sessionId) return;
      if (res.success && res.state) {
        if (res.state.dashboards) {
          setDashboards(res.state.dashboards);
          setActiveDashId(res.state.activeId || Object.keys(res.state.dashboards)[0] || null);
        } else if (res.state.panels) {
          const defaultDash = {
            id: 'default',
            title: res.state.title || 'Session Dashboard',
            panels: res.state.panels || [],
          };
          setDashboards({ default: defaultDash });
          setActiveDashId('default');
        }
      }
    } catch (e) {
      console.error('Failed to load dashboard state', e);
    }
  };

  // Persist to an explicit session. UI state is only touched if that session
  // is still the one on screen.
  const persist = async (sessionId, updatedDict, newActiveId) => {
    if (activeSessionRef.current === sessionId) {
      dashboardsRef.current = updatedDict;
      activeDashIdRef.current = newActiveId;
      setDashboards(updatedDict);
      setActiveDashId(newActiveId);
    }
    try {
      await api.saveDashboard(sessionId, {
        dashboards: updatedDict,
        activeId: newActiveId,
        built: Object.keys(updatedDict).length > 0,
      });
    } catch (e) {
      console.error('Failed to save dashboard state', e);
      if (activeSessionRef.current === sessionId) setNotice('Could not save the dashboard. Changes may be lost on reload.');
    }
  };

  const closeModal = () => {
    setIsModalOpen(false);
    setModalError('');
    setNewDashTitle('');
    setNewDashPrompt('');
  };

  const handleCreateDashboard = async (e) => {
    e.preventDefault();
    const sessionId = activeSessionId;
    const title = newDashTitle.trim();
    const prompt = newDashPrompt.trim();
    if (!title || !sessionId) return;

    setIsGenerating(true);
    setModalError('');
    const panels = [];

    if (prompt) {
      const { panel, error } = await buildPanel(sessionId, prompt);
      if (activeSessionRef.current !== sessionId) return; // user left; drop it
      if (error) {
        // Keep the modal open so the user can rephrase (or clear the prompt
        // to create an empty dashboard) instead of silently getting nothing.
        setModalError(error);
        setIsGenerating(false);
        return;
      }
      panels.push(panel);
    }

    const newId = `dash_${Date.now()}`;
    await persist(sessionId, {
      ...dashboardsRef.current,
      [newId]: { id: newId, title, prompt, panels },
    }, newId);
    setIsGenerating(false);
    closeModal();
  };

  const handleAddChart = async (e) => {
    e.preventDefault();
    const sessionId = activeSessionId;
    const dashId = activeDashIdRef.current;
    const prompt = addPrompt.trim();
    if (!prompt || !sessionId || !dashId) return;

    setIsAdding(true);
    setNotice('');
    const { panel, error } = await buildPanel(sessionId, prompt);
    if (activeSessionRef.current !== sessionId) return;
    setIsAdding(false);
    if (error) { setNotice(error); return; }

    const current = dashboardsRef.current;
    const dash = current[dashId];
    if (!dash) return;
    setAddPrompt('');
    await persist(sessionId, { ...current, [dashId]: { ...dash, panels: [...dash.panels, panel] } }, dashId);
  };

  const refreshPanel = async (dashId, idx) => {
    const sessionId = activeSessionId;
    const panel = dashboardsRef.current[dashId]?.panels?.[idx];
    if (!panel?.query) return null;
    setRefreshing((r) => ({ ...r, [idx]: true }));
    const { panel: fresh, error } = await buildPanel(sessionId, panel.query);
    if (activeSessionRef.current !== sessionId) return null;
    setRefreshing((r) => ({ ...r, [idx]: false }));
    if (error) { setNotice(`Refresh failed for "${panel.title}": ${error}`); return null; }

    const current = dashboardsRef.current;
    const dash = current[dashId];
    if (!dash || !dash.panels[idx]) return null;
    const panels = [...dash.panels];
    panels[idx] = { ...fresh, title: panel.title };
    await persist(sessionId, { ...current, [dashId]: { ...dash, panels } }, dashId);
    return true;
  };

  const refreshAll = async () => {
    const dashId = activeDashIdRef.current;
    const dash = dashboardsRef.current[dashId];
    if (!dash) return;
    setNotice('');
    // Sequential: each refresh persists the latest state before the next.
    for (let i = 0; i < dash.panels.length; i++) {
      if (dash.panels[i].query) await refreshPanel(dashId, i);
    }
  };

  const handleDeleteDashboard = (dashId) => {
    const dash = dashboardsRef.current[dashId];
    if (dash?.panels?.length && !window.confirm(`Delete dashboard "${dash.title}" and its ${dash.panels.length} chart(s)?`)) return;
    const nextDict = { ...dashboardsRef.current };
    delete nextDict[dashId];
    const keys = Object.keys(nextDict);
    const nextActive = activeDashIdRef.current === dashId ? (keys[0] || null) : activeDashIdRef.current;
    persist(activeSessionId, nextDict, nextActive);
  };

  const handleRemovePanel = (dashId, panelIndex) => {
    const dash = dashboardsRef.current[dashId];
    if (!dash) return;
    const updatedPanels = [...dash.panels];
    updatedPanels.splice(panelIndex, 1);
    persist(activeSessionId, { ...dashboardsRef.current, [dashId]: { ...dash, panels: updatedPanels } }, dashId);
  };

  const selectDashboard = (dashId) => {
    setActiveDashId(dashId);
    setNotice('');
  };

  const dashList = Object.values(dashboards);
  const activeDash = activeDashId ? dashboards[activeDashId] : null;
  const anyRefreshable = activeDash?.panels?.some((p) => p.query);
  const anyRefreshing = Object.values(refreshing).some(Boolean);

  return (
    <div className="flex-1 flex flex-col h-full overflow-hidden" style={{ background: 'var(--bg)' }}>
      {/* Dashboard tab strip */}
      <div
        className="flex items-center gap-1 px-5 h-12 border-b shrink-0 overflow-x-auto"
        style={{ borderColor: 'var(--border)', background: 'var(--surface)' }}
      >
        {dashList.map((d) => {
          const isActive = d.id === activeDashId;
          return (
            <button
              key={d.id}
              onClick={() => selectDashboard(d.id)}
              className="group flex items-center gap-2 px-3 py-1.5 rounded-md text-[12.5px] font-medium shrink-0 transition-colors cursor-pointer"
              style={{
                background: isActive ? 'var(--accent-soft)' : 'transparent',
                color: isActive ? 'var(--accent-text)' : 'var(--text-secondary)',
              }}
            >
              <span>{d.title}</span>
              <span className="text-[10.5px] opacity-60">{d.panels?.length || 0}</span>
              <X
                className="w-3 h-3 opacity-0 group-hover:opacity-60 hover:!opacity-100 transition-opacity"
                onClick={(e) => { e.stopPropagation(); handleDeleteDashboard(d.id); }}
              />
            </button>
          );
        })}

        <button
          onClick={() => setIsModalOpen(true)}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-[12.5px] font-medium shrink-0 transition-colors cursor-pointer"
          style={{ color: 'var(--text-muted)' }}
          onMouseEnter={(e) => (e.currentTarget.style.background = 'var(--bg-subtle)')}
          onMouseLeave={(e) => (e.currentTarget.style.background = 'transparent')}
        >
          <Plus className="w-3.5 h-3.5" />
          <span>New dashboard</span>
        </button>
      </div>

      {/* Add-chart bar for the active dashboard */}
      {activeDash && (
        <div className="px-6 pt-4 shrink-0">
          <div className="max-w-5xl mx-auto flex items-center gap-2">
            <form onSubmit={handleAddChart} className="flex-1 flex items-center gap-2">
              <input
                type="text"
                value={addPrompt}
                onChange={(e) => setAddPrompt(e.target.value)}
                placeholder='Add a chart, e.g. "revenue by region" or "units sold per month"'
                className="input flex-1"
                disabled={isAdding}
              />
              <button type="submit" disabled={isAdding || !addPrompt.trim()} className="btn btn-primary shrink-0">
                {isAdding ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Plus className="w-3.5 h-3.5" />}
                <span>Add chart</span>
              </button>
            </form>
            {anyRefreshable && (
              <button onClick={refreshAll} disabled={anyRefreshing} className="btn btn-secondary shrink-0" title="Re-run every chart against the current data">
                <RefreshCw className={`w-3.5 h-3.5 ${anyRefreshing ? 'animate-spin' : ''}`} />
                <span>Refresh all</span>
              </button>
            )}
          </div>
          {notice && (
            <div className="max-w-5xl mx-auto mt-2 flex items-start gap-2 text-[12.5px] px-3 py-2 rounded-md"
                 style={{ background: 'var(--bg-subtle)', color: 'var(--text-secondary)', border: '1px solid var(--border)' }}>
              <AlertCircle className="w-3.5 h-3.5 mt-0.5 shrink-0" style={{ color: 'var(--red)' }} />
              <span className="flex-1">{notice}</span>
              <X className="w-3.5 h-3.5 cursor-pointer opacity-60 hover:opacity-100" onClick={() => setNotice('')} />
            </div>
          )}
        </div>
      )}

      {/* Main Grid View */}
      <div className="flex-1 overflow-y-auto p-6">
        {!activeDash || activeDash.panels.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center gap-4">
            <div className="w-12 h-12 rounded-xl flex items-center justify-center" style={{ background: 'var(--blue-soft)' }}>
              <BarChart2 className="w-5 h-5" style={{ color: 'var(--blue)' }} />
            </div>
            <div className="text-center">
              <h3 className="text-[14.5px] font-semibold text-[var(--text)]">
                {!activeDash ? 'No dashboard selected' : 'This dashboard is empty'}
              </h3>
              <p className="text-[13px] text-[var(--text-muted)] mt-1.5 max-w-sm">
                {!activeDash
                  ? 'Create a dashboard to start pinning charts.'
                  : 'Add a chart above, or pin one from any chat answer.'}
              </p>
              {!activeDash && (
                <button onClick={() => setIsModalOpen(true)} className="btn btn-primary mt-4">
                  <Plus className="w-4 h-4" />
                  <span>New dashboard</span>
                </button>
              )}
            </div>
          </div>
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-5 max-w-5xl mx-auto">
            {activeDash.panels.map((panel, idx) => (
              <div key={`${activeDash.id}-${idx}-${panel.title}`} className="card p-4 flex flex-col">
                <div className="flex items-center justify-between pb-3 mb-3 border-b gap-2" style={{ borderColor: 'var(--border)' }}>
                  <h4 className="text-[13px] font-semibold text-[var(--text)] truncate" title={panel.query || panel.title}>{panel.title}</h4>
                  <div className="flex items-center gap-2 shrink-0">
                    {panel.query && (
                      <button
                        onClick={() => refreshPanel(activeDash.id, idx)}
                        disabled={!!refreshing[idx]}
                        title="Re-run against the current data"
                        className="text-[var(--text-muted)] hover:text-[var(--accent)] transition-colors cursor-pointer"
                      >
                        <RefreshCw className={`w-3.5 h-3.5 ${refreshing[idx] ? 'animate-spin' : ''}`} />
                      </button>
                    )}
                    <button
                      onClick={() => handleRemovePanel(activeDash.id, idx)}
                      title="Remove chart"
                      className="text-[var(--text-muted)] hover:text-[var(--red)] transition-colors cursor-pointer"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                  </div>
                </div>
                <div className="flex-1 min-h-[260px]">
                  <ChartCanvas plotSpec={panel.plot} />
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Create New Dashboard Modal */}
      {isModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4" style={{ background: 'rgba(10,10,14,.5)', backdropFilter: 'blur(2px)' }}>
          <div className="card w-full max-w-md p-6 space-y-4" style={{ boxShadow: 'var(--shadow-lg)' }}>
            <h3 className="text-[14.5px] font-semibold text-[var(--text)] flex items-center gap-2">
              <Sparkles className="w-4 h-4" style={{ color: 'var(--blue)' }} />
              New dashboard
            </h3>

            <form onSubmit={handleCreateDashboard} className="space-y-4">
              <div>
                <label className="block text-[12.5px] font-medium text-[var(--text-secondary)] mb-1.5">Title</label>
                <input
                  type="text"
                  required
                  autoFocus
                  value={newDashTitle}
                  onChange={(e) => setNewDashTitle(e.target.value)}
                  placeholder="e.g. Sales Regional Breakdown"
                  className="input"
                />
              </div>

              <div>
                <label className="block text-[12.5px] font-medium text-[var(--text-secondary)] mb-1.5">Initial chart prompt (optional)</label>
                <input
                  type="text"
                  value={newDashPrompt}
                  onChange={(e) => { setNewDashPrompt(e.target.value); setModalError(''); }}
                  placeholder="e.g. Total revenue by region"
                  className="input"
                />
                {modalError && (
                  <p className="mt-2 text-[12px] flex items-start gap-1.5" style={{ color: 'var(--red)' }}>
                    <AlertCircle className="w-3.5 h-3.5 mt-0.5 shrink-0" />
                    <span>{modalError} Clear the prompt to create an empty dashboard.</span>
                  </p>
                )}
              </div>

              <div className="flex justify-end gap-2 pt-1">
                <button type="button" onClick={closeModal} className="btn btn-secondary">
                  Cancel
                </button>
                <button type="submit" disabled={isGenerating || !newDashTitle.trim()} className="btn btn-primary">
                  {isGenerating ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Plus className="w-3.5 h-3.5" />}
                  <span>{isGenerating ? 'Building chart…' : 'Create'}</span>
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
