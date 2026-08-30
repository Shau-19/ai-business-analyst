import React, { useState, useEffect } from 'react';
import { Plus, LayoutGrid, Sparkles, Trash2, Download, RefreshCw, BarChart2 } from 'lucide-react';
import ChartCanvas from './ChartCanvas';
import { api } from '../api';

export default function DashboardView({ activeSessionId }) {
  // sessionDashboards structure: { [dashId]: { id, title, prompt, panels: [{ title, plot }] } }
  const [dashboards, setDashboards] = useState({});
  const [activeDashId, setActiveDashId] = useState(null);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [newDashTitle, setNewDashTitle] = useState('');
  const [newDashPrompt, setNewDashPrompt] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);

  // Load dashboards when session changes
  useEffect(() => {
    if (!activeSessionId) return;
    loadDashboardState();
  }, [activeSessionId]);

  const loadDashboardState = async () => {
    try {
      const res = await api.getDashboard(activeSessionId);
      if (res.success && res.state) {
        // If state is multi-dashboard dict
        if (res.state.dashboards) {
          setDashboards(res.state.dashboards);
          setActiveDashId(res.state.activeId || Object.keys(res.state.dashboards)[0] || null);
        } else if (res.state.panels) {
          // Backward compatibility: single dashboard object converted to multi
          const defaultDash = {
            id: 'default',
            title: res.state.title || 'Session Dashboard',
            panels: res.state.panels || [],
          };
          const dict = { default: defaultDash };
          setDashboards(dict);
          setActiveDashId('default');
        }
      } else {
        setDashboards({});
        setActiveDashId(null);
      }
    } catch (e) {
      console.error('Failed to load dashboard state', e);
    }
  };

  const saveDashboardState = async (updatedDict, newActiveId) => {
    setDashboards(updatedDict);
    setActiveDashId(newActiveId);
    if (!activeSessionId) return;
    try {
      await api.saveDashboard(activeSessionId, {
        dashboards: updatedDict,
        activeId: newActiveId,
        built: Object.keys(updatedDict).length > 0,
      });
    } catch (e) {
      console.error('Failed to save dashboard state', e);
    }
  };

  const handleCreateDashboard = async (e) => {
    e.preventDefault();
    if (!newDashTitle.trim() || !activeSessionId) return;

    setIsGenerating(true);
    const newId = `dash_${Date.now()}`;
    let panels = [];

    // Optional: generate widgets from prompt
    if (newDashPrompt.trim()) {
      try {
        const queryRes = await api.runQuery(activeSessionId, newDashPrompt.trim());
        if (queryRes.success && queryRes.plot) {
          panels.push({ title: queryRes.plot.title || 'Overview Chart', plot: queryRes.plot });
        }
      } catch (err) {
        console.error('Auto query failed', err);
      }
    }

    const updatedDict = {
      ...dashboards,
      [newId]: {
        id: newId,
        title: newDashTitle.trim(),
        prompt: newDashPrompt.trim(),
        panels: panels,
      },
    };

    await saveDashboardState(updatedDict, newId);
    setIsGenerating(false);
    setIsModalOpen(false);
    setNewDashTitle('');
    setNewDashPrompt('');
  };

  const handleDeleteDashboard = async (dashId) => {
    const nextDict = { ...dashboards };
    delete nextDict[dashId];
    const keys = Object.keys(nextDict);
    const nextActiveId = keys.length > 0 ? keys[0] : null;
    await saveDashboardState(nextDict, nextActiveId);
  };

  const handleRemovePanel = async (dashId, panelIndex) => {
    const dash = dashboards[dashId];
    if (!dash) return;
    const updatedPanels = [...dash.panels];
    updatedPanels.splice(panelIndex, 1);
    const updatedDict = {
      ...dashboards,
      [dashId]: { ...dash, panels: updatedPanels },
    };
    await saveDashboardState(updatedDict, dashId);
  };

  const activeDash = activeDashId ? dashboards[activeDashId] : null;

  return (
    <div className="flex-1 flex flex-col h-full bg-slate-950 overflow-hidden">
      {/* Top Bar - Dashboard Selector & Create Actions */}
      <div className="p-4 bg-slate-900 border-b border-slate-800 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <LayoutGrid className="w-5 h-5 text-cyan-400" />
          <div className="flex items-center gap-2">
            <label className="text-xs font-mono text-slate-400">DASHBOARD:</label>
            {Object.keys(dashboards).length > 0 ? (
              <select
                value={activeDashId || ''}
                onChange={(e) => setActiveDashId(e.target.value)}
                className="bg-slate-950 border border-slate-700 rounded-lg px-3 py-1.5 font-mono text-xs text-slate-100 focus:outline-none focus:border-cyan-500"
              >
                {Object.values(dashboards).map((d) => (
                  <option key={d.id} value={d.id}>
                    {d.title} ({d.panels?.length || 0} charts)
                  </option>
                ))}
              </select>
            ) : (
              <span className="text-xs font-mono text-slate-500">No dashboards created</span>
            )}
          </div>
        </div>

        <div className="flex items-center gap-2">
          {activeDash && (
            <button
              onClick={() => handleDeleteDashboard(activeDash.id)}
              className="px-3 py-1.5 bg-rose-500/10 hover:bg-rose-500/20 text-rose-400 border border-rose-500/30 rounded-lg font-mono text-xs flex items-center gap-1.5 transition-colors cursor-pointer"
            >
              <Trash2 className="w-3.5 h-3.5" />
              <span>DELETE</span>
            </button>
          )}

          <button
            onClick={() => setIsModalOpen(true)}
            className="px-4 py-1.5 bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-mono font-bold text-xs rounded-lg flex items-center gap-1.5 transition-colors cursor-pointer shadow-lg shadow-cyan-500/10"
          >
            <Plus className="w-4 h-4" />
            <span>NEW DASHBOARD</span>
          </button>
        </div>
      </div>

      {/* Main Grid View */}
      <div className="flex-1 overflow-y-auto p-6">
        {!activeDash || activeDash.panels.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center text-slate-500 gap-4">
            <BarChart2 className="w-12 h-12 text-slate-700" />
            <div className="text-center">
              <h3 className="font-mono text-sm font-semibold text-slate-300">
                {!activeDash ? 'No Dashboard Selected' : 'This Dashboard is Empty'}
              </h3>
              <p className="text-xs font-mono text-slate-500 mt-1 max-w-sm">
                Click "+ NEW DASHBOARD" above or click "📌 PIN TO DASHBOARD" on any chart in the Chat tab.
              </p>
            </div>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-2 gap-6">
            {activeDash.panels.map((panel, idx) => (
              <div key={idx} className="bg-slate-900 border border-slate-800 rounded-xl p-4 flex flex-col">
                <div className="flex items-center justify-between pb-3 mb-3 border-b border-slate-800">
                  <h4 className="font-mono text-xs font-semibold text-slate-200">{panel.title}</h4>
                  <button
                    onClick={() => handleRemovePanel(activeDash.id, idx)}
                    className="text-slate-500 hover:text-rose-400 transition-colors"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
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
        <div className="fixed inset-0 bg-slate-950/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-slate-900 border border-slate-800 rounded-xl w-full max-w-md p-6 space-y-4 shadow-2xl">
            <h3 className="font-mono text-sm font-bold text-slate-100 flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-cyan-400" />
              CREATE NEW DASHBOARD OBJECT
            </h3>

            <form onSubmit={handleCreateDashboard} className="space-y-4">
              <div>
                <label className="block font-mono text-xs text-slate-400 mb-1">Dashboard Title</label>
                <input
                  type="text"
                  required
                  value={newDashTitle}
                  onChange={(e) => setNewDashTitle(e.target.value)}
                  placeholder="e.g. Sales Regional Breakdown"
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs font-mono text-slate-100 focus:outline-none focus:border-cyan-500"
                />
              </div>

              <div>
                <label className="block font-mono text-xs text-slate-400 mb-1">Initial Visualization Prompt (Optional)</label>
                <input
                  type="text"
                  value={newDashPrompt}
                  onChange={(e) => setNewDashPrompt(e.target.value)}
                  placeholder="e.g. Show a bar chart of total revenue by region"
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs font-mono text-slate-100 focus:outline-none focus:border-cyan-500"
                />
              </div>

              <div className="flex justify-end gap-2 pt-2">
                <button
                  type="button"
                  onClick={() => setIsModalOpen(false)}
                  className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 font-mono text-xs rounded-lg transition-colors"
                >
                  CANCEL
                </button>
                <button
                  type="submit"
                  disabled={isGenerating || !newDashTitle.trim()}
                  className="px-4 py-2 bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-mono font-bold text-xs rounded-lg transition-colors flex items-center gap-1.5"
                >
                  {isGenerating ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Plus className="w-3.5 h-3.5" />}
                  <span>CREATE</span>
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
