import React, { useState, useEffect } from 'react';
import { LineChart, Play, AlertTriangle, Sparkles, Loader2 } from 'lucide-react';
import ChartCanvas from './ChartCanvas';
import { api } from '../api';
import { marked } from 'marked';

export default function ForecastView({ activeSessionId }) {
  const [columns, setColumns] = useState([]);
  const [selectedCol, setSelectedCol] = useState('');
  const [periods, setPeriods] = useState(6);
  const [isLoading, setIsLoading] = useState(false);
  const [forecastResult, setForecastResult] = useState(null);

  useEffect(() => {
    if (!activeSessionId) return;
    fetchNumericColumns();
  }, [activeSessionId]);

  const fetchNumericColumns = async () => {
    try {
      const res = await api.getNumericColumns(activeSessionId);
      if (res.success && res.columns) {
        setColumns(res.columns);
        if (res.columns.length > 0) setSelectedCol(res.columns[0]);
      }
    } catch (e) {
      console.error('Failed to fetch numeric columns', e);
    }
  };

  const handleRunForecast = async () => {
    if (!activeSessionId) return;
    setIsLoading(true);
    const question = selectedCol
      ? `forecast ${selectedCol} for next ${periods} periods`
      : `forecast the main metric for next ${periods} periods`;

    try {
      const res = await api.runQuery(activeSessionId, question, selectedCol);
      if (res.success) {
        setForecastResult(res);
      }
    } catch (e) {
      console.error('Forecast failed', e);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex-1 flex flex-col h-full bg-slate-950 overflow-hidden">
      {/* Top Controls Bar */}
      <div className="p-4 bg-slate-900 border-b border-slate-800 flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <LineChart className="w-5 h-5 text-purple-400" />
            <span className="font-mono text-xs font-bold text-purple-400">PROPHET ENGINE</span>
          </div>

          {/* Column Selector */}
          <div className="flex items-center gap-2">
            <label className="text-xs font-mono text-slate-400">COLUMN:</label>
            <select
              value={selectedCol}
              onChange={(e) => setSelectedCol(e.target.value)}
              className="bg-slate-950 border border-slate-700 rounded-lg px-3 py-1.5 font-mono text-xs text-slate-100 focus:outline-none focus:border-purple-500"
            >
              {columns.length === 0 ? (
                <option value="">No numeric columns found</option>
              ) : (
                columns.map((c) => (
                  <option key={c} value={c}>
                    {c}
                  </option>
                ))
              )}
            </select>
          </div>

          {/* Periods Selector */}
          <div className="flex items-center gap-2">
            <label className="text-xs font-mono text-slate-400">HORIZON:</label>
            <div className="flex gap-1">
              {[3, 6, 12, 24].map((p) => (
                <button
                  key={p}
                  onClick={() => setPeriods(p)}
                  className={`px-2.5 py-1 rounded text-xs font-mono transition-colors border cursor-pointer ${
                    periods === p
                      ? 'bg-purple-950 text-purple-400 border-purple-500/50'
                      : 'bg-slate-950 text-slate-400 border-slate-800 hover:text-slate-200'
                  }`}
                >
                  {p}m
                </button>
              ))}
            </div>
          </div>
        </div>

        <button
          onClick={handleRunForecast}
          disabled={isLoading || !activeSessionId}
          className="px-5 py-2 bg-purple-500 hover:bg-purple-400 disabled:opacity-40 text-slate-950 font-mono font-bold text-xs rounded-lg flex items-center gap-2 transition-colors cursor-pointer shadow-lg shadow-purple-500/10"
        >
          {isLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
          <span>RUN FORECAST</span>
        </button>
      </div>

      {/* Main Studio Area */}
      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {!forecastResult ? (
          <div className="h-full flex flex-col items-center justify-center text-slate-500 gap-3">
            <LineChart className="w-12 h-12 text-slate-700" />
            <p className="font-mono text-sm">Select a numeric column above and click RUN FORECAST</p>
          </div>
        ) : (
          <div className="space-y-6 max-w-5xl mx-auto">
            {/* Forecast Visual Canvas */}
            {forecastResult.plot && (
              <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">
                <h3 className="font-mono text-sm font-bold text-purple-400 mb-4 flex items-center gap-2">
                  <Sparkles className="w-4 h-4" />
                  HISTORICAL VS PREDICTED PROJECTION
                </h3>
                <ChartCanvas plotSpec={forecastResult.plot} />
              </div>
            )}

            {/* Insight Text */}
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">
              <h3 className="font-mono text-xs font-bold text-slate-400 uppercase tracking-wider mb-3">
                Forecast Analysis & Insights
              </h3>
              <div
                className="prose prose-invert prose-sm max-w-none font-sans"
                dangerouslySetInnerHTML={{ __html: marked.parse(forecastResult.explanation || '') }}
              />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
