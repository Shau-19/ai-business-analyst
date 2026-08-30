import React from 'react';
import { MessageSquare, LayoutDashboard, LineChart, Database, Sparkles } from 'lucide-react';

export default function Header({ activeTab, onTabChange, activeSessionId, documentsCount }) {
  const shortSession = activeSessionId ? `${activeSessionId.substring(0, 8)}...` : 'NO SESSION';

  return (
    <header className="h-14 bg-slate-900 border-b border-slate-800 flex items-center justify-between px-4 shrink-0">
      {/* Scope Navigation Tabs */}
      <div className="flex items-center gap-1">
        <button
          onClick={() => onTabChange('chat')}
          className={`flex items-center gap-2 px-4 py-2 font-mono text-xs font-semibold rounded-lg transition-all cursor-pointer ${
            activeTab === 'chat'
              ? 'bg-slate-800 text-emerald-400 border border-emerald-500/30'
              : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/40'
          }`}
        >
          <MessageSquare className="w-3.5 h-3.5" />
          <span>CHAT</span>
        </button>

        <button
          onClick={() => onTabChange('dashboard')}
          className={`flex items-center gap-2 px-4 py-2 font-mono text-xs font-semibold rounded-lg transition-all cursor-pointer ${
            activeTab === 'dashboard'
              ? 'bg-slate-800 text-cyan-400 border border-cyan-500/30'
              : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/40'
          }`}
        >
          <LayoutDashboard className="w-3.5 h-3.5" />
          <span>DASHBOARD SUITE</span>
          <span className="text-[9px] px-1.5 py-0.5 rounded bg-cyan-950 text-cyan-400 border border-cyan-800">MULTI</span>
        </button>

        <button
          onClick={() => onTabChange('forecast')}
          className={`flex items-center gap-2 px-4 py-2 font-mono text-xs font-semibold rounded-lg transition-all cursor-pointer ${
            activeTab === 'forecast'
              ? 'bg-slate-800 text-purple-400 border border-purple-500/30'
              : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/40'
          }`}
        >
          <LineChart className="w-3.5 h-3.5" />
          <span>FORECAST STUDIO</span>
        </button>
      </div>

      {/* Session Metadata & Badges */}
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-2 px-3 py-1 bg-slate-950/60 border border-slate-800 rounded-md font-mono text-xs text-slate-400">
          <span className="text-[10px] text-slate-500">SESSION:</span>
          <span className="text-emerald-400 font-semibold">{shortSession}</span>
        </div>

        <div className="flex items-center gap-1.5 px-3 py-1 bg-slate-950/60 border border-slate-800 rounded-md font-mono text-xs text-slate-400">
          <Database className="w-3.5 h-3.5 text-cyan-400" />
          <span>{documentsCount} Docs</span>
        </div>
      </div>
    </header>
  );
}
