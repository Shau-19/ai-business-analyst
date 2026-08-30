import React, { useState } from 'react';
import { Plus, MessageSquare, FileText, Upload, CheckCircle2, Loader2, Sparkles, FolderOpen } from 'lucide-react';

export default function Sidebar({
  conversations,
  activeSessionId,
  onSelectSession,
  onNewSession,
  onFileUpload,
  documents,
  isUploading,
}) {
  const [isDragging, setIsDragging] = useState(false);

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      onFileUpload(e.dataTransfer.files);
    }
  };

  const getFileIcon = (name) => {
    const ext = name.split('.').pop().toLowerCase();
    return ['csv', 'xlsx'].includes(ext) ? '📊' : ['pdf', 'docx'].includes(ext) ? '📄' : '📝';
  };

  return (
    <aside className="w-72 bg-slate-900 border-r border-slate-800 flex flex-col h-full shrink-0">
      {/* Brand Header */}
      <div className="p-4 border-b border-slate-800 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-emerald-500/10 border border-emerald-500/30 flex items-center justify-center">
            <Sparkles className="w-4 h-4 text-emerald-400" />
          </div>
          <div>
            <h1 className="font-mono font-bold text-sm text-slate-100 tracking-wider">ANALYST</h1>
            <p className="text-[10px] font-mono text-emerald-400">BUSINESS INTELLIGENCE</p>
          </div>
        </div>
      </div>

      {/* New Session Action */}
      <div className="p-3">
        <button
          onClick={onNewSession}
          className="w-full py-2.5 px-4 bg-emerald-500 hover:bg-emerald-400 text-slate-950 font-mono font-semibold text-xs rounded-lg flex items-center justify-center gap-2 transition-colors shadow-lg shadow-emerald-500/10 cursor-pointer"
        >
          <Plus className="w-4 h-4" />
          <span>NEW SESSION</span>
        </button>
      </div>

      {/* Session List */}
      <div className="flex-1 overflow-y-auto px-3 py-2 space-y-1">
        <div className="px-2 py-1.5 text-[10px] font-mono font-semibold text-slate-500 uppercase tracking-wider">
          Sessions
        </div>
        {conversations.length === 0 ? (
          <div className="p-4 text-center text-xs font-mono text-slate-600">No sessions yet</div>
        ) : (
          conversations.map((conv) => {
            const isActive = conv.conversation_id === activeSessionId;
            return (
              <button
                key={conv.conversation_id}
                onClick={() => onSelectSession(conv.conversation_id)}
                className={`w-full text-left p-2.5 rounded-lg flex flex-col gap-1 transition-all border cursor-pointer ${
                  isActive
                    ? 'bg-slate-800/90 border-emerald-500/40 text-slate-100 shadow-sm'
                    : 'bg-transparent border-transparent hover:bg-slate-800/40 text-slate-400 hover:text-slate-200'
                }`}
              >
                <div className="flex items-center gap-2">
                  <MessageSquare className={`w-3.5 h-3.5 shrink-0 ${isActive ? 'text-emerald-400' : 'text-slate-500'}`} />
                  <span className="text-xs font-medium truncate">{conv.title || 'Untitled Session'}</span>
                </div>
                <div className="flex items-center justify-between text-[10px] font-mono text-slate-500 pl-5">
                  <span>{conv.message_count || 0} msgs</span>
                  <span>{conv.updated_at ? new Date(conv.updated_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : ''}</span>
                </div>
              </button>
            );
          })
        )}
      </div>

      {/* File Attachment & Session Data Area */}
      <div className="p-3 border-t border-slate-800 bg-slate-900/60">
        <div className="flex items-center justify-between mb-2">
          <span className="text-[10px] font-mono font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
            <FolderOpen className="w-3 h-3 text-cyan-400" /> Session Data
          </span>
          <span className="text-[10px] font-mono text-slate-500">{documents.length} files</span>
        </div>

        {/* Drop Zone */}
        <div
          onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={handleDrop}
          className={`border border-dashed rounded-lg p-3 text-center transition-colors ${
            isDragging ? 'border-emerald-400 bg-emerald-500/10' : 'border-slate-800 bg-slate-950/40'
          }`}
        >
          <input
            type="file"
            id="sidebarFileInput"
            multiple
            accept=".csv,.xlsx,.pdf,.docx,.txt"
            className="hidden"
            onChange={(e) => e.target.files?.length && onFileUpload(e.target.files)}
          />
          <label htmlFor="sidebarFileInput" className="cursor-pointer flex flex-col items-center gap-1">
            {isUploading ? (
              <Loader2 className="w-5 h-5 text-emerald-400 animate-spin" />
            ) : (
              <Upload className="w-5 h-5 text-slate-500 hover:text-emerald-400 transition-colors" />
            )}
            <span className="text-[11px] font-mono text-slate-300">
              {isUploading ? 'Indexing files...' : 'Upload CSV / PDF'}
            </span>
            <span className="text-[9px] text-slate-500">Drag & drop or click to browse</span>
          </label>
        </div>

        {/* Attached Files List */}
        {documents.length > 0 && (
          <div className="mt-2 space-y-1 max-h-32 overflow-y-auto">
            {documents.map((doc, idx) => (
              <div key={idx} className="flex items-center justify-between p-1.5 bg-slate-950/60 border border-slate-800/80 rounded text-xs font-mono text-slate-300">
                <div className="flex items-center gap-1.5 truncate">
                  <span>{getFileIcon(doc)}</span>
                  <span className="truncate text-[11px]">{doc}</span>
                </div>
                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
              </div>
            ))}
          </div>
        )}
      </div>
    </aside>
  );
}
