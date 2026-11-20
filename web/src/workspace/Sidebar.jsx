import { useRef, useState } from 'react';
import { Plus, Trash2, Upload, LoaderCircle, Moon, Sun, ArrowLeft, X, Sheet } from 'lucide-react';
import Logo from '../components/Logo';
import { relativeTime } from '../lib/format';

export default function Sidebar({ conversations, activeId, onSelect, onNew, onDelete, onUpload, uploading, theme, onClose }) {
  const [dragging, setDragging] = useState(false);
  const [confirming, setConfirming] = useState(null);
  const input = useRef(null);

  const drop = (e) => {
    e.preventDefault();
    setDragging(false);
    if (e.dataTransfer.files?.length) onUpload(e.dataTransfer.files);
  };

  return (
    <aside className="sb" aria-label="Analyses">
      <div className="sb-head">
        <a href="#/" className="sb-brand" aria-label="Back to the Analyst home page"><Logo size={26} /></a>
        <button className="icon-btn sb-close" onClick={onClose} aria-label="Close menu"><X /></button>
      </div>

      <button className="btn btn-primary sb-new" onClick={onNew}><Plus /> New analysis</button>

      <div className="sb-label eyebrow">Analyses</div>
      <nav className="sb-list">
        {conversations.length === 0 && (
          <p className="sb-empty">Nothing here yet. Upload a file or try the sample to start your first analysis.</p>
        )}
        {conversations.map((c) => {
          const active = c.conversation_id === activeId;
          const asking = confirming === c.conversation_id;
          return (
            <div key={c.conversation_id} className={`sb-item ${active ? 'is-active' : ''}`}>
              <button className="sb-item-main" onClick={() => onSelect(c.conversation_id)} aria-current={active ? 'page' : undefined}>
                <span className="sb-item-title">{c.title && c.title !== 'New Chat' ? c.title : 'Untitled analysis'}</span>
                <span className="sb-item-meta num">
                  {c.documents > 0 && <span><Sheet aria-hidden="true" />{c.documents} file{c.documents === 1 ? '' : 's'}</span>}
                  <span>{relativeTime(c.updated_at)}</span>
                </span>
              </button>
              {asking ? (
                <span className="sb-confirm">
                  <button className="btn btn-sm sb-confirm-yes" onClick={() => { setConfirming(null); onDelete(c.conversation_id); }}>Delete</button>
                  <button className="btn btn-sm btn-quiet" onClick={() => setConfirming(null)}>Keep</button>
                </span>
              ) : (
                <button className="icon-btn sb-del" onClick={() => setConfirming(c.conversation_id)} aria-label={`Delete ${c.title || 'analysis'}`}>
                  <Trash2 />
                </button>
              )}
            </div>
          );
        })}
      </nav>

      <div
        className={`sb-drop ${dragging ? 'is-drag' : ''}`}
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={drop}
      >
        <input ref={input} type="file" multiple hidden accept=".csv,.xlsx,.xls,.pdf,.docx,.txt"
          onChange={(e) => { onUpload(e.target.files); e.target.value = ''; }} />
        <button className="sb-drop-btn" onClick={() => input.current?.click()} disabled={uploading}>
          {uploading ? <LoaderCircle className="spin" /> : <Upload />}
          <span>{uploading ? 'Reading your files…' : 'Add files'}</span>
          <small>Drop CSV, Excel, PDF or Word here</small>
        </button>
      </div>

      <div className="sb-foot">
        <a href="#/" className="btn btn-quiet"><ArrowLeft /> Home</a>
        <button className="icon-btn" onClick={theme.cycle} aria-label={theme.isDark ? 'Switch to light theme' : 'Switch to dark theme'}>
          {theme.isDark ? <Sun /> : <Moon />}
        </button>
      </div>
    </aside>
  );
}
