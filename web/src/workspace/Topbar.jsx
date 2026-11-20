import { useRef } from 'react';
import { Menu, Plus, LayoutDashboard, MessageSquareText, LayoutGrid, TrendingUp, Table2 } from 'lucide-react';

const TABS = [
  { id: 'overview', label: 'Overview', icon: LayoutDashboard },
  { id: 'ask', label: 'Ask', icon: MessageSquareText },
  { id: 'dashboards', label: 'Dashboards', icon: LayoutGrid },
  { id: 'forecast', label: 'Forecast', icon: TrendingUp },
  { id: 'data', label: 'Data', icon: Table2 },
];

export default function Topbar({ title, profile, tab, setTab, onUpload, onMenu }) {
  const input = useRef(null);
  const files = profile?.files || [];
  const name = title && title !== 'New Chat' ? title : files[0]?.replace(/\.[^.]+$/, '') || 'New analysis';

  return (
    <header className="tb">
      <div className="tb-row">
        <button className="icon-btn tb-menu" onClick={onMenu} aria-label="Open menu"><Menu /></button>
        <div className="tb-title">
          <h1 className="display">{name}</h1>
          {files.length > 0 && <span className="tb-files">{files.length} file{files.length === 1 ? '' : 's'} · {files.slice(0, 3).join(', ')}{files.length > 3 ? '…' : ''}</span>}
        </div>
        <input ref={input} type="file" multiple hidden accept=".csv,.xlsx,.xls,.pdf,.docx,.txt"
          onChange={(e) => { onUpload(e.target.files); e.target.value = ''; }} />
        <button className="btn btn-ghost tb-add" onClick={() => input.current?.click()}><Plus /> Add files</button>
      </div>
      <nav className="tabs" aria-label="Views">
        {TABS.map(({ id, label, icon: Icon }) => (
          <button key={id} className={`tab ${tab === id ? 'is-on' : ''}`} onClick={() => setTab(id)} aria-current={tab === id ? 'page' : undefined}>
            <Icon aria-hidden="true" /> {label}
          </button>
        ))}
      </nav>
    </header>
  );
}
