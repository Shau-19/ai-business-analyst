import { useEffect } from 'react';
import { CircleCheck, CircleAlert, TriangleAlert, Info, X } from 'lucide-react';

const ICON = { good: CircleCheck, error: CircleAlert, warn: TriangleAlert, info: Info };

export default function Toast({ toast, onClose }) {
  useEffect(() => {
    const t = setTimeout(onClose, toast.kind === 'error' ? 9000 : toast.action ? 6000 : 3500);
    return () => clearTimeout(t);
  }, [toast, onClose]);
  const Icon = ICON[toast.kind] || Info;
  return (
    <div className={`toast toast-${toast.kind}`} role={toast.kind === 'error' ? 'alert' : 'status'}>
      <Icon className="toast-icon" aria-hidden="true" />
      <span className="toast-text">{toast.text}</span>
      {toast.action && (
        <button className="btn btn-sm btn-ghost" onClick={() => { toast.action.run(); onClose(); }}>{toast.action.label}</button>
      )}
      <button className="icon-btn" onClick={onClose} aria-label="Dismiss"><X /></button>
    </div>
  );
}
