import { useState } from 'react';
import { Sheet, FileText, Database } from 'lucide-react';
import { compact } from '../lib/format';

const ROLE = {
  numeric: { label: 'Measure', cls: 'chip-forest' },
  categorical: { label: 'Category', cls: 'chip-coral' },
  date: { label: 'Date', cls: '' },
  month: { label: 'Month', cls: '' },
  id: { label: 'ID', cls: '' },
  text: { label: 'Text', cls: '' },
};

function stat(c) {
  if (c.role === 'numeric') return `${compact(c.min)} – ${compact(c.max)} · avg ${compact(c.mean)}`;
  if (c.role === 'date') return c.min && c.max ? `${c.min} → ${c.max}` : '';
  if (c.role === 'categorical') return c.top?.slice(0, 3).map((t) => `${t.value} (${t.count})`).join(', ');
  return '';
}

export default function DataView({ profile, onUpload }) {
  const [i, setI] = useState(0);
  const datasets = (profile?.datasets || []).filter((d) => !d.error);
  const docs = profile?.documents || [];

  if (!datasets.length) {
    return (
      <div className="fc-empty">
        <Database aria-hidden="true" />
        <h2 className="display">No tables yet.</h2>
        <p>Upload a CSV or Excel file to see its columns, types and a preview of the rows.</p>
        <label className="btn btn-primary">
          <input type="file" hidden multiple accept=".csv,.xlsx,.xls" onChange={(e) => { onUpload(e.target.files); e.target.value = ''; }} />
          Choose a file
        </label>
      </div>
    );
  }

  const ds = datasets[Math.min(i, datasets.length - 1)];
  const cols = ds.sample?.length ? Object.keys(ds.sample[0]) : ds.columns.map((c) => c.name);

  return (
    <div className="data">
      <div className="ov-switch">
        {datasets.map((d, k) => (
          <button key={d.table} className={`chip ${k === i ? 'chip-forest' : ''}`} onClick={() => setI(k)} aria-pressed={k === i}>
            <Sheet aria-hidden="true" /> {d.source || d.table}
          </button>
        ))}
        {docs.map((d) => <span key={d} className="chip"><FileText aria-hidden="true" /> {d}</span>)}
      </div>

      <section className="tile">
        <div className="tile-head">
          <div>
            <h3 className="tile-title">Columns</h3>
            <span className="tile-sub num">
              {ds.columns.length} columns · {ds.rows.toLocaleString('en-US')} rows · table <span className="mono">{ds.table}</span>
              {ds.quality?.duplicates_removed ? ` · ${ds.quality.duplicates_removed} duplicate rows removed on import` : ''}
              {ds.quality?.identical_rows_kept ? ` · ${ds.quality.identical_rows_kept} identical rows kept` : ''}
            </span>
          </div>
        </div>
        <div className="table-wrap">
          <table className="tbl">
            <thead><tr><th>Column</th><th>Type</th><th className="r">Distinct</th><th className="r">Empty</th><th>Summary</th><th>Cleaning</th></tr></thead>
            <tbody>
              {ds.columns.map((c) => {
                const q = ds.quality?.columns?.find((x) => x.column === c.name);
                return (
                <tr key={c.name}>
                  <td><b>{c.label}</b>{c.label !== c.name && <> <span className="mono tbl-dim">{c.name}</span></>}</td>
                  <td><span className={`chip ${ROLE[c.role]?.cls || ''}`}>{ROLE[c.role]?.label || c.role}</span></td>
                  <td className="r num">{c.distinct.toLocaleString('en-US')}</td>
                  <td className="r num">{c.nulls ? c.nulls.toLocaleString('en-US') : '–'}</td>
                  <td className="tbl-dim num">{stat(c)}</td>
                  <td className="tbl-clean">
                    {q?.changes?.map((t) => <span key={t}>{t}</span>)}
                    {q?.warnings?.map((t) => <span key={t} className="is-warn">{t}</span>)}
                    {!q?.changes?.length && !q?.warnings?.length && <span className="tbl-dim">–</span>}
                  </td>
                </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>

      <section className="tile">
        <div className="tile-head">
          <div><h3 className="tile-title">Preview</h3><span className="tile-sub">First {ds.sample.length} of {ds.rows.toLocaleString('en-US')} rows</span></div>
        </div>
        <div className="table-wrap">
          <table className="tbl num">
            <thead><tr>{cols.map((c) => <th key={c} className={typeof ds.sample[0]?.[c] === 'number' ? 'r' : ''}>{c}</th>)}</tr></thead>
            <tbody>
              {ds.sample.map((row, r) => (
                <tr key={r}>{cols.map((c) => (
                  <td key={c} className={typeof row[c] === 'number' ? 'r' : ''}>
                    {row[c] === null ? '–' : typeof row[c] === 'number' ? row[c].toLocaleString('en-US', { maximumFractionDigits: 4 }) : String(row[c])}
                  </td>
                ))}</tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
