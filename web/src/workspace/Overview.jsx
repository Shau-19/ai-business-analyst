import { useRef, useState } from 'react';
import { Upload, Sparkles, Pin, MessageSquareText, ArrowRight, FileText, Sheet, ArrowUpRight, ArrowDownRight, LoaderCircle, CircleAlert, ShieldCheck, TriangleAlert } from 'lucide-react';
import Chart from '../components/Chart';
import Sparkline from '../components/Sparkline';
import { formatFor, signedPct } from '../lib/format';

const ROLE_LABEL = { numeric: 'Measures', categorical: 'Categories', date: 'Dates', month: 'Months', id: 'IDs', text: 'Text' };

function EmptyState({ onUpload, onSample }) {
  const input = useRef(null);
  const [drag, setDrag] = useState(false);
  return (
    <div className="ov-empty">
      <div
        className={`tile ov-drop ${drag ? 'is-drag' : ''}`}
        onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => { e.preventDefault(); setDrag(false); onUpload(e.dataTransfer.files); }}
      >
        <span className="eyebrow">Start here</span>
        <h2 className="display">Drop a spreadsheet to grow your first answers.</h2>
        <p>CSV or Excel for the numbers; add PDFs, Word or text files for the context behind them. You’ll get KPIs and charts straight away, no questions needed.</p>
        <input ref={input} type="file" multiple hidden accept=".csv,.xlsx,.xls,.pdf,.docx,.txt"
          onChange={(e) => { onUpload(e.target.files); e.target.value = ''; }} />
        <div className="ov-drop-actions">
          <button className="btn btn-primary btn-lg" onClick={() => input.current?.click()}><Upload /> Choose files</button>
          <span className="ov-drop-hint">or drag them anywhere on this card</span>
        </div>
      </div>
      <div className="tile tile-feature ov-sample">
        <Sparkles className="ov-sample-icon" aria-hidden="true" />
        <h2 className="display">No file handy?</h2>
        <p>Load a six-month sales report and the strategy memo that explains it, then ask why revenue dipped in March.</p>
        <ul className="ov-sample-files">
          <li><Sheet aria-hidden="true" /> Sales_Report_H1_2025.csv</li>
          <li><FileText aria-hidden="true" /> Strategic_Planning_Jan2025.docx</li>
        </ul>
        <button className="btn btn-coral" onClick={onSample}>Load sample data <ArrowRight /></button>
      </div>
      <div className="ov-promise">
        <div className="tile"><b>Instant overview</b><p>Totals, averages and trends, computed directly from your file.</p></div>
        <div className="tile"><b>Plain-language questions</b><p>Answers show the SQL and the documents they came from.</p></div>
        <div className="tile"><b>Dashboards and forecasts</b><p>Pin the charts you need and project any metric forward.</p></div>
      </div>
    </div>
  );
}

function Loading({ files }) {
  return (
    <div className="ov-grid" aria-busy="true">
      <div className="tile ov-status span-6">
        <LoaderCircle className="spin" aria-hidden="true" />
        <div>
          <b>Reading {files?.length ? files.join(', ') : 'your files'}…</b>
          <p>Importing tables and indexing documents. Larger files take a minute.</p>
        </div>
      </div>
      {[0, 1, 2].map((i) => <div key={i} className="tile span-2"><div className="skeleton" style={{ height: 14, width: '40%' }} /><div className="skeleton" style={{ height: 34, width: '60%', marginTop: 12 }} /><div className="skeleton" style={{ height: 30, marginTop: 14 }} /></div>)}
      <div className="tile span-4"><div className="skeleton" style={{ height: 260 }} /></div>
      <div className="tile span-2"><div className="skeleton" style={{ height: 260 }} /></div>
    </div>
  );
}

// For these metrics a fall is the good outcome (churn down = green).
const LOWER_IS_BETTER = /(churn|cost|expense|spend|defect|return|complaint|attrition|error|latency|bounce|refund|delay|downtime)/i;

// Data-cleaning report for one dataset: what was fixed on import, what to watch.
function DataHealth({ quality }) {
  if (!quality) return null;
  const fixed = (quality.columns || []).filter((c) => c.changes?.length);
  const warnings = (quality.columns || []).flatMap((c) => (c.warnings || []).map((w) => ({ column: c.column, text: w })));
  if (!fixed.length && !warnings.length && !quality.duplicates_removed && !quality.identical_rows_kept) return null;
  return (
    <div className="tile health span-6">
      <div className="tile-head">
        <div>
          <h3 className="tile-title"><ShieldCheck aria-hidden="true" /> Data health</h3>
          <span className="tile-sub num">
            Cleaned on import: {fixed.length} column{fixed.length === 1 ? '' : 's'} fixed
            {quality.duplicates_removed ? ` · ${quality.duplicates_removed} duplicate rows removed (${quality.rows_in.toLocaleString('en-US')} → ${quality.rows_out.toLocaleString('en-US')})` : ''}
            {quality.identical_rows_kept ? ` · ${quality.identical_rows_kept} identical rows kept (no ID column to confirm they’re duplicates)` : ''}
          </span>
        </div>
      </div>
      <ul className="health-list">
        {fixed.map((c) => (
          <li key={c.column}><b>{c.column}</b><span>{c.changes.join('; ')}</span></li>
        ))}
        {warnings.map((w) => (
          <li key={w.column + w.text} className="is-warn"><b><TriangleAlert aria-hidden="true" /> {w.column}</b><span>{w.text}</span></li>
        ))}
      </ul>
    </div>
  );
}

function Kpi({ kpi, warning }) {
  const up = kpi.change_pct > 0, down = kpi.change_pct < 0;
  const good = LOWER_IS_BETTER.test(kpi.column) ? down : up;
  const bad = LOWER_IS_BETTER.test(kpi.column) ? up : down;
  return (
    <div className="tile kpi span-2">
      <span className="kpi-label">{kpi.agg} {kpi.label.toLowerCase()}</span>
      <span className="kpi-value display num">{formatFor(kpi.column, kpi.value)}</span>
      {warning && <span className="chip chip-warn kpi-warn" title={warning}><TriangleAlert aria-hidden="true" /> Includes extreme values</span>}
      <div className="kpi-foot">
        {kpi.change_pct !== undefined && kpi.change_pct !== null ? (
          <span className="kpi-change num">
            <span className={`chip ${good ? 'chip-good' : bad ? 'chip-bad' : ''}`}>
              {up ? <ArrowUpRight aria-hidden="true" /> : down ? <ArrowDownRight aria-hidden="true" /> : null}
              {signedPct(kpi.change_pct)}
            </span>
            <span className="kpi-range" title={kpi.trend_note ? `Trend ${kpi.trend_note.toLowerCase()}` : undefined}>
              {kpi.from_label} → {kpi.to_label}{kpi.trend_note ? ' · typical values' : ''}
            </span>
          </span>
        ) : <span className="kpi-range">Across all rows</span>}
        {kpi.spark && <Sparkline values={kpi.spark} width={110} height={34} />}
      </div>
    </div>
  );
}

function ChartTile({ chart, span, onPin, onAsk }) {
  return (
    <div className={`tile chart-tile ${span}`}>
      <div className="tile-head">
        <div>
          <h3 className="tile-title">{chart.plot.title}</h3>
          <span className="tile-sub">{chart.plot.y_label}{chart.note ? ` · ${chart.note}` : ''}</span>
        </div>
        <div className="tile-actions">
          <button className="icon-btn" onClick={() => onPin(chart.plot, chart.question)} aria-label="Pin chart to dashboard" title="Pin to dashboard"><Pin /></button>
          <button className="btn btn-sm btn-ghost" onClick={() => onAsk(chart.question)}><MessageSquareText /> Ask</button>
        </div>
      </div>
      <div className="chart-fill"><Chart spec={chart.plot} height="100%" /></div>
    </div>
  );
}

export default function Overview({ profile, processing, onUpload, onSample, onAsk, onPin, setTab }) {
  const [dsIndex, setDsIndex] = useState(0);
  const [draft, setDraft] = useState('');

  if (processing?.status === 'processing') return <Loading files={processing.files} />;
  if (!profile) return <Loading />;
  const datasets = (profile.datasets || []).filter((d) => !d.error);
  if (!datasets.length && !(profile.documents || []).length) {
    return (
      <>
        {processing?.status === 'error' && (
          <div className="banner banner-bad"><CircleAlert aria-hidden="true" /> {processing.error}</div>
        )}
        <EmptyState onUpload={onUpload} onSample={onSample} />
      </>
    );
  }

  const ds = datasets[Math.min(dsIndex, datasets.length - 1)];
  const docs = profile.documents || [];
  const questions = ds?.questions?.length ? ds.questions : ['Summarize the key points in my documents'];
  const roles = (ds?.columns || []).reduce((acc, c) => { (acc[c.role] ||= []).push(c.label); return acc; }, {});
  const trend = ds?.charts?.find((c) => c.kind === 'trend');
  const breakdown = ds?.charts?.find((c) => c.kind === 'breakdown');
  const extraChart = !trend ? breakdown : null;

  return (
    <div className="ov">
      {datasets.length > 1 && (
        <div className="ov-switch" role="tablist" aria-label="Datasets">
          {datasets.map((d, i) => (
            <button key={d.table} role="tab" aria-selected={i === dsIndex} className={`chip ${i === dsIndex ? 'chip-forest' : ''}`} onClick={() => setDsIndex(i)}>
              <Sheet aria-hidden="true" /> {d.source || d.table}
            </button>
          ))}
        </div>
      )}

      <div className="ov-grid">
        {ds?.kpis?.map((k) => (
          <Kpi key={k.column} kpi={k}
            warning={ds.quality?.columns?.find((c) => c.column === k.column)?.warnings?.[0]} />
        ))}
        {ds && ds.kpis?.length < 3 && (
          <div className={`tile kpi ${ds.kpis.length === 2 ? 'span-2' : ds.kpis.length === 1 ? 'span-4' : 'span-6'}`}>
            <span className="kpi-label">Rows in {ds.source || 'this dataset'}</span>
            <span className="kpi-value display num">{ds.rows.toLocaleString('en-US')}</span>
            <span className="kpi-range">{ds.columns.length} columns</span>
          </div>
        )}

        {(trend || extraChart) && <ChartTile chart={trend || extraChart} span="span-4" onPin={onPin} onAsk={onAsk} />}

        <form
          className={`tile tile-feature ov-ask ${trend || extraChart ? 'span-2' : 'span-6'}`}
          onSubmit={(e) => { e.preventDefault(); if (draft.trim()) onAsk(draft); }}
        >
          <span className="eyebrow ov-ask-eyebrow">Ask your data</span>
          <h3 className="display">What do you want to know?</h3>
          <ul className="ov-questions">
            {questions.slice(0, 4).map((q) => (
              <li key={q}><button type="button" onClick={() => onAsk(q)}>{q} <ArrowRight aria-hidden="true" /></button></li>
            ))}
          </ul>
          <label className="sr-only" htmlFor="ov-ask-input">Your question</label>
          <div className="ov-ask-input">
            <input id="ov-ask-input" value={draft} onChange={(e) => setDraft(e.target.value)} placeholder="Type a question…" />
            <button type="submit" aria-label="Ask" disabled={!draft.trim()}><ArrowRight /></button>
          </div>
        </form>

        {trend && breakdown && <ChartTile chart={breakdown} span="span-3" onPin={onPin} onAsk={onAsk} />}

        {ds && (
          <div className={`tile ov-facts ${trend && breakdown ? 'span-3' : 'span-6'}`}>
            <div className="tile-head">
              <div>
                <h3 className="tile-title">{ds.source || ds.table}</h3>
                <span className="tile-sub num">{ds.rows.toLocaleString('en-US')} rows · {ds.columns.length} columns</span>
              </div>
              <button className="btn btn-sm btn-ghost" onClick={() => setTab('data')}>Explore data <ArrowRight /></button>
            </div>
            <dl className="ov-roles">
              {Object.entries(roles).map(([role, cols]) => (
                <div key={role}>
                  <dt>{ROLE_LABEL[role] || role}</dt>
                  <dd>{cols.map((c) => <span key={c} className="chip">{c}</span>)}</dd>
                </div>
              ))}
            </dl>
            {docs.length > 0 && (
              <div className="ov-docs">
                <span className="eyebrow">Context documents</span>
                {docs.map((d) => <span key={d} className="chip chip-forest"><FileText aria-hidden="true" /> {d}</span>)}
              </div>
            )}
          </div>
        )}

        {ds && <DataHealth quality={ds.quality} />}

        {!ds && docs.length > 0 && (
          <div className="tile ov-facts span-6">
            <h3 className="tile-title">Documents ready to ask about</h3>
            <div className="ov-docs">{docs.map((d) => <span key={d} className="chip chip-forest"><FileText aria-hidden="true" /> {d}</span>)}</div>
            <p className="tile-sub">Add a CSV or Excel file to get KPIs, charts and forecasts.</p>
          </div>
        )}
      </div>
    </div>
  );
}
