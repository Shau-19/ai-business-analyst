import { useEffect, useMemo, useRef, useState } from 'react';
import { TrendingUp, LoaderCircle, Pin, CircleAlert, TriangleAlert } from 'lucide-react';
import Chart from '../components/Chart';
import { api } from '../lib/api';
import { renderMarkdown } from '../lib/markdown';
import { formatFor, signedPct } from '../lib/format';

export default function Forecast({ activeId, profile, onPin }) {
  const options = useMemo(() => {
    const out = [];
    for (const d of profile?.datasets || []) {
      const hasTime = d.columns?.some((c) => c.role === 'date' || c.role === 'month');
      for (const c of d.columns || []) if (c.role === 'numeric') out.push({ value: c.name, label: c.label, source: d.source || d.table, hasTime });
    }
    return out;
  }, [profile]);
  const [col, setCol] = useState(() => options.find((o) => o.hasTime)?.value || options[0]?.value || '');
  // Profile can arrive after this view mounts: pick a sensible default then.
  useEffect(() => {
    if (!options.some((o) => o.value === col)) setCol(options.find((o) => o.hasTime)?.value || options[0]?.value || '');
  }, [options, col]);
  const [periods, setPeriods] = useState(6);
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const sessionRef = useRef(activeId);
  sessionRef.current = activeId;

  const run = async (e) => {
    e.preventDefault();
    if (!col || !activeId) return;
    const cid = activeId;
    setRunning(true); setError(''); setResult(null);
    try {
      const res = await api.silentQuery(cid, `forecast ${col} for the next ${periods} periods`, { valueCol: col });
      if (sessionRef.current !== cid) return;
      if (!res.success) setError(res.error || 'The forecast couldn’t be produced.');
      else setResult(res);
    } catch (err) {
      if (sessionRef.current === cid) setError(err.message);
    } finally {
      if (sessionRef.current === cid) setRunning(false);
    }
  };

  if (!options.length) {
    return (
      <div className="fc-empty">
        <TrendingUp aria-hidden="true" />
        <h2 className="display">Forecasts need a numeric column over time.</h2>
        <p>Upload a CSV or Excel file with a date or month column and at least one number, like revenue or units, to project it forward.</p>
      </div>
    );
  }

  const opt = options.find((o) => o.value === col);
  const fc = result?.data?.filter((r) => r.forecast !== null && r.forecast !== undefined) || [];
  const endFc = fc.at(-1)?.forecast;
  const change = result?.last_actual && endFc !== undefined ? ((endFc - result.last_actual) / Math.abs(result.last_actual)) * 100 : null;

  return (
    <div className="fc">
      <form className="tile fc-controls" onSubmit={run}>
        <div className="fc-field">
          <label htmlFor="fc-col">Metric</label>
          <select id="fc-col" className="input" value={col} onChange={(e) => setCol(e.target.value)}>
            {options.map((o) => <option key={`${o.source}-${o.value}`} value={o.value}>{o.label}{options.some((x) => x.source !== o.source) ? ` · ${o.source}` : ''}</option>)}
          </select>
        </div>
        <div className="fc-field">
          <span id="fc-h-label">Horizon</span>
          <div className="seg" role="group" aria-labelledby="fc-h-label">
            {[3, 6, 12].map((n) => (
              <button type="button" key={n} aria-pressed={periods === n} className={periods === n ? 'is-on' : ''} onClick={() => setPeriods(n)}>{n} periods</button>
            ))}
          </div>
        </div>
        <button className="btn btn-primary btn-lg fc-run" type="submit" disabled={running}>
          {running ? <LoaderCircle className="spin" /> : <TrendingUp />} {running ? 'Fitting model…' : 'Run forecast'}
        </button>
        {opt && !opt.hasTime && <p className="fc-warn"><TriangleAlert aria-hidden="true" /> {opt.source} has no date or month column, so this metric can’t be forecast.</p>}
      </form>

      {error && <div className="banner banner-bad"><CircleAlert aria-hidden="true" /> {error}</div>}

      {running && <div className="tile"><div className="skeleton" style={{ height: 320 }} /></div>}

      {result && (
        <div className="fc-grid">
          <div className="tile kpi"><span className="kpi-label">Last actual</span><span className="kpi-value display num">{formatFor(col, result.last_actual)}</span><span className="kpi-range">{result.data_points} data points</span></div>
          <div className="tile kpi"><span className="kpi-label">Next period</span><span className="kpi-value display num">{formatFor(col, result.next_forecast)}</span><span className="kpi-range">Model estimate</span></div>
          <div className="tile kpi"><span className="kpi-label">End of horizon</span><span className="kpi-value display num">{formatFor(col, endFc)}</span>
            {change !== null && <span className={`chip ${change >= 0 ? 'chip-good' : 'chip-bad'} num`}>{signedPct(change)} vs last actual</span>}
          </div>
          <div className="tile kpi"><span className="kpi-label">Anomalies</span><span className="kpi-value display num">{result.anomalies?.length || 0}</span><span className="kpi-range">Outside the 95% range</span></div>

          <div className="tile fc-chart">
            <div className="tile-head">
              <div>
                <h3 className="tile-title">{result.plot?.title}</h3>
                <span className="tile-sub">Shaded band: 95% confidence range{result.date_note ? `. ${result.date_note}` : ''}</span>
              </div>
              <button className="btn btn-sm btn-ghost" onClick={() => onPin(result.plot, `forecast ${col} for the next ${periods} periods`)}><Pin /> Pin</button>
            </div>
            <Chart spec={result.plot} height={320} />
          </div>

          <div className="tile fc-narrative">
            <span className="eyebrow">What it means</span>
            <div className="prose" dangerouslySetInnerHTML={{ __html: renderMarkdown(result.explanation) }} />
          </div>

          {result.anomalies?.length > 0 && (
            <div className="tile fc-anoms">
              <span className="eyebrow">Anomalies</span>
              <div className="table-wrap">
                <table className="tbl num">
                  <thead><tr><th>Date</th><th>Actual</th><th>Expected</th><th>Range</th></tr></thead>
                  <tbody>{result.anomalies.map((a) => (
                    <tr key={a.ds}><td>{a.ds}</td><td>{formatFor(col, a.y)}</td><td>{formatFor(col, a.yhat)}</td><td>{formatFor(col, a.lower)} – {formatFor(col, a.upper)}</td></tr>
                  ))}</tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
