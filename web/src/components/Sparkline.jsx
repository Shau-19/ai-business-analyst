// Tiny trend line for KPI tiles: area fill, faint baseline, emphasised endpoint.
export default function Sparkline({ values = [], width = 120, height = 36, color = 'var(--series-1)' }) {
  const pts = values.filter((v) => v !== null && v !== undefined);
  if (pts.length < 2) return null;
  const min = Math.min(...pts), max = Math.max(...pts);
  const pad = 3, span = max - min || 1;
  const x = (i) => pad + (i * (width - pad * 2)) / (values.length - 1);
  const y = (v) => pad + (height - pad * 2) * (1 - (v - min) / span);
  const coords = values.map((v, i) => (v === null || v === undefined ? null : [x(i), y(v)])).filter(Boolean);
  const line = coords.map(([cx, cy], i) => `${i ? 'L' : 'M'}${cx.toFixed(1)},${cy.toFixed(1)}`).join('');
  const area = `${line}L${coords.at(-1)[0].toFixed(1)},${height}L${coords[0][0].toFixed(1)},${height}Z`;
  const [ex, ey] = coords.at(-1);
  return (
    <svg className="spark" viewBox={`0 0 ${width} ${height}`} width={width} height={height} aria-hidden="true">
      <path d={area} fill={color} opacity="0.12" />
      <path d={line} fill="none" stroke={color} strokeWidth="2" strokeLinejoin="round" strokeLinecap="round" />
      <circle cx={ex} cy={ey} r="3" fill={color} stroke="var(--surface)" strokeWidth="1.5" />
    </svg>
  );
}
