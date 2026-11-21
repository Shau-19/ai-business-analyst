const MONEY_HINT = /(revenue|sales|amount|price|cost|profit|spend|budget|salary|income|deal|value|usd|inr)/i;
const PCT_HINT = /(rate|pct|percent|churn|margin|ratio|discount)/i;

// Compact numbers for KPIs and axes: 3,210,000 → 3.21M
export function compact(v, digits = 2) {
  if (v === null || v === undefined || Number.isNaN(v)) return '–';
  const a = Math.abs(v);
  if (a >= 1e9) return (v / 1e9).toFixed(digits).replace(/\.?0+$/, '') + 'B';
  if (a >= 1e6) return (v / 1e6).toFixed(digits).replace(/\.?0+$/, '') + 'M';
  if (a >= 1e4) return (v / 1e3).toFixed(1).replace(/\.0$/, '') + 'K';
  if (Number.isInteger(v)) return v.toLocaleString('en-US');
  return v.toLocaleString('en-US', { maximumFractionDigits: 2 });
}

// Value formatted for its column: currency, percentage, or plain
export function formatFor(column, v) {
  if (v === null || v === undefined) return '–';
  if (PCT_HINT.test(column || '')) {
    // Cleaned percentage columns are fractions (0.09 = 9%); others are already percents (1.65)
    const pct = Math.abs(v) <= 1 && !/rate|churn/i.test(column) ? v * 100 : Number(v);
    return `${pct.toFixed(pct < 10 ? 2 : 1)}%`;
  }
  if (MONEY_HINT.test(column || '')) return `$${compact(v)}`;
  return compact(v);
}

export function signedPct(p) {
  if (p === null || p === undefined) return '';
  return `${p > 0 ? '+' : p < 0 ? '−' : ''}${Math.abs(p).toFixed(1)}%`;
}

export function relativeTime(iso) {
  if (!iso) return '';
  const t = new Date(iso.endsWith('Z') || iso.includes('+') ? iso : iso.replace(' ', 'T') + 'Z');
  const s = (Date.now() - t.getTime()) / 1000;
  if (s < 60) return 'just now';
  if (s < 3600) return `${Math.floor(s / 60)} min ago`;
  if (s < 86400) return `${Math.floor(s / 3600)} h ago`;
  if (s < 86400 * 7) return `${Math.floor(s / 86400)} d ago`;
  return t.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
}

export const ROUTE_LABEL = {
  sql: 'SQL',
  document: 'Documents',
  hybrid: 'SQL + documents',
  forecast: 'Forecast',
  analysis: 'Why analysis',
  general: 'General',
};

export function fileKind(name = '') {
  const ext = name.split('.').pop().toLowerCase();
  if (['csv', 'xlsx', 'xls'].includes(ext)) return 'data';
  return 'doc';
}
