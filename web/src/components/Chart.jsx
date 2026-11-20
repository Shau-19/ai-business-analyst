import { useEffect, useRef, useState } from 'react';
import {
  Chart as ChartJS, CategoryScale, LinearScale, BarElement, LineElement, PointElement,
  ArcElement, Tooltip, Legend, Filler, BarController, LineController, DoughnutController,
} from 'chart.js';
import { compact } from '../lib/format';

ChartJS.register(
  CategoryScale, LinearScale, BarElement, LineElement, PointElement, ArcElement,
  Tooltip, Legend, Filler, BarController, LineController, DoughnutController,
);

const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

// Re-render charts when the theme flips (toggle or OS setting).
export function useThemeVersion() {
  const [v, setV] = useState(0);
  useEffect(() => {
    const bump = () => setV((x) => x + 1);
    const mo = new MutationObserver(bump);
    mo.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
    const mq = window.matchMedia('(prefers-color-scheme: dark)');
    mq.addEventListener('change', bump);
    return () => { mo.disconnect(); mq.removeEventListener('change', bump); };
  }, []);
  return v;
}

/**
 * Normalise the three plot shapes the backend sends to { labels, series }:
 *  1. SQL / profile:  { x, datasets: [{label, data}] }
 *  2. older single:   { x, y, y_label }
 *  3. forecast:       { data: [{date, actual, forecast, lower, upper}], series, x_key }
 */
function normalize(spec) {
  if (!spec) return { labels: [], series: [], forecast: false };
  if (Array.isArray(spec.data) && spec.series && !spec.x) {
    const key = spec.x_key || 'date';
    const rows = spec.data;
    const series = spec.series.map((s, i) => ({
      label: s.label || s.key, slot: i + 1, data: rows.map((r) => r[s.key] ?? null), dashed: s.key === 'forecast',
    }));
    // Start the forecast (and its band) at the last actual point so the
    // dashed line continues the solid one instead of floating after a gap.
    const last = rows.map((r) => r.actual).findLastIndex((v) => v !== null && v !== undefined);
    const bridge = (key) => rows.map((r, i) => (i === last && r[key] == null ? r.actual : r[key] ?? null));
    const fcSeries = series.find((s) => s.dashed);
    if (fcSeries && last >= 0) fcSeries.data = bridge('forecast');
    if (rows.some((r) => r.upper != null)) {
      series.push({ label: 'Upper 95%', band: 'upper', data: last >= 0 ? bridge('upper') : rows.map((r) => r.upper ?? null) });
      series.push({ label: 'Lower 95%', band: 'lower', data: last >= 0 ? bridge('lower') : rows.map((r) => r.lower ?? null) });
    }
    return { labels: rows.map((r) => String(r[key] ?? '')), series, forecast: true };
  }
  const labels = spec.x || [];
  if (Array.isArray(spec.datasets) && spec.datasets.length) {
    return { labels, series: spec.datasets.map((d, i) => ({ label: d.label, slot: i + 1, data: d.data || [] })) };
  }
  return { labels, series: [{ label: spec.y_label || 'Value', slot: 1, data: spec.y || [] }] };
}

export default function Chart({ spec, height = 240, compactAxis = false }) {
  const canvas = useRef(null);
  const chart = useRef(null);
  const themeV = useThemeVersion();

  useEffect(() => {
    if (!canvas.current || !spec) return undefined;
    chart.current?.destroy();

    const { labels, series, forecast } = normalize(spec);
    const type = spec.type === 'pie' || spec.type === 'doughnut' ? 'doughnut' : spec.type === 'line' || forecast ? 'line' : 'bar';
    const isPie = type === 'doughnut';
    const ink = css('--ink'), muted = css('--muted'), grid = css('--grid'), surface = css('--surface'), border = css('--border');
    const slot = (n) => css(`--series-${((n - 1) % 6) + 1}`);
    const font = { family: css('--font-ui') || 'sans-serif', size: 11 };
    const multi = series.filter((s) => !s.band).length > 1;

    const datasets = series.map((s) => {
      if (s.band) {
        const c = css('--series-3');
        return {
          label: s.label, data: s.data, borderWidth: 0, pointRadius: 0,
          backgroundColor: c + '26', borderColor: 'transparent',
          fill: s.band === 'upper' ? '+1' : false, spanGaps: false,
        };
      }
      const color = forecast && s.dashed ? css('--series-3') : slot(s.slot);
      return {
        label: s.label,
        data: s.data,
        backgroundColor: isPie ? labels.map((_, i) => slot(i + 1)) : type === 'line' ? color + '1f' : color,
        borderColor: isPie ? surface : color,
        borderWidth: isPie ? 2 : type === 'line' ? 2 : 0,
        borderRadius: type === 'bar' ? { topLeft: 4, topRight: 4 } : 0,
        borderSkipped: 'bottom',
        borderDash: s.dashed ? [6, 4] : undefined,
        pointRadius: type === 'line' ? (labels.length > 24 ? 0 : 2.5) : 0,
        pointHoverRadius: 5,
        pointBackgroundColor: color,
        fill: type === 'line' && !multi && !forecast ? 'origin' : false,
        tension: 0.3,
        spanGaps: false,
        maxBarThickness: 46,
        categoryPercentage: multi ? 0.75 : 0.8,
        barPercentage: 0.86,
      };
    });

    chart.current = new ChartJS(canvas.current.getContext('2d'), {
      type,
      data: { labels, datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 500, easing: 'easeOutQuart' },
        interaction: { mode: isPie ? 'nearest' : 'index', intersect: false },
        cutout: isPie ? '58%' : undefined,
        plugins: {
          legend: {
            display: Boolean(isPie || multi || forecast),   // single series: the title names it
            position: 'bottom',
            labels: {
              color: muted, font, boxWidth: 8, boxHeight: 8, usePointStyle: true, pointStyle: 'rectRounded', padding: 14,
              filter: (item) => !/95%/.test(item.text),
            },
          },
          tooltip: {
            backgroundColor: surface, borderColor: border, borderWidth: 1, titleColor: ink, bodyColor: ink,
            padding: 10, cornerRadius: 10, boxPadding: 4, titleFont: { ...font, weight: '600' }, bodyFont: font,
            filter: (item) => item.raw !== null && item.raw !== undefined,
            callbacks: { label: (c) => ` ${c.dataset.label}: ${compact(c.parsed.y ?? c.parsed)}` },
          },
        },
        scales: isPie ? {} : {
          x: {
            grid: { display: false }, border: { color: border },
            ticks: { color: muted, font, maxRotation: 0, autoSkip: true, autoSkipPadding: 12 },
          },
          y: {
            beginAtZero: !forecast, grid: { color: grid }, border: { display: false },
            ticks: { color: muted, font, maxTicksLimit: compactAxis ? 4 : 6, callback: (v) => compact(v) },
          },
        },
      },
    });
    return () => chart.current?.destroy();
  }, [spec, themeV, compactAxis]);

  return (
    <div className="chart" style={{ height }}>
      <canvas ref={canvas} role="img" aria-label={spec?.title || 'Chart'} />
    </div>
  );
}
