import React, { useEffect, useRef } from 'react';
import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, LineElement, PointElement, ArcElement, Title, Tooltip, Legend, Filler } from 'chart.js';

ChartJS.register(CategoryScale, LinearScale, BarElement, LineElement, PointElement, ArcElement, Title, Tooltip, Legend, Filler);

const PALETTE = ['#6366f1', '#0ea5e9', '#f59e0b', '#ef4444', '#a855f7', '#10b981', '#fb923c', '#818cf8'];

function cssVar(name, fallback) {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}

// Appending a hex alpha only works for #rrggbb colors.
function withAlpha(color, alphaHex) {
  return /^#[0-9a-f]{6}$/i.test(color) ? `${color}${alphaHex}` : color;
}

/**
 * The backend sends three chart shapes; normalise them to { labels, series }.
 *  1. SQL multi-series:  { x, datasets: [{label, data}] }
 *  2. SQL single-series: { x, y, y_label }               (older saved panels)
 *  3. Forecast:          { data: [{date, actual, forecast, lower, upper}], series, x_key }
 * Shape 3 used to render as an empty chart because only x/y were read.
 */
function normalizeSpec(spec) {
  if (!spec) return { labels: [], series: [] };

  if (Array.isArray(spec.data) && spec.series && !spec.x) {
    const xKey = spec.x_key || 'date';
    const rows = spec.data;
    const labels = rows.map((r) => String(r[xKey] ?? ''));
    const series = spec.series.map((s) => ({
      label: s.label || s.key,
      color: s.color,
      data: rows.map((r) => (r[s.key] ?? null)),
      dashed: s.key === 'forecast',
    }));
    // 95% confidence band around the forecast, when present
    if (rows.some((r) => r.upper != null)) {
      series.push({ label: 'Upper 95%', data: rows.map((r) => r.upper ?? null), color: '#f59e0b', hidePoints: true, fill: '+1' });
      series.push({ label: 'Lower 95%', data: rows.map((r) => r.lower ?? null), color: '#f59e0b', hidePoints: true });
    }
    return { labels, series };
  }

  const labels = spec.x || [];
  if (Array.isArray(spec.datasets) && spec.datasets.length) {
    return { labels, series: spec.datasets.map((d) => ({ label: d.label, data: d.data || [] })) };
  }
  return { labels, series: [{ label: spec.y_label || 'Value', data: spec.y || [] }] };
}

export default function ChartCanvas({ plotSpec }) {
  const canvasRef = useRef(null);
  const chartRef = useRef(null);

  useEffect(() => {
    if (!canvasRef.current || !plotSpec) return;

    if (chartRef.current) {
      chartRef.current.destroy();
    }

    const ctx = canvasRef.current.getContext('2d');
    const isPie = plotSpec.type === 'pie' || plotSpec.type === 'doughnut';

    const text = cssVar('--text', '#18181b');
    const muted = cssVar('--text-muted', '#94949c');
    const border = cssVar('--border', '#e8e8ea');
    const accent = cssVar('--accent', '#6366f1');
    const surfaceRaised = cssVar('--surface-raised', '#ffffff');

    const { labels, series } = normalizeSpec(plotSpec);
    const multi = series.length > 1;

    chartRef.current = new ChartJS(ctx, {
      type: plotSpec.type || 'bar',
      data: {
        labels,
        datasets: series.map((s, i) => {
          const color = s.color || (i === 0 ? accent : PALETTE[i % PALETTE.length]);
          return {
            label: s.label,
            data: s.data,
            backgroundColor: isPie ? PALETTE.slice(0, labels.length) : withAlpha(color, s.fill ? '22' : '33'),
            borderColor: isPie ? PALETTE.slice(0, labels.length) : color,
            borderWidth: 1.5,
            borderRadius: isPie ? 0 : 4,
            borderDash: s.dashed ? [5, 4] : undefined,
            pointRadius: s.hidePoints ? 0 : 2.5,
            fill: s.fill || false,
            spanGaps: false,          // missing values show as gaps, not zeros
            tension: 0.3,
          };
        }),
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          title: {
            display: true,
            text: plotSpec.title || 'Data Visualization',
            color: text,
            font: { family: 'Inter, sans-serif', size: 13, weight: '600' },
            padding: { bottom: 12 },
          },
          legend: {
            display: isPie || multi,
            labels: { color: muted, font: { family: 'Inter, sans-serif', size: 11 } },
          },
          tooltip: {
            backgroundColor: surfaceRaised,
            borderColor: border,
            borderWidth: 1,
            titleColor: accent,
            bodyColor: text,
            padding: 8,
            cornerRadius: 6,
          },
        },
        scales: isPie
          ? {}
          : {
              x: {
                ticks: { color: muted, font: { family: 'Inter, sans-serif', size: 10.5 } },
                grid: { color: border },
              },
              y: {
                ticks: { color: muted, font: { family: 'Inter, sans-serif', size: 10.5 } },
                grid: { color: border },
                beginAtZero: true,
              },
            },
      },
    });

    return () => {
      if (chartRef.current) chartRef.current.destroy();
    };
  }, [plotSpec]);

  return (
    <div className="w-full">
      <div className="relative w-full h-64">
        <canvas ref={canvasRef} />
      </div>
      {plotSpec?.truncated && (
        <p className="mt-1 text-[11px] text-[var(--text-muted)]">
          Showing the first {plotSpec.x?.length} rows of a larger result.
        </p>
      )}
    </div>
  );
}
