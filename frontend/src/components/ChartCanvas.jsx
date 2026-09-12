import React, { useEffect, useRef } from 'react';
import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, LineElement, PointElement, ArcElement, Title, Tooltip, Legend } from 'chart.js';

ChartJS.register(CategoryScale, LinearScale, BarElement, LineElement, PointElement, ArcElement, Title, Tooltip, Legend);

const PALETTE = ['#6366f1', '#0ea5e9', '#f59e0b', '#ef4444', '#a855f7', '#10b981', '#fb923c', '#818cf8'];

function cssVar(name, fallback) {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
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

    chartRef.current = new ChartJS(ctx, {
      type: plotSpec.type || 'bar',
      data: {
        labels: plotSpec.x || [],
        datasets: [
          {
            label: plotSpec.y_label || 'Value',
            data: plotSpec.y || [],
            backgroundColor: isPie ? PALETTE.slice(0, (plotSpec.x || []).length) : `${accent}33`,
            borderColor: isPie ? PALETTE.slice(0, (plotSpec.x || []).length) : accent,
            borderWidth: 1.5,
            borderRadius: isPie ? 0 : 4,
            tension: 0.3,
          },
        ],
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
            display: isPie,
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
    <div className="relative w-full h-64">
      <canvas ref={canvasRef} />
    </div>
  );
}
