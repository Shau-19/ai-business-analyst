import React, { useEffect, useRef } from 'react';
import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, LineElement, PointElement, ArcElement, Title, Tooltip, Legend } from 'chart.js';

ChartJS.register(CategoryScale, LinearScale, BarElement, LineElement, PointElement, ArcElement, Title, Tooltip, Legend);

export default function ChartCanvas({ plotSpec }) {
  const canvasRef = useRef(null);
  const chartRef = useRef(null);

  useEffect(() => {
    if (!canvasRef.current || !plotSpec) return;

    if (chartRef.current) {
      chartRef.current.destroy();
    }

    const ctx = canvasRef.current.getContext('2d');
    const colors = ['#4ade80', '#38bdf8', '#fbbf24', '#f87171', '#a78bfa', '#fb923c', '#34d399', '#818cf8'];
    const isPie = plotSpec.type === 'pie' || plotSpec.type === 'doughnut';

    chartRef.current = new ChartJS(ctx, {
      type: plotSpec.type || 'bar',
      data: {
        labels: plotSpec.x || [],
        datasets: [
          {
            label: plotSpec.y_label || 'Value',
            data: plotSpec.y || [],
            backgroundColor: isPie ? colors.slice(0, (plotSpec.x || []).length) : '#4ade8033',
            borderColor: isPie ? colors.slice(0, (plotSpec.x || []).length) : '#4ade80',
            borderWidth: 1.5,
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
            color: '#86efac',
            font: { family: 'ui-monospace, monospace', size: 13, weight: '600' },
            padding: { bottom: 12 },
          },
          legend: {
            display: isPie,
            labels: { color: '#94a3b8', font: { family: 'ui-monospace, monospace', size: 11 } },
          },
          tooltip: {
            backgroundColor: '#0f172a',
            borderColor: '#334155',
            borderWidth: 1,
            titleColor: '#4ade80',
            bodyColor: '#f8fafc',
          },
        },
        scales: isPie
          ? {}
          : {
              x: {
                ticks: { color: '#64748b', font: { family: 'ui-monospace, monospace', size: 10 } },
                grid: { color: '#1e293b' },
              },
              y: {
                ticks: { color: '#64748b', font: { family: 'ui-monospace, monospace', size: 10 } },
                grid: { color: '#1e293b' },
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
