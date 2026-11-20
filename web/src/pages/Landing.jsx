import { ArrowRight, Database, FileText, LineChart, Languages, LayoutGrid, ShieldCheck, Moon, Sun } from 'lucide-react';
import Logo from '../components/Logo';
import Sparkline from '../components/Sparkline';

const REVENUE = [450, 520, 480, 550, 590, 620];
const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'];

function openWorkspace(withSample = false) {
  try {
    if (withSample) sessionStorage.setItem('analyst_load_sample', '1');
  } catch { /* storage blocked: sample button in the workspace still works */ }
  window.location.hash = '#/app';
}

// Hero illustration: a coral sun, a forest pot, and a plant whose stems are
// the H1 revenue bars, with a rotating label ring.
function HeroArt() {
  // Stems are the six H1 revenue months. Heights are scaled from a baseline
  // just under the minimum so the March dip and the growth read clearly.
  const lo = 400, hi = 620, minH = 70, maxH = 200;
  const R = 58; // label ring radius
  return (
    <div className="hero-art" aria-hidden="true">
      <svg viewBox="0 0 420 440" className="hero-svg">
        <defs>
          <path id="ring" d={`M 346,${100 - R} a ${R},${R} 0 1,1 -0.01,0`} />
        </defs>
        <circle cx="210" cy="196" r="168" fill="var(--coral)" />
        {/* stems + leaves */}
        {REVENUE.map((v, i) => {
          const x = 108 + i * 38, h = minH + ((v - lo) / (hi - lo)) * (maxH - minH), top = 314 - h;
          const side = i % 2 ? 1 : -1, ly = top + h * 0.42, lx = side > 0 ? x + 20 : x;
          return (
            <g key={MONTHS[i]} className="stem" style={{ animationDelay: `${120 + i * 70}ms` }}>
              <path d={`M${lx} ${ly} q ${side * 22} -18 ${side * 30} -6 q ${-side * 10} 16 ${-side * 30} 6 z`} fill="var(--forest)" />
              <rect x={x} y={top} width="20" height={h + 4} rx="10" fill="var(--forest)" />
              <circle cx={x + 10} cy={top + 10} r="3" fill="var(--coral)" opacity=".9" />
            </g>
          );
        })}
        {/* pot */}
        <path d="M112 318 H308 L292 412 Q290 424 278 424 H142 Q130 424 128 412 Z" fill="var(--forest-deep)" />
        <rect x="102" y="306" width="216" height="22" rx="8" fill="var(--forest)" />
        {/* label ring: always on a cream disc so it reads in both themes */}
        <g className="ring">
          <circle cx="346" cy="100" r="72" fill="var(--cream)" />
          <text fontSize="10.5" fontWeight="600" fill="var(--forest)" fontFamily="var(--font-ui)">
            <textPath href="#ring" textLength={2 * Math.PI * R - 4} lengthAdjust="spacing">
              SQL · DOCUMENTS · CHARTS · FORECASTS ·
            </textPath>
          </text>
        </g>
        <circle cx="346" cy="100" r="44" fill="none" stroke="var(--forest)" strokeWidth="1.2" />
        <text x="346" y="96" textAnchor="middle" fontFamily="var(--font-display)" fontStyle="italic" fontSize="16" fill="var(--forest)">from</text>
        <text x="346" y="114" textAnchor="middle" fontFamily="var(--font-display)" fontStyle="italic" fontSize="16" fill="var(--forest)">your files</text>
      </svg>

      <div className="hero-kpi card">
        <span className="eyebrow">Revenue · H1 2025</span>
        <div className="hero-kpi-row">
          <span className="hero-kpi-val display num">$3.21M</span>
          <span className="chip chip-good num">+37.8%</span>
        </div>
        <Sparkline values={REVENUE} width={150} height={34} />
        <span className="hero-kpi-note">Jan → Jun, from <span className="mono">Sales_Report_H1_2025.csv</span></span>
      </div>
    </div>
  );
}

function DemoChat() {
  const max = 620;
  return (
    <div className="demo card">
      <div className="demo-q">
        <span className="demo-avatar" aria-hidden="true">You</span>
        <p>Why did revenue dip in March?</p>
      </div>
      <div className="demo-a">
        <div className="demo-a-head"><span className="chip chip-forest">SQL</span><span className="demo-meta">Answered from Sales_Report_H1_2025.csv</span></div>
        <p>
          Revenue fell <strong>7.7%</strong> in March, to $480K from $520K in February. It was the only down month in H1, and churn peaked
          at <strong>2.3%</strong> the same month. Revenue recovered in April and grew every month through June.
        </p>
        <div className="demo-bars" role="img" aria-label="Monthly revenue January to June; March is the only decline">
          {REVENUE.map((v, i) => (
            <div key={MONTHS[i]} className="demo-bar">
              <span className="demo-bar-val num">${v}K</span>
              <span className={`demo-bar-fill ${i === 2 ? 'is-dip' : ''}`} style={{ height: `${(v / max) * 100}%` }} />
              <span className="demo-bar-label">{MONTHS[i]}</span>
            </div>
          ))}
        </div>
        <pre className="demo-sql mono">{`SELECT Month, Revenue, Churn_Rate
FROM sales_report_h1_2025
ORDER BY month_number;`}</pre>
      </div>
    </div>
  );
}

const STATS = [
  { value: '8/8', label: 'SQL answers exactly right', note: 'Results diffed against ground truth computed separately' },
  { value: '5/5', label: 'Out-of-scope questions refused', note: 'It says “not available” instead of inventing an answer' },
  { value: '9', label: 'Languages detected', note: 'Answers come back in the language you asked in' },
  { value: '13/15', label: 'Questions routed correctly', note: 'The two misses are reported, not tuned away' },
];

export default function Landing({ theme }) {
  return (
    <div className="landing">
      <header className="l-nav">
        <div className="l-inner l-nav-inner">
          <a href="#/" className="l-brand" aria-label="Analyst home"><Logo /></a>
          <nav className="l-links" aria-label="Sections">
            <a href="#how">How it works</a>
            <a href="#features">What it does</a>
            <a href="#proof">Proof</a>
          </nav>
          <div className="l-nav-actions">
            <button className="icon-btn" onClick={theme.cycle} aria-label="Switch light or dark theme">
              {theme.isDark ? <Sun /> : <Moon />}
            </button>
            <button className="btn btn-primary" onClick={() => openWorkspace()}>Open workspace</button>
          </div>
        </div>
      </header>

      <main>
        <section className="hero">
          <div className="l-inner hero-grid">
            <div className="hero-copy">
              <span className="eyebrow">AI business analyst · spreadsheets and the reports around them</span>
              <h1 className="display hero-title">
                Grow answers<br />from your <em>data.</em>
              </h1>
              <p className="hero-lede">
                Drop in a CSV and the documents that explain it. Ask in plain language and get the number, the chart and the
                reason, with the SQL behind every figure.
              </p>
              <div className="hero-ctas">
                <button className="btn btn-coral btn-lg" onClick={() => openWorkspace(true)}>
                  Try it with sample data <ArrowRight />
                </button>
                <button className="btn btn-ghost btn-lg" onClick={() => openWorkspace()}>Upload your own</button>
              </div>
              <p className="hero-fine">No setup. CSV, Excel, PDF, Word and text files.</p>
            </div>
            <HeroArt />
          </div>
        </section>

        <section className="l-section" aria-labelledby="ask-title">
          <div className="l-inner ask-grid">
            <div className="section-copy">
              <span className="eyebrow">Ask it like a colleague</span>
              <h2 id="ask-title" className="display section-title">The number, the chart <em>and</em> the reason.</h2>
              <p>
                Analytical questions run as read-only SQL on your data, so figures are exact. “Why” questions also read the
                documents you uploaded. Every answer shows where it came from.
              </p>
            </div>
            <DemoChat />
          </div>
        </section>

        <section className="l-section" id="features" aria-labelledby="features-title">
          <div className="l-inner">
            <div className="section-copy narrow">
              <span className="eyebrow">What it does</span>
              <h2 id="features-title" className="display section-title">A full analyst, in one workspace.</h2>
            </div>
            <div className="bento-l">
              <article className="b-tile b-sql">
                <Database className="b-icon" />
                <h3 className="display">Numbers come from SQL, never guesses</h3>
                <p>Every figure is computed by a read-only query against your file. Open any answer to see the exact SQL.</p>
                <pre className="mono b-code">{`SELECT region,
       ROUND(SUM(revenue), 2) AS total
FROM   upload_sales
GROUP  BY region
ORDER  BY total DESC;`}</pre>
                <div className="b-result" aria-label="Example result">
                  {[['North', 812], ['West', 741], ['South', 705], ['East', 626]].map(([r, v]) => (
                    <div key={r} className="b-result-row">
                      <span>{r}</span>
                      <span className="b-result-bar"><i style={{ width: `${(v / 812) * 100}%` }} /></span>
                      <span className="num">${v}K</span>
                    </div>
                  ))}
                </div>
              </article>
              <article className="b-tile b-docs">
                <FileText className="b-icon" />
                <h3 className="display">Reasons come from your documents</h3>
                <p>Ask “why”, and it reads the PDFs and Word files you added alongside the numbers, and cites the file it used.</p>
                <div className="b-files">
                  <span className="chip">Strategic_Planning_Jan2025.docx</span>
                  <span className="chip">Q4_Review.pdf</span>
                </div>
              </article>
              <article className="b-tile b-forecast">
                <LineChart className="b-icon" />
                <h3 className="display">Forecasts with honest ranges</h3>
                <p>Project any numeric column forward with a 95% range, and see which past points were anomalies.</p>
                <svg viewBox="0 0 240 70" className="b-forecast-art" aria-hidden="true">
                  <path d="M0 52 L40 46 L80 50 L120 34 V34 L160 22 L200 24 L240 12 L240 40 L200 44 L160 40 L120 34 Z" fill="var(--forest)" opacity=".18" />
                  <path d="M0 52 L40 46 L80 50 L120 34" fill="none" stroke="var(--forest)" strokeWidth="2.5" strokeLinecap="round" />
                  <path d="M120 34 L160 31 L200 34 L240 26" fill="none" stroke="var(--forest)" strokeWidth="2.5" strokeDasharray="6 5" strokeLinecap="round" />
                  <circle cx="120" cy="34" r="4" fill="var(--forest)" />
                </svg>
              </article>
              <article className="b-tile b-lang">
                <Languages className="b-icon" />
                <h3 className="display">Ask in your language</h3>
                <p>Answers come back in the language you asked in.</p>
                <div className="b-langs">
                  <span lang="es">¿Cuál fue el ingreso total?</span>
                  <span lang="hi">कुल राजस्व कितना था?</span>
                  <span lang="fr">Quel est le chiffre d’affaires&nbsp;?</span>
                </div>
              </article>
              <article className="b-tile b-dash">
                <LayoutGrid className="b-icon" />
                <h3 className="display">Dashboards that stay current</h3>
                <p>Pin any chart. Refresh it later and it re-runs against the latest data.</p>
                <div className="b-mini" aria-hidden="true"><span /><span /><span /><span /></div>
              </article>
              <article className="b-tile b-safe">
                <ShieldCheck className="b-icon" />
                <h3 className="display">Says no instead of making things up</h3>
                <p>If the answer isn’t in your files, it tells you. Queries are read-only and each workspace’s data stays separate.</p>
              </article>
            </div>
          </div>
        </section>

        <section className="l-section" id="how" aria-labelledby="how-title">
          <div className="l-inner">
            <div className="section-copy narrow">
              <span className="eyebrow">How it works</span>
              <h2 id="how-title" className="display section-title">Three steps, about a minute.</h2>
            </div>
            <ol className="steps">
              <li><span className="step-n display">1</span><h3>Upload</h3><p>Drop a CSV or Excel file, plus any PDFs or Word docs that explain it. You get an overview with KPIs and charts right away.</p></li>
              <li><span className="step-n display">2</span><h3>Ask</h3><p>Type a question the way you’d ask a colleague. It picks SQL, documents or both, and shows its working.</p></li>
              <li><span className="step-n display">3</span><h3>Pin</h3><p>Keep the charts that matter on a dashboard, and forecast any metric forward when you need to plan.</p></li>
            </ol>
          </div>
        </section>

        <section className="l-section" id="proof" aria-labelledby="proof-title">
          <div className="l-inner">
            <div className="section-copy narrow">
              <span className="eyebrow">Proof</span>
              <h2 id="proof-title" className="display section-title">Measured, not claimed.</h2>
              <p>From the project’s evaluation suite, run end to end with real model calls.</p>
            </div>
            <dl className="stats">
              {STATS.map((s) => (
                <div key={s.label} className="stat">
                  <dt className="stat-label">{s.label}</dt>
                  <dd className="stat-value display num">{s.value}</dd>
                  <dd className="stat-note">{s.note}</dd>
                </div>
              ))}
            </dl>
          </div>
        </section>

        <section className="cta-band">
          <div className="l-inner cta-inner">
            <h2 className="display">Your data already has the answers.</h2>
            <div className="hero-ctas">
              <button className="btn btn-coral btn-lg" onClick={() => openWorkspace(true)}>Try it with sample data <ArrowRight /></button>
              <button className="btn btn-ghost-light btn-lg" onClick={() => openWorkspace()}>Open the workspace</button>
            </div>
          </div>
        </section>
      </main>

      <footer className="l-footer">
        <div className="l-inner l-footer-inner">
          <Logo size={22} />
          <span>Built with FastAPI, LangChain, FAISS and Prophet.</span>
        </div>
      </footer>
    </div>
  );
}
