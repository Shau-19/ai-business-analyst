import { useEffect, useRef, useState } from 'react';
import { ArrowUp, Square, Pin, Code2, FileText, LoaderCircle, CircleAlert } from 'lucide-react';
import Chart from '../components/Chart';
import { LogoMark } from '../components/Logo';
import { renderMarkdown } from '../lib/markdown';
import { ROUTE_LABEL } from '../lib/format';

function Answer({ msg, question, onPin }) {
  const meta = msg.metadata || {};
  const plot = meta.plot || msg.plot;
  const sources = (meta.sources || []).filter((s) => s.source);
  // The backend appends "*Source: file*" to document answers; the chips below
  // already say that, so drop the duplicate line.
  const text = sources.length ? (msg.content || '').replace(/\n*\*?Sources?:[^\n]*\*?\s*$/i, '') : msg.content;
  return (
    <article className="msg msg-ai">
      <div className="msg-avatar" aria-hidden="true"><LogoMark size={26} /></div>
      <div className="msg-body">
        <div className="msg-head">
          <span className="msg-who">Analyst</span>
          {msg.routing && <span className={`chip ${(msg.routing === 'hybrid' || msg.routing === 'analysis') ? 'chip-coral' : 'chip-forest'}`}>{ROUTE_LABEL[msg.routing] || msg.routing}</span>}
          {msg.pending && !msg.content && <span className="msg-thinking"><LoaderCircle className="spin" aria-hidden="true" /> Working it out…</span>}
        </div>
        {msg.failed && !msg.content ? null : (
          <div className={`answer ${msg.failed ? 'is-failed' : ''}`}>
            {msg.failed && <CircleAlert className="answer-icon" aria-hidden="true" />}
            <div className="prose" dangerouslySetInnerHTML={{ __html: renderMarkdown(text) }} />
          </div>
        )}
        {plot && (
          <div className="answer-chart">
            <div className="tile-head">
              <h4 className="tile-title">{plot.title}</h4>
              <button className="btn btn-sm btn-ghost" onClick={() => onPin(plot, question)}><Pin /> Pin</button>
            </div>
            <Chart spec={plot} height={230} />
          </div>
        )}
        {(meta.sql_query || sources.length > 0) && (
          <div className="answer-meta">
            {meta.sql_query && (
              <details className="answer-sql">
                <summary><Code2 aria-hidden="true" /> SQL used</summary>
                <pre className="mono">{meta.sql_query}</pre>
              </details>
            )}
            {sources.length > 0 && (
              <div className="answer-sources">
                <span className="eyebrow">Sources</span>
                {[...new Map(sources.map((s) => [s.source, s])).values()].map((s) => (
                  <span key={s.source} className="chip" title={s.preview}>
                    <FileText aria-hidden="true" /> {s.source}
                  </span>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </article>
  );
}

export default function Ask({ activeId, profile, messages, streaming, onAsk, onStop, onPin }) {
  const [draft, setDraft] = useState('');
  const end = useRef(null);
  const box = useRef(null);

  useEffect(() => { end.current?.scrollIntoView({ behavior: 'smooth', block: 'end' }); }, [messages]);
  useEffect(() => {
    const el = box.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 180) + 'px';
  }, [draft]);

  const suggestions = profile?.datasets?.[0]?.questions?.length
    ? profile.datasets[0].questions
    : profile?.documents?.length ? ['Summarize the key points in my documents', 'What are the action items?'] : [];
  const send = () => { if (draft.trim() && !streaming) { onAsk(draft); setDraft(''); } };

  return (
    <div className="ask">
      <div className="ask-thread" aria-live="polite">
        {messages.length === 0 ? (
          <div className="ask-empty">
            <LogoMark size={44} />
            <h2 className="display">Ask anything about your data.</h2>
            <p>{activeId && (profile?.files?.length)
              ? 'Numbers come from SQL on your files; “why” questions also read your documents.'
              : 'Add a CSV or document first, or ask a general question.'}</p>
            {suggestions.length > 0 && (
              <div className="ask-suggest">
                {suggestions.slice(0, 5).map((q) => <button key={q} className="chip ask-suggest-chip" onClick={() => onAsk(q)}>{q}</button>)}
              </div>
            )}
          </div>
        ) : messages.map((m, i) => (m.role === 'user' ? (
          <div key={i} className="msg msg-user"><p>{m.content}</p></div>
        ) : (
          <Answer key={i} msg={m} question={messages[i - 1]?.role === 'user' ? messages[i - 1].content : null} onPin={onPin} />
        )))}
        <div ref={end} />
      </div>

      <div className="composer-dock">
      <form className="composer" onSubmit={(e) => { e.preventDefault(); send(); }}>
        <label className="sr-only" htmlFor="composer-input">Ask a question</label>
        <textarea
          id="composer-input" ref={box} rows={1} value={draft}
          placeholder="Ask about revenue, trends, causes… in any language"
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); } }}
        />
        {streaming ? (
          <button type="button" className="composer-btn is-stop" onClick={onStop} aria-label="Stop"><Square /></button>
        ) : (
          <button type="submit" className="composer-btn" disabled={!draft.trim()} aria-label="Send"><ArrowUp /></button>
        )}
      </form>
      <p className="composer-hint">Enter to send · Shift + Enter for a new line</p>
      </div>
    </div>
  );
}
