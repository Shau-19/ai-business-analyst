import React, { useState, useRef, useEffect } from 'react';
import { Send, Pin, Download, Bot, User, Sparkles, AlertCircle } from 'lucide-react';
import { marked } from 'marked';
import ChartCanvas from './ChartCanvas';
import { api } from '../api';

marked.setOptions({ breaks: true, gfm: true });

export default function ChatView({
  activeSessionId,
  messages,
  onSendMessage,
  onPinChart,
  isLoading,
}) {
  const [inputText, setInputText] = useState('');
  const chatEndRef = useRef(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!inputText.trim() || isLoading || !activeSessionId) return;
    onSendMessage(inputText.trim());
    setInputText('');
  };

  const exportChartPNG = (canvasId, title) => {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const a = document.createElement('a');
    a.download = `${title || 'chart'}.png`;
    a.href = canvas.toDataURL('image/png');
    a.click();
  };

  return (
    <div className="flex-1 flex flex-col h-full bg-slate-950 overflow-hidden">
      {/* Messages Feed */}
      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {!activeSessionId ? (
          <div className="h-full flex flex-col items-center justify-center text-slate-500 gap-3">
            <Sparkles className="w-10 h-10 text-emerald-400/40" />
            <p className="font-mono text-sm">Select or create a session to start</p>
          </div>
        ) : messages.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center text-slate-500 gap-3">
            <Bot className="w-10 h-10 text-slate-600" />
            <p className="font-mono text-sm text-slate-400">Query your CSV data or documents using natural language</p>
            <div className="flex gap-2 text-xs font-mono text-slate-500">
              <span className="px-2 py-1 bg-slate-900 border border-slate-800 rounded">"Who is the top sales rep?"</span>
              <span className="px-2 py-1 bg-slate-900 border border-slate-800 rounded">"Show revenue by region"</span>
            </div>
          </div>
        ) : (
          messages.map((msg, index) => {
            const isUser = msg.role === 'user';
            const plotData = msg.metadata?.plot || msg.plot;
            const routing = msg.routing;

            return (
              <div key={index} className={`flex flex-col gap-2 ${isUser ? 'items-end' : 'items-start'}`}>
                {/* Header Badge */}
                <div className="flex items-center gap-2 px-1 text-[11px] font-mono text-slate-400">
                  {isUser ? (
                    <>
                      <span>▸ YOU</span>
                      <User className="w-3 h-3 text-cyan-400" />
                    </>
                  ) : (
                    <>
                      <Bot className="w-3.5 h-3.5 text-emerald-400" />
                      <span className="text-emerald-400 font-semibold">◈ ANALYST</span>
                      {routing && (
                        <span className="px-1.5 py-0.5 rounded text-[9px] font-bold tracking-wider uppercase border bg-emerald-950/80 text-emerald-400 border-emerald-800">
                          {routing}
                        </span>
                      )}
                    </>
                  )}
                </div>

                {/* Message Bubble */}
                <div
                  className={`p-4 rounded-xl max-w-2xl text-sm leading-relaxed border ${
                    isUser
                      ? 'bg-cyan-950/40 border-cyan-500/30 text-slate-100 rounded-tr-none font-mono text-xs'
                      : 'bg-slate-900/90 border-slate-800 text-slate-200 rounded-tl-none'
                  }`}
                >
                  {/* Markdown Content */}
                  {isUser ? (
                    <div>{msg.content}</div>
                  ) : (
                    <div
                      className="prose prose-invert prose-sm max-w-none prose-pre:bg-slate-950 prose-pre:border prose-pre:border-slate-800"
                      dangerouslySetInnerHTML={{ __html: marked.parse(msg.content || '') }}
                    />
                  )}

                  {/* Render Graphical Chart if present */}
                  {plotData && (
                    <div className="mt-4 p-4 bg-slate-950 border border-slate-800 rounded-lg">
                      <ChartCanvas plotSpec={plotData} />

                      {/* Chart Actions */}
                      <div className="mt-3 flex items-center justify-end gap-2 pt-2 border-t border-slate-900">
                        <button
                          onClick={() => onPinChart(plotData)}
                          className="px-2.5 py-1 bg-purple-500/10 hover:bg-purple-500/20 text-purple-400 border border-purple-500/30 rounded text-xs font-mono flex items-center gap-1.5 transition-colors cursor-pointer"
                        >
                          <Pin className="w-3 h-3" />
                          <span>PIN TO DASHBOARD</span>
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            );
          })
        )}
        <div ref={chatEndRef} />
      </div>

      {/* Input Area */}
      <div className="p-4 bg-slate-900 border-t border-slate-800">
        <form onSubmit={handleSubmit} className="flex gap-3">
          <input
            type="text"
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            disabled={isLoading || !activeSessionId}
            placeholder={
              !activeSessionId
                ? 'Create a session first...'
                : 'Ask a question (e.g. "Show a bar chart of total revenue by region")'
            }
            className="flex-1 bg-slate-950 border border-slate-800 rounded-lg px-4 py-3 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:border-emerald-500 font-mono disabled:opacity-50"
          />
          <button
            type="submit"
            disabled={isLoading || !inputText.trim() || !activeSessionId}
            className="px-6 py-3 bg-emerald-500 hover:bg-emerald-400 disabled:opacity-40 text-slate-950 font-mono font-bold text-xs rounded-lg flex items-center gap-2 transition-colors cursor-pointer shadow-lg shadow-emerald-500/10"
          >
            <span>SEND</span>
            <Send className="w-4 h-4" />
          </button>
        </form>
      </div>
    </div>
  );
}
