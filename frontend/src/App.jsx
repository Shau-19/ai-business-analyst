import React, { useState, useEffect } from 'react';
import Sidebar from './components/Sidebar';
import Header from './components/Header';
import ChatView from './components/ChatView';
import DashboardView from './components/DashboardView';
import ForecastView from './components/ForecastView';
import { api } from './api';

export default function App() {
  const [userId] = useState(() => {
    const saved = localStorage.getItem('analyst_user_id');
    if (saved) return saved;
    const newId = `user_${Math.random().toString(36).substring(2, 10)}`;
    localStorage.setItem('analyst_user_id', newId);
    return newId;
  });

  const [conversations, setConversations] = useState([]);
  const [activeSessionId, setActiveSessionId] = useState(null);
  const [activeTab, setActiveTab] = useState('chat');
  const [messages, setMessages] = useState([]);
  const [documents, setDocuments] = useState([]);
  const [isUploading, setIsUploading] = useState(false);
  const [isQueryLoading, setIsQueryLoading] = useState(false);

  // Load conversations list on mount
  useEffect(() => {
    loadSessions();
  }, []);

  // Load session data when active session changes
  useEffect(() => {
    if (!activeSessionId) {
      setMessages([]);
      setDocuments([]);
      return;
    }
    loadSessionDetails(activeSessionId);
  }, [activeSessionId]);

  const loadSessions = async () => {
    try {
      const res = await api.getConversations(userId);
      if (res.conversations) {
        setConversations(res.conversations);
        if (!activeSessionId && res.conversations.length > 0) {
          setActiveSessionId(res.conversations[0].conversation_id);
        }
      }
    } catch (e) {
      console.error('Failed to load sessions', e);
    }
  };

  const loadSessionDetails = async (sessionId) => {
    try {
      const res = await api.getConversation(sessionId);
      if (res.success && res.conversation) {
        setMessages(res.conversation.messages || []);
      }
      const docsRes = await api.getDocuments(sessionId);
      if (docsRes.success) {
        setDocuments(docsRes.documents || []);
      }
    } catch (e) {
      console.error('Failed to load session details', e);
    }
  };

  const handleNewSession = async () => {
    try {
      const res = await api.createConversation(userId, 'New Session');
      if (res.success) {
        setActiveSessionId(res.conversation_id);
        await loadSessions();
      }
    } catch (e) {
      console.error('Failed to create new session', e);
    }
  };

  const handleFileUpload = async (files) => {
    if (!activeSessionId) return;
    setIsUploading(true);
    try {
      const res = await api.uploadFiles(activeSessionId, files);
      if (res.success || res.loaded_files) {
        pollStatus(activeSessionId);
      }
    } catch (e) {
      console.error('Upload failed', e);
      setIsUploading(false);
    }
  };

  const pollStatus = (sessionId) => {
    let attempts = 0;
    const interval = setInterval(async () => {
      attempts++;
      if (attempts > 60) {
        clearInterval(interval);
        setIsUploading(false);
        return;
      }
      try {
        const res = await api.getProcessingStatus(sessionId);
        if (res.status === 'ready') {
          clearInterval(interval);
          setIsUploading(false);
          loadSessionDetails(sessionId);
        } else if (res.status === 'error') {
          clearInterval(interval);
          setIsUploading(false);
        }
      } catch (e) {
        console.error('Status poll error', e);
      }
    }, 1000);
  };

  const handleSendMessage = async (text) => {
    if (!activeSessionId) return;

    // Optimistically add user message
    const userMsg = { role: 'user', content: text, timestamp: new Date().toISOString() };
    setMessages((prev) => [...prev, userMsg]);
    setIsQueryLoading(true);

    let assistantMsg = {
      role: 'assistant',
      content: '',
      routing: '',
      metadata: {},
    };

    setMessages((prev) => [...prev, assistantMsg]);

    await api.streamQuery(activeSessionId, text, null, {
      onMeta: (meta) => {
        setMessages((prev) => {
          const updated = [...prev];
          const last = { ...updated[updated.length - 1] };
          last.routing = meta.routing;
          last.metadata = { ...last.metadata, plot: meta.plot, sql_query: meta.sql_query };
          updated[updated.length - 1] = last;
          return updated;
        });
      },
      onToken: (chunk) => {
        setMessages((prev) => {
          const updated = [...prev];
          const last = { ...updated[updated.length - 1] };
          last.content = (last.content || '') + chunk;
          updated[updated.length - 1] = last;
          return updated;
        });
      },
      onDone: () => {
        setIsQueryLoading(false);
        loadSessions();
      },
      onError: (err) => {
        setIsQueryLoading(false);
        setMessages((prev) => {
          const updated = [...prev];
          const last = { ...updated[updated.length - 1] };
          last.content = `Error: ${err.message || err.error || 'Failed to get response'}`;
          updated[updated.length - 1] = last;
          return updated;
        });
      },
    });
  };

  const handlePinChart = async (plotSpec) => {
    if (!activeSessionId) return;
    try {
      const res = await api.getDashboard(activeSessionId);
      let state = res.state || { dashboards: {}, activeId: null, built: true };
      if (!state.dashboards) state.dashboards = {};

      const dashKeys = Object.keys(state.dashboards);
      let targetId = state.activeId || (dashKeys.length > 0 ? dashKeys[0] : 'dash_main');

      if (!state.dashboards[targetId]) {
        state.dashboards[targetId] = {
          id: targetId,
          title: 'Main Dashboard',
          panels: [],
        };
      }

      state.dashboards[targetId].panels.push({
        title: plotSpec.title || 'Pinned Chart',
        plot: plotSpec,
      });

      state.activeId = targetId;
      state.built = true;

      await api.saveDashboard(activeSessionId, state);
      setActiveTab('dashboard');
    } catch (e) {
      console.error('Failed to pin chart', e);
    }
  };

  return (
    <div className="flex h-screen w-screen bg-slate-950 text-slate-100 font-sans overflow-hidden">
      {/* Sidebar */}
      <Sidebar
        conversations={conversations}
        activeSessionId={activeSessionId}
        onSelectSession={setActiveSessionId}
        onNewSession={handleNewSession}
        onFileUpload={handleFileUpload}
        documents={documents}
        isUploading={isUploading}
      />

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col h-full overflow-hidden">
        <Header
          activeTab={activeTab}
          onTabChange={setActiveTab}
          activeSessionId={activeSessionId}
          documentsCount={documents.length}
        />

        <div className="flex-1 overflow-hidden relative">
          {activeTab === 'chat' && (
            <ChatView
              activeSessionId={activeSessionId}
              messages={messages}
              onSendMessage={handleSendMessage}
              onPinChart={handlePinChart}
              isLoading={isQueryLoading}
            />
          )}

          {activeTab === 'dashboard' && <DashboardView activeSessionId={activeSessionId} />}

          {activeTab === 'forecast' && <ForecastView activeSessionId={activeSessionId} />}
        </div>
      </div>
    </div>
  );
}
