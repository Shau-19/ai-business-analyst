import { useEffect, useState, useCallback } from 'react';
import Landing from './pages/Landing';
import Workspace from './workspace/Workspace';
import AccessKeyModal from './components/AccessKeyModal';
import { api, AUTH_EVENT } from './lib/api';

// Hash routes: #/ (landing) and #/app (workspace). Hash routing keeps the
// build servable from any static host without rewrite rules.
function useRoute() {
  const read = () => (window.location.hash.startsWith('#/app') ? 'app' : 'landing');
  const [route, setRoute] = useState(read);
  useEffect(() => {
    const on = () => { setRoute(read()); window.scrollTo(0, 0); };
    window.addEventListener('hashchange', on);
    return () => window.removeEventListener('hashchange', on);
  }, []);
  return route;
}

// Theme: 'system' (no attribute) | 'light' | 'dark', remembered per browser.
export function useTheme() {
  const [theme, setTheme] = useState(() => {
    try { return localStorage.getItem('analyst_theme') || 'system'; } catch { return 'system'; }
  });
  useEffect(() => {
    const root = document.documentElement;
    if (theme === 'system') root.removeAttribute('data-theme');
    else root.setAttribute('data-theme', theme);
    try { localStorage.setItem('analyst_theme', theme); } catch { /* storage blocked */ }
  }, [theme]);
  const [osDark, setOsDark] = useState(() => window.matchMedia('(prefers-color-scheme: dark)').matches);
  useEffect(() => {
    const mq = window.matchMedia('(prefers-color-scheme: dark)');
    const on = (e) => setOsDark(e.matches);
    mq.addEventListener('change', on);
    return () => mq.removeEventListener('change', on);
  }, []);
  const isDark = theme === 'dark' || (theme === 'system' && osDark);
  const cycle = useCallback(() => setTheme(isDark ? 'light' : 'dark'), [isDark]);
  return { theme, isDark, cycle };
}

export default function App() {
  const route = useRoute();
  const theme = useTheme();
  const [needsKey, setNeedsKey] = useState(false);

  useEffect(() => {
    if (route !== 'app') return undefined;
    api.checkAuth()
      .then((r) => { if (r.required && !r.valid) setNeedsKey(true); })
      .catch(() => { /* backend down: requests will show errors */ });
    const on = () => setNeedsKey(true);
    window.addEventListener(AUTH_EVENT, on);
    return () => window.removeEventListener(AUTH_EVENT, on);
  }, [route]);

  return (
    <>
      {route === 'app' ? <Workspace theme={theme} authReady={!needsKey} /> : <Landing theme={theme} />}
      {route === 'app' && needsKey && <AccessKeyModal onAuthorized={() => setNeedsKey(false)} />}
    </>
  );
}
