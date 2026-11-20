import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Every backend route the app calls must be proxied in dev, or the request
// hits Vite and comes back as index.html.
const API = process.env.VITE_PROXY_TARGET || 'http://localhost:8000';
const backendPaths = ['/query', '/upload', '/conversations', '/auth', '/status', '/api', '/mcp', '/a2a'];

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: Object.fromEntries(backendPaths.map((p) => [p, API])),
  },
});
