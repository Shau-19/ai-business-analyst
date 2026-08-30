import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 3000,
    proxy: {
      '/query': 'http://localhost:8000',
      '/upload': 'http://localhost:8000',
      '/conversations': 'http://localhost:8000',
      '/mcp': 'http://localhost:8000',
      '/api': 'http://localhost:8000',
    }
  }
})
