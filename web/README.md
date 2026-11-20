# Analyst web app

The standalone frontend for the AI Business Analyst: a landing page and a BI workspace (Overview, Ask, Dashboards, Forecast, Data). It talks to the FastAPI backend in the repo root. The old embedded page at `/ui` (`static/index.html`) is untouched.

## Run it

```bash
# 1. backend, from the repo root
python main.py            # serves the API on http://localhost:8000

# 2. this app
cd web
npm install
npm run dev               # http://localhost:5173
```

In development, Vite forwards API calls to `http://localhost:8000`. Point it elsewhere with `VITE_PROXY_TARGET`:

```bash
VITE_PROXY_TARGET=http://localhost:8001 npm run dev
```

If the backend sets `API_KEYS` in `.env`, the workspace asks for a key once and keeps it in that browser. No key is ever bundled into the app.

## Build and deploy

```bash
npm run build             # outputs dist/
```

`dist/` is a static site; routes are hash-based (`#/`, `#/app`), so any static host works without rewrite rules. When it isn't served from the same origin as the API, set the backend URL at build time:

```bash
VITE_API_BASE_URL=https://api.example.com npm run build
```

## Design

Palette: Fern & Co. (cream paper, forest ink, coral accent), validated for text contrast (body ≥ 7:1, labels ≥ 4.5:1) and colour-blind-safe chart series in light and dark mode. Tokens live in `src/styles/tokens.css`. Type: Fraunces (display), Instrument Sans (UI), JetBrains Mono (SQL).

The Overview is computed without the LLM by `GET /conversations/{cid}/profile` (`tools/data_profile.py`), so KPIs and charts appear as soon as a file is indexed.
