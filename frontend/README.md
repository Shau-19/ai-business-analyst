# AI Business Analyst - Modern React Frontend

A standalone, modern React Business Intelligence application with multi-dashboard management, persistent chat feeds, and time-series predictive forecasting.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
npm install
```

### 2. Configure Environment (Optional)
Copy `.env.example` to `.env`:
```bash
VITE_API_BASE_URL=http://localhost:8000
```

### 3. Run Development Server
```bash
npm run dev
```

Open `http://localhost:3000` in your browser.

### 4. Production Build
```bash
npm run build
```

---

## 🌐 Independent Deployment (Vercel / Netlify / Cloudflare Pages)

1. Deploy the `frontend/` directory to your preferred host (Vercel, Netlify, Cloudflare).
2. Set Environment Variable:
   - `VITE_API_BASE_URL` = `https://your-backend-api-url.com`
3. Build command: `npm run build`
4. Output directory: `dist`
