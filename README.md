# FanPulse AI — Smart Stadium & Tournament Operations Assistant
### Challenge 4: Smart Stadiums & Tournament Operations — FIFA World Cup 2026

FanPulse AI is a GenAI-powered assistant that helps fans, volunteers, and venue
staff navigate stadiums, avoid crowd bottlenecks, and get accessible,
multilingual support in real time.

## Problem → Solution mapping

| Challenge requirement            | What FanPulse does |
|-----------------------------------|---------------------|
| Navigation                        | `/chat` answers gate/seating/amenity questions in natural language |
| Crowd management                  | `/crowd-status` gives live occupancy per gate + AI-generated flow recommendations to reroute fans away from congested gates |
| Accessibility                     | `/accessibility/simplify` rewrites announcements into plain, screen-reader-friendly language; frontend built to WCAG-conscious patterns (labels, `aria-live`, focus states, skip-friendly structure) |
| Multilingual assistance           | 8 languages (English, Spanish, Portuguese, French, Hindi, Arabic, German, Japanese) via Gemini |
| Real-time decision support        | Crowd endpoint is designed to sit on top of live turnstile/CCTV feeds and continuously re-rank gate recommendations |
| Sustainability (secondary)        | System prompt nudges the assistant to mention transit/refill points when relevant |

## Architecture

```
frontend/index.html  →  FastAPI backend (main.py)  →  Gemini (google-genai SDK)
                              │
                              └── in-memory crowd/zone data (swap for real
                                  turnstile/IoT feed in production)
```

- **Backend**: FastAPI, Pydantic v2 validation, Gemini 2.0 Flash via `google-genai`.
- **Frontend**: dependency-free HTML/CSS/JS — runs by opening the file, no build step.
- **Tests**: pytest + FastAPI `TestClient`, Gemini calls mocked (16 tests, all passing, offline).

## Running locally

```bash
cd backend
cp .env.example .env        # add your real GEMINI_API_KEY
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Open `frontend/index.html` directly in a browser (or serve it with any static
server). It talks to `http://localhost:8000` by default; override with
`window.FANPULSE_API_BASE` before the script tag if deploying separately.

## Running tests

```bash
cd fanpulse-ai
GEMINI_API_KEY=dummy python -m pytest tests/ -v
```

## Security notes

- API key is read from environment only (`.env`, never committed — `.env.example` provided instead).
- All inputs validated with Pydantic (length limits, language-code allowlist, non-empty checks).
- CORS restricted to an explicit allowlist (`ALLOWED_ORIGINS` env var), not `*`.
- Per-IP sliding-window rate limiting (20 req/min) on all endpoints to reduce abuse and API-cost blowout — noted in code as suitable for single-instance demo; a Redis-backed limiter is recommended before multi-instance production deployment.
- Gemini failures are caught and returned as generic `502`/`503` errors — no stack traces or internal details leaked to clients.
- System instruction constrains the model to stadium-ops topics and forbids inventing gate numbers, prices, or team/player facts, reducing hallucination risk in an operational setting.

## Known limitations (being upfront, not overselling)

- Crowd occupancy data is simulated in-memory; a real deployment needs a live feed (turnstile counters, CCTV-based density estimation, or GPS-based transit data) wired into `_live_occupancy`.
- Rate limiter is in-process only — fine for a single container/demo, not for horizontally-scaled production.
- No persistent chat history/session — each `/chat` call is stateless by design, to avoid storing PII unnecessarily.
- Frontend is intentionally framework-free for portability; a production build would likely move to React for state management at scale.

## Deploying: Render (backend) + Vercel (frontend)

This repo is pre-configured for exactly this split.

### 1. Backend → Render
1. Push this repo to GitHub.
2. In Render: **New → Blueprint**, point it at the repo — `render.yaml` at the root will auto-configure the service (build command, start command, health check).
   - If you'd rather set it up manually instead of using the blueprint: Build command `pip install -r backend/requirements.txt`, Start command `uvicorn main:app --host 0.0.0.0 --port $PORT --app-dir backend`.
3. In the Render dashboard, set the env var `GEMINI_API_KEY` (marked `sync: false` in `render.yaml` on purpose — never commit real keys). Optionally adjust `ALLOWED_ORIGINS` once you have your Vercel URL.
4. Deploy. Confirm it's live: `https://<your-service>.onrender.com/health` → `{"status": "ok", ...}`.
   - Free-tier Render services spin down when idle — first request after inactivity can take ~30–50s to wake up. Fine for a hackathon demo, worth mentioning if judges test it cold.

### 2. Frontend → Vercel
1. Edit `frontend/config.js` and set `window.FANPULSE_API_BASE` to your live Render URL from step above.
2. In Vercel: **New Project → Import** this repo. `vercel.json` at the root already sets `outputDirectory: "frontend"`, so no extra config needed.
3. Deploy. Your frontend URL will be something like `https://fanpulse-ai.vercel.app`.
4. Go back to Render and update `ALLOWED_ORIGINS` to that exact Vercel URL (CORS will otherwise block it), then redeploy the backend.

### 3. Verify end-to-end
Open the Vercel URL, ask FanPulse a question, and check the crowd status panel loads. If the chat fails with a network error, it's almost always the `ALLOWED_ORIGINS` / `config.js` URL mismatch above.

## Alternative: Google Cloud stack

- Backend → Cloud Run (containerize with the included `requirements.txt`)
- Secrets → Secret Manager, injected as `GEMINI_API_KEY` env var
- Crowd feed → Pub/Sub ingesting turnstile/IoT events → Firestore → backend reads from Firestore instead of the in-memory dict
- Frontend → Firebase Hosting
