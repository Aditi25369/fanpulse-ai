"""
FanPulse AI — Smart Stadium & Tournament Operations Assistant
FIFA World Cup 2026

A GenAI-powered backend that helps fans, volunteers, and venue staff with:
  - Multilingual conversational assistance (navigation, amenities, transport, rules)
  - Real-time crowd density monitoring & AI-generated crowd-flow recommendations
  - Gate/route suggestions to reduce bottlenecks
  - Accessibility mode: plain-language + screen-reader-friendly responses
  - Sustainability nudges (transit, waste, water refill points)

Built with FastAPI + Google Gemini (google-genai SDK).
"""

import os
import time
import logging
from collections import defaultdict, deque
from typing import Optional, Literal

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from google import genai
from google.genai import types

# ---------------------------------------------------------------------------
# Logging (no PII, no raw prompts containing personal data logged at INFO)
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("fanpulse")

# ---------------------------------------------------------------------------
# Configuration — secrets from environment only, never hardcoded
# ---------------------------------------------------------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "http://localhost:5173").split(",")

if not GEMINI_API_KEY:
    # Fail loudly at startup rather than silently degrading in production
    logger.warning("GEMINI_API_KEY not set — /chat and /accessibility/simplify will return 503")

_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

app = FastAPI(
    title="FanPulse AI",
    description="GenAI-powered smart stadium assistant for FIFA World Cup 2026",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    allow_credentials=False,
)

# ---------------------------------------------------------------------------
# Very small in-memory rate limiter (per-IP sliding window).
# For production, swap for Redis-backed limiter — noted in README.
# ---------------------------------------------------------------------------
RATE_LIMIT = 20          # requests
RATE_WINDOW = 60         # seconds
_request_log: dict[str, deque] = defaultdict(deque)


def rate_limit(request: Request):
    ip = request.client.host if request.client else "unknown"
    now = time.time()
    log = _request_log[ip]
    while log and now - log[0] > RATE_WINDOW:
        log.popleft()
    if len(log) >= RATE_LIMIT:
        raise HTTPException(status_code=429, detail="Too many requests. Please slow down.")
    log.append(now)


# ---------------------------------------------------------------------------
# Static reference data — in production these would come from a stadium ops
# DB / IoT feed (turnstile counters, CCTV crowd-density models, GPS transit
# feeds). Kept in-memory here so the challenge submission is self-contained
# and runnable without external infra.
# ---------------------------------------------------------------------------
ZONES = {
    "gate-a": {"name": "Gate A — North Concourse", "capacity": 8000},
    "gate-b": {"name": "Gate B — East Concourse", "capacity": 6000},
    "gate-c": {"name": "Gate C — South Concourse", "capacity": 7000},
    "gate-d": {"name": "Gate D — West Concourse (Accessible Entry)", "capacity": 3000},
}

# simulated live occupancy (0-1 fraction of capacity); would be fed by real sensors
_live_occupancy = {"gate-a": 0.92, "gate-b": 0.55, "gate-c": 0.71, "gate-d": 0.30}

SUPPORTED_LANGUAGES = {
    "en": "English", "es": "Spanish", "pt": "Portuguese", "fr": "French",
    "hi": "Hindi", "ar": "Arabic", "de": "German", "ja": "Japanese",
}

SYSTEM_INSTRUCTION = (
    "You are FanPulse, an official-style stadium assistant for the FIFA World Cup 2026. "
    "You help fans with navigation, gates, seating, transport, accessibility, lost & found, "
    "food/water points, and sustainability tips. Be concise (max 4 sentences), factual, and "
    "friendly. Never invent specific gate numbers, ticket prices, or player/team information — "
    "if asked something you cannot verify, say so and suggest asking on-site staff or the "
    "official FIFA app. Do not discuss anything unrelated to stadium/tournament operations."
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=500)
    language: str = Field(default="en")
    accessibility_mode: bool = Field(default=False, description="Plain-language, short-sentence output")

    @field_validator("language")
    @classmethod
    def check_language(cls, v):
        if v not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language code '{v}'. Supported: {list(SUPPORTED_LANGUAGES)}")
        return v

    @field_validator("message")
    @classmethod
    def strip_message(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("Message cannot be empty")
        return v


class ChatResponse(BaseModel):
    reply: str
    language: str


class CrowdStatus(BaseModel):
    zone_id: str
    name: str
    occupancy_pct: float
    status: Literal["low", "moderate", "high", "critical"]
    recommendation: str


class SimplifyRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)
    target_language: str = Field(default="en")

    @field_validator("target_language")
    @classmethod
    def check_lang(cls, v):
        if v not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language code '{v}'")
        return v


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _require_gemini():
    if _client is None:
        raise HTTPException(status_code=503, detail="AI service not configured (missing GEMINI_API_KEY)")


def _occupancy_status(frac: float) -> str:
    if frac < 0.5:
        return "low"
    if frac < 0.75:
        return "moderate"
    if frac < 0.9:
        return "high"
    return "critical"


def _crowd_recommendation(zone_id: str, frac: float) -> str:
    status = _occupancy_status(frac)
    if status in ("low", "moderate"):
        return f"{ZONES[zone_id]['name']} is flowing normally. No rerouting needed."
    # find least busy alternative gate
    alt = min(_live_occupancy, key=lambda z: _live_occupancy[z] if z != zone_id else 2)
    return (
        f"{ZONES[zone_id]['name']} is congested. Recommend directing fans to "
        f"{ZONES[alt]['name']} ({_live_occupancy[alt]*100:.0f}% full) where possible."
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "gemini_configured": _client is not None}


@app.get("/languages")
def languages():
    return SUPPORTED_LANGUAGES


@app.get("/crowd-status", response_model=list[CrowdStatus])
def crowd_status(_: None = Depends(rate_limit)):
    """Real-time (simulated) crowd density across all gates with AI-style flow recommendations."""
    results = []
    for zone_id, meta in ZONES.items():
        frac = _live_occupancy[zone_id]
        results.append(CrowdStatus(
            zone_id=zone_id,
            name=meta["name"],
            occupancy_pct=round(frac * 100, 1),
            status=_occupancy_status(frac),
            recommendation=_crowd_recommendation(zone_id, frac),
        ))
    return results


@app.get("/crowd-status/{zone_id}", response_model=CrowdStatus)
def crowd_status_single(zone_id: str, _: None = Depends(rate_limit)):
    if zone_id not in ZONES:
        raise HTTPException(status_code=404, detail="Unknown zone_id")
    frac = _live_occupancy[zone_id]
    return CrowdStatus(
        zone_id=zone_id,
        name=ZONES[zone_id]["name"],
        occupancy_pct=round(frac * 100, 1),
        status=_occupancy_status(frac),
        recommendation=_crowd_recommendation(zone_id, frac),
    )


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, _: None = Depends(rate_limit)):
    _require_gemini()
    style_note = (
        "Use very short, simple sentences and avoid jargon (accessibility mode is ON)."
        if req.accessibility_mode else ""
    )
    prompt = (
        f"Respond in {SUPPORTED_LANGUAGES[req.language]}. {style_note}\n"
        f"Fan question: {req.message}"
    )
    try:
        response = _client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                max_output_tokens=300,
                temperature=0.4,
            ),
        )
        reply = (response.text or "").strip() or "Sorry, I couldn't generate a response. Please ask on-site staff."
    except Exception as exc:  # noqa: BLE001 — external API call, convert to safe HTTP error
        logger.error("Gemini call failed: %s", exc)
        raise HTTPException(status_code=502, detail="AI assistant is temporarily unavailable")
    return ChatResponse(reply=reply, language=req.language)


@app.post("/accessibility/simplify", response_model=ChatResponse)
def simplify(req: SimplifyRequest, _: None = Depends(rate_limit)):
    """Rewrite venue announcements/signage text into plain language + target language."""
    _require_gemini()
    prompt = (
        f"Rewrite the following stadium announcement in very simple, plain "
        f"{SUPPORTED_LANGUAGES[req.target_language]} suitable for a screen reader "
        f"and for fans with cognitive or hearing accessibility needs. Keep all factual "
        f"details (times, gates, directions) unchanged:\n\n{req.text}"
    )
    try:
        response = _client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(max_output_tokens=300, temperature=0.2),
        )
        reply = (response.text or "").strip() or req.text
    except Exception as exc:  # noqa: BLE001
        logger.error("Gemini call failed: %s", exc)
        raise HTTPException(status_code=502, detail="AI assistant is temporarily unavailable")
    return ChatResponse(reply=reply, language=req.target_language)
