"""
Test suite for FanPulse AI backend.
Gemini calls are mocked so tests run offline, deterministically, and without
consuming API quota or requiring a real key.
"""
import sys
import os
import types as pytypes
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

os.environ.setdefault("GEMINI_API_KEY", "test-key-not-real")

import main  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

client = TestClient(main.app)


def _mock_gemini_response(text="Mocked assistant reply."):
    fake_response = MagicMock()
    fake_response.text = text
    return fake_response


def setup_function():
    """Reset rate limiter and inject a mock Gemini client before each test."""
    main._request_log.clear()
    main._client = MagicMock()
    main._client.models.generate_content.return_value = _mock_gemini_response()


# ---------------------------------------------------------------------------
# Health & metadata
# ---------------------------------------------------------------------------
def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_languages():
    resp = client.get("/languages")
    assert resp.status_code == 200
    assert "en" in resp.json()
    assert "hi" in resp.json()


# ---------------------------------------------------------------------------
# Crowd status (real-time decision support)
# ---------------------------------------------------------------------------
def test_crowd_status_all_zones():
    resp = client.get("/crowd-status")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 4
    for zone in data:
        assert zone["status"] in ("low", "moderate", "high", "critical")
        assert 0 <= zone["occupancy_pct"] <= 100


def test_crowd_status_single_zone():
    resp = client.get("/crowd-status/gate-a")
    assert resp.status_code == 200
    assert resp.json()["zone_id"] == "gate-a"


def test_crowd_status_unknown_zone_returns_404():
    resp = client.get("/crowd-status/gate-z")
    assert resp.status_code == 404


def test_congested_gate_recommends_alternative():
    # gate-a is seeded at 92% occupancy (critical) in main.py
    resp = client.get("/crowd-status/gate-a")
    body = resp.json()
    assert body["status"] == "critical"
    assert "Recommend" in body["recommendation"]


# ---------------------------------------------------------------------------
# Chat (multilingual assistant)
# ---------------------------------------------------------------------------
def test_chat_happy_path():
    resp = client.post("/chat", json={"message": "Where is Gate B?", "language": "en"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["reply"] == "Mocked assistant reply."
    assert body["language"] == "en"


def test_chat_rejects_unsupported_language():
    resp = client.post("/chat", json={"message": "Hola", "language": "xx"})
    assert resp.status_code == 422


def test_chat_rejects_empty_message():
    resp = client.post("/chat", json={"message": "   ", "language": "en"})
    assert resp.status_code == 422


def test_chat_rejects_oversized_message():
    resp = client.post("/chat", json={"message": "a" * 501, "language": "en"})
    assert resp.status_code == 422


def test_chat_returns_502_when_gemini_fails():
    main._client.models.generate_content.side_effect = RuntimeError("upstream error")
    resp = client.post("/chat", json={"message": "Where is Gate B?", "language": "en"})
    assert resp.status_code == 502


def test_chat_503_when_gemini_not_configured():
    main._client = None
    resp = client.post("/chat", json={"message": "Where is Gate B?", "language": "en"})
    assert resp.status_code == 503


# ---------------------------------------------------------------------------
# Accessibility
# ---------------------------------------------------------------------------
def test_simplify_endpoint():
    resp = client.post(
        "/accessibility/simplify",
        json={"text": "Patrons are advised egress via the northern concourse is temporarily suspended.",
              "target_language": "en"},
    )
    assert resp.status_code == 200
    assert resp.json()["reply"] == "Mocked assistant reply."


def test_simplify_rejects_bad_language():
    resp = client.post("/accessibility/simplify", json={"text": "hello", "target_language": "zz"})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Security: rate limiting
# ---------------------------------------------------------------------------
def test_rate_limit_blocks_excess_requests():
    for _ in range(main.RATE_LIMIT):
        resp = client.get("/crowd-status")
        assert resp.status_code == 200
    resp = client.get("/crowd-status")
    assert resp.status_code == 429


# ---------------------------------------------------------------------------
# Security: input validation guards against injection-style payloads
# ---------------------------------------------------------------------------
def test_chat_handles_special_characters_safely():
    resp = client.post(
        "/chat",
        json={"message": "<script>alert(1)</script> Where is the nearest exit?", "language": "en"},
    )
    # Should be accepted as plain text and passed to the model, not executed/interpreted
    assert resp.status_code == 200
