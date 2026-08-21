import base64

import pytest
from fastapi.testclient import TestClient

import src.security as security
from src.api import app
from src.security import RateLimiter, auth_enabled


client = TestClient(app)


# ---------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------


def test_limiter_allows_up_to_the_limit_then_blocks():
    limiter = RateLimiter(limit=3, window=60)

    assert [limiter.check("1.1.1.1")[0] for _ in range(3)] == [True] * 3
    assert limiter.check("1.1.1.1")[0] is False


def test_limiter_tracks_clients_independently():
    limiter = RateLimiter(limit=2, window=60)

    limiter.check("1.1.1.1")
    limiter.check("1.1.1.1")

    assert limiter.check("1.1.1.1")[0] is False
    assert limiter.check("2.2.2.2")[0] is True


def test_limiter_window_slides():
    limiter = RateLimiter(limit=2, window=10)

    # Drive the clock explicitly rather than sleeping.
    assert limiter.check("1.1.1.1", now=0.0)[0] is True
    assert limiter.check("1.1.1.1", now=1.0)[0] is True
    assert limiter.check("1.1.1.1", now=2.0)[0] is False

    # Once the first hits age out of the window, capacity returns.
    assert limiter.check("1.1.1.1", now=11.5)[0] is True


def test_limiter_reports_remaining_and_retry_after():
    limiter = RateLimiter(limit=2, window=10)

    _, remaining, _ = limiter.check("1.1.1.1", now=0.0)
    assert remaining == 1

    limiter.check("1.1.1.1", now=1.0)

    allowed, remaining, retry_after = limiter.check("1.1.1.1", now=2.0)

    assert allowed is False
    assert remaining == 0
    assert retry_after == pytest.approx(8.0)


def test_limiter_memory_does_not_grow_without_bound():
    limiter = RateLimiter(limit=5, window=10)

    for tick in range(50):
        limiter.check("1.1.1.1", now=float(tick))

    # Entries older than the window are discarded, so the deque stays small
    # rather than accumulating one entry per request forever.
    assert len(limiter._hits["1.1.1.1"]) <= 5


# ---------------------------------------------------------------------
# Enforcement through the app
# ---------------------------------------------------------------------


def test_responses_carry_remaining_budget():
    response = client.get("/health")

    assert response.status_code == 200

    # /health is exempt, so it reports the full allowance.
    assert int(response.headers["X-RateLimit-Remaining"]) == security.limiter.limit


def test_exceeding_the_limit_returns_429_with_retry_after(monkeypatch):
    monkeypatch.setattr(security.limiter, "limit", 5)

    statuses = [client.get("/model").status_code for _ in range(7)]

    assert statuses[:5] == [200] * 5
    assert statuses[5:] == [429, 429]

    blocked = client.get("/model")

    assert blocked.status_code == 429
    assert int(blocked.headers["Retry-After"]) >= 1
    assert "Rate limit" in blocked.json()["detail"]


def test_health_stays_reachable_when_rate_limited(monkeypatch):
    # The container health check must not be throttled, or the orchestrator
    # kills a container that is merely popular.
    monkeypatch.setattr(security.limiter, "limit", 2)

    for _ in range(5):
        client.get("/model")

    assert client.get("/model").status_code == 429
    assert client.get("/health").status_code == 200


def test_forwarded_header_separates_clients(monkeypatch):
    monkeypatch.setattr(security.limiter, "limit", 2)

    for _ in range(3):
        client.get("/model", headers={"X-Forwarded-For": "10.0.0.1"})

    assert client.get(
        "/model", headers={"X-Forwarded-For": "10.0.0.1"}
    ).status_code == 429

    # A different client behind the same proxy is unaffected.
    assert client.get(
        "/model", headers={"X-Forwarded-For": "10.0.0.2"}
    ).status_code == 200


# ---------------------------------------------------------------------
# Optional authentication
# ---------------------------------------------------------------------


def test_auth_is_off_by_default():
    # The demo is meant to be openable by strangers; both switches ship off.
    status = auth_enabled()

    assert status["password_protected"] is False
    assert status["api_key_required"] is False
    assert status["rate_limit_per_window"] > 0


def test_health_reports_active_protection():
    body = client.get("/health").json()

    assert "protection" in body
    assert set(body["protection"]) == {
        "password_protected",
        "api_key_required",
        "rate_limit_per_window",
        "rate_window_seconds",
    }


def _basic(username: str, password: str) -> dict:
    token = base64.b64encode(f"{username}:{password}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


def test_password_blocks_and_admits(monkeypatch):
    monkeypatch.setattr(security, "PASSWORD", "s3cret")
    monkeypatch.setattr(security, "USERNAME", "demo")

    unauthenticated = client.get("/model")
    assert unauthenticated.status_code == 401
    assert "Basic" in unauthenticated.headers.get("WWW-Authenticate", "")

    assert client.get("/model", headers=_basic("demo", "wrong")).status_code == 401
    assert client.get("/model", headers=_basic("nobody", "s3cret")).status_code == 401
    assert client.get("/model", headers=_basic("demo", "s3cret")).status_code == 200


def test_password_rejects_malformed_credentials(monkeypatch):
    monkeypatch.setattr(security, "PASSWORD", "s3cret")

    assert client.get(
        "/model", headers={"Authorization": "Basic !!!not-base64!!!"}
    ).status_code == 401

    assert client.get(
        "/model", headers={"Authorization": "Bearer something"}
    ).status_code == 401


def test_password_does_not_lock_out_the_health_check(monkeypatch):
    monkeypatch.setattr(security, "PASSWORD", "s3cret")

    assert client.get("/health").status_code == 200


def test_api_key_guards_scoring_endpoints(monkeypatch):
    monkeypatch.setattr(security, "API_KEY", "key-123")

    payload = {
        "age": 35, "country": "DE", "total_orders": 2, "total_spent": 150.0,
        "days_since_last_order": 75, "has_previous_order": 1, "total_events": 5,
        "add_to_cart_count": 1, "checkout_count": 0, "login_count": 2,
        "product_view_count": 2, "tenure_days": 180, "events_last_30_days": 0,
    }

    assert client.post("/predict", json=payload).status_code == 401
    assert client.post(
        "/predict", json=payload, headers={"X-API-Key": "wrong"}
    ).status_code == 401

    ok = client.post("/predict", json=payload, headers={"X-API-Key": "key-123"})
    assert ok.status_code == 200
    assert "churn_probability" in ok.json()


def test_api_key_does_not_guard_the_dashboard(monkeypatch):
    # The key protects scoring, not browsing. Locking the read-only views
    # behind it would make the public demo pointless while doing nothing
    # for the endpoint that actually costs compute.
    monkeypatch.setattr(security, "API_KEY", "key-123")

    assert client.get("/model").status_code == 200
