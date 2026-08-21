"""Rate limiting and optional authentication.

This service is deployed publicly so the dashboard can be opened from any
device. That is a defensible choice *here* because every customer in it is
synthetic -- there is no real person to expose. It would not be defensible
with real data, and the README says so rather than leaving the reader to
assume the deployment is a template.

What being public actually costs, and what this module does about it:

**Compute abuse.** An open endpoint is an open invitation to make someone
else's server do work. A per-IP sliding-window limit caps that.

**Model extraction.** Query a prediction endpoint enough times and you can
reconstruct a usable copy of the model behind it, or probe what its training
data looked like. Rate limiting is the cheap mitigation; a real service would
also authenticate. Here the model is a toy trained on invented data, so the
exposure is academic -- but the limit is what makes it academic rather than
merely unmeasured.

**Everything else** is handled by the service being strictly read-only. There
is no endpoint that writes, so there is nothing to corrupt.

Authentication is available and off by default. Set ``CHURN_PASSWORD`` to put
HTTP Basic in front of the whole site, or ``CHURN_API_KEY`` to require a header
on the scoring endpoints only. Off by default is the right default for a
portfolio demo whose entire purpose is being opened by strangers; both switches
exist so the same image can be deployed somewhere it is not.
"""

import base64
import binascii
import os
import secrets
import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request, status


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


# Requests per window per client IP. Generous enough that a human clicking
# around the dashboard never notices, tight enough that scraping the customer
# list or harvesting the model is slow and obvious.
RATE_LIMIT = _int_env("CHURN_RATE_LIMIT", 120)
RATE_WINDOW_SECONDS = _int_env("CHURN_RATE_WINDOW", 60)

# Optional credentials. Empty means the corresponding check is disabled.
PASSWORD = os.environ.get("CHURN_PASSWORD", "").strip()
USERNAME = os.environ.get("CHURN_USERNAME", "demo").strip()
API_KEY = os.environ.get("CHURN_API_KEY", "").strip()

# Paths that must stay reachable regardless, or the container health check
# and load balancers start failing for the wrong reason.
EXEMPT_PATHS = {"/health"}


class RateLimiter:
    """
    Fixed-memory sliding window, per client IP.

    In-process on purpose: one container, one counter. A multi-replica
    deployment needs shared state (Redis, or the load balancer's own
    limiter), because per-process counters let a client multiply their
    allowance by the number of replicas.
    """

    def __init__(self, limit: int = RATE_LIMIT, window: int = RATE_WINDOW_SECONDS):
        self.limit = limit
        self.window = window
        self._hits: dict[str, deque] = defaultdict(deque)

    def check(self, client: str, now: float | None = None) -> tuple[bool, int, float]:
        """Return ``(allowed, remaining, retry_after_seconds)``."""

        now = time.monotonic() if now is None else now
        hits = self._hits[client]

        cutoff = now - self.window

        while hits and hits[0] <= cutoff:
            hits.popleft()

        if len(hits) >= self.limit:
            return False, 0, max(0.0, hits[0] + self.window - now)

        hits.append(now)

        return True, self.limit - len(hits), 0.0

    def reset(self) -> None:
        self._hits.clear()

    def forget(self, client: str) -> None:
        self._hits.pop(client, None)


limiter = RateLimiter()


def client_ip(request: Request) -> str:
    """
    Best-effort client identity.

    Behind the reverse proxy the deployment uses, the socket peer is the
    proxy, so the real client is the first entry of X-Forwarded-For. That
    header is trivially spoofable by a direct caller, which is exactly why
    the container should not be reachable except through the proxy -- see
    DEPLOYMENT.md. Treating it as authoritative is a deliberate,
    documented trade, not an oversight.
    """

    forwarded = request.headers.get("x-forwarded-for")

    if forwarded:
        return forwarded.split(",")[0].strip()

    return request.client.host if request.client else "unknown"


def check_rate_limit(request: Request) -> tuple[int, float]:
    """Raise 429 if the caller is over the limit; else return headroom."""

    if request.url.path in EXEMPT_PATHS:
        return limiter.limit, 0.0

    allowed, remaining, retry_after = limiter.check(client_ip(request))

    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                f"Rate limit of {limiter.limit} requests per "
                f"{limiter.window}s exceeded. Retry in "
                f"{retry_after:.0f}s."
            ),
            headers={"Retry-After": str(int(retry_after) + 1)},
        )

    return remaining, retry_after


def _unauthorized(detail: str, basic: bool) -> HTTPException:
    headers = {"WWW-Authenticate": 'Basic realm="Churn Risk Console"'} if basic else {}

    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers=headers,
    )


def check_password(request: Request) -> None:
    """
    HTTP Basic over the whole site, when CHURN_PASSWORD is set.

    Basic rather than a login form because it works everywhere without
    session state: a browser prompts for it, curl takes ``-u``, and there is
    no cookie to secure. It sends credentials base64-encoded, not encrypted,
    so it is only safe behind HTTPS -- which is why the deployment terminates
    TLS at the proxy.
    """

    if not PASSWORD:
        return

    if request.url.path in EXEMPT_PATHS:
        return

    header = request.headers.get("authorization", "")

    if not header[:6].lower() == "basic ":
        raise _unauthorized("Authentication required.", basic=True)

    try:
        decoded = base64.b64decode(header[6:], validate=True).decode("utf-8")
        username, _, password = decoded.partition(":")
    except (binascii.Error, UnicodeDecodeError):
        raise _unauthorized("Malformed credentials.", basic=True) from None

    # Both compared in constant time, and both compared unconditionally, so
    # a wrong username costs the same as a wrong password.
    user_ok = secrets.compare_digest(username, USERNAME)
    password_ok = secrets.compare_digest(password, PASSWORD)

    if not (user_ok and password_ok):
        raise _unauthorized("Invalid credentials.", basic=True)


def check_api_key(request: Request) -> None:
    """Require X-API-Key on scoring endpoints, when CHURN_API_KEY is set."""

    if not API_KEY:
        return

    supplied = request.headers.get("x-api-key", "")

    # Constant-time compare: a naive == leaks how many leading characters
    # matched through timing, which is enough to recover a key byte by byte.
    if not secrets.compare_digest(supplied, API_KEY):
        raise _unauthorized("Valid X-API-Key header required.", basic=False)


def auth_enabled() -> dict:
    """What protection is actually switched on, for /health to report."""

    return {
        "password_protected": bool(PASSWORD),
        "api_key_required": bool(API_KEY),
        "rate_limit_per_window": limiter.limit,
        "rate_window_seconds": limiter.window,
    }
