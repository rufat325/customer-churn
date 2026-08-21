import pytest

from src.security import limiter


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    """
    Give every test a fresh rate-limit budget.

    The limiter is a module-level singleton keyed by client IP, and every
    request from TestClient arrives from the same host. Without this, the
    API tests share one 120-request-per-minute allowance between them and
    start failing with 429s as the suite grows -- a failure that would look
    like a bug in whichever test happened to run last.
    """

    limiter.reset()
    yield
    limiter.reset()
