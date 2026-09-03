"""Space-Track.org API client: cookie session, rate limiting, on-disk cache.

Space-Track enforces 30 requests/minute and 300/hour account-wide, with
per-class throttles on top, and suspends accounts that violate them. Every
request here goes through a conservative limiter and the cache in
`data/spacetrack_cache/`.

Credentials come from `.env` (gitignored) -- never hardcode them.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from collections import deque
from pathlib import Path

import requests

BASE = "https://www.space-track.org"
LOGIN_URL = f"{BASE}/ajaxauth/login"
QUERY_URL = f"{BASE}/basicspacedata/query"

ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT / ".env"
CACHE_DIR = ROOT / "data" / "spacetrack_cache"

# Held under the published 30/min and 300/hr so bursts can't trip suspension.
MAX_PER_MINUTE = 20
MAX_PER_HOUR = 200
USER_AGENT = "orbital-analysis/0.1 (conjunction assessment portfolio project)"


def load_env(path: Path = ENV_PATH) -> dict[str, str]:
    """Minimal KEY=VALUE parser, so the project doesn't need python-dotenv."""
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        env[key.strip()] = value.strip().strip("\"'")
    return env


def credentials() -> tuple[str, str]:
    """Read credentials from the environment, falling back to `.env`."""
    env = load_env()
    identity = os.environ.get("SPACETRACK_IDENTITY") or env.get("SPACETRACK_IDENTITY")
    password = os.environ.get("SPACETRACK_PASSWORD") or env.get("SPACETRACK_PASSWORD")
    if not identity or not password:
        raise RuntimeError(
            "Space-Track credentials missing. Copy .env.example to .env and fill "
            "in SPACETRACK_IDENTITY and SPACETRACK_PASSWORD."
        )
    if password.startswith("your-"):
        raise RuntimeError("`.env` still holds the placeholder password from .env.example.")
    return identity, password


class RateLimiter:
    """Sliding-window limiter over both the per-minute and per-hour caps."""

    def __init__(self, per_minute: int = MAX_PER_MINUTE, per_hour: int = MAX_PER_HOUR):
        self.per_minute, self.per_hour = per_minute, per_hour
        self.calls: deque[float] = deque()

    def acquire(self, verbose: bool = True) -> None:
        while True:
            now = time.time()
            while self.calls and now - self.calls[0] > 3600:
                self.calls.popleft()
            in_minute = sum(1 for t in self.calls if now - t <= 60)
            waits = []
            if in_minute >= self.per_minute:
                waits.append(60 - (now - [t for t in self.calls if now - t <= 60][0]))
            if len(self.calls) >= self.per_hour:
                waits.append(3600 - (now - self.calls[0]))
            if not waits:
                self.calls.append(now)
                return
            delay = max(0.1, max(waits))
            if verbose:
                print(f"  [rate limit] sleeping {delay:.1f}s")
            time.sleep(delay)


class SpaceTrack:
    """Authenticated Space-Track session.

    Usage:
        with SpaceTrack() as st:
            rows = st.query("gp", NORAD_CAT_ID=25544)
    """

    def __init__(self, timeout: float = 60.0):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        self.limiter = RateLimiter()
        self._logged_in = False

    def __enter__(self) -> "SpaceTrack":
        self.login()
        return self

    def __exit__(self, *exc) -> None:
        self.logout()

    def login(self) -> None:
        if self._logged_in:
            return
        identity, password = credentials()
        self.limiter.acquire()
        response = self.session.post(
            LOGIN_URL,
            data={"identity": identity, "password": password},
            timeout=self.timeout,
        )
        response.raise_for_status()
        # A failed login still returns 200, with an error message in the body.
        if "failed" in response.text.lower() or "invalid" in response.text.lower():
            raise RuntimeError(f"Space-Track login failed: {response.text.strip()[:200]}")
        self._logged_in = True

    def logout(self) -> None:
        if self._logged_in:
            try:
                self.session.get(f"{BASE}/ajaxauth/logout", timeout=self.timeout)
            except requests.RequestException:
                pass
            self._logged_in = False

    def _build_path(self, request_class: str, predicates: dict, fmt: str) -> str:
        parts = [QUERY_URL, "class", request_class]
        for key, value in predicates.items():
            if value is None:
                continue
            if isinstance(value, (list, tuple)):
                value = ",".join(str(v) for v in value)
            parts += [str(key), str(value)]
        parts += ["format", fmt]
        return "/".join(parts)

    def query(
        self,
        request_class: str,
        fmt: str = "json",
        max_age_hours: float = 6.0,
        force: bool = False,
        **predicates,
    ):
        """Run one query, served from cache when a fresh copy exists.

        Predicates are passed as keyword args in Space-Track's URL style, e.g.
        `query("gp", NORAD_CAT_ID=25544, orderby="EPOCH desc", limit=1)`.
        Returns parsed JSON for fmt="json", otherwise raw text.
        """
        url = self._build_path(request_class, predicates, fmt)
        digest = hashlib.sha256(url.encode()).hexdigest()[:16]
        path = CACHE_DIR / f"{request_class}_{digest}.{fmt}"

        if not force and path.exists() and path.stat().st_size > 0:
            if (time.time() - path.stat().st_mtime) < max_age_hours * 3600:
                text = path.read_text()
                return json.loads(text) if fmt == "json" else text

        self.login()
        self.limiter.acquire()
        response = self.session.get(url, timeout=self.timeout)
        if response.status_code == 401:
            # Session cookies expire; re-authenticate once and retry.
            self._logged_in = False
            self.login()
            self.limiter.acquire()
            response = self.session.get(url, timeout=self.timeout)
        response.raise_for_status()
        text = response.text

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return json.loads(text) if fmt == "json" else text
