"""CelesTrak TLE retrieval with local caching.

CelesTrak refreshes its GP data every ~2 hours and rate-limits/blocks
aggressive clients, so every pull goes through the on-disk cache in
`data/tle_cache/` and refetches only when the cached copy is stale.
"""
from __future__ import annotations

import time
from pathlib import Path

import requests

from tle import TLE, parse_tle_file

GP_URL = "https://celestrak.org/NORAD/elements/gp.php"
CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "tle_cache"
DEFAULT_MAX_AGE_HOURS = 2.0
USER_AGENT = "orbital-analysis/0.1 (conjunction assessment portfolio project)"


def _cache_path(key: str, value: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(value))
    return CACHE_DIR / f"{key}_{safe}.tle"


def _is_fresh(path: Path, max_age_hours: float) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    return (time.time() - path.stat().st_mtime) < max_age_hours * 3600.0


def fetch_gp(
    catnr: int | None = None,
    group: str | None = None,
    max_age_hours: float = DEFAULT_MAX_AGE_HOURS,
    force: bool = False,
    timeout: float = 30.0,
) -> str:
    """Return raw TLE text for one satellite (`catnr`) or a group (`group`).

    Served from cache unless the cached copy is older than `max_age_hours`.
    If the network call fails but a stale cache exists, the stale copy is
    returned rather than failing outright.
    """
    if (catnr is None) == (group is None):
        raise ValueError("pass exactly one of catnr or group")

    key, value = ("CATNR", catnr) if catnr is not None else ("GROUP", group)
    path = _cache_path(key.lower(), value)

    if not force and _is_fresh(path, max_age_hours):
        return path.read_text()

    params = {key: str(value), "FORMAT": "tle"}
    try:
        response = requests.get(
            GP_URL, params=params, timeout=timeout, headers={"User-Agent": USER_AGENT}
        )
        response.raise_for_status()
        text = response.text
    except requests.RequestException:
        if path.exists():
            return path.read_text()   # stale beats nothing
        raise

    # CelesTrak returns 200 with a plaintext message, not an error status,
    # when a query matches nothing or the client is being throttled.
    if not text.lstrip().startswith(("0 ", "1 ")) and "\n1 " not in text:
        raise RuntimeError(f"CelesTrak returned no TLE data: {text.strip()[:200]!r}")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return text


def fetch_tles(
    catnr: int | None = None,
    group: str | None = None,
    **kwargs,
) -> list[TLE]:
    """Fetch and parse. See `fetch_gp` for caching behavior."""
    return parse_tle_file(fetch_gp(catnr=catnr, group=group, **kwargs))


def fetch_tle(catnr: int, **kwargs) -> TLE:
    """Fetch a single satellite's TLE by NORAD catalog number."""
    results = fetch_tles(catnr=catnr, **kwargs)
    if len(results) != 1:
        raise RuntimeError(f"expected 1 TLE for CATNR {catnr}, got {len(results)}")
    return results[0]


ISS_CATNR = 25544
