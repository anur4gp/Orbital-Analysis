"""CelesTrak GP fetch, cached in ``data/tle_cache/`` (CelesTrak refreshes every ~2 h)."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import requests

from orbital.paths import DATA_DIR
from orbital.sgp4tools.tle import TLE, parse_tle_file

GP_URL = "https://celestrak.org/NORAD/elements/gp.php"
CACHE_DIR = DATA_DIR / "tle_cache"
DEFAULT_MAX_AGE_HOURS = 2.0
USER_AGENT = "orbital-analysis/0.1 (conjunction assessment portfolio project)"


def _cache_path(key: str, value: str | int) -> Path:
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
    """Raw TLE text for one ``catnr`` or one ``group`` (exactly one).

    Served from cache when fresh; falls back to a stale cache on network error.
    """
    key: str
    value: str | int
    if catnr is not None and group is None:
        key, value = "CATNR", catnr
    elif group is not None and catnr is None:
        key, value = "GROUP", group
    else:
        raise ValueError("pass exactly one of catnr or group")

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
            return path.read_text()
        raise

    # No-match and throttling both come back as HTTP 200 with a plaintext message.
    if not text.lstrip().startswith(("0 ", "1 ")) and "\n1 " not in text:
        raise RuntimeError(f"CelesTrak returned no TLE data: {text.strip()[:200]!r}")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return text


def fetch_tles(
    catnr: int | None = None,
    group: str | None = None,
    **kwargs: Any,
) -> list[TLE]:
    """Fetch and parse element sets via :func:`fetch_gp`."""
    return parse_tle_file(fetch_gp(catnr=catnr, group=group, **kwargs))


def fetch_tle(catnr: int, **kwargs: Any) -> TLE:
    """Fetch a single satellite's TLE by NORAD catalog number."""
    results = fetch_tles(catnr=catnr, **kwargs)
    if len(results) != 1:
        raise RuntimeError(f"expected 1 TLE for CATNR {catnr}, got {len(results)}")
    return results[0]


ISS_CATNR = 25544
