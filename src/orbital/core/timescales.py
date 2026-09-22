"""Julian dates from timezone-aware UTC datetimes."""
from __future__ import annotations

from datetime import datetime, timedelta

J2000_JD: float = 2451545.0
SECONDS_PER_DAY: float = 86400.0


def require_utc(epoch: datetime) -> datetime:
    """Reject naive datetimes rather than guess their time zone."""
    if epoch.tzinfo is None or epoch.utcoffset() != timedelta(0):
        raise ValueError("epoch must be a timezone-aware UTC datetime")
    return epoch


def julian_date(epoch: datetime) -> tuple[float, float]:
    """Julian date as ``(jd, fr)``: jd ends in .5, fr in [0, 1) (Fliegel-Van Flandern)."""
    e = require_utc(epoch)
    a = (14 - e.month) // 12
    y = e.year + 4800 - a
    m = e.month + 12 * a - 3
    jdn = e.day + (153 * m + 2) // 5 + 365 * y + y // 4 - y // 100 + y // 400 - 32045
    seconds = e.hour * 3600 + e.minute * 60 + e.second + e.microsecond * 1e-6
    return float(jdn) - 0.5, seconds / SECONDS_PER_DAY
