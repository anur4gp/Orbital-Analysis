"""Propagate the ISS and sanity-check altitude, period and inclination.

Run: python scripts/validate_iss.py
"""
from __future__ import annotations

import numpy as np

from orbital.sgp4tools.celestrak import ISS_CATNR, fetch_tle
from orbital.sgp4tools.propagation import (
    altitude_km,
    apsides_km,
    period_minutes,
    propagate_series,
    radius_km,
    satrec_from_tle,
    semi_major_axis_km,
)

DAYS = 3


def main() -> int:
    t = fetch_tle(ISS_CATNR)
    sat = satrec_from_tle(t)

    print(f"{t.name}  (NORAD {t.catalog_number}, {t.international_designator})")
    print(f"  epoch            {t.epoch:%Y-%m-%d %H:%M:%S} UTC  ({t.age_days():+.2f} d old)")
    print(f"  inclination      {t.inclination:.4f} deg")
    print(f"  eccentricity     {t.eccentricity:.7f}")
    print(f"  semi-major axis  {semi_major_axis_km(sat):.2f} km")
    print(f"  period           {period_minutes(sat):.2f} min")
    peri, apo = apsides_km(sat)
    print(f"  perigee/apogee   {peri:.1f} / {apo:.1f} km altitude")

    times, r, v, _ = propagate_series(
        sat, t.epoch, duration_minutes=DAYS * 1440, step_minutes=1.0
    )
    alt, speed = altitude_km(r), np.linalg.norm(v, axis=1)
    print(f"\npropagated {DAYS} d from epoch, {len(times)} steps, 1 min cadence")
    print(f"  altitude   {alt.min():.1f} - {alt.max():.1f} km   (mean {alt.mean():.1f})")
    print(f"  speed      {speed.min():.3f} - {speed.max():.3f} km/s")

    # Empirical period from successive radius minima.
    rad = radius_km(r)
    minima = np.flatnonzero((rad[1:-1] < rad[:-2]) & (rad[1:-1] < rad[2:])) + 1
    empirical = np.diff(minima).mean() if len(minima) > 2 else float("nan")
    print(f"  revolutions in window   {len(minima)}")
    print(f"  empirical period        {empirical:.2f} min (from radius minima)")

    checks = [
        ("altitude in 380-440 km", 380 <= alt.mean() <= 440),
        ("period in 92-93 min", 92.0 <= period_minutes(sat) <= 93.0),
        ("empirical period matches elements", abs(empirical - period_minutes(sat)) < 1.0),
        ("speed in 7.6-7.8 km/s", 7.6 <= speed.mean() <= 7.8),
        ("inclination ~51.6 deg", abs(t.inclination - 51.6) < 0.2),
        ("near-circular (e < 0.01)", t.eccentricity < 0.01),
        ("TLE fresher than 7 d", abs(t.age_days()) < 7),
    ]
    print()
    for label, ok in checks:
        print(("  PASS  " if ok else "  FAIL  ") + label)

    failed = [label for label, ok in checks if not ok]
    print("\n" + ("all checks passed" if not failed else f"{len(failed)} check(s) failed"))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
