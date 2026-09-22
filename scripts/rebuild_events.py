"""Rebuild cdm_public conjunction geometry from TLEs and compare miss distances.

Run: python scripts/rebuild_events.py
"""
from __future__ import annotations

import numpy as np

from orbital.conjunction.events import build_geometry, fetch_tles_for, load_events, tractable
from orbital.sgp4tools.spacetrack import SpaceTrack

N_EVENTS = 12


def main() -> int:
    with SpaceTrack() as st:
        events = load_events(st, limit=500)
        usable = tractable(events)
        print(f"{len(events)} events, {len(usable)} with both objects in the public catalog")

        subset = usable[:N_EVENTS]
        ids = {i for e in subset for i in e.object_ids}
        print(f"fetching TLEs for {len(ids)} objects in one request...")
        tles = fetch_tles_for(st, ids)
        print(f"got {len(tles)} TLEs\n")

    header = f"{'CDM':>11}  {'objects':>13}  {'rebuilt':>9}  {'reported':>9}  {'err':>8}  {'vrel':>7}  {'TLE age':>8}"
    print(header)
    print("-" * len(header))

    errors, geoms = [], []
    for event in subset:
        g = build_geometry(event, tles)
        if g is None:
            print(f"{event.cdm_id:>11}  {event.sat1_id:>6}/{event.sat2_id:<6}  (TLE missing)")
            continue
        geoms.append(g)
        rep = g.reported_miss_km
        err = g.miss_error_km
        if err is not None:
            errors.append(err)
        age = max(abs(g.tle_age1_days), abs(g.tle_age2_days))
        print(
            f"{event.cdm_id:>11}  {event.sat1_id:>6}/{event.sat2_id:<6}  "
            f"{g.miss_km:>8.3f}  {rep:>9.3f}  {err:>+8.3f}  "
            f"{g.relative_velocity_km_s:>6.2f}  {age:>7.2f}d"
        )

    if errors:
        a = np.abs(errors)
        print(f"\nrebuilt vs reported miss distance, {len(errors)} events (km):")
        print(f"  mean |error|   {a.mean():.3f}")
        print(f"  median |error| {np.median(a):.3f}")
        print(f"  min / max      {a.min():.3f} / {a.max():.3f}")
        print("\nAll distances km; vrel km/s.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
