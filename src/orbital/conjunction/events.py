"""cdm_public screening records and their geometry rebuilt from TLEs via SGP4."""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np

from orbital.sgp4tools.propagation import propagate, satrec_from_tle
from orbital.sgp4tools.spacetrack import SpaceTrack
from orbital.sgp4tools.tle import TLE, parse_tle


def _parse_tca(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "")).replace(tzinfo=UTC)


def _to_float(value: Any) -> float | None:
    """Parse a Space-Track field, returning None when it is absent or blank."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass
class Event:
    """One screening record from cdm_public."""

    cdm_id: int
    tca: datetime
    pc: float | None            # 18 SDS collision probability
    min_rng_m: float | None     # reported miss distance, metres
    sat1_id: int
    sat2_id: int
    sat1_name: str
    sat2_name: str
    sat1_type: str
    sat2_type: str
    sat1_rcs: str               # RCS size class: SMALL / MEDIUM / LARGE
    sat2_rcs: str
    sat1_excl_vol: float | None
    sat2_excl_vol: float | None

    @property
    def object_ids(self) -> tuple[int, int]:
        """NORAD ids of the two objects, primary first."""
        return self.sat1_id, self.sat2_id

    @property
    def combined_excl_vol_km(self) -> float | None:
        """Sum of exclusion volumes, km (a screening volume, not a hard-body radius)."""
        if self.sat1_excl_vol is None or self.sat2_excl_vol is None:
            return None
        return self.sat1_excl_vol + self.sat2_excl_vol


def parse_event(row: dict) -> Event:
    """Build an :class:`Event` from one cdm_public JSON row."""
    return Event(
        cdm_id=int(row["CDM_ID"]),
        tca=_parse_tca(row["TCA"]),
        pc=_to_float(row.get("PC")),
        min_rng_m=_to_float(row.get("MIN_RNG")),
        sat1_id=int(row["SAT_1_ID"]),
        sat2_id=int(row["SAT_2_ID"]),
        sat1_name=(row.get("SAT_1_NAME") or "").strip(),
        sat2_name=(row.get("SAT_2_NAME") or "").strip(),
        sat1_type=(row.get("SAT1_OBJECT_TYPE") or "").strip(),
        sat2_type=(row.get("SAT2_OBJECT_TYPE") or "").strip(),
        sat1_rcs=(row.get("SAT1_RCS") or "").strip(),
        sat2_rcs=(row.get("SAT2_RCS") or "").strip(),
        sat1_excl_vol=_to_float(row.get("SAT_1_EXCL_VOL")),
        sat2_excl_vol=_to_float(row.get("SAT_2_EXCL_VOL")),
    )


def load_events(st: SpaceTrack, limit: int = 500, **kwargs: Any) -> list[Event]:
    """Most recent cdm_public records, newest first."""
    rows = st.query("cdm_public", orderby="TCA desc", limit=limit, **kwargs)
    return [parse_event(r) for r in rows]


def tractable(events: list[Event]) -> list[Event]:
    """Drop events involving analyst objects (UNKNOWN type, six-digit ids), which have no public TLEs."""
    return [
        e for e in events
        if "UNKNOWN" not in (e.sat1_type, e.sat2_type)
        and max(e.object_ids) < 100000
    ]


def fetch_tles_for(
    st: SpaceTrack, norad_ids: Iterable[int], **kwargs: Any
) -> dict[int, TLE]:
    """Latest TLE per catalog number, fetched in a single gp request."""
    ids = sorted({int(i) for i in norad_ids})
    rows = st.query(
        "gp",
        NORAD_CAT_ID=",".join(str(i) for i in ids),
        orderby="NORAD_CAT_ID",
        **kwargs,
    )
    out: dict[int, TLE] = {}
    for row in rows:
        line1, line2 = row.get("TLE_LINE1"), row.get("TLE_LINE2")
        if not line1 or not line2:
            continue
        t = parse_tle(line1, line2, name=(row.get("OBJECT_NAME") or "").strip())
        out[t.catalog_number] = t
    return out


@dataclass
class Geometry:
    """Rebuilt encounter geometry at TCA, from SGP4-propagated TLEs."""

    event: Event
    r1: np.ndarray
    v1: np.ndarray
    r2: np.ndarray
    v2: np.ndarray
    tle_age1_days: float
    tle_age2_days: float

    @property
    def miss_km(self) -> float:
        """Rebuilt miss distance at TCA, km."""
        return float(np.linalg.norm(self.r1 - self.r2))

    @property
    def relative_velocity_km_s(self) -> float:
        """Rebuilt relative speed at TCA, km/s."""
        return float(np.linalg.norm(self.v1 - self.v2))

    @property
    def reported_miss_km(self) -> float | None:
        """18 SDS's own miss distance, km, or None when absent."""
        return None if self.event.min_rng_m is None else self.event.min_rng_m / 1000.0

    @property
    def miss_error_km(self) -> float | None:
        """Rebuilt minus reported miss distance, km (km-scale, from TLE error)."""
        reported = self.reported_miss_km
        return None if reported is None else self.miss_km - reported


def build_geometry(event: Event, tles: dict) -> Geometry | None:
    """Propagate both objects to TCA; None if either TLE is missing."""
    t1, t2 = tles.get(event.sat1_id), tles.get(event.sat2_id)
    if t1 is None or t2 is None:
        return None
    r1, v1 = propagate(satrec_from_tle(t1), event.tca)
    r2, v2 = propagate(satrec_from_tle(t2), event.tca)
    return Geometry(
        event=event,
        r1=r1, v1=v1, r2=r2, v2=v2,
        tle_age1_days=t1.age_days(event.tca),
        tle_age2_days=t2.age_days(event.tca),
    )


def deduplicate(events: list[Event], tca_tolerance_s: float = 900.0) -> list[Event]:
    """Collapse repeated filings of one conjunction, sorted by TCA.

    cdm_public files each event once per primary and again per revision, with
    shifting TCAs, so filings of the same unordered pair within
    ``tca_tolerance_s`` are merged.
    """
    by_pair: dict[frozenset, list[Event]] = {}
    for e in events:
        by_pair.setdefault(frozenset(e.object_ids), []).append(e)

    out = []
    for group in by_pair.values():
        group.sort(key=lambda e: e.tca)
        kept: list[Event] = []
        for e in group:
            if kept and (e.tca - kept[-1].tca).total_seconds() <= tca_tolerance_s:
                continue
            kept.append(e)
        out.extend(kept)
    return sorted(out, key=lambda e: e.tca)
