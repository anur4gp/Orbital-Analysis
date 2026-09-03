"""TLE parsing and validation.

A TLE is a fixed-column record: fields are located by column position, not
by delimiter, so parsing is slicing. See CLAUDE.md for the column tables.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone


def checksum(line: str) -> int:
    """TLE checksum: sum of digits mod 10, minus signs count as 1."""
    total = 0
    for ch in line[:68]:
        if ch.isdigit():
            total += int(ch)
        elif ch == "-":
            total += 1
    return total % 10


def check_line(line: str) -> bool:
    """True if the line is 69 chars and its checksum digit matches."""
    return len(line) == 69 and line[68].isdigit() and checksum(line) == int(line[68])


def _decimal_point_assumed(field: str) -> float:
    """Decode a TLE exponential field, e.g. '30074-3' -> 0.30074e-3.

    The leading decimal point and the exponent's 'e' are both omitted.
    """
    field = field.strip()
    if not field or set(field) <= {"0", "+", "-", " "}:
        return 0.0
    sign = -1.0 if field[0] == "-" else 1.0
    if field[0] in "+-":
        field = field[1:]
    mantissa, exponent = field[:-2], field[-2:]
    return sign * float("0." + mantissa) * 10.0 ** int(exponent)


def _epoch_to_datetime(two_digit_year: int, day_of_year: float) -> datetime:
    """TLE epoch -> UTC datetime. Years 57-99 are 19xx, 00-56 are 20xx."""
    year = 1900 + two_digit_year if two_digit_year >= 57 else 2000 + two_digit_year
    start = datetime(year, 1, 1, tzinfo=timezone.utc)
    return start + timedelta(days=day_of_year - 1.0)


@dataclass(frozen=True)
class TLE:
    """A parsed TLE. `line1`/`line2` are kept verbatim for sgp4."""

    name: str
    line1: str
    line2: str

    catalog_number: int
    classification: str
    international_designator: str
    epoch: datetime

    # Line 1 drag / bookkeeping
    mean_motion_dot: float      # rev/day^2, already halved by the format
    mean_motion_ddot: float     # rev/day^3, already divided by 6
    bstar: float                # 1/earth radii
    element_set_number: int

    # Line 2: the six orbital elements
    inclination: float          # deg
    raan: float                 # deg
    eccentricity: float
    arg_perigee: float          # deg
    mean_anomaly: float         # deg
    mean_motion: float          # rev/day
    revolution_number: int

    @property
    def period_minutes(self) -> float:
        return 1440.0 / self.mean_motion

    def age_days(self, at: datetime | None = None) -> float:
        """Days between the TLE epoch and `at` (default: now).

        SGP4 error grows roughly 1-3 km/day past epoch, so this is the first
        thing to check before trusting a propagated state.
        """
        at = at or datetime.now(timezone.utc)
        return (at - self.epoch).total_seconds() / 86400.0


def parse_tle(line1: str, line2: str, name: str = "") -> TLE:
    """Parse one TLE by column position. Raises ValueError on a bad checksum."""
    line1, line2 = line1.rstrip("\r\n"), line2.rstrip("\r\n")
    for n, line in ((1, line1), (2, line2)):
        if not check_line(line):
            raise ValueError(f"line {n} failed checksum/length validation: {line!r}")
    if line1[0] != "1" or line2[0] != "2":
        raise ValueError("lines are out of order or mislabeled")
    if line1[2:7] != line2[2:7]:
        raise ValueError("line 1 and line 2 catalog numbers disagree")

    return TLE(
        name=name.strip(),
        line1=line1,
        line2=line2,
        catalog_number=int(line1[2:7]),
        classification=line1[7],
        international_designator=line1[9:17].strip(),
        epoch=_epoch_to_datetime(int(line1[18:20]), float(line1[20:32])),
        mean_motion_dot=float(line1[33:43]),
        mean_motion_ddot=_decimal_point_assumed(line1[44:52]),
        bstar=_decimal_point_assumed(line1[53:61]),
        element_set_number=int(line1[64:68]),
        inclination=float(line2[8:16]),
        raan=float(line2[17:25]),
        eccentricity=float("0." + line2[26:33].strip()),
        arg_perigee=float(line2[34:42]),
        mean_anomaly=float(line2[43:51]),
        mean_motion=float(line2[52:63]),
        revolution_number=int(line2[63:68]),
    )


def parse_tle_file(text: str) -> list[TLE]:
    """Parse a CelesTrak TLE response: repeating name/line1/line2 triples.

    Bulk GROUP pulls include a name line; single-satellite pulls in TLE
    format do too. Files without name lines are handled as well.
    """
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    out: list[TLE] = []
    i = 0
    while i < len(lines):
        if lines[i].startswith("1 ") and i + 1 < len(lines) and lines[i + 1].startswith("2 "):
            out.append(parse_tle(lines[i], lines[i + 1]))
            i += 2
        elif i + 2 < len(lines) and lines[i + 1].startswith("1 ") and lines[i + 2].startswith("2 "):
            out.append(parse_tle(lines[i + 1], lines[i + 2], name=lines[i]))
            i += 3
        else:
            raise ValueError(f"unrecognized TLE structure at line {i}: {lines[i]!r}")
    return out
