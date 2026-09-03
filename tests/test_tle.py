"""Offline unit tests for TLE parsing. No network.

Run: ./venv/bin/python tests/test_tle.py
"""
import sys
from datetime import timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tle import _decimal_point_assumed, check_line, checksum, parse_tle, parse_tle_file

# An ISS TLE frozen so these tests never depend on a live pull. Field values are
# representative of a real April 2024 element set; the two check digits were
# recomputed, so treat this as a fixture rather than a verbatim archived record.
NAME = "ISS (ZARYA)"
L1 = "1 25544U 98067A   24117.51782528  .00016717  00000-0  30074-3 0  9992"
L2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.49309239448471"

failures = []


def check(label, condition):
    print(("PASS  " if condition else "FAIL  ") + label)
    if not condition:
        failures.append(label)


def close(a, b, tol=1e-9):
    return abs(a - b) <= tol


check("line 1 checksum valid", check_line(L1))
check("line 2 checksum valid", check_line(L2))
check("corrupted line rejected", not check_line(L1[:20] + "9" + L1[21:]))
check("checksum is a digit 0-9", 0 <= checksum(L1) <= 9)

check("assumed-decimal exponential", close(_decimal_point_assumed("30074-3"), 0.30074e-3))
check("assumed-decimal zero field", close(_decimal_point_assumed("00000-0"), 0.0))
check("assumed-decimal negative", close(_decimal_point_assumed("-11606-4"), -0.11606e-4))

t = parse_tle(L1, L2, NAME)
check("name", t.name == NAME)
check("catalog number", t.catalog_number == 25544)
check("classification", t.classification == "U")
check("international designator", t.international_designator == "98067A")
check("bstar", close(t.bstar, 0.30074e-3))
check("mean motion dot", close(t.mean_motion_dot, 0.00016717))
check("element set number", t.element_set_number == 999)

# Epoch 24117.51782528 -> 2024 day 117 = April 26, at 0.51782528 of a day.
check("epoch year", t.epoch.year == 2024)
check("epoch month/day", (t.epoch.month, t.epoch.day) == (4, 26))
check("epoch is UTC-aware", t.epoch.tzinfo == timezone.utc)
check("epoch fraction -> 12:25", (t.epoch.hour, t.epoch.minute) == (12, 25))

check("inclination", close(t.inclination, 51.6416))
check("raan", close(t.raan, 247.4627))
check("eccentricity implied decimal", close(t.eccentricity, 0.0006703))
check("arg perigee", close(t.arg_perigee, 130.5360))
check("mean anomaly", close(t.mean_anomaly, 325.0288))
check("mean motion", close(t.mean_motion, 15.49309239))
check("revolution number", t.revolution_number == 44847)
check("period ~92.9 min", close(t.period_minutes, 1440.0 / 15.49309239, 1e-6))

check("parse with name line", len(parse_tle_file(f"{NAME}\n{L1}\n{L2}\n")) == 1)
check("parse without name line", len(parse_tle_file(f"{L1}\n{L2}\n")) == 1)
check("parse multiple", len(parse_tle_file(f"{NAME}\n{L1}\n{L2}\n{NAME}\n{L1}\n{L2}\n")) == 2)

try:
    parse_tle(L1, L2.replace("2 25544", "2 25545"))
    check("mismatched catalog numbers rejected", False)
except ValueError:
    check("mismatched catalog numbers rejected", True)

try:
    parse_tle(L2, L1)
    check("swapped lines rejected", False)
except ValueError:
    check("swapped lines rejected", True)

print()
print(f"{len(failures)} failure(s)" if failures else "all tests passed")
sys.exit(1 if failures else 0)
