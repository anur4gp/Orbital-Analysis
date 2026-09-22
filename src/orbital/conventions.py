"""Frames, time systems and units used throughout the package.

Units: km, km/s, s, kg, kg m^2, rad (``*_deg`` names at TLE/report boundaries).
Epochs are timezone-aware UTC ``datetime``; Julian dates are ``(jd, fr)`` pairs,
since a single float64 JD loses ~20 us (~15 cm along-track).

Frames
------
TEME       SGP4 output frame.
ECI_J2000  6-DOF dynamics and estimation frame.
ECEF       Ground stations.
RTN        Per-object radial / in-track / cross-track; uncertainty only.
BODY       Principal axes; attitude is the rotation BODY -> ECI_J2000.

TEME and ECI_J2000 differ by hundreds of metres to kilometres in LEO, so states
carry their frame and conversions are explicit.

Quaternions are scalar-first ``[w, x, y, z]``, Hamilton convention.
"""
from __future__ import annotations

from enum import StrEnum


class Frame(StrEnum):
    """Reference frames."""

    TEME = "TEME"
    ECI_J2000 = "ECI_J2000"
    ECEF = "ECEF"
    RTN = "RTN"
    BODY = "BODY"


class TimeSystem(StrEnum):
    """Time systems."""

    UTC = "UTC"
    UT1 = "UT1"
    TT = "TT"


QUATERNION_ORDER = "scalar-first (w, x, y, z)"
QUATERNION_CONVENTION = "Hamilton"
