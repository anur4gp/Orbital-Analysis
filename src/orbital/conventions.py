"""Frames, time systems, and units -- the single source of truth.

Every physical quantity in this package carries explicit units in its
docstring. The conventions below are stated once here and are not restated
(or silently varied) elsewhere.

Units
-----
Length          kilometres (km)
Velocity        kilometres per second (km/s)
Time            seconds (s); epochs as timezone-aware UTC ``datetime``
Angle           radians internally; degrees only at TLE/report boundaries,
                always named ``*_deg``
Mass            kilograms (kg)
Inertia         kg m^2
Angular rate    radians per second (rad/s)

Frames
------
``Frame.TEME``
    True Equator, Mean Equinox. **The frame SGP4 outputs**, and therefore the
    frame of every state derived from a TLE.
``Frame.ECI_J2000``
    Earth-centred inertial, J2000 equator and equinox. The natural frame for
    the 6-DOF rigid-body dynamics, which integrate Newton's and Euler's
    equations directly.
``Frame.ECEF``
    Earth-centred, Earth-fixed. Ground-station positions live here.
``Frame.RTN``
    Radial / in-track / cross-track, defined per object about its own orbit.
    Rotating, non-inertial: valid for expressing uncertainty, never for
    integrating dynamics.
``Frame.BODY``
    Body-fixed principal axes. Attitude is the rotation BODY -> ECI_J2000.

**TEME and ECI_J2000 are not interchangeable.** They differ by the equation
of the equinoxes -- hundreds of metres to kilometres in low Earth orbit. SGP4
output is TEME; the 6-DOF propagator works in ECI_J2000. Mixing them without
an explicit conversion is a silent kilometre-scale error, so states carry
their frame and conversions are always an explicit call.

Time systems
------------
``TimeSystem.UTC``
    The external interface. All epochs crossing a public boundary are
    timezone-aware UTC datetimes; naive datetimes are rejected rather than
    assumed.
``TimeSystem.UT1``
    Needed for Earth rotation angle (TEME -> ECEF). Differs from UTC by
    |DUT1| < 0.9 s.
``TimeSystem.TT``
    Terrestrial Time, for precession/nutation. TT = TAI + 32.184 s.

Julian dates are carried as a ``(jd, fr)`` pair -- integer-ish day plus
fraction -- because a single float64 Julian date loses roughly 20 microseconds
of resolution at current epochs, which is 15 cm of along-track position at
orbital speed.

Attitude
--------
Quaternions are **scalar-first**, ``q = [w, x, y, z]``, unit norm, and
represent the rotation from BODY to the stated inertial frame. The
Hamilton convention is used throughout (not JPL).
"""
from __future__ import annotations

from enum import StrEnum


class Frame(StrEnum):
    """Reference frames used in this package. See the module docstring."""

    TEME = "TEME"
    ECI_J2000 = "ECI_J2000"
    ECEF = "ECEF"
    RTN = "RTN"
    BODY = "BODY"


class TimeSystem(StrEnum):
    """Time systems used in this package. See the module docstring."""

    UTC = "UTC"
    UT1 = "UT1"
    TT = "TT"


#: Quaternion convention, stated so downstream code can assert on it.
QUATERNION_ORDER = "scalar-first (w, x, y, z)"
QUATERNION_CONVENTION = "Hamilton"
