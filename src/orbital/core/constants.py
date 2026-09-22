"""EGM96 / WGS-84 Earth constants (km, s).

SGP4 uses its own WGS-72 constants; do not mix the two.
"""
from __future__ import annotations

#: Earth gravitational parameter GM, km^3/s^2 (EGM96).
MU_EARTH_KM3_S2: float = 398600.4415

#: Earth equatorial radius, km (EGM96 reference radius).
R_EARTH_KM: float = 6378.1363

#: Unnormalised second zonal harmonic, dimensionless (EGM96, C20 = -J2).
J2_EARTH: float = 1.0826266835531513e-3

#: Earth rotation rate, rad/s (IERS, mean sidereal).
OMEGA_EARTH_RAD_S: float = 7.292115146706979e-5

#: WGS-84 ellipsoid, used only for geodetic station coordinates.
WGS84_A_KM: float = 6378.137
WGS84_F: float = 1.0 / 298.257223563
