"""Earth-fixed <-> inertial rotation and geodetic station coordinates.

Model: Earth rotates about the ECI_J2000 z-axis at a constant rate, with the
angle given by IAU-1982 Greenwich mean sidereal time. Precession, nutation
and polar motion are neglected -- the same pole assumption the J2 model
makes (see :mod:`orbital.dynamics.forces`). UT1 is taken as UTC unless
``dut1_s`` is given; |DUT1| < 0.9 s is up to 66 micro-radians of Earth
rotation, about 0.4 km at the surface.

Within a simulation this model is self-consistent: truth and filter use the
same rotation, so it does not bias estimation results. It is not accurate
enough to process real tracking data, which would need the full IAU-2006
reduction; that is deliberately out of scope here.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from orbital.attitude.dcm import rot_z
from orbital.attitude.quaternion import FloatArray
from orbital.core.constants import OMEGA_EARTH_RAD_S, WGS84_A_KM, WGS84_F
from orbital.core.timescales import J2000_JD, SECONDS_PER_DAY, julian_date

EARTH_SPIN_RAD_S: FloatArray = np.array([0.0, 0.0, OMEGA_EARTH_RAD_S])


def gmst_rad(epoch: datetime, dut1_s: float = 0.0) -> float:
    """Greenwich mean sidereal time (IAU-1982), rad in ``[0, 2 pi)``.

    Parameters
    ----------
    epoch
        UTC epoch, timezone-aware.
    dut1_s
        UT1 - UTC, s.
    """
    jd, fr = julian_date(epoch)
    t = ((jd - J2000_JD) + fr + dut1_s / SECONDS_PER_DAY) / 36525.0
    seconds = (
        67310.54841
        + (876600.0 * 3600.0 + 8640184.812866) * t
        + 0.093104 * t**2
        - 6.2e-6 * t**3
    )
    return float(np.mod(seconds * 2.0 * np.pi / SECONDS_PER_DAY, 2.0 * np.pi))


def geodetic_to_ecef(lat_deg: float, lon_deg: float, alt_km: float) -> FloatArray:
    """WGS-84 geodetic coordinates to ECEF position, km."""
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    e2 = WGS84_F * (2.0 - WGS84_F)
    n = WGS84_A_KM / np.sqrt(1.0 - e2 * np.sin(lat) ** 2)
    return np.array([
        (n + alt_km) * np.cos(lat) * np.cos(lon),
        (n + alt_km) * np.cos(lat) * np.sin(lon),
        (n * (1.0 - e2) + alt_km) * np.sin(lat),
    ])


def geodetic_up(lat_deg: float, lon_deg: float) -> FloatArray:
    """Unit ellipsoid normal (local vertical) in ECEF."""
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    return np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])


@dataclass(frozen=True)
class EarthRotation:
    """Earth orientation relative to a reference epoch.

    Simulation times are seconds since ``epoch``; the rotation angle is
    ``gmst(epoch) + omega_E t``.
    """

    epoch: datetime
    dut1_s: float = 0.0

    def angle_rad(self, t_s: float) -> float:
        """Earth rotation angle at ``t_s`` s past the epoch, rad."""
        return gmst_rad(self.epoch, self.dut1_s) + OMEGA_EARTH_RAD_S * t_s

    def ecef_to_eci(self, t_s: float) -> FloatArray:
        """DCM taking ECEF components to ECI_J2000 components."""
        return rot_z(self.angle_rad(t_s))
