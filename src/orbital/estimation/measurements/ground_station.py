"""Ground-station geometry: position, velocity and elevation in ECI_J2000."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from orbital.attitude.quaternion import FloatArray
from orbital.core.frames import EARTH_SPIN_RAD_S, EarthRotation, geodetic_to_ecef, geodetic_up


@dataclass(frozen=True)
class GroundStation:
    """A fixed site on the WGS-84 ellipsoid.

    Parameters
    ----------
    name
        Site label.
    lat_deg, lon_deg
        Geodetic latitude and east longitude, degrees.
    alt_km
        Height above the ellipsoid, km.
    earth
        Earth orientation model shared with the rest of the simulation.
    min_elevation_deg
        Elevation mask, degrees. Below it the site cannot track.
    """

    name: str
    lat_deg: float
    lon_deg: float
    alt_km: float
    earth: EarthRotation
    min_elevation_deg: float = 10.0
    _r_ecef: FloatArray = field(init=False, repr=False)
    _up_ecef: FloatArray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_r_ecef", geodetic_to_ecef(self.lat_deg, self.lon_deg,
                                                             self.alt_km))
        object.__setattr__(self, "_up_ecef", geodetic_up(self.lat_deg, self.lon_deg))

    def position_eci(self, t_s: float) -> FloatArray:
        """Site position, km, ECI_J2000."""
        return self.earth.ecef_to_eci(t_s) @ self._r_ecef

    def velocity_eci(self, t_s: float) -> FloatArray:
        """Site inertial velocity ``omega_E x r``, km/s, ECI_J2000."""
        return np.cross(EARTH_SPIN_RAD_S, self.position_eci(t_s))

    def elevation_rad(self, t_s: float, r_km: FloatArray) -> float:
        """Elevation of an object at ``r_km`` (ECI_J2000) above the local horizon, rad."""
        c = self.earth.ecef_to_eci(t_s)
        los = np.asarray(r_km) - c @ self._r_ecef
        sin_el = np.dot(los, c @ self._up_ecef) / np.linalg.norm(los)
        # Round-off can push |sin| just past 1 at zenith.
        return float(np.arcsin(np.clip(sin_el, -1.0, 1.0)))

    def is_visible(self, t_s: float, r_km: FloatArray) -> bool:
        """True if the object is above the elevation mask."""
        return self.elevation_rad(t_s, r_km) >= np.radians(self.min_elevation_deg)
