"""Campaign design points and their mapping to physical settings."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from orbital.attitude.quaternion import FloatArray
from orbital.campaign.config import PARAMETERS, CampaignConfig
from orbital.surrogate import designs


def design_matrix(config: CampaignConfig) -> FloatArray:
    """Design points in the unit cube, shape ``(n_points, dimension)``.

    Uncached MaxPro sizes need the compiled extension; the container has none.
    """
    n, k = config.n_points, config.dimension
    if config.design == "maxpro":
        try:
            return np.atleast_2d(designs.load_or_generate(n, k))
        except RuntimeError as exc:
            raise RuntimeError(
                f"no cached MaxPro design for n={n}, k={k}, and the "
                f"pt_maxpro extension is unavailable here. Cached sizes: "
                f"{cached_maxpro_sizes(k) or 'none'}. Generate it where the "
                f"extension is built (see README), commit the CSV under "
                f"reference/lhd/designs/, and rebuild the image."
            ) from exc
    rng = np.random.default_rng(config.seed)
    if config.design == "random_lhd":
        return designs.random_lhd(n, k, rng)
    return designs.uniform_sample(n, k, rng)


def cached_maxpro_sizes(k: int) -> list[int]:
    """Design sizes available as cached CSV for dimension ``k``, ascending."""
    sizes = []
    for path in designs.DESIGN_DIR.glob(f"maxpro_n*_k{k}.csv"):
        stem = path.stem.split("_")
        sizes.append(int(stem[1][1:]))
    return sorted(sizes)


@dataclass(frozen=True)
class PointSettings:
    """Physical settings of one design point.

    Attributes
    ----------
    sigma_r0_km, sigma_v0_km_s
        Initial 1-sigma uncertainty per axis, km and km/s.
    sigma_range_km, sigma_range_rate_km_s
        Measurement noise, km and km/s.
    cadence_s
        Interval between measurement opportunities, s.
    min_elevation_deg
        Station elevation mask, degrees.
    """

    sigma_r0_km: float
    sigma_v0_km_s: float
    sigma_range_km: float
    sigma_range_rate_km_s: float
    cadence_s: float
    min_elevation_deg: float

    def to_dict(self) -> dict[str, float]:
        """Settings keyed by parameter name, for the results table."""
        return {p.name: getattr(self, p.name) for p in PARAMETERS}


def settings_for(u: FloatArray) -> PointSettings:
    """Map one unit-cube row onto :class:`PointSettings`."""
    u = np.asarray(u, dtype=float)
    if u.shape != (len(PARAMETERS),):
        raise ValueError(f"expected {len(PARAMETERS)} coordinates, got {u.shape}")
    values = {p.name: float(p.from_unit(ui)) for p, ui in zip(PARAMETERS, u, strict=True)}
    return PointSettings(**values)
