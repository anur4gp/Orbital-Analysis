"""Campaign configuration and provenance.

A campaign is fully determined by this config: the same config and seed give
the same rows, on any number of workers or shards. The config is written
beside the results and fingerprinted, so a stored run can always be traced
back to what produced it.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from typing import Any

import numpy as np
import scipy

from orbital.paths import PROJECT_ROOT


@dataclass(frozen=True)
class Parameter:
    """One swept parameter.

    Attributes
    ----------
    name
        Column name in the results table.
    low, high
        Range limits, in ``unit``.
    log
        If true the parameter is sampled geometrically, so the design covers
        orders of magnitude evenly rather than favouring the top decade.
    unit
        Unit string, for tables and axis labels.
    """

    name: str
    low: float
    high: float
    log: bool
    unit: str

    def from_unit(self, u: float | np.ndarray) -> float | np.ndarray:
        """Map ``u`` in [0, 1] onto the parameter range."""
        if self.log:
            return float(self.low) * (float(self.high) / float(self.low)) ** u
        return float(self.low) + (float(self.high) - float(self.low)) * u


#: The swept space. Position and velocity uncertainty span the range from
#: GNSS-grade to worse than TLE-grade; sensor noise spans metre-class radar
#: to poor; cadence and mask control how much data a pass yields.
PARAMETERS: tuple[Parameter, ...] = (
    Parameter("sigma_r0_km", 0.01, 10.0, True, "km"),
    Parameter("sigma_v0_km_s", 1e-5, 1e-2, True, "km/s"),
    Parameter("sigma_range_km", 0.001, 0.1, True, "km"),
    Parameter("sigma_range_rate_km_s", 1e-6, 1e-4, True, "km/s"),
    Parameter("cadence_s", 10.0, 120.0, False, "s"),
    Parameter("min_elevation_deg", 5.0, 20.0, False, "deg"),
)


@dataclass(frozen=True)
class CampaignConfig:
    """Everything that determines a campaign's results.

    Attributes
    ----------
    n_points
        Design size (number of parameter combinations).
    n_trials
        Monte Carlo trials per design point. Every filter sees the same
        trials, so filter differences are not sampling noise.
    design
        ``maxpro`` (parallel tempering, cached), ``random_lhd`` or ``uniform``.
    seed
        Base seed. Each point derives its own stream from ``(seed, index)``,
        so results do not depend on execution order.
    revolutions
        Arc length in orbital revolutions.
    semi_major_axis_km, inclination_deg, raan_deg
        The truth orbit.
    filters
        Filter names to compare.
    confidence
        Confidence level for the NEES/NIS acceptance bands.
    """

    n_points: int = 64
    n_trials: int = 8
    design: str = "maxpro"
    seed: int = 2026
    revolutions: float = 2.0
    semi_major_axis_km: float = 7000.0
    inclination_deg: float = 51.6
    raan_deg: float = 250.0
    filters: tuple[str, ...] = ("EKF", "UKF")
    confidence: float = 0.95

    def __post_init__(self) -> None:
        if self.design not in {"maxpro", "random_lhd", "uniform"}:
            raise ValueError(f"unknown design type {self.design!r}")
        if self.n_points < 1 or self.n_trials < 1:
            raise ValueError("n_points and n_trials must be positive")
        if not set(self.filters) <= {"EKF", "UKF"}:
            raise ValueError(f"unknown filters in {self.filters}")

    @property
    def dimension(self) -> int:
        """Number of swept parameters."""
        return len(PARAMETERS)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable form."""
        d = asdict(self)
        d["filters"] = list(self.filters)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CampaignConfig:
        """Inverse of :meth:`to_dict`, ignoring unknown keys."""
        fields = {f for f in cls.__dataclass_fields__}
        kwargs = {k: v for k, v in d.items() if k in fields}
        if "filters" in kwargs:
            kwargs["filters"] = tuple(kwargs["filters"])
        return cls(**kwargs)

    def replace(self, **changes: Any) -> CampaignConfig:
        """Copy with fields changed."""
        return replace(self, **changes)

    def fingerprint(self) -> str:
        """Short stable hash of the config -- the campaign's identity."""
        blob = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


@dataclass(frozen=True)
class Provenance:
    """Where and with what a campaign ran. Stored, never used as input."""

    git_commit: str = field(default_factory=_git_commit)
    python: str = field(default_factory=platform.python_version)
    numpy: str = field(default_factory=lambda: np.__version__)
    scipy: str = field(default_factory=lambda: scipy.__version__)
    platform: str = field(default_factory=platform.platform)
    started_utc: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, str]:
        """Fields as plain strings, for the stored JSON."""
        return asdict(self)
