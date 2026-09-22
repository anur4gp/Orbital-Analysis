"""Shared fixtures. Every test here is offline and deterministic."""
from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from orbital.sgp4tools.tle import TLE, parse_tle

# Representative ISS element set (April 2024) with recomputed check digits.
ISS_LINE1 = "1 25544U 98067A   24117.51782528  .00016717  00000-0  30074-3 0  9992"
ISS_LINE2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.49309239448471"
ISS_NAME = "ISS (ZARYA)"


@pytest.fixture
def iss_lines() -> tuple[str, str, str]:
    """(name, line1, line2) of the frozen ISS fixture."""
    return ISS_NAME, ISS_LINE1, ISS_LINE2


@pytest.fixture
def iss_tle() -> TLE:
    """Parsed frozen ISS element set."""
    return parse_tle(ISS_LINE1, ISS_LINE2, ISS_NAME)


@pytest.fixture
def epoch() -> datetime:
    """A fixed UTC epoch matching the ISS fixture."""
    return datetime(2024, 4, 26, 12, 25, 0, tzinfo=UTC)


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded generator, so failures reproduce."""
    return np.random.default_rng(20240426)


@pytest.fixture
def circular_equatorial() -> tuple[np.ndarray, np.ndarray]:
    """(r, v) for a circular equatorial orbit; its RTN basis equals the inertial axes."""
    return np.array([7000.0, 0.0, 0.0]), np.array([0.0, 7.5, 0.0])


@pytest.fixture
def inclined_orbit() -> tuple[np.ndarray, np.ndarray]:
    """(r, v) for an orbit tilted out of every coordinate plane."""
    return np.array([4000.0, 5000.0, 2000.0]), np.array([-5.0, 3.0, 2.0])
