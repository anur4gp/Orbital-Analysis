"""The measurement-model interface.

A model maps the 6-element orbit state ``x = [r (km), v (km/s)]`` in
ECI_J2000 at time ``t_s`` to a measurement vector. The EKF needs
:meth:`~MeasurementModel.jacobian`; the UKF uses only
:meth:`~MeasurementModel.predict`. Adding a sensor type means implementing
this protocol -- neither filter changes.
"""
from __future__ import annotations

from typing import Protocol

from orbital.attitude.quaternion import FloatArray


class MeasurementModel(Protocol):
    """A sensor: prediction, linearisation and noise."""

    @property
    def name(self) -> str:
        """Label used in histories and plots."""
        ...

    @property
    def dim(self) -> int:
        """Length of the measurement vector."""
        ...

    @property
    def noise_covariance(self) -> FloatArray:
        """Measurement noise covariance R, shape ``(dim, dim)``, in measurement units."""
        ...

    def predict(self, t_s: float, x: FloatArray) -> FloatArray:
        """Noise-free measurement h(x) at ``t_s``."""
        ...

    def jacobian(self, t_s: float, x: FloatArray) -> FloatArray:
        """dh/dx at ``x``, shape ``(dim, 6)``."""
        ...

    def is_available(self, t_s: float, x: FloatArray) -> bool:
        """Whether the sensor can observe state ``x`` at ``t_s`` (e.g. above the mask)."""
        ...

    def residual(self, z: FloatArray, z_pred: FloatArray) -> FloatArray:
        """Innovation ``z - z_pred``. Override for angle wrapping."""
        ...
