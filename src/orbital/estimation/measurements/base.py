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
        """Noise-free measurement ``h(x)``.

        Parameters
        ----------
        t_s
            Time since the reference epoch, s. Sensors fixed to the rotating
            Earth need it to place themselves.
        x
            Orbit state ``[r (km), v (km/s)]`` in ECI_J2000, shape (6,).

        Returns
        -------
        numpy.ndarray
            Predicted measurement, shape (dim,).
        """
        ...

    def jacobian(self, t_s: float, x: FloatArray) -> FloatArray:
        """Measurement Jacobian ``dh/dx``.

        Parameters
        ----------
        t_s
            Time since the reference epoch, s.
        x
            Linearisation point, shape (6,).

        Returns
        -------
        numpy.ndarray
            Shape (dim, 6).
        """
        ...

    def is_available(self, t_s: float, x: FloatArray) -> bool:
        """Whether the sensor can observe this state, e.g. above its mask.

        Parameters
        ----------
        t_s
            Time since the reference epoch, s.
        x
            True state being observed, shape (6,).

        Returns
        -------
        bool
        """
        ...

    def residual(self, z: FloatArray, z_pred: FloatArray) -> FloatArray:
        """Innovation ``z - z_pred``. Override for angle wrapping.

        Parameters
        ----------
        z
            Observed measurement, shape (dim,).
        z_pred
            Predicted measurement, shape (dim,).

        Returns
        -------
        numpy.ndarray
            Innovation, shape (dim,).
        """
        ...
