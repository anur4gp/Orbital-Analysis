"""Mass properties: validated inertia tensors.

A symmetric positive-definite matrix is not automatically a physical inertia
tensor. Principal moments of any real mass distribution also satisfy the
triangle inequality ``I_i <= I_j + I_k``, because ``I_j + I_k - I_i`` is twice
a second moment of mass. Tensors that violate it are rejected here, since
Euler's equations integrate them without complaint and give nonsense.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from orbital.attitude.quaternion import FloatArray

_TOL = 1e-9


@dataclass(frozen=True)
class InertiaTensor:
    """Inertia tensor about the centre of mass, BODY axes, kg m^2."""

    matrix: FloatArray

    def __post_init__(self) -> None:
        m = np.array(self.matrix, dtype=float)
        if m.shape != (3, 3):
            raise ValueError(f"inertia tensor must be 3x3, got {m.shape}")
        scale = float(np.max(np.abs(m)))
        if not np.allclose(m, m.T, atol=_TOL * max(scale, 1.0)):
            raise ValueError("inertia tensor must be symmetric")
        moments = np.linalg.eigvalsh(m)
        if moments[0] <= 0.0:
            raise ValueError("inertia tensor must be positive definite")
        if moments[2] > moments[0] + moments[1] + _TOL * scale:
            raise ValueError(
                "principal moments violate the triangle inequality "
                f"{moments[2]:g} > {moments[0]:g} + {moments[1]:g}; "
                "no physical mass distribution has this tensor"
            )
        m.setflags(write=False)
        inv = np.linalg.inv(m)
        inv.setflags(write=False)
        object.__setattr__(self, "matrix", m)
        object.__setattr__(self, "_inverse", inv)

    @property
    def inverse(self) -> FloatArray:
        """Inverse tensor, 1/(kg m^2). Computed once."""
        inv: FloatArray = self.__dict__["_inverse"]
        return inv

    @classmethod
    def diagonal(cls, i1: float, i2: float, i3: float) -> InertiaTensor:
        """Tensor already expressed in principal axes, kg m^2."""
        return cls(np.diag([i1, i2, i3]))

    @classmethod
    def cuboid(cls, mass_kg: float, x_m: float, y_m: float, z_m: float) -> InertiaTensor:
        """Uniform rectangular box with edge lengths in metres."""
        k = mass_kg / 12.0
        return cls.diagonal(k * (y_m**2 + z_m**2), k * (x_m**2 + z_m**2), k * (x_m**2 + y_m**2))

    def principal(self) -> tuple[FloatArray, FloatArray]:
        """Principal moments (ascending, kg m^2) and axes (columns, right-handed)."""
        moments, axes = np.linalg.eigh(self.matrix)
        if np.linalg.det(axes) < 0.0:
            axes[:, 2] *= -1.0
        return moments, axes

    def is_axisymmetric(self, rel_tol: float = 1e-9) -> bool:
        """True if two principal moments coincide."""
        m, _ = self.principal()
        return bool(np.isclose(m[0], m[1], rtol=rel_tol) or np.isclose(m[1], m[2], rtol=rel_tol))


@dataclass(frozen=True)
class MassProperties:
    """Mass (kg) and inertia tensor about the centre of mass (kg m^2)."""

    mass_kg: float
    inertia: InertiaTensor

    def __post_init__(self) -> None:
        if not self.mass_kg > 0.0:
            raise ValueError("mass_kg must be positive")
