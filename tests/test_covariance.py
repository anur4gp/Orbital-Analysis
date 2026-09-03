"""Unit tests for the RTN frame and covariance rotation. No network."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from covariance import RTNCovariance, combined_covariance, rtn_basis, sample_relative_offsets

failures = []


def check(label, ok):
    print(("PASS  " if ok else "FAIL  ") + label)
    if not ok:
        failures.append(label)


# A circular equatorial orbit: position +x, velocity +y.
r = np.array([7000.0, 0.0, 0.0])
v = np.array([0.0, 7.5, 0.0])
A = rtn_basis(r, v)

check("basis is orthonormal", np.allclose(A.T @ A, np.eye(3), atol=1e-12))
check("basis is right-handed (det +1)", np.isclose(np.linalg.det(A), 1.0))
check("R points along position", np.allclose(A[:, 0], [1, 0, 0]))
check("T points along velocity for circular orbit", np.allclose(A[:, 1], [0, 1, 0]))
check("N points along orbit normal", np.allclose(A[:, 2], [0, 0, 1]))

cov = RTNCovariance(sigma_r_km=0.2, k_t=10.0, k_n=1.5)
check("sigma_t = k_t * sigma_r", np.isclose(cov.sigma_t_km, 2.0))
check("sigma_n = k_n * sigma_r", np.isclose(cov.sigma_n_km, 0.3))

C_rtn = cov.matrix_rtn()
C_eci = cov.matrix_eci(r, v)
check("rtn matrix is diagonal", np.allclose(C_rtn, np.diag(np.diag(C_rtn))))
check("rotation preserves trace", np.isclose(np.trace(C_rtn), np.trace(C_eci)))
check("rotation preserves determinant", np.isclose(np.linalg.det(C_rtn), np.linalg.det(C_eci)))
check("rotated matrix is symmetric", np.allclose(C_eci, C_eci.T))
check("rotated matrix is positive definite", np.all(np.linalg.eigvalsh(C_eci) > 0))
check("eigenvalues preserved under rotation",
      np.allclose(np.sort(np.linalg.eigvalsh(C_rtn)), np.sort(np.linalg.eigvalsh(C_eci))))

# For this aligned orbit the ECI covariance should equal the RTN one.
check("aligned orbit: eci == rtn", np.allclose(C_eci, C_rtn))

# A tilted orbit must NOT give a diagonal ECI covariance.
r2 = np.array([4000.0, 5000.0, 2000.0])
v2 = np.array([-5.0, 3.0, 2.0])
C2 = cov.matrix_eci(r2, v2)
check("tilted orbit: eci is not diagonal", not np.allclose(C2, np.diag(np.diag(C2))))
check("tilted orbit: trace still preserved", np.isclose(np.trace(C2), np.trace(C_rtn)))

# Combining two objects.
comb = combined_covariance(r, v, r2, v2, cov, cov)
check("combined = sum of rotated parts", np.allclose(comb, C_eci + C2))
check("combined trace = sum of traces", np.isclose(np.trace(comb), 2 * np.trace(C_rtn)))
check("combining is symmetric in its arguments",
      np.allclose(comb, combined_covariance(r2, v2, r, v, cov, cov)))

# Sampling must reproduce the covariance it was given.
rng = np.random.default_rng(0)
draws = sample_relative_offsets(comb, 400_000, rng)
empirical = np.cov(draws, rowvar=False)
check("sample mean ~ 0", np.allclose(draws.mean(axis=0), 0.0, atol=0.02))
check("sample covariance recovers input", np.allclose(empirical, comb, rtol=0.03, atol=1e-3))

# A singular covariance must not crash the sampler.
singular = np.diag([1.0, 1.0, 0.0])
check("singular covariance handled", sample_relative_offsets(singular, 10, rng).shape == (10, 3))

print()
print(f"{len(failures)} failure(s)" if failures else "all tests passed")
sys.exit(1 if failures else 0)
