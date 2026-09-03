"""Unit tests for the encounter-plane projection and Pc estimators."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from montecarlo import (encounter_plane_basis, hard_body_radius_km, pc_analytic,
                        pc_monte_carlo, pc_small_disk, project_encounter)

failures = []


def check(label, ok):
    print(("PASS  " if ok else "FAIL  ") + label)
    if not ok:
        failures.append(label)


for vrel in ([7.5, 0, 0], [0, 0, 12.0], [3.0, -4.0, 5.0], [1e-3, 0, 0]):
    B = encounter_plane_basis(np.array(vrel, dtype=float))
    n = np.array(vrel, dtype=float) / np.linalg.norm(vrel)
    ok = (np.allclose(B.T @ B, np.eye(2), atol=1e-12)
          and np.allclose(B.T @ n, 0.0, atol=1e-12))
    check(f"encounter basis orthonormal and perpendicular, vrel={vrel}", ok)

# Projection drops exactly the along-track component.
dr = np.array([1.0, 2.0, 3.0])
vrel = np.array([0.0, 0.0, 10.0])
cov = np.diag([4.0, 9.0, 16.0])
mu2, cov2 = project_encounter(dr, vrel, cov)
check("projection removes the velocity-axis component",
      np.isclose(np.linalg.norm(mu2), np.linalg.norm(dr[:2])))
check("projected covariance is 2x2", cov2.shape == (2, 2))
check("projected covariance is symmetric", np.allclose(cov2, cov2.T))
check("projected covariance positive definite", np.all(np.linalg.eigvalsh(cov2) > 0))

# A large disk relative to sigma: Pc must approach 1.
mu0 = np.zeros(2)
tiny = np.diag([1e-4, 1e-4])
check("huge disk -> Pc ~ 1", np.isclose(pc_analytic(mu0, tiny, 10.0), 1.0, atol=1e-6))

# Quadrature and the small-disk closed form must agree when HBR << sigma.
cov2d = np.array([[1.0, 0.3], [0.3, 0.5]])
for mu in (np.zeros(2), np.array([0.4, -0.2]), np.array([1.5, 1.0])):
    a = pc_analytic(mu, cov2d, 1e-3)
    s = pc_small_disk(mu, cov2d, 1e-3)
    check(f"quadrature == small-disk form, mu={mu}", np.isclose(a, s, rtol=1e-6))

# Monte Carlo must agree with quadrature within its own error bar.
rng = np.random.default_rng(7)
for mu, hbr in ((np.zeros(2), 0.05), (np.array([0.5, 0.0]), 0.08)):
    exact = pc_analytic(mu, cov2d, hbr)
    mc = pc_monte_carlo(mu, cov2d, hbr, 4_000_000, rng)
    z = abs(mc.pc - exact) / mc.stderr
    check(f"MC within 4 sigma of quadrature, mu={mu} (z={z:.2f})", z < 4.0)

# Pc scales as HBR^2 in the small-disk regime.
p1 = pc_small_disk(np.array([0.3, 0.1]), cov2d, 1e-3)
p2 = pc_small_disk(np.array([0.3, 0.1]), cov2d, 2e-3)
check("Pc scales as HBR^2", np.isclose(p2 / p1, 4.0, rtol=1e-9))

# Farther miss distance must lower Pc.
near = pc_small_disk(np.array([0.1, 0.0]), cov2d, 1e-3)
far = pc_small_disk(np.array([3.0, 0.0]), cov2d, 1e-3)
check("larger miss distance lowers Pc", far < near)

check("HBR from RCS classes", np.isclose(hard_body_radius_km("SMALL", "LARGE"), 0.0035))
check("unknown RCS falls back to default", np.isclose(hard_body_radius_km("", ""), 0.002))

print()
print(f"{len(failures)} failure(s)" if failures else "all tests passed")
sys.exit(1 if failures else 0)
