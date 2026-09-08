"""Tests for the 6-D parameter space, including the invariances the 4-D
reduction relies on. These are what make the reduction a verified claim
rather than an assumed one."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from montecarlo import pc_analytic
from paramspace import (BOUNDS, BOUNDS_4D, DIM, DIM_4D, from_unit_cube,
                        from_unit_cube_4d, reduce_to_4d, to_unit_cube, unpack,
                        unpack_4d)

failures = []


def check(label, ok):
    print(("PASS  " if ok else "FAIL  ") + label)
    if not ok:
        failures.append(label)


rng = np.random.default_rng(3)

# --- box mapping ---
u = rng.random((200, DIM))
x = from_unit_cube(u)
check("unit cube maps inside the box",
      bool(np.all(x >= BOUNDS[:, 0] - 1e-12) and np.all(x <= BOUNDS[:, 1] + 1e-12)))
check("round trip is exact", np.allclose(to_unit_cube(x), u, atol=1e-12))
check("corners map to bounds",
      np.allclose(from_unit_cube(np.zeros((1, DIM)))[0], BOUNDS[:, 0]) and
      np.allclose(from_unit_cube(np.ones((1, DIM)))[0], BOUNDS[:, 1]))

# --- every point yields a valid covariance ---
ok_pd = ok_sym = True
for row in x:
    _, cov, hbr = unpack(row)
    ok_pd &= bool(np.all(np.linalg.eigvalsh(cov) > 0))
    ok_sym &= bool(np.allclose(cov, cov.T))
    ok_pd &= hbr > 0
check("all 200 points give positive-definite covariance", ok_pd)
check("all covariances symmetric", ok_sym)

# Eigenvalues must come back as the sigmas that went in.
row = from_unit_cube(np.array([[0.3, 0.7, 0.25, 0.8, 0.4, 0.5]]))[0]
_, cov, _ = unpack(row)
vals = np.sort(np.sqrt(np.linalg.eigvalsh(cov)))
check("covariance eigenvalues recover the input sigmas",
      np.allclose(vals, np.sort([10**row[2], 10**row[3]])))

# --- the invariances the 4-D reduction depends on ---
# If either fails, reduce_to_4d is discarding real information.
rot_ok = scale_ok = True
max_rot_err = max_scale_err = 0.0
for row in from_unit_cube(rng.random((300, DIM))):
    mu, cov, hbr = unpack(row)
    base = pc_analytic(mu, cov, hbr)
    if base <= 0:
        continue

    # Rotation: rotate mu and cov together by an arbitrary angle.
    a = rng.uniform(0, 2 * np.pi)
    q = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    rot = pc_analytic(q @ mu, q @ cov @ q.T, hbr)
    max_rot_err = max(max_rot_err, abs(rot - base) / base)

    # Scale: multiply every length by s (covariance by s^2).
    s = rng.uniform(0.2, 5.0)
    sc = pc_analytic(s * mu, s * s * cov, s * hbr)
    max_scale_err = max(max_scale_err, abs(sc - base) / base)

check(f"Pc invariant under rotation (max rel err {max_rot_err:.2e})", max_rot_err < 1e-6)
check(f"Pc invariant under uniform scaling (max rel err {max_scale_err:.2e})", max_scale_err < 1e-6)

# --- the reduction preserves Pc ---
max_red_err = 0.0
for row in from_unit_cube(rng.random((200, DIM))):
    mu, cov, hbr = unpack(row)
    base = pc_analytic(mu, cov, hbr)
    if base <= 0:
        continue
    m1, m2, s1, s2 = reduce_to_4d(mu, cov, hbr)
    red = pc_analytic(np.array([m1, m2]), np.diag([s1 ** 2, s2 ** 2]), 1.0)
    max_red_err = max(max_red_err, abs(red - base) / base)
check(f"4-D reduction reproduces Pc exactly (max rel err {max_red_err:.2e})",
      max_red_err < 1e-6)

# Two encounters differing only by rotation/scale must reduce identically.
mu, cov, hbr = unpack(from_unit_cube(np.array([[0.2, 0.6, 0.3, 0.7, 0.15, 0.4]]))[0])
a, s = 0.9, 2.7
q = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
r1 = reduce_to_4d(mu, cov, hbr)
r2 = reduce_to_4d(s * (q @ mu), s * s * (q @ cov @ q.T), s * hbr)
check("reduction is invariant to rotation and scale", np.allclose(r1, r2, atol=1e-9))

# --- sign symmetry, which lets the 4-D box use d1, d2 >= 0 ---
max_sign_err = 0.0
for row in from_unit_cube(rng.random((200, DIM))):
    mu, cov, hbr = unpack(row)
    # Work in the eigenbasis, where the symmetry holds componentwise.
    vals, vecs = np.linalg.eigh(cov)
    mu_e = vecs.T @ mu
    cov_e = np.diag(vals)
    base = pc_analytic(mu_e, cov_e, hbr)
    if base <= 0:
        continue
    for flip in ([-1, 1], [1, -1], [-1, -1]):
        alt = pc_analytic(mu_e * np.array(flip), cov_e, hbr)
        max_sign_err = max(max_sign_err, abs(alt - base) / base)
check(f"Pc even in each mu component in the eigenbasis (max rel err {max_sign_err:.2e})",
      max_sign_err < 1e-6)

# --- 4-D primary space ---
u4 = rng.random((200, DIM_4D))
x4 = from_unit_cube_4d(u4)
check("4-D cube maps inside its box",
      bool(np.all(x4 >= BOUNDS_4D[:, 0] - 1e-12) and np.all(x4 <= BOUNDS_4D[:, 1] + 1e-12)))

ok_order = ok_pd4 = True
for row in x4:
    mu4, cov4, hbr4 = unpack_4d(row)
    vals4 = np.linalg.eigvalsh(cov4)
    ok_pd4 &= bool(np.all(vals4 > 0)) and hbr4 == 1.0
    ok_order &= bool(cov4[1, 1] <= cov4[0, 0] + 1e-12)
check("4-D points give positive-definite covariance with HBR = 1", ok_pd4)
check("4-D anisotropy parameterization enforces sigma_2 <= sigma_1", ok_order)

# A 6-D encounter reduced to 4-D, then unpacked, must give the same Pc.
max_rt = 0.0
for row in from_unit_cube(rng.random((200, DIM))):
    mu, cov, hbr = unpack(row)
    base = pc_analytic(mu, cov, hbr)
    if base <= 0:
        continue
    m1, m2, s1, s2 = reduce_to_4d(mu, cov, hbr)
    x = np.array([np.log10(s1), np.log10(s2 / s1), abs(m1) / s1, abs(m2) / s2])
    mu_r, cov_r, hbr_r = unpack_4d(x)
    max_rt = max(max_rt, abs(pc_analytic(mu_r, cov_r, hbr_r) - base) / base)
check(f"6-D -> 4-D -> Pc round trip is exact (max rel err {max_rt:.2e})", max_rt < 1e-6)

print()
print(f"{len(failures)} failure(s)" if failures else "all tests passed")
sys.exit(1 if failures else 0)
