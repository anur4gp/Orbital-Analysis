"""Space-filling designs: parallel-tempering MaxPro (``reference/lhd``) and baselines.

Designs are cached as CSV in ``reference/lhd/designs/``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from orbital.paths import REFERENCE_DIR

LHD_DIR = REFERENCE_DIR / "lhd"
DESIGN_DIR = LHD_DIR / "designs"

if str(LHD_DIR) not in sys.path:
    sys.path.insert(0, str(LHD_DIR))

try:
    import pt_maxpro
    HAVE_PT_MAXPRO = True
except ImportError:
    HAVE_PT_MAXPRO = False


def temperature_schedule(m: int, t_min: float = 0.05, t_max: float = 5.0) -> np.ndarray:
    """Geometric ladder of ``m`` replica temperatures."""
    return np.geomspace(t_min, t_max, m)


def maxpro_criterion(design: np.ndarray) -> float:
    """MaxPro criterion psi (lower is better); mirrors ``psi()`` in pt_maxpro.cpp."""
    n, k = design.shape
    diff = design[:, None, :] - design[None, :, :]
    prod = np.prod(diff ** 2, axis=2)
    iu = np.triu_indices(n, k=1)
    return float((2.0 / (n * (n - 1)) * np.sum(1.0 / prod[iu])) ** (1.0 / k))


def maximin_criterion(design: np.ndarray, p: float = 15.0) -> float:
    """Morris-Mitchell phi_p (lower is better); mirrors ``phi()`` in maximin_PTLHD.cpp."""
    n = design.shape[0]
    diff = design[:, None, :] - design[None, :, :]
    d2 = np.sum(diff ** 2, axis=2)
    iu = np.triu_indices(n, k=1)
    return float((2.0 / (n * (n - 1)) * np.sum(d2[iu] ** (-p * 0.5))) ** (1.0 / p))


def min_distance(design: np.ndarray) -> float:
    """Smallest pairwise Euclidean distance."""
    n = design.shape[0]
    diff = design[:, None, :] - design[None, :, :]
    d = np.sqrt(np.sum(diff ** 2, axis=2))
    iu = np.triu_indices(n, k=1)
    return float(d[iu].min())


def random_lhd(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """Unoptimized Latin hypercube (n, k): one random permutation per column."""
    levels = (np.arange(n) + 0.5) / n
    return np.column_stack([rng.permutation(levels) for _ in range(k)])


def uniform_sample(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """I.i.d. uniform points in the unit cube, shape (n, k)."""
    return rng.random((n, k))


def generate_maxpro(
    n: int, k: int, m: int = 10, n_max: int = 100_000,
    n_swap: int = 100, tolerance: int = 20_000,
) -> tuple[np.ndarray, float, int]:
    """Run the parallel-tempering MaxPro search; returns ``(design, psi, iterations)``.

    Parameters
    ----------
    m
        Number of replicas.
    n_max
        Iteration cap.
    n_swap
        Iterations between replica swaps.
    tolerance
        Iterations without improvement before stopping.
    """
    if not HAVE_PT_MAXPRO:
        raise RuntimeError(
            "pt_maxpro extension not built. From reference/lhd/ run:\n"
            "  python setup.py build_ext --inplace"
        )
    result = pt_maxpro.pt_maxpro_lhd(
        n, k, m, n_max, n_swap, tolerance, temperature_schedule(m)
    )
    return np.ascontiguousarray(result["design"]), float(result["measure"]), int(result["ntotal"])


def design_path(name: str, n: int, k: int) -> Path:
    """Cache path for the design of size ``n`` in ``k`` dimensions."""
    return DESIGN_DIR / f"{name}_n{n}_k{k}.csv"


def load_or_generate(n: int, k: int, name: str = "maxpro", force: bool = False,
                     **kwargs: int) -> np.ndarray:
    """Load a cached design, generating and caching it on first use."""
    path = design_path(name, n, k)
    if path.exists() and not force:
        return np.loadtxt(path, delimiter=",")
    design, _, _ = generate_maxpro(n, k, **kwargs)
    DESIGN_DIR.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, design, delimiter=",", fmt="%.10f")
    return design


def is_latin_hypercube(design: np.ndarray, tol: float = 1e-9) -> bool:
    """Whether every column is a permutation of the ``(i - 0.5)/n`` levels."""
    n, k = design.shape
    expected = (np.arange(n) + 0.5) / n
    return all(np.allclose(np.sort(design[:, j]), expected, atol=tol) for j in range(k))
