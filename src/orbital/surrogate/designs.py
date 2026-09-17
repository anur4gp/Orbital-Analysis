"""Space-filling designs for the Phase 3 surrogate.

Wraps the RUSIS parallel-tempering MaxPro code (reference/lhd/pt_maxpro.cpp,
built as a pybind11 extension) and provides the comparison baselines the
benchmark needs: a plain random Latin hypercube and plain uniform sampling.

Designs are cached as CSV in reference/lhd/designs/ -- a design is a static
artifact, so the optimizer runs once and never again on the runtime path.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from orbital.paths import REFERENCE_DIR

LHD_DIR = REFERENCE_DIR / "lhd"
DESIGN_DIR = LHD_DIR / "designs"

# The compiled extension lives next to its source, not on the default path.
if str(LHD_DIR) not in sys.path:
    sys.path.insert(0, str(LHD_DIR))

try:
    import pt_maxpro
    HAVE_PT_MAXPRO = True
except ImportError:
    HAVE_PT_MAXPRO = False


def temperature_schedule(m: int, t_min: float = 0.05, t_max: float = 5.0) -> np.ndarray:
    """Geometric ladder of replica temperatures.

    Parameters
    ----------
    m
        Number of replicas.
    t_min, t_max
        Coldest and hottest temperatures, matching the reference test.

    Returns
    -------
    numpy.ndarray
        Temperatures, shape (m,).
    """
    return np.geomspace(t_min, t_max, m)


def maxpro_criterion(design: np.ndarray) -> float:
    """MaxPro measure psi -- lower is better.

    Mirrors psi() in pt_maxpro.cpp: the average reciprocal product of squared
    per-dimension gaps. Recomputed here in numpy so designs can be scored
    without the extension, and so the extension's own value is checked.
    """
    n, k = design.shape
    diff = design[:, None, :] - design[None, :, :]
    prod = np.prod(diff ** 2, axis=2)
    iu = np.triu_indices(n, k=1)
    return float((2.0 / (n * (n - 1)) * np.sum(1.0 / prod[iu])) ** (1.0 / k))


def maximin_criterion(design: np.ndarray, p: float = 15.0) -> float:
    """Morris-Mitchell phi_p criterion -- lower is better.

    Approaches 1 / min-distance as ``p`` grows. Mirrors phi() in
    maximin_PTLHD.cpp.

    Parameters
    ----------
    design
        Design points in the unit cube, shape (n, k).
    p
        Exponent; larger values weight the closest pair more heavily.

    Returns
    -------
    float
    """
    n = design.shape[0]
    diff = design[:, None, :] - design[None, :, :]
    d2 = np.sum(diff ** 2, axis=2)
    iu = np.triu_indices(n, k=1)
    return float((2.0 / (n * (n - 1)) * np.sum(d2[iu] ** (-p * 0.5))) ** (1.0 / p))


def min_distance(design: np.ndarray) -> float:
    """Smallest pairwise Euclidean distance -- the quantity maximin maximizes."""
    n = design.shape[0]
    diff = design[:, None, :] - design[None, :, :]
    d = np.sqrt(np.sum(diff ** 2, axis=2))
    iu = np.triu_indices(n, k=1)
    return float(d[iu].min())


def random_lhd(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """Unoptimized Latin hypercube: one random permutation per column.

    The baseline that isolates what the parallel-tempering search buys --
    same stratification, no space-filling optimization.

    Parameters
    ----------
    n
        Number of design points.
    k
        Number of dimensions.
    rng
        Random generator; seed it for reproducibility.

    Returns
    -------
    numpy.ndarray
        Design in the unit cube, shape (n, k).
    """
    levels = (np.arange(n) + 0.5) / n
    return np.column_stack([rng.permutation(levels) for _ in range(k)])


def uniform_sample(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """Plain i.i.d. uniform sampling -- the Monte Carlo baseline.

    Parameters
    ----------
    n
        Number of points.
    k
        Number of dimensions.
    rng
        Random generator; seed it for reproducibility.

    Returns
    -------
    numpy.ndarray
        Points in the unit cube, shape (n, k).
    """
    return rng.random((n, k))


def generate_maxpro(
    n: int, k: int, m: int = 10, n_max: int = 100_000,
    n_swap: int = 100, tolerance: int = 20_000,
) -> tuple[np.ndarray, float, int]:
    """Run the parallel-tempering MaxPro search.

    Parameters
    ----------
    n
        Number of design points.
    k
        Number of dimensions.
    m
        Replica count for the temperature ladder.
    n_max
        Iteration cap.
    n_swap
        Iterations between replica swap attempts.
    tolerance
        Iterations without improvement before stopping.

    Returns
    -------
    design : numpy.ndarray
        Optimised design in the unit cube, shape (n, k).
    measure : float
        Its MaxPro criterion value.
    iterations : int
        Iterations actually performed.

    Raises
    ------
    RuntimeError
        If the pt_maxpro extension is not built.
    """
    if not HAVE_PT_MAXPRO:
        raise RuntimeError(
            "pt_maxpro extension not built. From reference/lhd/ run:\n"
            "  ../../venv/bin/python setup.py build_ext --inplace"
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
    """Load a cached design, generating and saving it on first use.

    Parameters
    ----------
    n
        Number of design points.
    k
        Number of dimensions.
    name
        Cache name; ``maxpro`` for the optimised designs.
    force
        Regenerate and overwrite an existing cache entry.
    **kwargs
        Passed to :func:`generate_maxpro`.

    Returns
    -------
    numpy.ndarray
        Design in the unit cube, shape (n, k).
    """
    path = design_path(name, n, k)
    if path.exists() and not force:
        return np.loadtxt(path, delimiter=",")
    design, _, _ = generate_maxpro(n, k, **kwargs)
    DESIGN_DIR.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, design, delimiter=",", fmt="%.10f")
    return design


def is_latin_hypercube(design: np.ndarray, tol: float = 1e-9) -> bool:
    """Whether every column is a permutation of the ``(i - 0.5)/n`` levels.

    Parameters
    ----------
    design
        Candidate design, shape (n, k).
    tol
        Absolute tolerance on each level.

    Returns
    -------
    bool
    """
    n, k = design.shape
    expected = (np.arange(n) + 0.5) / n
    return all(np.allclose(np.sort(design[:, j]), expected, atol=tol) for j in range(k))
