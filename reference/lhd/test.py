"""
Quick sanity test for the pt_maxpro pybind11 extension.
 
Run after building the extension (see build instructions), from the same
directory the .pyd / .so file was placed in, e.g.:
 
    python test_pt_maxpro.py
"""
import numpy as np
import time
import pt_maxpro
 


# depending on the research you are doing, you might want to customize this.
# usually experimenting with temperature steepness.
def make_temp_schedule(M, t_min=0.05, t_max=5.0):
    return np.geomspace(t_min, t_max, M)

 
def main():
    n = 20          # number of design points (rows)
    k = 4           # number of factors/dimensions (columns)
    M = 10           # number of parallel-tempering replicas
    Nmax = 20000      # max iterations
    Nswap = 10        # swap attempt frequency
    tolerance = 5000  # stop early if no global-min improvement for this many iters
 
    temp_sched = make_temp_schedule(M)

    print(f"Design points: {n}")
    print(f"Factors: {k}")
    print(f"Replicas: {M}")
    print(f"Max iterations: {Nmax}")
    print(f"Iterations til swap: {Nswap}")
    print(f"Tolerance {tolerance}")
    print("Initial temp_sched:", temp_sched)
 
    t0 = time.perf_counter()
    result = pt_maxpro.pt_maxpro_lhd(n, k, M, Nmax, Nswap, tolerance, temp_sched)
    elapsed = time.perf_counter() - t0
 
    design = result["design"]
    measure = result["measure"]
    t0_final = result["t0"]
    ntotal = result["ntotal"]
 
    print(f"\nRan in {elapsed:.4f}s, {ntotal} iterations")
    print("Design shape:", design.shape, "dtype:", design.dtype)
    print("Measure (psi, maxpro):", measure)
    print("Final temp_sched:", t0_final)
 
    assert design.shape == (n, k)
    assert np.all(design >= 0.0) and np.all(design <= 1.0), "design values out of [0,1] range"


    print("final design")
    print(design)
 
 
main()
