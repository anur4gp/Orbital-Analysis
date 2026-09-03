"""One-time probe: what does this account actually get from Space-Track?

The decisive question for Phase 2 is whether `cdm_public` includes the state
covariance matrices. If it does, the Monte Carlo baseline samples real
operational uncertainty; if not, covariances have to be synthesized and that
becomes a stated assumption in the writeup.

Run: ./venv/bin/python src/probe_cdm.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from spacetrack import SpaceTrack

# CDM covariance is reported in the RTN/RSW frame: 21 lower-triangular terms
# per object, named like CR_R, CT_R, CT_T, ... for objects 1 and 2.
COV_PREFIXES = ("CR_", "CT_", "CN_", "CRDOT_", "CTDOT_", "CNDOT_")


def main() -> int:
    with SpaceTrack() as st:
        print("login OK\n")

        rows = st.query("cdm_public", orderby="TCA asc", limit=5)
        print(f"cdm_public: {len(rows)} row(s) returned")
        if not rows:
            print("  no rows -- account may not be entitled, or the class is empty")
            return 1

        fields = sorted(rows[0].keys())
        print(f"  {len(fields)} fields\n")

        cov = [f for f in fields if f.startswith(COV_PREFIXES) or "CO" == f[:2] and "VAR" in f]
        print(f"COVARIANCE FIELDS PRESENT: {len(cov)}")
        for f in cov:
            print(f"    {f} = {rows[0].get(f)!r}")
        if not cov:
            print("    none -- Phase 2 will need a synthesized covariance model")

        print("\nall fields:")
        for f in fields:
            value = str(rows[0].get(f))
            print(f"    {f:<28} {value[:48]}")

        print("\nsample event:")
        for key in ("TCA", "MISS_DISTANCE", "PC", "COLLISION_PROBABILITY",
                    "SAT_1_ID", "SAT_1_NAME", "SAT_2_ID", "SAT_2_NAME"):
            if key in rows[0]:
                print(f"    {key:<24} {rows[0][key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
