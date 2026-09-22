"""Probe which fields Space-Track's cdm_public returns (notably covariance).

Run: python scripts/probe_cdm.py
"""
from __future__ import annotations

from orbital.sgp4tools.spacetrack import SpaceTrack

# CCSDS CDM RTN covariance terms: CR_R, CT_R, CT_T, ...
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
            print("    none -- covariance must be synthesized")

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
