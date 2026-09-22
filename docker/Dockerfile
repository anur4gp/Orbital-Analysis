# Campaign worker (no plotting, no compiler; MaxPro designs are committed CSV).
#
# Build (from the repository root):
#   docker build -f docker/Dockerfile -t orbital-campaign .
# Run locally:
#   docker run --rm -v "$PWD/data:/data" orbital-campaign --points 16
# Results land in /data/<campaign id>; pass --out to choose the directory.
# See docker/README.md for AWS Batch and GCP Batch.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    ORBITAL_PROJECT_ROOT=/app

WORKDIR /app

COPY pyproject.toml README.md ./
# Pinned to match the local environment. Results match macOS to ~1e-5
# relative (OpenBLAS vs Accelerate); within a platform they are bit-identical.
RUN python -m pip install --upgrade pip \
 && python -m pip install \
      "numpy==2.0.2" "scipy==1.13.1" "pandas==2.3.3" "pyarrow==25.0.1"

COPY src ./src
COPY reference/lhd/designs ./reference/lhd/designs
RUN python -m pip install --no-deps -e .

# Parallelism comes from processes; avoid BLAS oversubscription.
ENV OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    ORBITAL_CAMPAIGN_OUT=/data

RUN useradd --create-home --uid 10001 runner && mkdir -p /data && chown runner /data
USER runner
VOLUME ["/data"]

ENTRYPOINT ["python", "-m", "orbital.campaign"]
CMD ["--help"]
