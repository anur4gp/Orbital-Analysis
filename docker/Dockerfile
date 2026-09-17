# Campaign worker: computes design points and writes parquet. No plotting, no
# display, no compiler -- the MaxPro designs are committed CSV artifacts, so
# the image never needs the C++ extension or its toolchain.
#
# Build (from the repository root):
#   docker build -f docker/Dockerfile -t orbital-campaign .
# Run locally:
#   docker run --rm -v "$PWD/data:/data" orbital-campaign --points 16
# Results land in /data/<campaign id>; pass --out to choose the directory.
# See docker/README.md for AWS Batch and GCP Batch.
FROM python:3.12-slim

# Fail fast and keep logs unbuffered, so batch-service log tails are live.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    ORBITAL_PROJECT_ROOT=/app

WORKDIR /app

# Dependencies first: this layer is cached unless pyproject.toml changes.
COPY pyproject.toml README.md ./
# Pinned to the versions the local environment uses, so a container result
# matches a local one as closely as the platform allows. Exact bitwise
# equality across platforms is not achievable here: macOS links Accelerate
# and this image links OpenBLAS, which differ in the last bits of each dot
# product. Measured effect: parameters identical, metrics within ~1e-5
# relative. Determinism *within* a platform is exact and is tested.
RUN python -m pip install --upgrade pip \
 && python -m pip install \
      "numpy==2.0.2" "scipy==1.13.1" "pandas==2.3.3" "pyarrow==25.0.1"

# Source and the cached space-filling designs.
COPY src ./src
COPY reference/lhd/designs ./reference/lhd/designs
RUN python -m pip install --no-deps -e .

# One thread per BLAS process: the runner gets its parallelism from separate
# processes, and nested threading oversubscribes a batch vCPU allocation.
ENV OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    ORBITAL_CAMPAIGN_OUT=/data

# Unprivileged, and /data is where results are expected to be mounted.
RUN useradd --create-home --uid 10001 runner && mkdir -p /data && chown runner /data
USER runner
VOLUME ["/data"]

ENTRYPOINT ["python", "-m", "orbital.campaign"]
CMD ["--help"]
