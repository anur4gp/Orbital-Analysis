# Running the campaign in a container

The image is a **campaign worker**: it computes design points and writes
parquet. It has no plotting stack and no display. Aggregation and figures are
a separate local step (`scripts/run_campaign.py --load <dir>`), which keeps
the image small and the batch job free of anything it does not need.

The MaxPro designs are committed CSV files (`reference/lhd/designs/`), so the
image needs neither pybind11 nor a C++ toolchain. A design is a static
artifact: it is optimised once and reused.

The consequence is that the image runs **only design sizes that are already
cached** (`--points 4, 16, 32, 64, 128, 256` for this 6-parameter sweep). Any
other size needs the parallel-tempering extension, which the image does not
carry; it fails with a message listing what is available rather than quietly
substituting a different design. To add a size, generate it where the
extension is built, commit the CSV, and rebuild.

## Build and run locally

```bash
docker build -f docker/Dockerfile -t orbital-campaign .   # from the repo root

docker run --rm -v "$PWD/data:/data" orbital-campaign \
    --points 64 --trials 8 --out /data/campaign/local
```

The image runs as an unprivileged user (uid 10001). On Linux a bind-mounted
host directory is owned by root, so add `--user "$(id -u):$(id -g)"` to write
into it; Docker Desktop on macOS maps ownership for you and does not need it.
Managed volumes (EFS, GCS via the batch services) are mounted writable.

Results land in `data/campaign/local/` (`runs.parquet` + `config.json`).
Plot them with:

```bash
python scripts/run_campaign.py --load data/campaign/local
```

`--workers` defaults to every visible core. Under a CPU quota, set it to the
vCPUs the job actually reserved; the image already pins BLAS to one thread
per process, so processes do not oversubscribe.

## Sharding

`--shards N` splits the design; `--shard i` picks the slice. Shard `i` takes
design points `i, i+N, i+2N, ...`, so shards stay similar in runtime even
though cost per point varies with cadence and elevation mask. Each shard
writes `runs.shard<i>of<N>.parquet` into the output directory, and the merge
step concatenates whatever is there.

`--shard` defaults to `AWS_BATCH_JOB_ARRAY_INDEX`, or `BATCH_TASK_INDEX` on
GCP, so the same command line works on both services unmodified.

Each design point derives its random stream from `(seed, index)`. Results
therefore do not depend on the number of shards, the number of workers, or
completion order: **a 20-container array job returns the same rows as one
local process.** Verified for this image: a 2-shard, 2-worker run is
bit-identical to a single-worker unsharded run.

Across *platforms* the numbers agree to about 1e-5 relative rather than
exactly. Dependency versions are pinned in the Dockerfile, but macOS links
Accelerate while this image links OpenBLAS, and the two differ in the last
bits of a dot product; a filter covariance amplifies that. Swept parameters
and acceptance bands are bit-identical. Pin one platform if you need
identical digits across machines.

## AWS Batch

Push the image, then register a job definition (Fargate or EC2; 1 vCPU and
2 GB is enough per shard):

```json
{
  "jobDefinitionName": "orbital-campaign",
  "type": "container",
  "platformCapabilities": ["FARGATE"],
  "containerProperties": {
    "image": "<account>.dkr.ecr.<region>.amazonaws.com/orbital-campaign:latest",
    "command": ["--points", "64", "--trials", "8", "--shards", "20",
                "--workers", "1", "--out", "/data/campaign/run1"],
    "resourceRequirements": [
      {"type": "VCPU", "value": "1"},
      {"type": "MEMORY", "value": "2048"}
    ],
    "executionRoleArn": "arn:aws:iam::<account>:role/ecsTaskExecutionRole",
    "jobRoleArn": "arn:aws:iam::<account>:role/orbital-campaign-task",
    "volumes": [{"name": "data",
                 "efsVolumeConfiguration": {"fileSystemId": "fs-xxxx",
                                            "rootDirectory": "/"}}],
    "mountPoints": [{"sourceVolume": "data", "containerPath": "/data"}]
  }
}
```

Submit it as an array job; AWS sets `AWS_BATCH_JOB_ARRAY_INDEX` per child, so
no per-shard command is needed:

```bash
aws batch submit-job \
  --job-name campaign-run1 \
  --job-queue orbital-queue \
  --job-definition orbital-campaign \
  --array-properties size=20
```

Shared storage: the snippet above mounts EFS so every shard writes into one
directory. The alternative, and usually the cheaper one, is to give each
shard `--out /tmp/out`, copy that file to S3 in a wrapper command, and let
the merge step read the prefix. Keep `--shards` equal to the array size, or
part of the design goes uncomputed — nothing in the image can detect that
for you.

## GCP Batch

The same image; `BATCH_TASK_INDEX` is read automatically.

```json
{
  "taskGroups": [{
    "taskCount": 20,
    "parallelism": 20,
    "taskSpec": {
      "runnables": [{
        "container": {
          "imageUri": "<region>-docker.pkg.dev/<project>/orbital/orbital-campaign:latest",
          "commands": ["--points", "64", "--trials", "8", "--shards", "20",
                       "--workers", "1", "--out", "/mnt/share/campaign/run1"]
        }
      }],
      "volumes": [{
        "gcs": {"remotePath": "my-bucket/campaign"},
        "mountPath": "/mnt/share"
      }],
      "computeResource": {"cpuMilli": 1000, "memoryMib": 2048}
    }
  }],
  "logsPolicy": {"destination": "CLOUD_LOGGING"}
}
```

```bash
gcloud batch jobs submit campaign-run1 --location us-central1 --config job.json
```

## Collecting the results

Put every shard's parquet in one directory and load it:

```bash
aws s3 sync s3://my-bucket/campaign/run1 data/campaign/run1   # or gsutil rsync
python scripts/run_campaign.py --load data/campaign/run1
```

`config.json` travels with the results and records the config, its
fingerprint, the git commit, and the library versions that produced them.
