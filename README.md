# Packaging surface: `memo/` + `infra/`

Branch for the **training image** (`memo/`) and the **control plane** (`infra/` / `trainerctl`).  
Legacy AAAI tree (`rtrrl/`) is intentionally absent here so it is not confused with the memo worker.

Full detail: [`docs/trainerctl-manual.md`](docs/trainerctl-manual.md). Contract: [`docs/contract.md`](docs/contract.md).

## Layout

| Path | Role |
|------|------|
| `memo/` | Image contents: entries, worker, `catalog.json`, Dockerfiles |
| `infra/` | `trainerctl` — sample HPO, submit Batch/local rounds, score |
| `experiments/` | Example / probe experiment YAMLs |
| `tests/contracts/` | Shared contract fixtures used by infra tests |
| `.github/workflows/` | Memo image build + memo/infra CI |

Experiment YAMLs for a study live **outside** this tree (or under `experiments/`). Point `trainerctl` at the file path you use.

## Prerequisites

- [uv](https://docs.astral.sh/uv/) (Python 3.11 for `infra`)
- For Batch: AWS credentials for account `007122174918`, region **`eu-north-1`** (fixed in code)
- Aim URL reachable from Batch workers (private IP, not `127.0.0.1`)
- ECR image + Batch job definition registered for that digest (see Ops below)

## Install control plane

```bash
git clone -b packaging/memo-infra https://github.com/le2333/RTRRL.git
cd RTRRL/infra
uv sync
uv run trainerctl --help
```

## Run an experiment

Catalog shipped with the image source (also baked into the image):

```bash
cd infra
uv run trainerctl run /path/to/experiment.yml \
  --backend batch \
  --catalog ../memo/catalog.json \
  --database /path/to/studies.sqlite \
  --queues run
```

Local backend (needs exchange + workspace; ignores Batch queues):

```bash
uv run trainerctl run /path/to/experiment.yml \
  --backend local \
  --catalog ../memo/catalog.json \
  --database /path/to/studies.sqlite \
  --exchange /path/to/exchange \
  --workspace /path/to/workspace
```

Recover scores for trials a stopped controller left running (no new submits):

```bash
uv run trainerctl settle /path/to/experiment.yml \
  --launch-id <launch_id> \
  --backend batch \
  --catalog ../memo/catalog.json \
  --database /path/to/studies.sqlite \
  --queues run
```

`experiment.yml` must set `image:` to an ECR digest that exists, `entry:` from the catalog, and `storage:` (`s3://…` for Batch, `file://…` for local). See the manual for the full schema.

## Build / publish the memo image

Images are built in GitHub Actions (not on the laptop).

- **Verify only:** push commits that touch `memo/**` to a branch that triggers the workflow, or dispatch without push.
- **Publish to ECR** (manual):

```bash
gh workflow run build-memo-image.yml \
  --ref packaging/memo-infra \
  -f push=true \
  -f confirm_account=007122174918
```

CPU Dockerfile: `memo/docker/Dockerfile.cpu`. GPU variant is parked (XLA fusion abort).

After a new digest is in ECR, register the Batch job definition  
`trainer-{profile}-{digest without sha256:}` for each instance profile you use — `trainerctl` submits to that name; it does not create it.

## Tests

```bash
cd infra
uv run pytest
```

Memo suites and infra CI also run via `.github/workflows/memo-ci.yml` and `tests.yml`.

## Note on job names

Experiment `name` values may contain `.` (e.g. Adam beta tags). This branch sanitizes AWS Batch `jobName` to `[A-Za-z0-9_-]` before `submit_job`.
