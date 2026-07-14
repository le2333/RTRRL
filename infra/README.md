# Shared AWS infra (multi-project)

Standalone, **jax-free** orchestration + HPO shared by multiple training
projects (e.g. `streaming-rtrrl`, `memorax-rtrl`). Because none of this layer
imports jax, the projects' incompatible JAX versions never meet here — each
project ships its own training **image** (its `infra/docker/Dockerfile[.gpu]`),
distinguished only by the ECR **tag**. AWS resources (ECR repo, Batch queues,
IAM roles, S3, the Aim server, the W&B secret) are shared across projects.

## Per-project layering (`project.env`)

`env.sh` sets the common AWS resources, then sources the calling project's
`project.env` (found via `PROJECT_DIR`, default `$PWD`), which sets the
per-project bits: `PROJECT_NAME`, `IMAGE_TAG`, `GPU_IMAGE_TAG`, `DEFAULT_ENTRY`,
`WANDB_PROJECT`. Run the scripts from a project dir (or pass `--project PATH`):

```
cd ../streaming-rtrrl && ../infra/build-and-push.sh --gpu
cd ../streaming-rtrrl && ../infra/submit.sh --config config/rtrrl_hop.yml
cd ../memorax-rtrl    && ../infra/submit.sh --config config/stream_ac_rtrl_hopper_maskP.yml \
                          --entry experiments/stream_ac_mujoco_masked/run.py --queue rtrrl-gpu-queue --job-def rtrrl-gpu-job
```

`build-and-push.sh` stages the shared in-container helper `run_many.py` into the
project build context so it bakes into the image (`/app/run_many.py`); the staged
copy is gitignored in each project.

Pipeline: run training on **AWS Batch** (EC2-backed; CPU `c7a.xlarge` or GPU
`g5.2xlarge`), with metrics streamed live to **Aim** (on the jump host) and/or
**Weights & Biases**. W&B also drives **hyperparameter sweeps** (see below); the
main HPO lives in `hpo/` (Optuna + Aim, also project-agnostic — see `hpo/`).

**Not wired up yet (by choice):** S3 checkpoint sync and Aim run-resume across
Spot interruptions. Jobs run on on-demand EC2, so a run is not interrupted and
resume is unnecessary for now.

## Layout

```
infra/                     # <- this shared repo (jax-free)
├── env.sh                 # common AWS config; sources <project>/project.env
├── build-and-push.sh      # build a project's image -> ECR (--project, --gpu)
├── submit.sh              # submit one training run (--project, --entry, --config)
├── submit_many.sh         # submit N configs run sequentially in one container
├── run_many.py            # in-container helper (staged into the image at build)
├── sweep.yaml / sweep.sh  # W&B sweep definition + launcher (backup HPO path)
├── benchmark.sh           # instance-type benchmark helper
├── batch/create-batch.sh  # compute environment + queue + job definition
├── iam/                   # IAM roles/policies (see iam/README.md)
├── monitoring/            # CloudWatch agent config for the jump host
└── hpo/                   # Optuna+Aim HPO ENGINE (project-agnostic; see hpo/)

<project>/                 # e.g. streaming-rtrrl, memorax-rtrl
├── project.env            # IMAGE_TAG / GPU_IMAGE_TAG / DEFAULT_ENTRY / WANDB_PROJECT
├── infra/docker/          # THIS project's image (Dockerfile[.gpu] + entrypoint.sh)
└── hpo/                   # THIS project's HPO data: specs/ runs/ studies/ snapshots/
```

## How the image is built

Primary path is **GitHub Actions** (`.github/workflows/build-image.yml`): on a
push to `main` that touches `**.py`, `pyproject.toml`, `uv.lock`,
`infra/docker/**`, or the workflow itself, it assumes `rtrrl-github-actions-role`
via OIDC, creates the ECR repo if needed, builds the CPU image, and pushes
`rtrrl:cpu` and `rtrrl:<sha>`. Just push your changes; no local Docker needed.

`infra/build-and-push.sh` is only a manual fallback (needs Docker locally).

## One-time setup order

1. **IAM** — `infra/iam/setup-iam.sh` (creates `rtrrl-ecs-instance-role`,
   `rtrrl-batch-job-role`, `rtrrl-batch-execution-role`; control-plane perms
   live on the `controller` role).
2. **GitHub OIDC** — `infra/iam/setup-github-oidc.sh` (OIDC provider +
   `rtrrl-github-actions-role`, so Actions can push to ECR).
3. **(For W&B) Create the API-key secret** — once, with admin/console creds
   (keeps the key out of git and job overrides):
   ```bash
   aws secretsmanager create-secret --name rtrrl/wandb-api-key \
     --secret-string '<your-wandb-api-key>' --region eu-north-1
   ```
   Paste the returned **full ARN** into `WANDB_SECRET_ARN` in `infra/env.sh`.
   The execution role (`rtrrl-batch-execution-role`) already has read access.
4. **Edit `infra/env.sh`** — set `SUBNET_IDS`, `SECURITY_GROUP_ID`, `AIM_SERVER`
   (jump host private IP), `WANDB_SECRET_ARN` / `WANDB_PROJECT` / `WANDB_ENTITY`,
   default `LOGGING`, confirm `ECR_REPO`, `INSTANCE_TYPE`.
5. **Build image** — push to `main` and let GitHub Actions build it.
6. **Create Batch resources** — `infra/batch/create-batch.sh` (idempotent;
   re-registers the job definition each run). Re-run it after setting
   `WANDB_SECRET_ARN` so the job definition injects `WANDB_API_KEY`.
7. **(Optional) S3 bucket** — only needed once checkpointing is added:
   ```bash
   aws s3api create-bucket --bucket "$S3_BUCKET" --region eu-north-1 \
     --create-bucket-configuration LocationConstraint=eu-north-1
   ```

## Run training

1. **Start the Aim server on the jump host** (once; leave it running). It must be
   up *before* submitting, or metric logging from the job will fail:

   ```bash
   nohup uv run aim server --repo .aim --host 0.0.0.0 --port 53800 \
     > logs/aim-server.log 2>&1 &
   ```

2. **Submit runs** (one per config; loop for parallel runs). Each `--name`
   becomes its own run. Logging defaults to `$LOGGING` (`aim+wandb`); override
   with `--logging aim|wandb|aim+wandb`:

   ```bash
   # RTRRL (dual logging by default)
   infra/submit.sh --config config/rtrrl_brax_halfcheetah_paral1.yml --name hc_p1

   # PPO baseline, Aim only
   infra/submit.sh --entry ppo_baseline.py --logging aim \
     --config config/ppo_hopper_default_2m.yml --name ppo_hopper

   # Override params without a new config file (e.g. a seed sweep)
   for s in 1 2 3; do
     infra/submit.sh --config config/rtrrl_brax_halfcheetah_paral1.yml \
       --name hc_seed$s -- --seed $s
   done

   # Smoke test: a few episodes to validate the path quickly
   infra/submit.sh --config config/rtrrl_brax_halfcheetah_paral1.yml \
     --name smoke -- --episodes 30
   ```

3. **Watch a job**:

   ```bash
   aws batch describe-jobs --region eu-north-1 --jobs <jobId> \
     --query "jobs[0].{status:status,exit:container.exitCode,reason:container.reason,log:container.logStreamName}"

   # full container logs (CloudWatch)
   aws logs get-log-events --region eu-north-1 --log-group-name /aws/batch/job \
     --log-stream-name "<logStreamName>" --start-from-head \
     --query "events[].message" --output text
   ```

4. **View metrics in Aim** — on the jump host, start the UI and forward the port
   (see the repo root README "Logging" section):

   ```bash
   uv run aim up --repo .aim --host 0.0.0.0 --port 43800
   ```

   W&B runs appear at `https://wandb.ai/<entity>/RTRRL-PPO` (PPO) /
   `RTRRL` (rtrrl.py).

## Logging cadence (number of points)

Aim and W&B receive **exactly the same points** — one per *evaluation*, not per
training step. `brax` PPO calls `progress_fn` once per eval (plus an initial eval
at step 0), and `ppo_baseline.py` logs one point per call; there is also a single
`final/*` summary at the end. Per-gradient-step metrics are not logged.

The number of evals (= number of curve points) is:

- **`num_evals` in the config's `ppo_overrides`** if set (it overrides the
  default, because `ppo_overrides` is spread last into `ppo.train`), otherwise
- the **default** from `_default_num_evals`: `brax-hopper` → 1 eval per **1M**
  steps; other envs → 2 evals per **5M** steps.

So examples:

| config | steps | evals | points you see |
|---|---|---|---|
| `ppo_smoke.yml` (`num_evals: 1`) | 64 | 1 | **1** (single point — expected, not a bug) |
| `ppo_hopper_default_2m.yml` (default) | 2M | ~2 | ~2–3 |
| hopper 20M (default) | 20M | ~20 | ~20 |
| hopper 20M + `num_evals: 40` | 20M | 40 | ~40 (smoother curve) |

If a run shows only one point in W&B, check Aim too — if both show one point,
it's the eval cadence (e.g. a smoke config), not a logging problem. For a smooth
curve, raise `num_evals`:

```yaml
ppo_overrides:
  num_evals: 40   # 20M steps -> ~40 points
```

## Hyperparameter sweeps (W&B)

W&B Sweeps drive HPO; agents run as Batch jobs and pull trials from the W&B
controller. The default sweep is tuned for PPO HalfCheetah and maximizes
`hc_target_score`: high reward near 20M steps, peak reward close to 2500, and a
penalty for peaks that keep moving far past 25M.

Sweep parameters use **dotted keys** (e.g. `ppo_overrides.learning_rate`) so they
land in nested `PPOParams` fields. `logging_util.with_logger` reads
`wandb.config`, expands the dotted keys, and merges them into the dataclass
before training — so they reach `brax` `ppo.train()`. The sweep `command` has no
`${args}`, so params flow only through `wandb.config` (not the CLI).

```bash
# One-time on the jump host: authenticate W&B (key stored in ~/.netrc).
uv run wandb login

# 1) Create the sweep (edit infra/sweep.yaml to adjust ranges first).
infra/sweep.sh create
#    -> prints a sweep id like <entity>/RTRRL-PPO/abc123

# 2) Launch N parallel agents on Batch, each with a base HalfCheetah config
#    (env + num_timesteps come from this config; the sweep overrides
#    ppo_overrides.*).
infra/sweep.sh launch <entity>/RTRRL-PPO/abc123 4 \
  --config config/ppo_hc_036.yml --count 1
```

Each agent injects the base config (`CONFIG_B64` -> `/tmp/run-config.yml`) and
runs `wandb agent`, which executes `ppo_baseline.py` with the trial's params.
Increase `--count` to have one agent run several trials sequentially, or raise
the number of agents (and Batch `maxvCpus`) for more parallelism. Sweep runs log
to **W&B only** by default (the sweep lives there); add `aim` to the `command` in
`infra/sweep.yaml` if you also want them in the Aim repo.

## HPO control plane

Optuna-style HPO is intentionally kept out of the training containers. The jump
host can run a local control-plane tool in `control/hpo`: it reads Aim history,
uses Optuna with a local SQLite file to propose the next batch of configs, and
then optionally submits those configs to Batch. Workers only train and log to
Aim; they do not need Optuna or database credentials.

For short RTRRL sweeps, avoid repeatedly scanning the live Aim repo. Refresh the
local result cache only when needed:

```bash
uv run --project control/hpo python control/hpo/src/hpo_control/scheduler.py \
  sync-aim --spec control/hpo/specs/rtrrl_hop_stream.yaml
```

`suggest` reads `control/hpo/studies/<study>/results.jsonl` first, then falls
back to the Optuna SQLite history. It no longer scans live Aim by default.

If Aim UI/API queries slow down, snapshot the repo and clean bad RTRRL runs from
the active repo:

```bash
# Refuses to copy a live Aim repo unless --allow-live is passed.
uv run --project control/hpo python control/hpo/src/hpo_control/scheduler.py \
  snapshot-aim

uv run --project control/hpo python control/hpo/src/hpo_control/scheduler.py \
  aim-manifest --spec control/hpo/specs/rtrrl_hop_stream.yaml

uv run --project control/hpo python control/hpo/src/hpo_control/scheduler.py \
  cleanup-aim --spec control/hpo/specs/rtrrl_hop_stream.yaml --dry-run
```

Stop Aim before destructive cleanup, or pass `--allow-live` only when you have
confirmed no Batch jobs are writing to it.

RTRRL jobs keep Aim enabled for local UI, but `AimLogger.finalize()` no longer
blocks by default. Set `AIM_FINALIZE_BLOCK=1` only for debugging a single run
where you prefer a hard flush over avoiding stuck short sweep containers. If Aim
handshake timeouts continue after cleanup and the non-blocking image is deployed,
switch RTRRL sweeps to W&B-only or add an S3/JSONL summary path for HPO scoring.

## Caveats / gotchas

- **Do NOT upgrade JAX past 0.5.0.** `pyproject.toml` pins `jax==0.5.0` /
  `jaxlib==0.5.0` on purpose. JAX 0.6+ removed the `jax.jax` self-alias used by
  `envs/wrappers.py` (`jax.jax.tree.map`) and changed array types so `aim` 3.28
  can no longer track them. Re-running `uv lock` without the pin will silently
  re-break training. If you regenerate the lock, keep the pin.
- **The Aim server must be running on the jump host before you submit.** The job
  connects to `AIM_SERVER` (`aim://<jump-host-private-ip>:53800`) from
  `infra/env.sh`. If the jump host IP changes, update `AIM_SERVER`.
- **Compute environment scales to zero when idle** (`minvCpus=0`). The first job
  after idle waits ~1-2 min for an instance to launch; there is no cost while no
  job runs.
- **Configs are injected at submit time, not baked into the image.** `submit.sh`
  base64-encodes the chosen YAML into `CONFIG_B64`; the entrypoint decodes it to
  `/tmp/run-config.yml`. Editing or adding a config needs no image rebuild — just
  re-submit. (Configs are tiny YAML, so inlining beats S3.)
- **Builder needs a C++ toolchain.** `pytinyrenderer` (via `brax`) has no wheel
  and compiles from source, so the Dockerfile installs `build-essential` in the
  builder stage (stripped from the runtime image).
- **GitHub Actions build cache** requires the `docker-container` buildx driver
  (the workflow sets it up). The default docker driver cannot export `type=gha`.
- **W&B needs the API key in the container.** The job definition injects
  `WANDB_API_KEY` from Secrets Manager via the execution role
  (`rtrrl-batch-execution-role`). If `WANDB_SECRET_ARN` is empty, `create-batch.sh`
  registers the job def WITHOUT it — then `--logging wandb`/`aim+wandb` jobs fail.
  Set the ARN and re-run `create-batch.sh`. The key is never put in git or job
  overrides.

## Networking prerequisites

- The compute subnet must reach **ECR** and **S3**. Using a public subnet with
  auto-assigned public IPs (current setup) needs no NAT/VPC endpoints.
- The jump host security group must allow inbound TCP **53800** from the compute
  security group (Aim); the compute SG needs egress for ECR/S3 and to the jump
  host.
