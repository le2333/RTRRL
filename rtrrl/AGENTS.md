# Agent guide

Notes for AI agents working in this repository. Keep this short and current.

## What this is

RTRRL / PPO reinforcement-learning code (JAX/Flax/Brax) plus an `infra/` pipeline
to run training on AWS Batch with metrics streamed to an Aim server on a jump
host. Entry points: `rtrrl.py`, `ppo_baseline.py`. Configs in `config/*.yml`.

## Dependencies: uv, and JAX is pinned — do not bump

- Use **uv** (`uv sync`, `uv run ...`), not pip/poetry.
- **`pyproject.toml` pins `jax==0.5.0` / `jaxlib==0.5.0`. Do NOT upgrade.**
  JAX 0.6+ removed the `jax.jax` self-alias used by `envs/wrappers.py`
  (`jax.jax.tree.map`) and changed array types so `aim` 3.28 can no longer track
  metrics. If you regenerate `uv.lock`, **keep the pin** or training breaks at
  runtime (not at lock/build time).
- Lesson learned: when the upstream training code fails after a dependency
  change, **fix the dependency versions to match upstream** rather than patching
  the training scripts. The only intentional source change for the pipeline is
  one line in `rtrrl.py` (`aim_repo=hparams.log_repo`) needed for remote Aim.

## AWS pipeline (see `../infra/README.md` for details)

- **Shared infra:** the orchestration scripts + HPO engine were extracted to the
  shared `../infra/` repo (jax-free). This repo keeps only `infra/docker/` (its
  image), `project.env` (its tags/entry/W&B project), and `hpo/` (its HPO data).
  Run scripts from this dir, e.g. `../infra/submit.sh ...`; `env.sh` sources
  `project.env`.
- **Image build:** GitHub Actions (`.github/workflows/build-image.yml`) builds
  and pushes `rtrrl:cpu` to ECR on push to `main`. No local Docker needed.
- **Configs are injected at submit time, not baked into the image.** `submit.sh`
  base64-encodes the YAML into `CONFIG_B64`; `infra/docker/entrypoint.sh` decodes
  it to `/tmp/run-config.yml`. `config/`, `docs/`, `figures/` are excluded via
  `.dockerignore`. Editing a config needs no rebuild.
- **Submit a run:** `../infra/submit.sh --config <cfg> --name <run>`; each `--name`
  is its own run. `--logging aim|wandb|aim+wandb` (default `$LOGGING`). Loop for
  parallel runs.
- **Aim server must be running on the jump host before submitting** (it listens
  on `:53800`; the job connects via `AIM_SERVER` in `../infra/env.sh`).
- Batch compute scales to zero when idle; first job after idle waits ~1-2 min.
- IAM: control-plane perms are on the `controller` role (jump host). Some changes
  require updating its inline policies via the AWS console (see `infra/iam/`).

## Logging + W&B sweeps

- `logging_util.with_logger` supports **dual logging**: Aim (local, for
  programmatic/AI reading) + W&B (cloud, for analysis). `MultiLogger` fans calls
  out to both; reads come from the first logger.
- **Logging cadence:** one point per *eval* (not per step), same points to Aim
  and W&B. Count = `ppo_overrides.num_evals` if set, else `_default_num_evals`
  (hopper: 1/1M steps; others: 2/5M steps). A smoke run (`num_evals=1`) logging a
  single point in both backends is expected, not a bug. Raise `num_evals` for a
  denser curve. See "Logging cadence" in `infra/README.md`.
- **W&B sweeps** (`../infra/sweep.yaml`, `../infra/sweep.sh`) do HPO on `ppo_baseline.py`,
  maximizing `eval/best_eval_reward`. Sweep params use **dotted keys**
  (`ppo_overrides.learning_rate`); `with_logger` expands them (`_expand_dotted`)
  and merges into the dataclass so they reach `brax` `ppo.train()`. The sweep
  `command` must NOT include `${args}` (params flow via `wandb.config`, not CLI —
  `simple_parsing` can't parse nested flags). PPO hyperparameters live in the
  free-form `ppo_overrides: dict`, NOT top-level `PPOParams` fields; a flat sweep
  key would be silently dropped by dacite.
- **`WANDB_API_KEY`** is injected into Batch jobs from Secrets Manager
  (`rtrrl/wandb-api-key`) via `rtrrl-batch-execution-role`; never put it in git or
  job overrides. `create-batch.sh` only wires it when `WANDB_SECRET_ARN` is set in
  `../infra/env.sh` (re-run after setting it). `wandb.init` uses `mode=disabled` when
  `debug`, else honors `WANDB_MODE` (default online).

## Experiment execution rules (binding)

These rules are mandatory for every AWS Batch experiment.

1. **Run only from `config/`.** Any config submitted to AWS Batch must be a
   repo-root `config/*.yml` file named by the experiment ID. Selected scan
   candidates must be copied/renamed into `config/` before submission. Never run
   from `/tmp`, an HPO run dir, or an in-memory config.

2. **Never delete experiment configs.** Generated, selected, or launched configs
   are audit records. Do not remove them to clean a round. If a config is missing,
   say so; do not hide the loss.

3. **Only prompted confirmation authorizes a run.** A run is authorized only when
   the AI explicitly asks whether to run/submit a clearly listed set of configs
   and the user replies yes to that question. User messages such as "选",
   "按老规矩选", "看 1/3", "继续", "可以", "同意", or an unprompted "跑" do not
   authorize submission. If scope or authorization is ambiguous, ask before doing
   anything.

4. **No automatic submit/re-submit.** Do not auto-continue experiments, retry a
   failed/terminated job, cancel-and-resubmit, or launch extra variants without a
   fresh explicit run command.

5. **Non-scan runs require a pre-flight diff and a stop.** For extensions or
   reproductions, derive the new config from the on-disk baseline config in
   `config/`, show `diff -u <baseline> <new_config>`, then stop and wait. Do not
   submit in the same turn as the diff. Optuna/scan candidates are exempt from
   this diff requirement.

6. **Use on-disk configs to build configs.** Do not build new run configs from
   Aim hparams or memory. Aim/Optuna may be used only to verify historical values
   or scores when needed.

7. **Before any submit, show only the required audit view and ask for confirmation.**
   For a non-scan run, show the diff against the base config. For a sweep/scan
   selection, show only the selected experiment numbers and their selection
   scores (e.g. acquisition and/or distance). Do not include other submission
   details unless the user asks. End with an explicit yes/no run question and wait.

8. **Sweep/scan display has two stages.** First show the full candidate list and
   wait for the user to confirm the selection rule. Only then show the filtered
   list. The filtered list must emphasize the mapping from suggestion index
   (`config_00N`) to final experiment ID (`PPO-HC-NNN` / `config/ppo_hc_NNN.yml`).

## Conventions

- Don't commit unless asked. Don't commit Aim run data (`.aim/`, `logs/aim/`).
- Keep changes minimal and faithful to the upstream paper code.
