# Experiment Tracking

This project uses Aim as the interactive tracking UI for local PPO/RTRRL
comparison work.

## PPO Baseline

Run the PPO baseline with the same config-loading style as `rtrrl.py`:

```bash
uv run python ppo_baseline.py --config_path config/ppo_smoke.yml
```

`config/ppo_smoke.yml` is only for smoke verification. It uses small explicit
`ppo_overrides` so the run finishes quickly. It is not a paper or reproduction
configuration.

For real PPO experiments, provide a separate YAML config with:

- `env_params.env_name`
- `env_params.max_ep_length`
- `env_params.obs_mask` when testing partial observations
- `env_params.init_kwargs.backend` for Brax backend selection
- `num_timesteps`
- `ppo_overrides` only for PPO settings you intentionally override

Unspecified PPO hyperparameters are left to Brax PPO defaults.
`ppo_overrides` cannot replace entrypoint-controlled arguments such as
`environment`, `num_timesteps`, `episode_length`, `progress_fn`, or `seed`,
because those fields must stay aligned with the config and Aim records.

`ppo_baseline.py` sets a default evaluation recording density as a probe, not
as a PPO training hyperparameter. Hopper uses 1 evaluation point per 1M
requested environment steps. Other environments use 2 evaluation points per 5M
requested environment steps. If needed, set `ppo_overrides.num_evals`
explicitly to choose a different logging density for a run.

## Aim UI

Start the Aim UI from this repository directory:

```bash
uv run aim up --repo logs/aim_smoke/.aim
```

Use the same `log_repo` value in the config and UI command. Smoke runs should
use a clearly named local repo such as `logs/aim_smoke/.aim`; real experiments
can use a separate repo path. Use the same repository path for training and the
Aim UI to avoid inspecting a different run store.

Aim runtime data is local generated output and should not be committed.

The project currently pins Aim to `3.28.0`. In the local Python 3.10
environment, Aim `3.29.1` created run metadata but did not make tracked metrics
queryable after the process exited.

## Shared Comparison Fields

PPO records shared comparison fields for later PPO/RTRRL analysis:

- `algorithm`
- `env_name`
- `obs_mask`
- `backend`
- `seed`
- `requested_steps`
- `actual_final_step`
- `train/env_steps`
- `train/native_step`
- `eval/episode_reward`
- `eval/best_eval_reward`
- `final/best_eval_reward`

`train/env_steps` is the main comparison axis for PPO. PPO currently uses the
Brax PPO `progress_fn` `num_steps` value for both `train/env_steps` and
`train/native_step`.

## PPO/RTRRL Mapping

PPO and RTRRL should share comparison fields where semantics match, but their
native algorithm fields remain separate.

| Meaning | PPO field | RTRRL current field or mapping |
| --- | --- | --- |
| Main comparison axis | `train/env_steps` | RTRRL computes `total_steps = episodes * steps * batch_size` in metrics as `steps`; logger step remains native `log_steps` |
| Native algorithm step | `train/native_step` | RTRRL logger `step` argument currently uses `log_steps = episodes * steps` |
| Evaluation reward | `eval/episode_reward` | RTRRL currently logs `eval/rewards` |
| Best evaluation reward | `eval/best_eval_reward` | RTRRL currently logs `eval/best_eval_reward` |
| Final best reward | `final/best_eval_reward` | RTRRL returns `best_eval_reward`; final field is not standardized yet |

This Issue does not change RTRRL logging behavior. It documents the mapping so
future comparison or plotting code can treat PPO and RTRRL carefully instead of
forcing mismatched step semantics into the same field.

The shared Aim logger records the numeric step passed by training code as Aim's
`step` field. RTRRL still passes its existing native `log_steps` value; this
change only affects how Aim stores the axis, not RTRRL learning behavior.

## JSONL Status

`ppo_baseline.py` does not write a default hand-made JSONL runtime log. Aim is
the intended runtime tracker for PPO runs in this Issue.

The historical `logs/ppo_runs` directories (early JSONL-phase PPO baseline
artifacts, not Aim runs) have been removed; Aim is now the single source of
truth for PPO baselines.
