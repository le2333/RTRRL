# Real-Time Recurrent Reinforcement Learning

Code Appendix for the Paper "[Real-Time Recurrent Reinforcement Learning](https://arxiv.org/abs/2311.04830)" accepted for AAAI 2025.

## Install

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/)
2. Sync dependencies from the repository root:

```
uv sync
```

## Run

```
uv run python rtrrl.py
```

## Loggging

You can log results using [`aim`](https://aimstack.readthedocs.io/en/latest/index.html), [`wandb`](https://wandb.ai/), or both at once. Aim runs locally on the jump host; W&B is cloud-hosted and also drives hyperparameter sweeps.

You can enable logging by providing the `--logging` argument: `aim`, `wandb`, or `aim+wandb` (dual logging — metrics go to both).

```
uv run python rtrrl.py --logging aim
uv run python rtrrl.py --logging aim+wandb   # dual logging
```

W&B needs an API key: `wandb login` locally, or set `WANDB_API_KEY` (on AWS Batch it is injected from Secrets Manager — see `infra/README.md`). For hyperparameter sweeps see the "Hyperparameter sweeps" section of `infra/README.md`.

For PPO runs, set `logging: "aim"` and `log_repo` in the YAML config. For
example, `config/ppo_hopper_default_2m.yml` writes to `.aim`.

To view Aim results when this project is running on a remote machine, start the
Aim UI from the repository root in the remote project terminal:

```
uv run aim up --repo .aim --host 0.0.0.0 --port 43800
```

Then forward remote port `43800` through Cursor/SSH and open the forwarded local
address in your browser, usually:

```
http://localhost:43800
```

If Cursor shows an "Open in Browser" prompt for the forwarded port, use that
link. Use Aim's run list, metric filters, or the run hash printed by your
tracking workflow to find a specific run.

## Running on AWS Batch

To run training at scale on AWS Batch (EC2-backed CPU compute) with metrics
streamed to Aim and/or W&B (and W&B-driven hyperparameter sweeps), see the
**shared infra** repo [`../infra/README.md`](../infra/README.md). The infra +
HPO were extracted out of this repo into `../infra/` so they can be shared with
sibling projects (e.g. `memorax-rtrl`); this repo keeps only its image
definition (`infra/docker/`), its `project.env`, and its HPO data (`hpo/`).
After the one-time setup, submit runs from this project dir:

```
../infra/submit.sh --config config/rtrrl_brax_halfcheetah_paral1.yml --name my_run
```

Configs are injected at submit time (not baked into the image), so changing a
config needs no rebuild. **Note:** the pipeline pins `jax==0.5.0`; see the
caveats in `infra/README.md` before upgrading.

## CUDA support

A GPU can speed up training when using large batch sizes but will slow it down for smaller ones. 

The project pins JAX through its dependency graph. For JAX `0.5.0`, CUDA support
is provided by the plugin-based `jax[cuda12]` extra rather than a separate CUDA
`jaxlib` wheel. Install the CUDA 12 plugin into the uv environment with:

```
uv run poe install_jax_cuda12
```

Check the active backend and devices:

```
uv run python -c "import jax; print(jax.default_backend()); print(jax.devices())"
```

To force CPU execution even when a GPU backend is installed:

```
JAX_PLATFORM_NAME=cpu uv run python -c "import jax; print(jax.default_backend()); print(jax.devices())"
```

## Algorithm

![RTRRL steps](figures/RTRRL_steps.png)

## Hyperparameters
| Symbol      | Description                          | Default Value |
| ----------- | ------------------------------------ | ------------- |
| $\gamma$    | Discount factor.                     | 0.99          |
| $\alpha_{TD}$  | TD($\lambda$) learning rate.                 | 1e-5          |
| $\alpha_R$  | RNN learning rate.                   | 1e-5          |
| $\eta_H$    | Entropy rate.                        | 1e-5          |
| $\eta_A$    | Actor trace scaling.                        | 1.0          |
| $\lambda_A$ | Lambda for actor eligibility trace.  | 0.99           |
| $\lambda_C$ | Lambda for critic eligibility trace. | 0.99           |
| $\lambda_R$ | Lambda for RNN eligibility trace.    | 0.99          |


![3 step diagram](figures/Time_line.png)

## Configurables
This is an incomplete table of configurables.
Run `uv run python rtrrl.py --help` to find out more.

There is a preset for `brax` environments that can be used by providing the config path:

```
uv run python rtrrl.py --config_path configs/brax.yml
```

|Name | Description | Default Value |
| --- | ------- | -------- |
|debug| Enables debugging functionality. |False|
|env_name | Environment ID as defined by `gymnax` | 'CartPole-v1' |
|obs_mask| Allows masking of observation. Allowed values are None, 'even', 'odd', 'first_half' or a List of indices. | None|
|env_init_args| Arguments passed to environment constructor (e.g. size=16 for `DeepSea-bsuite`)   | - |
|env_params| Environment parameters passed to `step` and `reset` methods. (e.g. memory_length=32 for `MemoryChain-bsuite`) | - |
|rnn_model| Determines which RNN model is used. Set to None for vanilla TD($\lambda$). | 'CTRNN_simple'|
|hidden_size| RNN hidden state size. | 16|
|seed| Random seed for jax. Set to `None` for a random integer. |None|
|optimizer_params_td.opt_name| Optimizer used for tD($\lambda$). |'adam'|
|optimizer_params_rnn.opt_name| Optimizer used for the RNN. |'adam'|
|episodes| Total training episodes. |150_000|
|eval_every| Number of episodes between evaluation. |100|
|eval_steps| Number of evaluation steps. |10000|
|steps| Number of training steps per episode. |10000|
|max_ep_length| Max number of steps in episode. Specific environments may supersede this. |1000|
|patience| Early stopping is triggered after this number of evaluation episodes without improvement. |20|
|batch_size| Number of parallel environments. | 1|
|eta| Can be used for infinite horizon tasks. If set, average reward $\bar r$ is maintained and updated as $\bar r \gets \bar r + \eta\ \delta$. | None| 
|eta_pi| Scale gradients of action probability passed to RNN. | 1|
|eta_f| Scale gradients of RNN. | 1|
|entropy_rate| Scale gradient of action entropy. | 1|
|var_scaling| If True, scales the gradients of action probability by the scale of the action distribution. Only works for continuous actions. | False|
|gradient_mode| Select method for online gradient computation: 'RTRL', 'RFLO' or 'LocalMSE'. Ignored when LRU is used for `rnn_model`.| 'RFLO'|
|trace_mode| Type of eligibility trace. 'accumulate' or 'dutch'  | 'accumulate'|
|wiring| Specify wiring of RNN. See `modles/jax/wirings_jax.py` for available options. | 'fully_connected'|
|dt| Determines number of steps for forward Euler. e.g. 0.2 results in 5 steps. | 1|


