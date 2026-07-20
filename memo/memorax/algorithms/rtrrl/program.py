"""Closed scan/JIT orchestration and scalar aggregation for strict RTRRL."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any, cast

from flax import struct
from flax.core import freeze
import jax
import jax.numpy as jnp

from memorax.online_ac.types import (
    ActionDecision,
    AgentProgram,
    EvalSummary,
)

from .compatibility import RTRRLComponentConfig
from .heads import make_action_distribution
from .lru import AAAI25LRU
from .state_machine import (
    _flat_recurrent_parameters,
    _pack_recurrent_state,
    _step_model_input,
    _unpack_recurrent_state,
    make_init_fn,
    make_step_fn,
)
from .types import RTRRLComponents, RTRRLState, TrainStepMetrics


@struct.dataclass
class RTRRLEnvironmentState:
    """Fixed JAX view of a legacy Gymnax-style vector environment."""

    obs: Any
    reward: Any
    done: Any
    inner_state: Any
    keys: Any
    observation_mean: Any
    observation_m2: Any
    observation_count: Any
    reward_mean: Any
    reward_m2: Any
    reward_count: Any
    discounted_return: Any


class LegacyRTRRLEnvironmentAdapter:
    """Translate legacy keyed reset/step calls outside the numerical core."""

    def __init__(
        self,
        env: Any,
        env_params: Any,
        num_envs: int,
        *,
        normalize_observation: bool = False,
        normalize_reward: bool = False,
        eps: float = 1e-8,
        reward_gamma: float = 0.99,
    ):
        self.env = env
        self.env_params = env_params
        self.num_envs = num_envs
        self.normalize_observation = normalize_observation
        self.normalize_reward = normalize_reward
        self.eps = eps
        self.reward_gamma = reward_gamma

    def reset(self, key):
        reset_root, step_root = jax.random.split(key)
        reset_keys = jax.random.split(reset_root, self.num_envs)
        step_keys = jax.random.split(step_root, self.num_envs)
        observation, inner_state = jax.vmap(
            self.env.reset, in_axes=(0, None)
        )(reset_keys, self.env_params)
        observation_mean = jnp.zeros_like(observation)
        observation_m2 = jnp.ones_like(observation)
        observation_count = jnp.ones(
            (self.num_envs,), dtype=jnp.float32
        )
        if self.normalize_observation:
            observation_count = observation_count + 1
            delta = observation - observation_mean
            observation_mean = (
                observation_mean
                + delta / observation_count[:, None]
            )
            delta2 = observation - observation_mean
            observation_m2 = observation_m2 + delta * delta2
            observation = (observation - observation_mean) / jnp.sqrt(
                observation_m2 / observation_count[:, None] + self.eps
            )
        return RTRRLEnvironmentState(
            obs=observation,
            reward=jnp.zeros((self.num_envs,), dtype=jnp.float32),
            done=jnp.zeros((self.num_envs,), dtype=jnp.bool_),
            inner_state=inner_state,
            keys=step_keys,
            observation_mean=observation_mean,
            observation_m2=observation_m2,
            observation_count=observation_count,
            reward_mean=jnp.zeros((self.num_envs,), dtype=jnp.float32),
            reward_m2=jnp.ones((self.num_envs,), dtype=jnp.float32),
            reward_count=jnp.ones((self.num_envs,), dtype=jnp.float32),
            discounted_return=jnp.zeros(
                (self.num_envs,), dtype=jnp.float32
            ),
        )

    def step(self, state: RTRRLEnvironmentState, action):
        split_keys = jax.vmap(lambda key: jax.random.split(key, 2))(state.keys)
        next_keys = split_keys[:, 0]
        environment_keys = split_keys[:, 1]
        observation, inner_state, reward, done, _ = jax.vmap(
            self.env.step, in_axes=(0, 0, 0, None)
        )(
            environment_keys,
            state.inner_state,
            action,
            self.env_params,
        )
        observation_mean = state.observation_mean
        observation_m2 = state.observation_m2
        observation_count = state.observation_count
        if self.normalize_observation:
            observation_count = observation_count + 1
            delta = observation - observation_mean
            observation_mean = (
                observation_mean
                + delta / observation_count[:, None]
            )
            delta2 = observation - observation_mean
            observation_m2 = observation_m2 + delta * delta2
            observation = (observation - observation_mean) / jnp.sqrt(
                observation_m2 / observation_count[:, None] + self.eps
            )
        reward = jnp.asarray(reward, dtype=jnp.float32)
        reward_mean = state.reward_mean
        reward_m2 = state.reward_m2
        reward_count = state.reward_count
        discounted_return = (
            reward
            + self.reward_gamma
            * state.discounted_return
            * (1 - done)
        )
        if self.normalize_reward:
            reward_count = reward_count + 1
            delta = discounted_return - reward_mean
            reward_mean = reward_mean + delta / reward_count
            delta2 = discounted_return - reward_mean
            reward_m2 = reward_m2 + delta * delta2
            reward = reward / jnp.sqrt(
                reward_m2 / reward_count + self.eps
            )
        discounted_return = discounted_return * (1 - done)
        return RTRRLEnvironmentState(
            obs=observation,
            reward=reward,
            done=jnp.asarray(done, dtype=jnp.bool_),
            inner_state=inner_state,
            keys=next_keys,
            observation_mean=observation_mean,
            observation_m2=observation_m2,
            observation_count=observation_count,
            reward_mean=reward_mean,
            reward_m2=reward_m2,
            reward_count=reward_count,
            discounted_return=discounted_return,
        )


@struct.dataclass
class RTRRLEpochSummary:
    """Historical epoch values represented only by scalar pytree leaves."""

    steps: Any
    mean_reward: Any
    num_episodes: Any
    mean_delta: Any
    mean_r_bar: Any
    mean_v: Any
    total_td_loss: Any
    actor_loss: Any
    critic_loss: Any
    entropy: Any
    v_targ: Any
    magnitude_loss: Any = None
    learning_rate_td: Any = None
    learning_rate_rnn: Any = None
    norms: Any = struct.field(default_factory=dict)


def _tree_leaf_norms(tree: Any) -> dict[str, Any]:
    flattened, _ = jax.tree.flatten_with_path(tree)
    return {
        jax.tree_util.keystr(path): jnp.asarray(
            jnp.linalg.norm(value), dtype=jnp.float32
        )
        for path, value in flattened
    }


def _register_environment_state_pytree(value: Any) -> None:
    """Translate ordinary adapter dataclasses at the JAX program boundary."""

    if not is_dataclass(value) or isinstance(value, type):
        return
    names = [field.name for field in fields(value)]
    for name in names:
        _register_environment_state_pytree(getattr(value, name))
    try:
        jax.tree_util.register_dataclass(
            type(value),
            data_fields=names,
            meta_fields=[],
        )
    except ValueError as error:
        if "Duplicate custom dataclass PyTreeDef type registration" not in str(
            error
        ):
            raise


def aggregate_epoch_summary(
    metrics: TrainStepMetrics,
    final_state: Any,
    *,
    num_steps: int,
    num_envs: int,
    learning_rate_td: Any = None,
    learning_rate_rnn: Any = None,
) -> RTRRLEpochSummary:
    """Apply the pinned AAAI25 epoch reductions to scalar step summaries."""

    environment_count = jnp.asarray(num_envs, dtype=jnp.float32)
    num_episodes = jnp.asarray(
        jnp.rint(jnp.sum(metrics.done) * environment_count),
        dtype=jnp.int32,
    )
    divisor = jnp.maximum(num_episodes, jnp.asarray(1, dtype=jnp.int32))
    divisor_float = divisor.astype(jnp.float32)
    actor_loss = jnp.asarray(
        jnp.mean(metrics.actor_loss_mean), dtype=jnp.float32
    )
    critic_loss = jnp.asarray(
        jnp.mean(metrics.value_mean), dtype=jnp.float32
    )
    norms = _tree_leaf_norms(
        {
            "z": final_state.traces,
            "params": final_state.parameters,
            "slow_params": final_state.slow_parameters,
        }
    )
    return RTRRLEpochSummary(
        steps=jnp.asarray(
            final_state.step_count * num_envs, dtype=jnp.int32
        ),
        mean_reward=jnp.asarray(
            jnp.sum(metrics.reward) * environment_count / divisor_float,
            dtype=jnp.float32,
        ),
        num_episodes=num_episodes,
        mean_delta=jnp.asarray(
            jnp.sum(metrics.td_error_mean)
            * environment_count
            / divisor_float,
            dtype=jnp.float32,
        ),
        mean_r_bar=jnp.asarray(
            jnp.sum(final_state.average_reward) / divisor_float,
            dtype=jnp.float32,
        ),
        mean_v=critic_loss,
        total_td_loss=jnp.asarray(
            actor_loss + critic_loss, dtype=jnp.float32
        ),
        actor_loss=actor_loss,
        critic_loss=critic_loss,
        entropy=jnp.asarray(
            jnp.mean(metrics.entropy_mean), dtype=jnp.float32
        ),
        v_targ=jnp.asarray(
            jnp.mean(metrics.value_target_mean), dtype=jnp.float32
        ),
        magnitude_loss=(
            jnp.asarray(jnp.mean(metrics.magnitude_loss_mean), dtype=jnp.float32)
            if metrics.magnitude_loss_mean is not None
            else None
        ),
        learning_rate_td=learning_rate_td,
        learning_rate_rnn=learning_rate_rnn,
        norms=norms,
    )


def _optimizer_learning_rate(
    optimizer_state: Any,
    label: str,
    fallback: float,
) -> Any:
    """Read an injected Optax group rate without depending on tuple offsets."""

    candidates = []
    for path, value in jax.tree_util.tree_leaves_with_path(optimizer_state):
        keys = tuple(
            getattr(key, "key", getattr(key, "name", None)) for key in path
        )
        if label in keys and "learning_rate" in keys:
            candidates.append(value)
    if len(candidates) > 1:
        raise ValueError(f"multiple learning rates found for optimizer group {label}")
    value = candidates[0] if candidates else -fallback
    return jnp.asarray(value, dtype=jnp.float32)


def _make_evaluate_fn(
    components: RTRRLComponents,
    config: RTRRLComponentConfig,
    env: Any,
):
    recurrent = cast(AAAI25LRU, components.recurrent)

    def evaluate(key, state: RTRRLState, num_steps: int):
        reset_key, _ = jax.random.split(key)
        environment_state = env.reset(reset_key)
        recurrent_state = state.initial_recurrent_state
        feedback_action = jnp.zeros_like(state.action)
        running_return = jnp.zeros_like(environment_state.reward)

        def evaluate_step(carry, _):
            (
                current_environment,
                current_recurrent,
                previous_action,
                current_return,
            ) = carry
            done = current_environment.done
            current_recurrent = jax.tree.map(
                lambda initial, current: jax.vmap(jnp.where)(
                    done, initial, current
                ),
                state.initial_recurrent_state,
                current_recurrent,
            )
            model_input, _ = _step_model_input(
                current_environment, previous_action, config
            )

            def act_one(one_recurrent, inputs):
                recurrent_carry, credit = _unpack_recurrent_state(
                    one_recurrent
                )
                next_carry, hidden = recurrent.forward(
                    _flat_recurrent_parameters(state.slow_parameters),
                    recurrent_carry,
                    inputs,
                    False,
                )
                actor_output, _ = components.head.apply(
                    freeze({"params": state.slow_parameters["params"]["td"]}),
                    hidden,
                )
                distribution = cast(
                    Any,
                    make_action_distribution(
                        actor_output, discrete=config.discrete
                    ),
                )
                action = (
                    jnp.argmax(distribution.logits, axis=-1)
                    if config.discrete
                    else distribution.mode()
                )
                return _pack_recurrent_state(next_carry, credit), action

            next_recurrent, action = jax.vmap(act_one)(
                current_recurrent, model_input
            )
            action = action.reshape((*model_input.shape[:-1], -1))
            decision = ActionDecision(
                sampled_action=action,
                logprob_action=action,
                env_action=action,
                bootstrap_feedback_action=action,
                persisted_feedback_action=action,
            )
            next_environment = env.step(current_environment, decision.env_action)
            episode_return = current_return + next_environment.reward
            returned_episode = next_environment.done
            returned_episode_returns = jnp.where(
                returned_episode,
                episode_return,
                jnp.zeros_like(episode_return),
            )
            next_return = jnp.where(
                returned_episode,
                jnp.zeros_like(episode_return),
                episode_return,
            )
            event = {
                "reward": next_environment.reward,
                "done": next_environment.done,
                "returned_episode": returned_episode,
                "returned_episode_returns": returned_episode_returns,
                "environment_state": next_environment,
                "action_decision": decision,
            }
            return (
                next_environment,
                next_recurrent,
                decision.persisted_feedback_action,
                next_return,
            ), event

        _, events = jax.lax.scan(
            evaluate_step,
            (
                environment_state,
                recurrent_state,
                feedback_action,
                running_return,
            ),
            xs=None,
            length=num_steps,
        )
        return state, EvalSummary(info=events)

    return evaluate


def build_rtrrl_program(
    config: RTRRLComponentConfig,
    components: RTRRLComponents,
    env: Any,
) -> AgentProgram:
    """Resolve strict components once and return closed lifecycle functions."""

    if not isinstance(config, RTRRLComponentConfig):
        raise TypeError("config must be RTRRLComponentConfig")
    if not isinstance(components, RTRRLComponents):
        raise TypeError("components must be RTRRLComponents")
    if config.profile != "aaai25_strict_lru":
        raise ValueError("Task 9 builds only the aaai25_strict_lru profile")
    if config.backbone != "aaai25_lru":
        raise ValueError("strict RTRRL requires the aaai25_lru backbone")

    _register_environment_state_pytree(env.reset(jax.random.key(0)))
    eager_init = make_init_fn(components, config, env)
    production_step = make_step_fn(components, config, env, debug=False)
    evaluate_fn = _make_evaluate_fn(components, config, env)

    def init_fn(key):
        state, _ = eager_init(key)
        return state

    def train_epoch_fn(key, state: RTRRLState, num_steps: int):
        def scan_step(carry, _):
            current_state, current_key = carry
            next_state, next_key, metrics = production_step(
                current_state, current_key
            )
            return (next_state, next_key), metrics

        (final_state, _), metrics = jax.lax.scan(
            scan_step,
            (state, key),
            xs=None,
            length=num_steps,
        )
        return final_state, aggregate_epoch_summary(
            cast(TrainStepMetrics, metrics),
            final_state,
            num_steps=num_steps,
            num_envs=config.num_envs,
            learning_rate_td=_optimizer_learning_rate(
                final_state.optimizer_state,
                "td",
                config.optimizer_params_td.learning_rate,
            ),
            learning_rate_rnn=_optimizer_learning_rate(
                final_state.optimizer_state,
                "rnn",
                config.optimizer_params_rnn.learning_rate,
            ),
        )

    return AgentProgram(
        init_fn=init_fn,
        train_epoch_fn=train_epoch_fn,
        evaluate_fn=evaluate_fn,
        state_schema=RTRRLState,
        metric_schema=RTRRLEpochSummary,
    )


__all__ = [
    "LegacyRTRRLEnvironmentAdapter",
    "RTRRLEnvironmentState",
    "RTRRLEpochSummary",
    "aggregate_epoch_summary",
    "build_rtrrl_program",
]
