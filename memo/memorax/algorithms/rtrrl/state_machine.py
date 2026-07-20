"""Historical initialization and eager strict-RTRRL transition composition."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any, cast

import flax.linen as nn
from flax.core import freeze
import jax
import jax.numpy as jnp
import optax

from .compatibility import RTRRLComponentConfig
from .heads import RTRRLTDHead, make_action_distribution
from .lru import (
    AAAI25LRU,
    LRUCarry,
    LRUCreditState,
    _InitializerLayer,
)
from .rules import (
    combine_update_directions,
    td_error,
    update_emphasis_or_average_reward,
    update_slow_target,
    update_traces,
)
from .types import (
    DebugStepMetrics,
    InitializationKeys,
    RTRRLComponents,
    RTRRLState,
    TrainStepMetrics,
)


class _HistoricalInitializer(nn.Module):
    """Recreate the pinned Flax scope and parameter RNG order."""

    input_dim: int
    hidden_dim: int
    action_dim: int
    discrete: bool

    @nn.compact
    def __call__(self, inputs):
        preactivation = _InitializerLayer(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
            name="rnn",
        )(inputs)
        hidden = jax.nn.silu(preactivation)
        actor, value = RTRRLTDHead(
            action_dim=self.action_dim,
            discrete=self.discrete,
            f_align=False,
            name="td",
        )(hidden)
        return hidden, actor, value


def _make_maximizing_optimizer(
    config: RTRRLComponentConfig,
) -> optax.GradientTransformation:
    @optax.inject_hyperparams
    def make_group(learning_rate):
        return optax.chain(
            optax.add_decayed_weights(0.0),
            optax.identity(),
            getattr(optax, config.optimizer_name)(learning_rate),
        )

    return optax.multi_transform(
        {
            "rnn": make_group(
                learning_rate=-config.recurrent_learning_rate
            ),
            "td": make_group(
                learning_rate=-config.actor_critic_learning_rate
            ),
        },
        {"rnn": "rnn", "td": "td"},
    )


def _flat_recurrent_parameters(parameters: Any) -> dict[str, Any]:
    root = cast(Mapping[str, Any], parameters["params"]["rnn"])
    online = cast(Mapping[str, Any], root["OnlineLRUCell_0"])
    cell = cast(Mapping[str, Any], online["LRUCell_0"])
    return {
        "nu_log": cell["nu_log"],
        "theta_log": cell["theta_log"],
        "gamma_log": cell["gamma_log"],
        "B_real": cell["B_real"],
        "B_img": cell["B_img"],
        "C_real": root["C_real"],
        "C_img": root["C_img"],
        "D": root["D"],
    }


def _pack_recurrent_state(
    carry: LRUCarry,
    credit: LRUCreditState,
) -> tuple[Any, tuple[Any, Any, Any]]:
    return (
        carry.hidden,
        (
            credit.lambda_sensitivity,
            credit.gamma_sensitivity,
            credit.B_sensitivity,
        ),
    )


def _unpack_recurrent_state(
    state: tuple[Any, tuple[Any, Any, Any]],
) -> tuple[LRUCarry, LRUCreditState]:
    hidden, (lambda_sensitivity, gamma_sensitivity, B_sensitivity) = state
    return (
        LRUCarry(hidden=hidden),
        LRUCreditState(
            lambda_sensitivity=lambda_sensitivity,
            gamma_sensitivity=gamma_sensitivity,
            B_sensitivity=B_sensitivity,
        ),
    )


def _initial_model_input(environment_state, config):
    observation = environment_state.obs
    if not config.meta_rl:
        return observation
    batch_shape = observation.shape[:-1]
    return jnp.concatenate(
        [
            observation,
            jnp.zeros(
                batch_shape + (config.action_dim,), dtype=observation.dtype
            ),
            environment_state.reward.reshape(batch_shape + (-1,)),
        ],
        axis=-1,
    )


def _zero_credit(component: AAAI25LRU) -> LRUCreditState:
    return LRUCreditState(
        lambda_sensitivity=jnp.zeros(
            (component.hidden_dim,), dtype=jnp.complex64
        ),
        gamma_sensitivity=jnp.zeros(
            (component.hidden_dim,), dtype=jnp.complex64
        ),
        B_sensitivity=jnp.zeros(
            (component.hidden_dim, component.input_dim),
            dtype=jnp.complex64,
        ),
    )


def make_init_fn(
    components: RTRRLComponents,
    config: RTRRLComponentConfig,
    env: Any,
):
    """Close over static construction and return historical ``init(key)``."""

    recurrent = cast(AAAI25LRU, components.recurrent)
    optimizer = _make_maximizing_optimizer(config)
    initializer = _HistoricalInitializer(
        input_dim=recurrent.input_dim,
        hidden_dim=config.hidden_dim,
        action_dim=config.action_dim,
        discrete=config.discrete,
    )

    def initialize(root_key):
        with jax.threefry_partitionable(False):
            _, model_key, step_key, carry_key, env_key, outer_key = (
                jax.random.split(root_key, 6)
            )
        environment_state = env.reset(env_key)
        model_input = _initial_model_input(environment_state, config)
        with jax.threefry_partitionable(False):
            parameters = initializer.init(model_key, model_input)
        recurrent_parameters = _flat_recurrent_parameters(parameters)
        batch_shape = model_input.shape[:-1]
        zero_carry = LRUCarry(
            hidden=jnp.zeros(
                batch_shape + (recurrent.hidden_dim,),
                dtype=jnp.complex64,
            )
        )

        def advance_one(carry_hidden, inputs):
            credit = _zero_credit(recurrent)
            next_carry, hidden = recurrent.forward(
                recurrent_parameters,
                LRUCarry(hidden=carry_hidden),
                inputs,
                False,
            )
            return _pack_recurrent_state(next_carry, credit), hidden

        recurrent_state, hidden = jax.vmap(advance_one)(
            zero_carry.hidden, model_input
        )
        head_variables = {"params": parameters["params"]["td"]}
        actor_output, value = components.head.apply(head_variables, hidden)
        distribution = make_action_distribution(
            actor_output,
            discrete=config.discrete,
        )
        with jax.threefry_partitionable(False):
            action = distribution.sample(seed=model_key)
        action = action.reshape((*batch_shape, -1))
        selected_parameters = {
            name: value
            for name, value in parameters["params"].items()
            if name in {"td", "rnn"}
        }
        traces = jax.tree.map(
            lambda value: jnp.zeros(batch_shape + value.shape, value.dtype),
            selected_parameters,
        )
        optimizer_state = optimizer.init(parameters["params"])
        state = RTRRLState(
            parameters=parameters,
            slow_parameters=parameters,
            optimizer_state=optimizer_state,
            environment_state=environment_state,
            action=action,
            recurrent_state=recurrent_state,
            traces=traces,
            value=value,
            average_reward=jnp.array([0.0], dtype=jnp.float32),
            emphasis=jnp.ones(batch_shape, dtype=jnp.float32),
            observation_statistics=None,
            reward_statistics=None,
            model_input=model_input,
            initial_recurrent_state=jax.tree.map(
                jnp.zeros_like, recurrent_state
            ),
            step_count=jnp.array(0, dtype=jnp.int32),
        )
        keys = InitializationKeys(
            step=step_key,
            outer=outer_key,
            model=model_key,
            carry=carry_key,
            environment=env_key,
        )
        return state, keys

    return initialize


def _domain_trees(tree):
    parameters = tree["params"]
    return {
        "actor": parameters["td"]["actor"],
        "critic": parameters["td"]["critic"],
        "recurrent": parameters["rnn"],
    }


def _state_trace_domains(tree):
    return {
        "actor": tree["td"]["actor"],
        "critic": tree["td"]["critic"],
        "recurrent": tree["rnn"],
    }


def _parameter_tree(domains):
    return {
        "params": {
            "td": {
                "actor": domains["actor"],
                "critic": domains["critic"],
            },
            "rnn": domains["recurrent"],
        }
    }


def _trace_tree(domains):
    return {
        "td": {
            "actor": domains["actor"],
            "critic": domains["critic"],
        },
        "rnn": domains["recurrent"],
    }


def _step_model_input(environment_state, action, config):
    observation = environment_state.obs
    if not config.meta_rl:
        return observation, action
    done = environment_state.done
    reward = environment_state.reward.reshape(-1)
    action, reward = jax.tree.map(
        lambda value: jax.vmap(jnp.where)(
            done, jnp.zeros_like(value), value
        ),
        (action, reward),
    )
    batch_shape = observation.shape[:-1]
    return (
        jnp.concatenate(
            [
                observation,
                action,
                reward.reshape(batch_shape + (-1,)),
            ],
            axis=-1,
        ),
        action,
    )


def make_step_fn(
    components: RTRRLComponents,
    config: RTRRLComponentConfig,
    env: Any,
    debug: bool,
):
    """Compose one eager environment-first AAAI25 online transition."""

    recurrent = cast(AAAI25LRU, components.recurrent)
    optimizer = _make_maximizing_optimizer(config)

    def step(state: RTRRLState, key):
        with jax.threefry_partitionable(False):
            next_key, action_key, _ = jax.random.split(key, 3)
        environment_action = state.action
        environment_state = env.step(
            state.environment_state, environment_action
        )
        done = environment_state.done
        recurrent_state = jax.tree.map(
            lambda initial, current: jax.vmap(jnp.where)(
                done, initial, current
            ),
            state.initial_recurrent_state,
            state.recurrent_state,
        )
        model_input, _ = _step_model_input(
            environment_state, environment_action, config
        )

        def gradients_for_environment(environment_recurrent_state, inputs):
            def recurrent_step(parameters):
                carry, credit = _unpack_recurrent_state(
                    environment_recurrent_state
                )
                # The pinned online cell computes updated sensitivities in its
                # custom-VJP residual but returns the incoming sensitivity
                # leaves from the primal unless force_trace_compute is set.
                # Preserve that persisted-state behavior while Task 6 supplies
                # the unbatched recurrent cotangent replacement.
                _, next_carry, hidden = (
                    recurrent.forward_with_credit(
                        _flat_recurrent_parameters(parameters),
                        credit,
                        carry,
                        inputs,
                        False,
                    )
                )
                return hidden, _pack_recurrent_state(
                    next_carry, credit
                )

            hidden, recurrent_pullback, next_recurrent_state = jax.vjp(
                recurrent_step,
                state.slow_parameters,
                has_aux=True,
            )

            def td_objective(parameters, head_input):
                actor_output, value = components.head.apply(
                    freeze({"params": parameters["params"]["td"]}),
                    head_input,
                )
                distribution = make_action_distribution(
                    actor_output,
                    discrete=config.discrete,
                )
                with jax.threefry_partitionable(False):
                    sampled_action = distribution.sample(seed=action_key)
                actor_loss = distribution.log_prob(sampled_action)
                objective = (
                    actor_loss.mean() * config.actor_scale + value.mean()
                )
                return objective, (
                    sampled_action,
                    value,
                    actor_loss.mean() * config.actor_scale,
                )

            (
                (gradients, hidden_gradient),
                (sampled_action, value, actor_loss),
            ) = jax.grad(
                td_objective,
                argnums=(0, 1),
                has_aux=True,
            )(state.slow_parameters, hidden)
            recurrent_gradients = recurrent_pullback(hidden_gradient)[0]
            gradients = jax.tree.map(
                lambda head, recurrent_value: head + recurrent_value,
                gradients,
                recurrent_gradients,
            )

            def direct_objective(parameters, head_input):
                actor_output, _ = components.head.apply(
                    freeze({"params": parameters["params"]["td"]}),
                    head_input,
                )
                distribution = make_action_distribution(
                    actor_output,
                    discrete=config.discrete,
                )
                entropy = distribution.entropy().mean()
                return entropy * config.entropy_rate, entropy

            (
                (direct_gradients, hidden_direct_gradient),
                entropy,
            ) = jax.grad(
                direct_objective,
                argnums=(0, 1),
                has_aux=True,
            )(state.slow_parameters, hidden)
            recurrent_direct_gradients = recurrent_pullback(
                hidden_direct_gradient
            )[0]
            direct_gradients = jax.tree.map(
                lambda head, recurrent_value: head + recurrent_value,
                direct_gradients,
                recurrent_direct_gradients,
            )
            return (
                next_recurrent_state,
                direct_gradients,
                sampled_action,
                value,
                gradients,
                actor_loss,
                entropy,
            )

        (
            recurrent_state,
            direct_gradients,
            sampled_action,
            next_value,
            gradients,
            actor_loss,
            entropy,
        ) = jax.vmap(gradients_for_environment)(
            recurrent_state, model_input
        )
        reward = environment_state.reward.reshape(-1)
        value_target = (
            reward
            + config.gamma * next_value.squeeze() * (1 - done)
        )
        delta = td_error(
            reward=reward,
            value=state.value.squeeze(),
            next_value=next_value.squeeze(),
            terminated=done,
            gamma=config.gamma,
            average_reward=state.average_reward,
        )
        trace_domains = _state_trace_domains(state.traces)
        gradient_domains = _domain_trees(gradients)
        direct_domains = _domain_trees(direct_gradients)
        emphasis_state = update_emphasis_or_average_reward(
            emphasis=state.emphasis,
            average_reward=state.average_reward,
            delta=delta,
            terminated=done,
            gamma=config.gamma,
            eta=config.average_reward_rate,
        )
        trace_directions = update_traces(
            trace_domains,
            gradient_domains,
            gamma=config.gamma,
            lambda_actor=config.lambda_actor,
            lambda_critic=config.lambda_critic,
            lambda_rnn=config.lambda_recurrent,
            trace_mode=config.trace_mode,
            critic_learning_rate=config.actor_critic_learning_rate,
            emphasis=emphasis_state.emphasis,
            terminated=done,
            timing=config.trace_timing,
        )
        mean_domains = combine_update_directions(
            trace_directions.update,
            direct_domains,
            delta=delta,
            recurrent_scale=config.recurrent_scale,
            trace_mode=config.trace_mode,
            critic_learning_rate=config.actor_critic_learning_rate,
            critic_value_difference=(
                next_value.squeeze() - state.value.squeeze()
            ),
        )
        mean_directions = _parameter_tree(mean_domains)
        optimizer_updates, optimizer_state = optimizer.update(
            mean_directions["params"],
            state.optimizer_state,
            state.parameters["params"],
        )
        parameters = {
            **state.parameters,
            "params": optax.apply_updates(
                state.parameters["params"], optimizer_updates
            ),
        }
        slow_recurrent = update_slow_target(
            fast_parameters=parameters["params"]["rnn"],
            previous_slow_parameters=state.slow_parameters["params"][
                "rnn"
            ],
            period=config.update_period,
        )
        slow_parameters = {
            **parameters,
            "params": {
                **parameters["params"],
                "rnn": slow_recurrent,
            },
        }
        sampled_action = sampled_action.reshape(
            (*model_input.shape[:-1], -1)
        )
        next_state = replace(
            state,
            parameters=parameters,
            slow_parameters=slow_parameters,
            optimizer_state=optimizer_state,
            environment_state=environment_state,
            action=sampled_action,
            recurrent_state=recurrent_state,
            traces=_trace_tree(trace_directions.carried),
            value=next_value,
            average_reward=emphasis_state.average_reward,
            emphasis=emphasis_state.emphasis,
            model_input=model_input,
            step_count=state.step_count + 1,
        )
        train_metrics = TrainStepMetrics(
            reward=jnp.mean(reward),
            done=jnp.mean(done.astype(jnp.float32)),
            td_error_mean=jnp.mean(delta),
            value_mean=jnp.mean(next_value),
            value_target_mean=jnp.mean(value_target),
            entropy_mean=jnp.mean(entropy),
            actor_loss_mean=jnp.mean(actor_loss),
        )
        if not debug or int(state.step_count) >= config.debug_max_steps:
            return next_state, next_key, train_metrics
        debug_metrics = DebugStepMetrics(
            train=train_metrics,
            environment_action=environment_action,
            model_input=model_input,
            sampled_next_action=sampled_action,
            value_target=value_target,
            td_error=delta,
            gradients=gradients,
            direct_gradients=direct_gradients,
            incoming_traces=state.traces,
            carried_traces=next_state.traces,
            mean_directions=mean_directions,
            optimizer_updates=optimizer_updates,
            fast_parameters=parameters,
            slow_parameters=slow_parameters,
            emphasis=emphasis_state.emphasis,
            average_reward=emphasis_state.average_reward,
            debug_active=jnp.array(True),
        )
        return next_state, next_key, debug_metrics

    return step


__all__ = ["make_init_fn", "make_step_fn"]
