"""The restored StreamAC runs, and hands back what upstream used to log.

Nothing here compares it to anything: its arithmetic is answered for block by
block in ``test_blocks.py``. What is checked here is that the file works at all
after being cut off from ``lox`` -- that an epoch trains, that an evaluation
rollout reports the environment's own reward, and that the transition record it
returns instead of logging is shaped the way a sink and a score need it.
"""

from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import TinyDiscreteEnv
from training_sdk.rollout import complete_episodes

from memorax.algorithms.slots import FeatureExtractor, Network
from memorax.algorithms.upstream_stream_ac import (
    UpstreamStreamAC,
    UpstreamStreamACConfig,
)
from memorax.networks import RNN, RTUCell, RTUConfig, heads

HORIZON = TinyDiscreteEnv().default_params.horizon
ENVS = 2


def agent(**overrides) -> UpstreamStreamAC:
    """Upstream's kernel on the smallest task that terminates during a test."""

    env = TinyDiscreteEnv()

    def network(head):
        return Network(
            feature_extractor=FeatureExtractor(
                observation_extractor=nn.Sequential((nn.Dense(3), nn.tanh))
            ),
            torso=RNN(cell=RTUCell(config=RTUConfig(features=3, hidden_dim=2))),
            head=head,
        )

    return UpstreamStreamAC(
        UpstreamStreamACConfig(
            num_envs=ENVS,
            gamma=0.89,
            trace_lambda=0.71,
            actor_lr=0.15,
            critic_lr=0.12,
            entropy_coefficient=0.02,
            beta2=0.95,
            eps=1e-6,
            **overrides,
        ),
        env,
        env.default_params,
        network(heads.Categorical(action_dim=2)),
        network(heads.VNetwork()),
    )


def finite(tree) -> bool:
    return all(jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(tree))


@pytest.mark.parametrize("adaptive", [False, True], ids=["obgd", "adaptive"])
def test_an_epoch_moves_parameters_and_hands_back_what_it_logged(adaptive):
    """Everything upstream sent to its logger is in the returned metrics.

    Upstream logged the TD error, the entropy and the value from inside the
    scan. Nothing here logs, so those have to come out as arrays with a step
    axis, or an epoch would be unreportable.
    """

    built = agent(adaptive=adaptive)
    steps = 8
    updates = steps // ENVS

    state = jax.jit(built.init)(jax.random.key(0))
    trained, metrics = jax.jit(built.train, static_argnums=2)(
        jax.random.key(1), state, steps
    )

    assert finite(metrics), "an epoch produced a non-finite observable"
    assert finite(trained)
    assert int(trained.step) == steps
    assert int(trained.update_step) == updates

    assert metrics.update.td_error.shape == (updates, ENVS)
    assert metrics.forward.value.shape == (updates, ENVS)
    assert metrics.forward.next_value.shape == (updates, ENVS)
    assert metrics.forward.log_prob.shape == (updates, ENVS)
    assert metrics.forward.entropy.shape == (updates,)
    assert metrics.interaction.info["step_count"].shape == (updates, ENVS)

    assert any(
        not jnp.allclose(old, new)
        for old, new in zip(
            jax.tree.leaves(state.actor_params), jax.tree.leaves(trained.actor_params)
        )
    ), "training left the actor untouched"


def test_the_evaluation_rollout_reports_the_reward_the_environment_gave():
    """The score is built from this, so it may not be the kernel's own idea.

    ``TinyDiscreteEnv`` pays 0.25 in the direction of the action plus 0.1 per
    step it has taken, which is little enough to recompute here from what the
    summary says was done.
    """

    built = agent()
    steps = HORIZON + 2
    state = jax.jit(built.init)(jax.random.key(0))
    _, summary = jax.jit(built.evaluate, static_argnums=2)(
        jax.random.key(2), state, steps
    )

    actions = np.asarray(summary.interaction.action)
    counts = np.asarray(summary.interaction.info["step_count"])
    expected = 0.25 * np.where(actions == 0, -1.0, 1.0) + 0.1 * counts

    assert actions.shape == (steps, ENVS)
    np.testing.assert_allclose(
        np.asarray(summary.interaction.reward), expected, rtol=0, atol=1e-6
    )
    # The rollout starts from a reset, so the step it reports is its own, and it
    # starts from one again at the horizon: the environment resets nothing, so
    # beginning the next episode is the acting step's business.
    restarted = np.array([(step % HORIZON) + 1 for step in range(steps)])
    np.testing.assert_array_equal(
        counts, np.broadcast_to(restarted[:, None], counts.shape)
    )


def test_the_rollout_cuts_into_the_episodes_a_sink_accepts():
    """The summary is the kernel's whole side of the contract with a sink."""

    built = agent()
    steps = HORIZON + 2
    state = jax.jit(built.init)(jax.random.key(0))
    _, summary = jax.jit(built.evaluate, static_argnums=2)(
        jax.random.key(2), state, steps
    )

    episodes = list(
        complete_episodes(
            summary,
            phase="eval",
            start_env_steps=0,
            num_envs=ENVS,
            transitions="interaction",
            reward="interaction.reward",
            terminal="interaction.terminal",
        )
    )
    assert episodes, "a rollout past the horizon held no complete episode"

    first = episodes[0]
    assert len(first.rewards) == HORIZON
    assert len(first.actions) == HORIZON
    # One observation more than there were actions: where the episode ended.
    assert len(first.observations) == HORIZON + 1
    assert first.terminals[-1] and not any(first.terminals[:-1])
    assert first.start_env_steps == 0
    assert first.end_env_steps == HORIZON * ENVS
