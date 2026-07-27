"""Upstream's StreamAC runs here, and our rewrite bounds a step the way it does.

Two things are checked, and they are different in kind.

The first is that the restored file works at all: it trains, it evaluates, and
what upstream handed to ``lox`` from inside the scan now comes back as returned
data with a fixed shape, which is what lets an SDK report it from outside JIT.

The second is the reason the file is worth keeping. ``StreamACRTRL`` is this
algorithm with recurrent credit swapped in, and the overshooting bound is meant
to be untouched by that swap. Upstream's ``_obgd_update`` is verbatim, so it can
answer for our ``make_obgd_rule`` directly, on the same traces, the same second
moment and the same TD error. The private method is reached on purpose: a rule
that agreed only after an epoch of training would tell us much less.
"""

from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import TinyDiscreteEnv, assert_within, flattened

from memorax.algorithms.stream_ac import StreamAC, StreamACConfig
from memorax.networks import (
    RNN,
    FeatureExtractor,
    Network,
    RTUCell,
    RTUConfig,
    heads,
)
from memorax.rl import make_obgd_rule
from runner.episodes import complete_episodes

HORIZON = TinyDiscreteEnv().default_params.horizon
ENVS = 2

# The product the two spell differently. Upstream writes ``ss * delta * z``,
# which multiplies the bounded step size into the TD error before the trace;
# our rule weights the trace by the TD error first, because it has to add
# untraced directions to that product before scaling it. Same factors, one
# reassociation, and the golden snapshot puts the cost of it at a single last
# bit on a bias vector. Four is loose enough not to fail on a machine that
# rounds differently and far tighter than a changed bound could hide under.
REASSOCIATED = 4.0


def agent(**overrides) -> StreamAC:
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

    return StreamAC(
        StreamACConfig(
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


def traces_and_errors(magnitude: float):
    """A trace tree wide enough to bind the step size, and one TD error per env.

    ``magnitude`` decides which side of the bound the step lands on: at 0.05
    the trace norm leaves the learning rate alone, and at 5.0 the maximum in
    the denominator is what sets the step, so both branches get exercised.
    """

    traces = {
        "kernel": magnitude * jnp.linspace(-1.0, 1.0, ENVS * 12).reshape(ENVS, 4, 3),
        "bias": magnitude * jnp.linspace(0.3, -0.7, ENVS * 3).reshape(ENVS, 3),
    }
    return traces, jnp.array([0.75, -2.5], dtype=jnp.float32)


@pytest.mark.parametrize("adaptive", [False, True], ids=["obgd", "adaptive"])
@pytest.mark.parametrize("magnitude", [0.05, 5.0], ids=["unbound", "bound"])
@pytest.mark.parametrize("step", [1, 40])
def test_our_rule_bounds_the_step_the_way_upstream_does(adaptive, magnitude, step):
    theirs = agent(adaptive=adaptive)
    cfg = theirs.cfg
    ours = make_obgd_rule(
        learning_rate=cfg.critic_lr,
        kappa=cfg.critic_kappa,
        beta2=cfg.beta2,
        eps=cfg.eps,
        adaptive=adaptive,
    )

    traces, td_error = traces_and_errors(magnitude)
    params = jax.tree.map(lambda leaf: jnp.mean(leaf, axis=0), traces)
    moment = ours.init(params=params, traces=traces)

    their_updates, their_moment = theirs._obgd_update(
        traces, moment, td_error, cfg.critic_lr, cfg.critic_kappa, step
    )
    output = ours.apply(traces, None, moment, delta=td_error, step=step, params=params)

    # The second moment is the same expression in both, so it is exact; the
    # update carries the one reassociated product.
    assert_within(
        flattened(output.state),
        flattened(their_moment),
        "second moment",
    )
    assert_within(
        flattened(output.updates),
        flattened(their_updates),
        "updates",
        allowed=REASSOCIATED,
    )
    # Both average the env axis out, since the parameters they step do not
    # carry one.
    assert output.updates["kernel"].shape == (4, 3)
    assert float(jnp.max(output.metrics["step_size"])) <= cfg.critic_lr


def test_the_second_moment_starts_where_upstream_starts_it():
    """Our rule builds its own state; upstream's caller passes one in."""

    traces, _ = traces_and_errors(1.0)
    ours = make_obgd_rule(learning_rate=0.1, kappa=2.0)
    initial = ours.init(params=None, traces=traces)
    upstream_initial = jax.tree.map(jnp.zeros_like, traces)
    assert_within(flattened(initial), flattened(upstream_initial), "initial moment")


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

    assert metrics.td_error.shape == (updates, ENVS)
    assert metrics.value.shape == (updates, ENVS)
    assert metrics.next_value.shape == (updates, ENVS)
    assert metrics.log_prob.shape == (updates, ENVS)
    assert metrics.entropy.shape == (updates,)
    assert metrics.info["step_count"].shape == (updates, ENVS)

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

    actions = np.asarray(summary.action)
    counts = np.asarray(summary.info["step_count"])
    expected = 0.25 * np.where(actions == 0, -1.0, 1.0) + 0.1 * counts

    assert actions.shape == (steps, ENVS)
    np.testing.assert_allclose(np.asarray(summary.reward), expected, rtol=0, atol=1e-6)
    # The rollout starts from a reset, so the step it reports is its own.
    np.testing.assert_array_equal(
        counts, np.broadcast_to(np.arange(1, steps + 1)[:, None], counts.shape)
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
        complete_episodes(summary, phase="eval", start_env_steps=0, num_envs=ENVS)
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
