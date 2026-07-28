"""Every block of arithmetic we rearranged, against the version it came from.

Our kernels are upstream's StreamAC taken apart: the overshooting bound became a
rule that any algorithm can step with, the TD error a primitive, the trace and
the objective methods on the kernel. Taking arithmetic apart is where it gets
quietly changed, so each block is driven here beside the one it was lifted from
and compared leaf by leaf.

The reference is ``memorax.algorithms.stream_ac``, which is upstream's file with
its arithmetic untouched. It is used rather than the read-only clone next door
because the clone imports ``lox`` and answers to the same package name.

One test per block. A block with no counterpart is not tested here -- RTRRL's
emphasis recurrence and three-domain trace exist in no other implementation, and
inventing an expression in a test to compare them against would only assert that
two copies of a guess agree. ``rl/credit.py`` and ``rl/targets.py`` are absent
for the opposite reason, written at the head of each.

Every input is drawn from a written-down seed, never from the clock: a bit-exact
assertion that fails one run in ten is worse than no assertion. The cases are
chosen to cross each branch -- terminal and live steps, a TD error of exactly
zero, a trace of exactly zero, both sides of the step-size bound.

Every block multiplies in the order the version it came from multiplies in, so
every one of them is exact, and a last bit of drift is a change of arithmetic
rather than a change of spelling. The gradient is the one exception, and it is
named at the block that carries it rather than granted file-wide.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import pytest
from conftest import TinyContinuousEnv, assert_within, deviations, flattened

from memorax.algorithms.stream_ac import StreamAC, StreamACConfig
from memorax.algorithms.upstream_stream_ac import (
    UpstreamStreamAC,
    UpstreamStreamACConfig,
)
from memorax.environments.wrappers.normalize_observation import (
    NormalizeObservationWrapper,
)
from memorax.environments.wrappers.normalize_reward import NormalizeRewardWrapper
from memorax.networks import RNN, FeatureExtractor, Network, RTUCell, RTUConfig, heads
from memorax.rl import NormalizationConfig, make_normalizer, make_obgd_rule, make_td0

ENVS = 3

# The shared settings both kernels are built from. Written once so a block can
# never be compared against a differently-configured version of itself.
SETTINGS = {
    "num_envs": ENVS,
    "gamma": 0.89,
    "trace_lambda": 0.71,
    "actor_lr": 0.15,
    "critic_lr": 0.12,
    "actor_kappa": 3.0,
    "critic_kappa": 2.0,
    "entropy_coefficient": 0.02,
    "beta2": 0.95,
    "eps": 1e-6,
}


def network(head):
    return Network(
        feature_extractor=FeatureExtractor(
            observation_extractor=nn.Sequential((nn.Dense(3), nn.tanh))
        ),
        torso=RNN(cell=RTUCell(config=RTUConfig(features=3, hidden_dim=2))),
        head=head,
    )


def upstream(**overrides) -> UpstreamStreamAC:
    """The version every block here answers to."""

    env = TinyContinuousEnv()
    return UpstreamStreamAC(
        UpstreamStreamACConfig(**{**SETTINGS, **overrides}),
        env,
        env.default_params,
        network(heads.Gaussian(action_dim=2)),
        network(heads.VNetwork()),
    )


def ours(**overrides) -> StreamAC:
    """Our kernel, built for its methods; nothing here initialises it."""

    env = TinyContinuousEnv()
    return StreamAC(
        StreamACConfig(**{**SETTINGS, **overrides}),
        env,
        env.default_params,
        network(heads.Gaussian(action_dim=2)),
        network(heads.VNetwork()),
    )


def vector(key, *, scale: float = 1.0):
    return scale * jax.random.normal(key, (ENVS,), dtype=jnp.float32)


def tree(key, *, scale: float = 1.0):
    """A parameter-shaped tree: one leaf with trailing axes, one without."""

    kernel, bias = jax.random.split(key)
    return {
        "kernel": scale * jax.random.normal(kernel, (ENVS, 4, 3), dtype=jnp.float32),
        "bias": scale * jax.random.normal(bias, (ENVS, 3), dtype=jnp.float32),
    }


def terminals(kind: str, key):
    """Which streams ended on this step."""

    if kind == "live":
        return jnp.zeros((ENVS,), dtype=bool)
    if kind == "terminal":
        return jnp.ones((ENVS,), dtype=bool)
    return jax.random.bernoulli(key, 0.5, (ENVS,))


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("done", ["mixed", "live", "terminal"])
def test_the_td_error_is_upstreams(seed, done):
    """Block: TD(0).

    Ours takes the bootstrap discount already formed, because an algorithm that
    discounts differently should not have to reimplement the difference. That
    makes the discount part of the comparison: it is spelled here the way both
    kernels spell it at the call site.
    """

    keys = jax.random.split(jax.random.key(seed), 4)
    reward, value, next_value = (vector(key) for key in keys[:3])
    next_done = terminals(done, keys[3])

    theirs = upstream()
    mine = make_td0()(
        reward=reward,
        value=value,
        next_value=next_value,
        bootstrap_discount=theirs.cfg.gamma * (1 - next_done),
    )
    assert_within(
        {"td": mine},
        {"td": theirs._td_error(reward, next_done, next_value, value)},
        f"td0 seed={seed} done={done}",
    )


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("done", ["mixed", "live", "terminal"])
@pytest.mark.parametrize("scale", [1.0, 0.0], ids=["gradient", "no_gradient"])
def test_the_trace_recurrence_is_upstreams(seed, done, scale):
    """Block: the eligibility trace.

    A trailing-axis leaf and a flat one, because the decay is broadcast over
    whatever axes a parameter has and getting that wrong is invisible on a
    vector.
    """

    keys = jax.random.split(jax.random.key(seed), 3)
    incoming = tree(keys[0])
    gradient = tree(keys[1], scale=scale)
    reset_before = terminals(done, keys[2])

    mine, theirs = ours(), upstream()
    assert mine.trace_decay == theirs.cfg.gamma * theirs.cfg.trace_lambda

    carried = mine._trace(incoming, gradient, reset_before=reset_before)
    expected = jax.tree.map(
        lambda z, g: theirs._update_trace(z, g, reset_before, mine.trace_decay),
        incoming,
        gradient,
    )
    assert_within(
        flattened(carried.carried),
        flattened(expected),
        f"trace seed={seed} done={done} scale={scale}",
    )
    # StreamAC resets before it acts, so the trace it carries away and the one
    # it steps with are the same object.
    assert_within(
        flattened(carried.update), flattened(expected), "trace used for the update"
    )


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize(
    "surprise", ["signed", "none", "positive"], ids=["signed", "zero_td", "positive_td"]
)
def test_the_actor_direction_is_upstreams(seed, surprise):
    """Block: the actor and critic ascent directions.

    A TD error of exactly zero is a case worth naming: ``sign`` returns zero
    there, so the entropy term drops out entirely, and an implementation that
    used ``where(td > 0, ...)`` instead would agree everywhere else.
    """

    keys = jax.random.split(jax.random.key(seed), 4)
    log_prob, entropy, value = (vector(key) for key in keys[:3])
    delta = {
        "signed": vector(keys[3]),
        "none": jnp.zeros((ENVS,), dtype=jnp.float32),
        "positive": jnp.abs(vector(keys[3])),
    }[surprise]

    mine, theirs = ours(), upstream()
    directions = mine._objective(
        log_prob=log_prob, value=value, entropy=entropy, delta=delta
    )
    assert_within(
        {"actor": directions.traced_by_domain["actor"]},
        {"actor": theirs._actor_direction(log_prob, entropy, delta)},
        f"actor direction seed={seed} td={surprise}",
    )
    # Upstream's critic objective is the value itself; ours has to route it
    # rather than recompute it, so it is exactly that value or it is wrong.
    assert_within(
        {"critic": directions.traced_by_domain["critic"]}, {"critic": value}, "critic"
    )


@pytest.mark.parametrize(
    "rule_name", ["obgd", "adaptive_obgd"], ids=["obgd", "adaptive"]
)
@pytest.mark.parametrize(
    "magnitude", [0.05, 5.0, 0.0], ids=["unbound", "bound", "no_trace"]
)
@pytest.mark.parametrize("step", [1, 40])
@pytest.mark.parametrize("surprise", [1.0, 0.0], ids=["surprised", "no_td"])
def test_the_bounded_step_is_upstreams(rule_name, magnitude, step, surprise):
    """Block: the overshooting-bounded update.

    The bound reads the TD error and the trace norm together, so the cases cross
    both: a trace small enough to leave the learning rate alone and one large
    enough for the bound to set the step, plus the degenerate pair where there
    is no trace or no surprise at all and the step must come out zero.

    Only the two rules upstream has are here. ``adaptive_obgd_fixed`` answers to
    the published optimiser instead, which upstream does not implement.
    """

    adaptive = rule_name != "obgd"
    theirs = upstream(adaptive=adaptive)
    cfg = theirs.cfg
    rule = make_obgd_rule(
        learning_rate=cfg.critic_lr,
        kappa=cfg.critic_kappa,
        beta2=cfg.beta2,
        eps=cfg.eps,
        rule=rule_name,
    )

    traces = tree(jax.random.key(7), scale=magnitude)
    td_error = surprise * jnp.array([0.75, -2.5, 0.0], dtype=jnp.float32)
    moment = rule.init(params=None, traces=traces)

    their_updates, their_moment = theirs._obgd_update(
        traces, moment, td_error, cfg.critic_lr, cfg.critic_kappa, step
    )
    output = rule.apply(traces, None, moment, delta=td_error, step=step, params=None)

    what = f"{rule_name} trace={magnitude} step={step} td={surprise}"
    assert_within(flattened(output.state), flattened(their_moment), f"{what} moment")
    assert_within(
        flattened(output.updates), flattened(their_updates), f"{what} updates"
    )
    assert output.updates["kernel"].shape == (4, 3), "the env axis survived the update"
    assert float(jnp.max(output.metrics["step_size"])) <= cfg.critic_lr


def test_the_observation_statistics_are_upstreams():
    """Block: running observation normalisation.

    Upstream normalises in an environment wrapper, one stream at a time, cold
    starting from a mean of zero and a second moment of one and folding the
    reset observation in before anything is normalised. We do the same inside
    the kernel over a batch of streams, so the comparison drives one stream at a
    time on their side and reads the matching row on ours.
    """

    epsilon = 1e-8
    wrapper = NormalizeObservationWrapper(TinyContinuousEnv(), eps=epsilon)
    normalizer = make_normalizer(
        NormalizationConfig(normalize_observation=True, eps=epsilon)
    )

    key = jax.random.key(3)
    rollout = [
        jax.random.normal(jax.random.fold_in(key, step), (ENVS, 2), dtype=jnp.float32)
        for step in range(5)
    ]

    def theirs(stats, observation):
        """One stream's step through upstream's wrapper, without the wrapper."""

        was_mean, was_m2, was_count = stats
        mean, m2, count = wrapper._welford_update(
            was_mean, was_m2, was_count, observation
        )
        return (mean, m2, count), (observation - mean) / jnp.sqrt(
            m2 / count + wrapper.eps
        )

    # Upstream's reset builds these three and then folds the first observation
    # in, which is what our cold start does too.
    upstream_stats = [
        (jnp.zeros_like(rollout[0][env]), jnp.ones_like(rollout[0][env]), 1.0)
        for env in range(ENVS)
    ]

    normalized, state = normalizer.reset(rollout[0])
    for step, observation in enumerate(rollout):
        if step:
            outcome = normalizer.step(
                state,
                observation=observation,
                reward=jnp.zeros((ENVS,), dtype=jnp.float32),
                done=jnp.zeros((ENVS,), dtype=bool),
            )
            normalized, state = outcome.observation, outcome.state

        statistics = state.observation
        assert statistics is not None, "the normaliser stopped keeping statistics"

        for env in range(ENVS):
            upstream_stats[env], expected = theirs(
                upstream_stats[env], observation[env]
            )
            mean, m2, count = upstream_stats[env]
            assert_within(
                flattened(
                    {
                        "normalized": normalized[env],
                        "mean": statistics.mean[env],
                        "M2": statistics.M2[env],
                        "count": statistics.count[env],
                    }
                ),
                flattened(
                    {"normalized": expected, "mean": mean, "M2": m2, "count": count}
                ),
                f"observation statistics step={step} env={env}",
            )


def test_the_reward_scale_is_upstreams():
    """Block: running reward normalisation, and the channel that survives it.

    Upstream scales the reward by the spread of the discounted return, in a
    wrapper that replaces the reward on its way out of the environment. Ours does
    the same arithmetic inside the kernel, which is what lets it keep what the
    environment paid; the wrapper had nothing to keep it in, so it now leaves it
    in ``info``, and this drives the wrapper for real rather than restating its
    six lines here.

    So three things are asserted at once, on the same rollout: the scale is theirs
    leaf for leaf, the statistics behind it are theirs, and the reward the wrapper
    parks in ``info`` is the one the environment paid rather than the one the
    agent learns on. The last is what an episode return is read off, and reading
    it off the wrong one is how a score in normalised units came to be compared
    against a recorded one in the task's.
    """

    gamma, epsilon = 0.91, 1e-8
    env = TinyContinuousEnv()
    # The wrapper is typed for gymnax's own parameters; this environment is small
    # enough to terminate inside a test and is not one of theirs.
    params: Any = env.default_params
    wrapper = NormalizeRewardWrapper(env, gamma=gamma, eps=epsilon)
    normalizer = make_normalizer(
        NormalizationConfig(normalize_reward=True, reward_gamma=gamma, eps=epsilon)
    )

    actions = jax.random.normal(jax.random.key(5), (6, ENVS, 2), dtype=jnp.float32)
    nothing = jnp.zeros((ENVS, 2), dtype=jnp.float32)

    def row(state, action):
        """One stream through the wrapper, which is written for one."""

        _, state, scaled, done, info = wrapper.step(
            jax.random.key(0), state, action, params
        )
        return state, (scaled, info["environment_reward"], done)

    states = [wrapper.reset(jax.random.key(env), params)[1] for env in range(ENVS)]
    rollout = []
    for step in actions:
        stepped = [row(states[env], step[env]) for env in range(ENVS)]
        states = [state for state, _ in stepped]
        readings = [reading for _, reading in stepped]
        rollout.append(
            {
                name: jnp.stack([jnp.asarray(reading[index]) for reading in readings])
                for index, name in enumerate(("scaled", "paid", "done"))
            }
            | {
                name: jnp.stack([getattr(state, name) for state in states])
                for name in ("mean", "M2", "count", "G")
            }
        )

    # Ours: the whole batch at once, fed the reward the wrapper says was paid.
    _, state = normalizer.reset(nothing)
    for step, theirs in enumerate(rollout):
        outcome = normalizer.step(
            state,
            observation=nothing,
            reward=theirs["paid"],
            done=theirs["done"],
        )
        state = outcome.state
        statistics = state.reward
        assert (
            statistics is not None
        ), "the normaliser stopped keeping reward statistics"
        assert_within(
            flattened(
                {
                    "scaled": outcome.reward,
                    "mean": statistics.mean,
                    "M2": statistics.M2,
                    "count": statistics.count,
                    "G": statistics.G,
                }
            ),
            flattened(
                {name: theirs[name] for name in ("scaled", "mean", "M2", "count", "G")}
            ),
            f"reward scale step={step}",
        )

    # And the two channels are two channels, or there was nothing here to
    # get wrong in the first place.
    assert any(
        deviations({"reward": theirs["paid"]}, {"reward": theirs["scaled"]})
        for theirs in rollout
    ), "the wrapper scaled nothing, so this rollout cannot tell the channels apart"


def test_the_initial_second_moment_is_upstreams():
    """Block: where the bound's second moment starts.

    Upstream's caller builds it in ``init``; our rule builds its own, and the
    bias correction divides by ``1 - beta2**step`` from the first step, so
    starting anywhere else would show up as a first update of the wrong size.
    """

    traces = tree(jax.random.key(11))
    rule = make_obgd_rule(learning_rate=0.1, kappa=2.0)
    assert_within(
        flattened(rule.init(params=None, traces=traces)),
        flattened(jax.tree.map(jnp.zeros_like, traces)),
        "initial second moment",
    )


def transition(seed: int, done: str):
    """A state both kernels can be handed, and the pieces an update needs.

    Built by upstream's ``init`` so that the parameters, the carry and the
    observation are one consistent set rather than three separate inventions,
    and because a tree of the right shape is not the same thing as a tree the
    networks would actually produce.
    """

    keys = jax.random.split(jax.random.key(seed), 4)
    state = upstream().init(keys[0])
    timestep = replace(state.timestep, done=terminals(done, keys[1]))
    action = jax.random.normal(keys[2], (ENVS, 2), dtype=jnp.float32)
    return state, timestep, action, vector(keys[3])


# Where the gradient is allowed to differ from upstream's, in float32 last bits.
# Everywhere not named here: nowhere at all.
#
# Upstream asks for the whole batch's Jacobian at once, which costs a backward
# pass per stream over the whole batch and so grows with the square of the number
# of streams. We differentiate one stream at a time and map that over the batch,
# which is the same derivative of the same expression and costs what one stream
# costs. It is not the same summation order: the recurrent decay's gradient sums
# over the hidden axis, and a batch of one lays that reduction out differently
# than a batch of many. The gap that leaves is a fraction of a last bit -- 0.5 at
# worst over the seeds below -- and it appears on that decay alone, because it is
# the only parameter whose gradient is a reduction rather than an outer product.
REASSOCIATED = {"params/torso/cell/nu_log": 1.0}


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("done", ["mixed", "live", "terminal"])
def test_the_truncated_gradient_is_upstreams(seed, done):
    """Block: the gradient, under the credit setting that claims to be theirs.

    This is the seam the other blocks leave open. They compare arithmetic on
    quantities that were already computed; this compares how the quantities a
    parameter is credited for are obtained, which is the only thing StreamAC
    and our kernel were ever supposed to disagree about. Under ``tbptt`` they
    are not supposed to disagree at all.

    Our kernel applies the three parts of a network separately, since the credit
    adapter has to reach the torso, while upstream applies the whole module at
    once, and it differentiates per stream where upstream differentiates the
    batch. Same composition and same derivative, so exact everywhere except the
    one reduction named above -- and if that ever stops being the only place, or
    stops being a fraction of a bit, this is where it shows.
    """

    state, timestep, action, delta = transition(seed, done)
    theirs, mine = upstream(), ours(credit="tbptt")

    apart = {
        "actor": deviations(
            flattened(
                mine._actor_gradient(
                    state.actor_params, timestep, state.actor_carry, None, action, delta
                )
            ),
            flattened(
                theirs._actor_gradient(
                    state.actor_params, timestep, state.actor_carry, action, delta
                )
            ),
        ),
        "critic": deviations(
            flattened(
                mine._critic_gradient(
                    state.critic_params, timestep, state.critic_carry, None, delta
                )
            ),
            flattened(
                theirs._critic_gradient(
                    state.critic_params, timestep, state.critic_carry
                )
            ),
        ),
    }
    unexplained = sorted(
        (
            (bits, f"{domain} {path}")
            for domain, found in apart.items()
            for bits, path in found
            if bits > REASSOCIATED.get(path, 0.0)
        ),
        reverse=True,
    )
    assert not unexplained, (
        f"gradient seed={seed} done={done}: leaves apart from upstream by more "
        "than this file explains, worst first:\n"
        + "\n".join(f"  {bits:.1f} last bits  {where}" for bits, where in unexplained)
    )


def test_exact_credit_is_not_the_truncated_one():
    """And the setting is a setting, not a label on one behaviour.

    A sensitivity of zero would make the two agree, which is why the one here is
    drawn rather than initialised: at the very first step of a run they do agree,
    since nothing has accumulated yet. The assertion is only that they part once
    something has.
    """

    state, timestep, action, delta = transition(0, "live")
    exact, truncated = ours(), ours(credit="tbptt")
    shape = (ENVS, None)
    empty = exact.actor_credit.initialize(jax.random.key(0), shape)
    assert empty is not None, "exact credit stopped carrying a sensitivity"
    sensitivity = jax.tree.map(
        lambda leaf: jax.random.normal(jax.random.key(1), leaf.shape, leaf.dtype),
        empty,
    )

    carried = exact._actor_gradient(
        state.actor_params, timestep, state.actor_carry, sensitivity, action, delta
    )
    cut = truncated._actor_gradient(
        state.actor_params, timestep, state.actor_carry, None, action, delta
    )
    assert deviations(flattened(carried), flattened(cut)), (
        "exact and truncated credit produced the same gradient from a "
        "sensitivity that was not zero, so one of them is not doing its job"
    )
