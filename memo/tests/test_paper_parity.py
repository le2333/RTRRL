"""Every block of StreamAC against the code the paper published.

Everything in ``test_blocks.py`` answers to memorax, which is itself a fork.
This file answers to the source: mohmdelsayed/streaming-drl, the code behind
"Streaming Deep Reinforcement Learning Finally Works". It is not vendored --
that repository is CC BY-NC and this one is not -- so CI clones it at a pinned
commit and points ``STREAMING_DRL`` at it. Without that the file skips, which is
the local case.

It began as the optimiser alone, and the gap that left is why it no longer is.
A fork can be faithful in the lines you read and unfaithful in the ones you did
not, and the only block whose reading had been checked against the source was
the one this file already covered. Reading the rest and finding them faithful is
not the same as driving them, so each block below is driven against the
published code that computes it:

- The TD error and the actor's ascent direction, by running their own
  ``update_params`` and intercepting what it hands the optimiser.
- The eligibility trace, through our kernel's ``_trace`` rather than a decay
  respelled in the test, over episode boundaries as well as within one.
- The bounded step, which is where the file started: ``obgd`` is the published
  ObGD, and ``adaptive_obgd_fixed`` is the published AdaptiveObGD while
  ``adaptive_obgd`` is not. That last is the reason two adaptive rules exist --
  the fork moved eps out of the square root, and while the second moment is near
  eps that changes the step by a factor rather than by a rounding.
Normalisation was compared here too, against ``normalization_upstream``, the arm
that reproduced their three differences. That arm is gone: the differences are
three fields on one estimator now -- ``cold_start``, ``variance`` and
``reset_on_done`` -- and nothing drives them against the published wrappers. The
comparison returns when there is a surface to write it against.

The comparison crosses frameworks, so most of it cannot be exact. Their z_sum
accumulates in float64 through ``.item()`` and ours in float32, and torch and
XLA reduce in their own orders. The budget below is what that costs, and it is
several orders of magnitude tighter than any difference of formula.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_blocks import SETTINGS as KERNEL_SETTINGS
from test_blocks import ours as ours_kernel

from memorax.rl import make_bounded_rule, make_td0
from memorax.rl.updates import (
    AdaptiveObBound,
    AdaptiveObBoundFixed,
    ObBound,
    Sgd,
)
from tests.support.numerics import assert_within, deviations, flattened

pytestmark = [pytest.mark.parity, pytest.mark.external]

# What crossing frameworks costs, in float32 last bits. Chosen to be loose
# enough for two float accumulation orders and tight enough that a misplaced
# eps, a missing bias correction or a reassociated bound cannot fit under it.
FRAMEWORKS = 64.0

SETTINGS = {"learning_rate": 0.12, "kappa": 2.0, "beta2": 0.95, "eps": 1e-6}
BOUNDS = {
    "obgd": ObBound(kappa=SETTINGS["kappa"]),
    "adaptive_obgd": AdaptiveObBound(
        kappa=SETTINGS["kappa"], beta2=SETTINGS["beta2"], eps=SETTINGS["eps"]
    ),
    "adaptive_obgd_fixed": AdaptiveObBoundFixed(
        kappa=SETTINGS["kappa"], beta2=SETTINGS["beta2"], eps=SETTINGS["eps"]
    ),
}
# gamma * lambda, as the trace recurrence uses it. Taken from the settings our
# kernel is built with, since the trace being driven below is now our kernel's.
DECAY = KERNEL_SETTINGS["gamma"] * KERNEL_SETTINGS["trace_lambda"]
STEPS = 6


def _published(filename: str, *, needs_torch: bool = True):
    """One of their files, loaded from the clone CI makes.

    Loaded by path rather than imported, because their package name is the one
    the fork answers to and putting theirs on the path would change which
    ``optim`` the rest of the suite sees.
    """

    root = os.environ.get("STREAMING_DRL")
    if not root:
        pytest.skip("set STREAMING_DRL to a streaming-drl checkout to compare")
    if needs_torch:
        pytest.importorskip("torch")
    spec = importlib.util.spec_from_file_location(
        f"streaming_drl_{Path(filename).stem}", Path(root) / filename
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # Their agent files do ``from optim import ObGD``, which only resolves if
    # their directory is importable while the file executes.
    sys.path.insert(0, root)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(root)
    return module


@pytest.fixture(scope="module")
def published():
    """The paper's own optimisers."""

    return _published("optim.py")


@pytest.fixture(scope="module")
def published_agent():
    """The paper's own continuous-control StreamAC, networks and all."""

    pytest.importorskip("gymnasium")
    return _published("stream_ac_continuous.py")


def grads(scale: float):
    """One gradient tree per step, from a written-down seed."""

    keys = jax.random.split(jax.random.key(11), STEPS)
    trees = []
    for key in keys:
        kernel, bias = jax.random.split(key)
        trees.append(
            {
                "kernel": scale * jax.random.normal(kernel, (4, 3), dtype=jnp.float32),
                "bias": scale * jax.random.normal(bias, (3,), dtype=jnp.float32),
            }
        )
    return trees


def surprises():
    """One TD error per step, crossing the bound and passing through zero."""

    return [0.75, -2.5, 0.0, 4.0, -0.125, 1.5]


def episodes():
    """Which of those steps ended an episode.

    Two of the six, so that the trace is required to restart more than once, and
    neither of them first or last: a reset on the opening step is invisible
    against a trace that starts at zero anyway, and one on the closing step is
    never read back.
    """

    return [False, True, False, False, True, False]


def theirs(module, optimiser: str, scale: float, ends: list | None = None):
    """Drive the published optimiser and collect what it did to the parameters."""

    ends = episodes() if ends is None else ends

    import torch

    tensors = {
        name: torch.zeros(tuple(leaf.shape), dtype=torch.float32)
        for name, leaf in grads(scale)[0].items()
    }
    for tensor in tensors.values():
        tensor.requires_grad_(True)
    kwargs = {"lr": SETTINGS["learning_rate"], "kappa": SETTINGS["kappa"]}
    if optimiser == "AdaptiveObGD":
        kwargs |= {"beta2": SETTINGS["beta2"], "eps": SETTINGS["eps"]}
    step = getattr(module, optimiser)(
        list(tensors.values()), gamma=0.89, lamda=0.71, **kwargs
    )

    collected = []
    for gradient, delta, done in zip(grads(scale), surprises(), ends):
        before = {name: tensor.detach().clone() for name, tensor in tensors.items()}
        for name, tensor in tensors.items():
            tensor.grad = torch.from_numpy(np.array(gradient[name], dtype=np.float32))
        step.step(delta, reset=done)
        # They descend a loss where we ascend an objective, so their parameter
        # change is our update with the sign turned around once.
        collected.append(
            {
                name: jnp.asarray(
                    (before[name] - tensor.detach()).numpy(), dtype=jnp.float32
                )
                for name, tensor in tensors.items()
            }
        )
    return collected


def mine(rule_name: str, scale: float):
    """Drive our rule over the same gradients, TD errors and episode boundaries.

    The trace comes from our kernel's own trace block rather than from a decay
    respelled here. That is the point of routing it through: their trace lives
    inside the optimiser, ours is a separate block, and a test that rebuilds ours
    in two lines of its own compares their recurrence against those two lines
    instead of against the block a run uses.

    The two sides mask on different steps and mean the same thing by it. Theirs
    zeroes the trace at the end of a step that ended an episode; ours masks the
    trace it inherits at the start of the next one. So ``reset_before`` here is
    the previous step's flag, and the first step inherits nothing.
    """

    # The trace is the parameter block's, not the kernel's: both heads own one
    # and they decay it identically, so either answers for the recurrence. Only
    # the recurrence is taken from it -- the rule below is built from this
    # file's own settings, so which head this is cannot reach the comparison.
    block = ours_kernel().core.actor.block
    assert block.trace_decay == DECAY, (
        f"our kernel decays the trace by {block.trace_decay} where the "
        f"published optimiser below is built with {DECAY}"
    )

    rule = make_bounded_rule(
        bound=BOUNDS[rule_name], base=Sgd(lr=SETTINGS["learning_rate"])
    )
    trace = {
        name: jnp.zeros((1,) + leaf.shape, dtype=jnp.float32)
        for name, leaf in grads(scale)[0].items()
    }
    moment = rule.init(params=None, traces=trace)
    ended = [False] + episodes()[:-1]

    collected = []
    for index, (gradient, delta, reset_before) in enumerate(
        zip(grads(scale), surprises(), ended), start=1
    ):
        # StreamAC resets before it acts, so the trace it carries away and the
        # one it steps with are one tree rather than two.
        trace = block.trace(
            trace,
            {name: leaf[None, ...] for name, leaf in gradient.items()},
            reset_before=jnp.asarray([reset_before]),
        )
        output = rule.apply(
            trace,
            None,
            moment,
            delta=jnp.asarray([delta], dtype=jnp.float32),
            step=index,
            params=None,
        )
        moment = output.state
        collected.append(output.updates)
    return collected


@pytest.mark.parametrize("magnitude", [0.05, 5.0], ids=["unbound", "bound"])
def test_obgd_is_the_published_obgd(published, magnitude):
    """Our default bounded rule is the one the paper's experiments all ran."""

    for index, (ours, reference) in enumerate(
        zip(mine("obgd", magnitude), theirs(published, "ObGD", magnitude)), start=1
    ):
        assert_within(
            flattened(ours),
            flattened(reference),
            f"obgd step={index} trace={magnitude}",
            allowed=FRAMEWORKS,
        )


@pytest.mark.parametrize("magnitude", [0.05, 5.0], ids=["unbound", "bound"])
def test_the_fixed_adaptive_rule_is_the_published_adaptive_obgd(published, magnitude):
    """``adaptive_obgd_fixed`` is AdaptiveObGD; the name means what it says."""

    for index, (ours, reference) in enumerate(
        zip(
            mine("adaptive_obgd_fixed", magnitude),
            theirs(published, "AdaptiveObGD", magnitude),
        ),
        start=1,
    ):
        assert_within(
            flattened(ours),
            flattened(reference),
            f"adaptive_obgd_fixed step={index} trace={magnitude}",
            allowed=FRAMEWORKS,
        )


# Eleven observations, which is Hopper's width and is not an arbitrary choice.
# Their ``sparse_init`` zeroes ``ceil(0.9 * fan_in)`` of each row's inputs, so at
# ten or fewer it zeroes every one of them and the first layer emits zeros
# whatever the observation is. At five, the width this was first written at, both
# value estimates come out exactly zero, the bootstrap term contributes nothing
# and the terminal case agrees with the live one for no reason at all.
TRANSITION = {
    "observation": [0.3, -1.2, 0.75, 0.0, 2.1, -0.5, 1.1, 0.4, -0.9, 1.6, -0.2],
    "next_observation": [-0.4, 0.9, -1.5, 0.25, 0.6, 1.3, -0.7, 0.1, 2.0, -1.1, 0.8],
    "action": [0.2, -0.7],
}

# Two rewards, chosen to land either side of zero. Their value head starts near
# zero under the sparse initialisation, so at the first transition the TD error
# is mostly the reward and takes its sign; that this held is asserted rather than
# relied on. The third sign needs the value it is cancelling and is reached in
# its own test below.
REWARDS = {"up": 1.375, "down": -2.25}


# Their agent is built with the settings our kernel is built with, so that the
# two objectives being compared are the same objective. Taken from the block
# tests rather than restated, and asserted because a silent drift between the
# two would make every comparison below vacuous rather than failing.
ENTROPY_COEFFICIENT = KERNEL_SETTINGS["entropy_coefficient"]
GAMMA = KERNEL_SETTINGS["gamma"]


def their_agent(module, *, done: bool, reward: float, seed: int = 0):
    """Their StreamAC, driven one transition, with what it computed read off.

    Their TD error and their two objectives are lines inside ``update_params``
    and are not functions anything can call. So the agent is run for real and
    every quantity is intercepted where their code hands it somewhere: the TD
    error at the optimiser, which is given it as a float, and the two objectives
    at ``backward``, which is given the scalar itself.

    Nothing here recomputes any of them. ``log_prob`` and ``entropy`` are read
    from their distribution because ours has to be fed the same two numbers to
    be comparing an objective rather than a network, but the objective ours is
    compared against is the tensor theirs called ``backward`` on.
    """

    import torch

    torch.manual_seed(seed)
    agent = module.StreamAC(
        n_obs=len(TRANSITION["observation"]),
        n_actions=len(TRANSITION["action"]),
        # Their default width. A narrow one is tempting for a single transition
        # and is wrong here: their initialiser zeroes ninety per cent of every
        # weight, and at eight units that leaves the critic outputting exactly
        # zero, which makes the bootstrap term and the zero-surprise case below
        # pass without being exercised.
        hidden_size=128,
        gamma=GAMMA,
        lamda=0.71,
    )

    stepped: dict = {}
    for name in ("optimizer_policy", "optimizer_value"):
        original = getattr(agent, name).step

        def capture(delta, reset=False, _name=name, _original=original):
            stepped[_name] = (float(delta), reset)
            return _original(delta, reset=reset)

        getattr(agent, name).step = capture

    # Their two ``backward`` calls, in the order ``update_params`` makes them:
    # the critic's objective first, then the actor's.
    descended: list = []
    original_backward = torch.Tensor.backward

    def record(self, *args, **kwargs):
        descended.append(float(self.detach()))
        return original_backward(self, *args, **kwargs)

    observation = np.asarray(TRANSITION["observation"], dtype=np.float32)
    next_observation = np.asarray(TRANSITION["next_observation"], dtype=np.float32)
    action = np.asarray(TRANSITION["action"], dtype=np.float32)

    # Before the update and not after it. ``update_params`` steps both
    # optimisers, so the networks it leaves behind are not the ones it computed
    # its TD error and its objectives from.
    with torch.no_grad():
        state = torch.from_numpy(observation)
        mu, std = agent.pi(state)
        distribution = torch.distributions.Normal(mu, std)
        log_prob = float(distribution.log_prob(torch.from_numpy(action)).sum())
        entropy = float(distribution.entropy().sum())
        value = float(agent.v(state))
        next_value = float(agent.v(torch.from_numpy(next_observation)))

    torch.Tensor.backward = record
    try:
        agent.update_params(
            observation,
            action,
            reward,
            next_observation,
            done,
            entropy_coeff=ENTROPY_COEFFICIENT,
        )
    finally:
        torch.Tensor.backward = original_backward

    assert len(descended) == 2, f"their update made {len(descended)} backward calls"
    critic_objective, actor_objective = descended

    delta, reset = stepped["optimizer_policy"]
    return {
        "delta": delta,
        "reset": reset,
        "reward": reward,
        "log_prob": log_prob,
        "entropy": entropy,
        "value": value,
        "next_value": next_value,
        # They descend where we ascend, so their direction is these negated.
        "actor": -actor_objective,
        "critic": -critic_objective,
    }


def our_directions(theirs):
    """Our two ascent directions, driven by the numbers their agent produced.

    Each objective takes a head's output rather than the readings inside it --
    the actor's a distribution, the critic's a value with its time and feature
    axes still on -- because that is the shape a head hands its own block. So
    the distribution below answers with their log-probability and their entropy
    and nothing else, which is what makes this a comparison of the arithmetic
    wrapped around the two readings rather than of two networks.
    """

    class Answers:
        def log_prob(self, action):
            del action
            return jnp.float32([[theirs["log_prob"]]])

        def entropy(self):
            return jnp.float32([[theirs["entropy"]]])

    core = ours_kernel().core
    return {
        "actor": core.actor.objective(
            (Answers(), {}),
            jnp.zeros((1, 1), dtype=jnp.float32),
            jnp.float32([theirs["delta"]]),
        ),
        "critic": core.critic.objective((jnp.float32([[[theirs["value"]]]]), {})),
    }


@pytest.mark.parametrize("done", [False, True], ids=["live", "terminal"])
def test_the_td_error_is_the_published_td_error(published_agent, done):
    """Block: TD(0), against the line their agent actually steps with.

    ``test_blocks.py`` compares this against the fork, which is a reading of the
    same three lines. This runs those lines. The terminal case is the one worth
    crossing: their mask is ``0 if done else 1`` written out, and an
    implementation that dropped it agrees everywhere else.
    """

    theirs = their_agent(published_agent, done=done, reward=REWARDS["up"])
    ours = make_td0()(
        reward=jnp.float32(theirs["reward"]),
        value=jnp.float32(theirs["value"]),
        next_value=jnp.float32(theirs["next_value"]),
        terminal=jnp.bool_(done),
        gamma=GAMMA,
    )
    assert_within(
        {"td": np.asarray(ours)},
        {"td": np.float32(theirs["delta"])},
        f"td error done={done}",
        allowed=FRAMEWORKS,
    )
    # Their optimiser is told to reset exactly when the transition ended an
    # episode, which is the fact our trace encodes by masking the step after.
    assert theirs["reset"] == done
    # And the bootstrap is carrying something, or the terminal case above would
    # agree with the live one for no reason.
    assert abs(theirs["next_value"]) > 1e-4


@pytest.mark.parametrize("surprise", sorted(REWARDS), ids=sorted(REWARDS))
def test_the_actor_and_critic_directions_are_the_published_ones(
    published_agent, surprise
):
    """Block: both ascent directions, against the scalars they call backward on.

    Their actor objective folds in the entropy coefficient and the sign of the
    TD error, and their critic objective is the negated value. Both are taken
    from the tensor their own line differentiates, so what is compared is their
    expression and not a reading of it.

    The sign their entropy term carries is the sign of the TD error their own
    transition produced, so it is reached by paying a different reward rather
    than by handing the objective a TD error from outside their code path. At
    initialisation their value head is near zero, so the reward is most of the
    TD error and its sign is the reward's; the test below the next one checks
    that the three rewards did land on three different signs rather than
    assuming they would.
    """

    theirs = their_agent(published_agent, done=False, reward=REWARDS[surprise])
    directions = our_directions(theirs)
    assert_within(
        {
            "actor": np.asarray(directions["actor"][0]),
            "critic": np.asarray(directions["critic"][0]),
        },
        {"actor": np.float32(theirs["actor"]), "critic": np.float32(theirs["critic"])},
        f"directions td={surprise}",
        allowed=FRAMEWORKS,
    )
    # The entropy term is carried by the coefficient and is not negligible, so
    # an implementation that dropped it would not have passed the assertion.
    assert abs(ENTROPY_COEFFICIENT * theirs["entropy"]) > 1e-4


def test_the_two_rewards_reach_both_signs_of_the_surprise(published_agent):
    """And the entropy term's sign was exercised both ways.

    The comparison above cannot hand their code a TD error, only a reward, so it
    pays two and takes what their agent makes of them. If both landed on the
    same sign, the ``sign`` in their entropy term would only ever have been
    checked one way round and an implementation using ``abs`` would have passed.
    """

    signs = {
        name: np.sign(their_agent(published_agent, done=False, reward=reward)["delta"])
        for name, reward in REWARDS.items()
    }
    assert set(signs.values()) == {-1.0, 1.0}, (
        f"the two rewards produced TD errors of sign {signs}, so the entropy "
        "term's sign was only exercised one way"
    )


def test_the_direction_is_the_published_one_at_a_surprise_of_exactly_zero(
    published_agent,
):
    """And at the third sign, which is the one that separates sign from a test.

    ``sign`` returns zero at zero, so their entropy term drops out entirely
    there, while an implementation written as ``where(td > 0, ...)`` would agree
    at every other TD error and disagree only here.

    Reached rather than stipulated. A terminal transition discards the bootstrap,
    so their TD error is the reward minus the value; paying exactly the value
    their own critic reports makes it exactly zero, and the assertion below
    checks it did rather than trusting the arithmetic to be exact.
    """

    probe = their_agent(published_agent, done=True, reward=0.0)
    theirs = their_agent(published_agent, done=True, reward=probe["value"])
    assert theirs["delta"] == 0.0, (
        f"paying their own value produced a TD error of {theirs['delta']:.3g} "
        "rather than exactly zero, so this is not the zero case"
    )

    directions = our_directions(theirs)
    assert_within(
        {"actor": np.asarray(directions["actor"][0])},
        {"actor": np.float32(theirs["actor"])},
        "actor direction at a TD error of exactly zero",
        allowed=FRAMEWORKS,
    )
    # And the entropy term really did drop out, so this is the branch it claims.
    assert_within(
        {"actor": np.asarray(directions["actor"][0])},
        {"actor": np.float32(theirs["log_prob"])},
        "the entropy term at a TD error of zero",
        allowed=FRAMEWORKS,
    )


def test_the_episode_boundaries_changed_what_was_compared(published):
    """And the resets in that comparison were doing something.

    Both sides now cross two episode boundaries, theirs by zeroing the trace
    after the step and ours by masking it before the next one. If the trace were
    small enough, or the boundaries placed where nothing reads them back, the
    comparison would hold identically without them and would be checking the
    decay alone. So the same six steps are run with the boundaries taken out and
    required to come out somewhere else.
    """

    with_resets = theirs(published, "ObGD", 5.0)
    without = theirs(published, "ObGD", 5.0, ends=[False] * STEPS)

    apart = [
        index
        for index, (left, right) in enumerate(zip(with_resets, without))
        if deviations(flattened(left), flattened(right))
    ]
    assert apart, (
        "the published optimiser produced the same six updates with the episode "
        "boundaries as without them, so the resets in the comparison above were "
        "not being exercised"
    )


def gap(ours, reference) -> float:
    """How far apart two updates are, relative to the size of the reference."""

    worst = 0.0
    for name, leaf in flattened(ours).items():
        expected = flattened(reference)[name]
        scale = float(jnp.max(jnp.abs(expected)))
        if scale == 0.0:
            continue
        worst = max(worst, float(jnp.max(jnp.abs(leaf - expected))) / scale)
    return worst


def test_the_forks_adaptive_rule_is_not_the_published_one(published):
    """And the fork's variant is a different rule, not a rounding of it.

    Kept as an assertion rather than a comment because it is the reason two
    adaptive rules exist. Which way the difference points is not obvious: the
    published denominator is the larger one while the second moment is near eps,
    which shrinks the update, but it also shrinks the normalised trace norm the
    bound is measured from, which grows it back. Measured, on a trace this
    small, the fork steps about half as far -- not the thousandfold the
    denominators alone would suggest.

    The one thing the algebra does settle is that the difference has to fade:
    once the second moment clears eps, both denominators are sqrt(v_hat) to
    within a rounding, and the second case checks that it does.
    """

    reference = theirs(published, "AdaptiveObGD", 0.001)
    fork = mine("adaptive_obgd", 0.001)
    fixed = mine("adaptive_obgd_fixed", 0.001)

    assert deviations(
        flattened(fork[0]), flattened(reference[0]), allowed=FRAMEWORKS
    ), "the fork's adaptive rule would have to differ from the published one"
    assert not deviations(
        flattened(fixed[0]), flattened(reference[0]), allowed=FRAMEWORKS
    ), "and the fixed one would not"

    near_eps = gap(fork[0], reference[0])
    assert near_eps > 0.2, (
        f"where the second moment sits near eps the two rules are {near_eps:.3g} "
        f"apart relative to the published step, which is not a rounding"
    )

    clear_of_eps = gap(
        mine("adaptive_obgd", 5.0)[0], theirs(published, "AdaptiveObGD", 5.0)[0]
    )
    assert clear_of_eps < near_eps / 100, (
        f"and once the second moment clears eps they should converge, but they "
        f"are still {clear_of_eps:.3g} apart"
    )
