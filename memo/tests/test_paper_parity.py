"""The bounded step against the optimisers the paper published.

Everything in ``test_blocks.py`` answers to memorax, which is itself a fork.
This file answers to the source: ``optim.py`` from mohmdelsayed/streaming-drl,
the code behind "Streaming Deep Reinforcement Learning Finally Works". It is
not vendored -- that repository is CC BY-NC and this one is not -- so CI clones
it at a pinned commit and points ``STREAMING_DRL`` at it. Without that the file
skips, which is the local case.

Two things are being established. That ``obgd`` is the published ObGD, since
every recorded run of ours used it and nothing had ever checked it against
anything but a fork. And that ``adaptive_obgd_fixed`` is the published
AdaptiveObGD while ``adaptive_obgd`` is not, which is the whole reason the two
exist separately: the fork moved eps out of the square root, and while the
second moment is near eps that changes the step by a factor, not by a rounding.

The comparison crosses frameworks, so it cannot be exact. Their z_sum
accumulates in float64 through ``.item()`` and ours in float32, and torch and
XLA reduce in their own orders. The budget below is what that costs, and it is
several orders of magnitude tighter than any difference of formula.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import assert_within, deviations, flattened

from memorax.rl import make_obgd_rule

# What crossing frameworks costs, in float32 last bits. Chosen to be loose
# enough for two float accumulation orders and tight enough that a misplaced
# eps, a missing bias correction or a reassociated bound cannot fit under it.
FRAMEWORKS = 64.0

SETTINGS = {"learning_rate": 0.12, "kappa": 2.0, "beta2": 0.95, "eps": 1e-6}
DECAY = 0.89 * 0.71  # gamma * lambda, as the trace recurrence uses it
STEPS = 6


@pytest.fixture(scope="module")
def published():
    """The paper's own optimisers, imported from the clone CI makes."""

    root = os.environ.get("STREAMING_DRL")
    if not root:
        pytest.skip("set STREAMING_DRL to a streaming-drl checkout to compare")
    pytest.importorskip("torch")
    spec = importlib.util.spec_from_file_location(
        "streaming_drl_optim", Path(root) / "optim.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def theirs(module, optimiser: str, scale: float):
    """Drive the published optimiser and collect what it did to the parameters."""

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
    for gradient, delta in zip(grads(scale), surprises()):
        before = {name: tensor.detach().clone() for name, tensor in tensors.items()}
        for name, tensor in tensors.items():
            tensor.grad = torch.from_numpy(np.array(gradient[name], dtype=np.float32))
        step.step(delta)
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
    """Drive our rule over the same gradients and TD errors, one stream."""

    rule = make_obgd_rule(**SETTINGS, rule=rule_name)
    trace = {
        name: jnp.zeros((1,) + leaf.shape, dtype=jnp.float32)
        for name, leaf in grads(scale)[0].items()
    }
    moment = rule.init(params=None, traces=trace)

    collected = []
    for index, (gradient, delta) in enumerate(zip(grads(scale), surprises()), start=1):
        trace = {
            name: DECAY * carried + gradient[name][None, ...]
            for name, carried in trace.items()
        }
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
