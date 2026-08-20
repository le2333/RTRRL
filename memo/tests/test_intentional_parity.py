"""The intentional update against the code the paper published.

``sharifnassab/Intentional_RL``, the implementation behind "Intentional Updates
for Streaming Reinforcement Learning". Not vendored -- CI clones it at a pinned
commit and points ``INTENTIONAL_RL`` at it -- so this file skips where the
clone is absent, which is the local case.

It exists because the other two oracles cannot settle a misreading. A NumPy
transcription of the paper written beside a JAX one agrees with it about
whatever both authors read the same way; only the published code can say what
the algorithm *is*. Everything the two implementations disagree about after
this file passes is a deliberate difference, and there are three:

``the trace``
    theirs is inside the optimizer and zeroed after an update whose ``reset``
    is set; ours is a component of the algorithm and drops the carried trace
    before the next derivative joins it. Those are the same episode boundary
    written one step apart, and the driver below lines them up rather than
    papering over them.
``the streams``
    theirs is one environment; ours carries an env axis and averages the
    finished update over it. Driven at one stream here, where the two are the
    same arithmetic.
``the entropy``
    theirs is added into the policy objective before ``backward``; ours is
    folded into the derivative by the algorithm before the trace sees it. Both
    put it in the trace multiplied by ``sign(delta)``, which is what the
    comparison below feeds in.

The comparison crosses frameworks, so it cannot be exact. Their ``norm_grad``
and ``z_sum`` accumulate in float64 through ``.item()`` and ours in float32,
and torch and XLA reduce in their own orders. The budget is what that costs and
is far tighter than any difference of formula could hide under.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from memorax.rl.intentional import (
    ADVANTAGE,
    TD,
    IntentionalOptimizer,
    IntentionalUpdate,
)
from memorax.rl.traces import CURRENT, Trace

pytestmark = [pytest.mark.parity, pytest.mark.external]

# What crossing frameworks costs, in float32 last bits.
FRAMEWORKS = 64.0

GAMMA = 0.99
LAMBDA = 0.8
DECAY = GAMMA * LAMBDA
WIDTH = 6
STEPS = 12

# The published actor-critic's own numbers.
ETA_POLICY = 0.05
ETA_VALUE = 0.5
SETTINGS = {
    TD: IntentionalUpdate(eta=ETA_VALUE),
    ADVANTAGE: IntentionalUpdate(eta=ETA_POLICY),
}


def _published():
    """Their optimizer, loaded from the clone CI makes.

    By path rather than by import: their module is called ``optimizer``, which
    is a name this suite should not be handing to the rest of the run.
    """

    root = os.environ.get("INTENTIONAL_RL")
    if not root:
        pytest.skip("set INTENTIONAL_RL to an Intentional_RL checkout to compare")
    pytest.importorskip("torch")
    spec = importlib.util.spec_from_file_location(
        "intentional_rl_optimizer", Path(root) / "optimizer.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, root)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(root)
    return module


@pytest.fixture(scope="module")
def published():
    return _published()


def transitions(seed: int = 0):
    """One fixed sequence of derivatives, TD errors and episode boundaries.

    The TD errors span three orders of magnitude and change sign, and one
    episode ends in the middle, because those are the places the clip, the
    advantage scale and the reset each do something.
    """

    generator = np.random.default_rng(seed)
    deltas = [0.4, -1.2, 0.05, 30.0, -0.3, 0.9, -2.5, 0.0, 1.1, -0.7, 0.2, 4.0]
    resets = [False] * STEPS
    resets[5] = True
    return [
        (
            generator.normal(size=WIDTH).astype(np.float32),
            np.float32(delta),
            reset,
        )
        for delta, reset in zip(deltas, resets)
    ]


def theirs(module, sequence, *, signal):
    """Their optimizer, driven a transition at a time.

    The gradient is written straight onto ``p.grad`` rather than produced by a
    backward pass: what is being compared is the optimizer, and feeding both
    implementations one fixed sequence of derivatives is what makes the
    comparison about the update rule and nothing else.
    """

    import torch

    parameter = torch.nn.Parameter(torch.zeros(WIDTH, dtype=torch.float64))
    build = (
        module.IntentionalOptimizerValue
        if signal == TD
        else module.IntentionalOptimizerPolicy
    )
    optimizer = build(
        [parameter],
        gamma=GAMMA,
        lamda=LAMBDA,
        eta=SETTINGS[signal].eta,
    )
    taken = []
    for gradient, delta, reset in sequence:
        before = parameter.data.clone()
        parameter.grad = torch.tensor(gradient, dtype=torch.float64)
        optimizer.step(float(delta), reset=reset)
        taken.append(
            (
                (parameter.data - before).numpy().copy(),
                float(optimizer.safe_delta),
            )
        )
    return taken


def ours(sequence, *, signal):
    """The same sequence through the trace component and the optimizer.

    Their reset zeroes the trace *after* the update it is passed with; ours
    drops the carried trace before the next derivative joins it. Shifting the
    flags by one is what makes those the same episode boundary.
    """

    trace = Trace(decay=DECAY, reads=CURRENT, emphasized=False)
    rule = IntentionalOptimizer(SETTINGS[signal], decay=DECAY, signal=signal)
    params = {"w": jnp.zeros((WIDTH,), dtype=jnp.float32)}
    carried = trace.initial(params, 1)
    state = rule.init(params, streams=1)
    ones = jnp.ones((1,), dtype=jnp.float32)

    dropped = [False] + [reset for _, _, reset in sequence[:-1]]
    taken = []
    for step, ((gradient, delta, _), reset) in enumerate(
        zip(sequence, dropped), start=1
    ):
        derivative = {"w": jnp.asarray(gradient)[None]}
        used, carried = trace.stepped(
            carried,
            derivative,
            reset=jnp.asarray([1.0 if reset else 0.0]),
            emphasis=ones,
        )
        updates, state, reading = rule.update(
            delta=jnp.asarray([delta]),
            trace=used,
            derivative=derivative,
            direct=None,
            step=step,
            params=params,
            state=state,
        )
        taken.append((np.asarray(updates["w"]), float(reading.signal[0]), state))
    return taken


def _within(ours_value, theirs_value, *, what: str):
    """Two float32 numbers, within the framework budget."""

    scale = max(abs(float(np.max(np.abs(theirs_value)))), 1e-6)
    apart = float(np.max(np.abs(np.asarray(ours_value) - np.asarray(theirs_value))))
    assert apart <= FRAMEWORKS * np.spacing(np.float32(scale)), (
        f"{what}: {apart} apart, budget "
        f"{FRAMEWORKS * np.spacing(np.float32(scale))}"
    )


@pytest.mark.parametrize("signal", [TD, ADVANTAGE], ids=["value", "policy"])
def test_every_step_is_the_published_optimizer_s_step(published, signal):
    """Intentional TD and the intentional policy gradient, transition by transition.

    Both are driven over one sequence with an outlier, a sign change, a zero
    and an episode boundary in it, and every parameter delta is compared. A
    misplaced bias correction, a denominator that forgot its floor, or a trace
    read a step early would all show here and nowhere else.
    """

    sequence = transitions()
    reference = theirs(published, sequence, signal=signal)
    taken = ours(sequence, signal=signal)

    for step, ((update, signal_value, _), (expected, safe_delta)) in enumerate(
        zip(taken, reference), start=1
    ):
        _within(signal_value, safe_delta, what=f"step {step} signal")
        _within(update, expected, what=f"step {step} update")


def test_the_running_statistics_are_the_published_ones(published):
    """The state behind the step, not only the step.

    Their ``sigma`` is the uncorrected average and ours is the corrected one,
    which is the same sequence read at two points: dividing theirs by
    ``1 - (gamma*lambda)^t`` is what makes them the same number, and asserting
    it is what says the two bias corrections agree rather than cancelling
    inside a product.
    """

    sequence = transitions()
    module = published
    import torch

    parameter = torch.nn.Parameter(torch.zeros(WIDTH, dtype=torch.float64))
    optimizer = module.IntentionalOptimizerValue(
        [parameter], gamma=GAMMA, lamda=LAMBDA, eta=ETA_VALUE
    )
    taken = ours(sequence, signal=TD)

    for step, ((gradient, delta, reset), (_, _, state)) in enumerate(
        zip(sequence, taken), start=1
    ):
        parameter.grad = torch.tensor(gradient, dtype=torch.float64)
        optimizer.step(float(delta), reset=reset)
        corrected = optimizer.sigma / (1 - DECAY**step)
        _within(float(state.sigma_bar[0]), corrected, what=f"step {step} sigma_bar")
        theirs_nu = optimizer.state[parameter]["entrywise_squared_grad"].numpy() / (
            1 - 0.999**step
        )
        _within(np.asarray(state.nu["w"][0]), theirs_nu, what=f"step {step} nu")
