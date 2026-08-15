"""``rtrrl_aaai.py`` against the algorithm it is a rebuild of.

Three comparisons, each answering a different question, because no single one of
them can answer all three:

``the algebra``
    The trace recurrence and what the TD error is multiplied into, against
    ``traces.py`` driving the same numbers. No networks at all, so a failure
    here is arithmetic and nothing else. This is the one that can run before
    anything is wired.

``the wiring``
    A whole step, against the published order re-hosted on this side's networks
    -- one vector-Jacobian product called twice, one combined objective
    differentiated once rather than two differentiated separately, the TD error
    read against a value carried from the previous step, the update taken along
    the trace before this step's gradient enters it. Both sides run the same
    networks and the same environment, so a mismatch can only be the wiring.

``the cell``
    memorax's LRU against the published one, forward and online gradient, with
    the published implementation's own BPTT mode as the judge. At the first step
    there is no history, so exact recurrent credit must equal plain
    backpropagation; whichever disagrees there is wrong.

They found different things, which is why all three are kept. The wiring
comparison found that the published actor differentiates *through* its
reparametrised draw -- ``action_dist.sample`` inside the differentiated function
-- which for a Gaussian sends the mean's gradient to exactly zero. The cell
comparison found that the published RTRL gradient disagrees with the published
BPTT gradient on ``B_real``, ``B_imag`` and ``gamma_log`` while this side agrees
with it on all eight leaves. Neither would have been visible to the other, and
neither was visible to twenty-one behavioural checks that all passed.

The published code is not vendored. CI points ``RTRRL_AAAI25`` at a clone; the
local case is a skip.

    pytest tests/test_rtrrl_parity.py       # a verdict
    python tests/test_rtrrl_parity.py       # the numbers
"""

from __future__ import annotations

import os
import sys

import jax
import jax.numpy as jnp
import pytest

# The kernel both files drive, built once. Two copies of a builder are two
# kernels the moment either drifts, and this file's whole job is to compare
# one thing against another.
from test_rtrrl import build, torso_network

from memorax.algorithms.rtrrl_aaai import RTRRLConfig, _advance_trace
from memorax.networks.sequence_models import LRUCell, LRUConfig, Memoroid
from memorax.networks.sequence_models.lru import LRUStructuredRTRL
from tests.support.numerics import assert_within, deviations, flattened

pytestmark = [pytest.mark.parity, pytest.mark.external]

ENVS = 3
STEPS = 12


def published(*names):
    """Modules from the RTRRL checkout, or the reason there are none."""

    root = os.environ.get("RTRRL_AAAI25")
    if not root:
        pytest.skip("set RTRRL_AAAI25 to an RTRRL-AAAI25 checkout to compare")
    if root not in sys.path:
        sys.path.insert(0, root)
    return [__import__(name, fromlist=["_"]) for name in names]


# --------------------------------------------------------------- the algebra


def _algebra():
    (traces,) = published("traces")
    cfg = RTRRLConfig(
        num_envs=ENVS, gamma=0.9, lambda_pi=0.8, lambda_v=0.6, lambda_rnn=0.4
    )
    shapes = {"A": (3, 2), "b": (3,)}

    def fake(seed):
        key = jax.random.key(seed)
        out = {}
        for name, shape in shapes.items():
            key, use = jax.random.split(key)
            out[name] = jax.random.normal(use, (ENVS, *shape))
        return out

    incoming, gradient = fake(1), fake(2)
    done = jnp.asarray([False, True, False])
    emphasis = jnp.asarray([1.0, 0.7, 0.49])

    found = {}
    for name, decay in (
        ("torso", cfg.lambda_rnn),
        ("actor", cfg.lambda_pi),
        ("critic", cfg.lambda_v),
    ):
        rate = cfg.gamma * decay
        # The published order: clear where the episode ended, then accumulate
        # with the emphasis riding on the gradient, one stream at a time.
        cleared = jax.tree.map(
            lambda old: jnp.where(
                done[(slice(None),) + (None,) * (old.ndim - 1)],
                jnp.zeros_like(old),
                old,
            ),
            incoming,
        )
        theirs = jax.vmap(
            lambda z, g, i: traces.trace_update(g, z, gamma_lambda=rate, _I=i)
        )(cleared, gradient, emphasis)
        # Jitted on both sides: ``trace_update`` carries ``@jax.jit`` and XLA
        # fuses its multiply-add, so an eager comparison is a sub-ulp apart for
        # reasons that are the format's and not the algorithm's.
        mine = jax.jit(
            lambda i, g, r, e, d=rate: _advance_trace(
                i, g, decay=d, reset_before=r, emphasis=e
            )
        )(incoming, gradient, done, emphasis)
        found[name] = (flattened(mine), flattened(theirs))
    return found


def test_the_trace_recurrence_is_the_published_one():
    for name, (mine, theirs) in _algebra().items():
        assert_within(mine, theirs, f"{name} trace", allowed=0.0)


# ---------------------------------------------------------------- the wiring


def _carried(state):
    core = state.core
    return flattened(
        {
            "torso/params": core.torso.params,
            "torso/slow": core.torso.slow_params,
            "torso/traces": core.torso.traces,
            "torso/carry": core.torso.recurrence.carry,
            "torso/differentiation_state": (
                core.torso.recurrence.differentiation_state
            ),
            "actor/params": core.actor.params,
            "actor/traces": core.actor.traces,
            "critic/params": core.critic.params,
            "critic/traces": core.critic.traces,
            "value": core.value,
            "emphasis": core.emphasis,
            "timestep": state.timestep,
        }
    )


def _wiring(steps, **overrides):
    published("traces")
    from reference.rtrrl_aaai25 import make_reference

    agent = build(**overrides)
    start = agent.init(jax.random.key(0))
    reference = make_reference(agent, torso_network())

    keys = jax.random.split(jax.random.key(7), steps)
    mine = theirs = start
    for step in range(steps):
        # ``train_step`` splits reset, action, env -- in that order.
        reset_key, action_key, env_key = jax.random.split(keys[step], 3)
        mine, _ = agent.train_step(mine, keys[step])
        theirs = reference(theirs, (action_key, env_key, reset_key))
    return _carried(mine), _carried(theirs)


def test_one_step_is_the_published_step_exactly():
    """Adding the two heads' cotangents is summing the objectives, to the bit."""

    mine, theirs = _wiring(1)
    assert_within(mine, theirs, "one step", allowed=0.0)


@pytest.mark.parametrize(
    "overrides", [{}, {"torso_follow": 1.0}, {"entropy_rate": 0.0}]
)
def test_a_rollout_stays_with_the_published_wiring(overrides):
    """Over many steps the two roundings drift, and only the roundings do.

    The reference sums the objectives before differentiating and this side adds
    the cotangents after, so the operations reach the same number by different
    fusions. Judged against the format rather than against zero.
    """

    mine, theirs = _wiring(STEPS, **overrides)
    assert_within(mine, theirs, f"{STEPS} steps {overrides}", allowed=8.0)


# ------------------------------------------------------------------ the cell


def _cell():
    (online_lru,) = published("models.online_lru")
    features, hidden, out = 3, 4, 3

    mine = Memoroid(
        cell=LRUCell(
            config=LRUConfig(features=features, hidden_dim=hidden, output_dim=out)
        )
    )
    differentiation = LRUStructuredRTRL(mine)
    carry = mine.initialize_carry(jax.random.key(0), (1, features))
    # The sensitivity belongs to the differentiation rule rather than to the cell
    # it differentiates, so it is asked for through the rule. Reaching past it to
    # the cell is what this file used to do, and is what stopped resolving.
    sensitivity = differentiation.initialize(jax.random.key(0), (1, features))
    done = jnp.zeros((1, 1), dtype=bool)
    params = mine.init(jax.random.key(2), jnp.zeros((1, 1, features)), done, carry)[
        "params"
    ]
    cell = params["cell"]

    # Same shapes throughout; only the spelling of the imaginary halves differs.
    theirs = {
        "params": {
            "C_real": cell["C_real"],
            "C_img": cell["C_imag"],
            "D": cell["D"],
            "OnlineLRUCell_0": {
                "LRUCell_0": {
                    "B_real": cell["B_real"],
                    "B_img": cell["B_imag"],
                    "nu_log": cell["nu_log"],
                    "theta_log": cell["theta_log"],
                    "gamma_log": cell["gamma_log"],
                }
            },
        }
    }

    x = jax.random.normal(jax.random.key(11), (features,))
    weight = jax.random.normal(jax.random.key(12), (out,))

    def spelled(grads):
        inner = grads["OnlineLRUCell_0"]["LRUCell_0"]
        return {
            "C_real": grads["C_real"],
            "C_imag": grads["C_img"],
            "D": grads["D"],
            "B_real": inner["B_real"],
            "B_imag": inner["B_img"],
            "nu_log": inner["nu_log"],
            "theta_log": inner["theta_log"],
            "gamma_log": inner["gamma_log"],
        }

    def theirs_grad(plasticity):
        layer = online_lru.OnlineLRULayer(
            d_output=out, d_hidden=hidden, plasticity=plasticity, activation=None
        )
        start = layer.initialize_carry(jax.random.key(0), (features,))
        return spelled(
            jax.grad(lambda p: (layer.apply(p, start, x)[1] * weight).sum())(theirs)[
                "params"
            ]
        )

    ours = jax.grad(
        lambda p: (
            differentiation(p, x[None, None], done, carry, sensitivity)[1][0, 0]
            * weight
        ).sum()
    )(params)["cell"]
    return ours, theirs_grad("rtrl"), theirs_grad("bptt")


def test_the_cells_first_gradient_is_plain_backpropagation():
    """No history yet, so exact recurrent credit has nothing to differ over.

    The judge is the published implementation's own BPTT mode. ``nu_log`` and
    ``theta_log`` are not evidence here and are excluded: with a zero initial
    carry the hidden state does not depend on the eigenvalue at all, so both
    sides report zero for reasons that have nothing to do with being right.
    """

    ours, _, bptt = _cell()
    judged = {k: v for k, v in bptt.items() if k not in ("nu_log", "theta_log")}
    assert_within(
        flattened(ours), flattened(judged), "memorax against BPTT", allowed=0.0
    )


def test_the_published_cell_disagrees_with_its_own_backpropagation():
    """Recorded because it is the reason this side does not copy that gradient.

    ``rtrl_gradient`` reads ``d_output_d_h = y_t[1][0]`` -- the first component
    of a hidden cotangent that has one entry per unit -- and broadcasts it over
    ``B`` and ``gamma``. The published ``grads_step`` runs inside ``jax.vmap``,
    where the carry is unbatched, so that is the path it takes.
    """

    _, rtrl, bptt = _cell()
    apart = [
        name
        for name in ("B_real", "B_imag", "gamma_log")
        if float(jnp.max(jnp.abs(rtrl[name] - bptt[name])))
        > 1e-6 * float(jnp.max(jnp.abs(bptt[name])))
    ]
    assert apart == ["B_real", "B_imag", "gamma_log"], (
        "the published RTRL gradient now agrees with its own BPTT on "
        f"{sorted(set(('B_real', 'B_imag', 'gamma_log')) - set(apart))}; if the "
        "checkout was fixed, this file's account of it is out of date"
    )


def main() -> None:
    if not os.environ.get("RTRRL_AAAI25"):
        print("set RTRRL_AAAI25 to an RTRRL-AAAI25 checkout to compare")
        return

    print("=== the algebra ===")
    for name, (mine, theirs) in _algebra().items():
        apart = deviations(mine, theirs, 0.0)
        print(f"{'ok  ' if not apart else 'FAIL'} {name} trace: {len(theirs)} leaves")

    print("\n=== the wiring ===")
    mine, theirs = _wiring(1)
    apart = deviations(mine, theirs, 0.0)
    print(f"{'ok  ' if not apart else 'FAIL'} one step: {len(theirs)} leaves, exact")
    for overrides in ({}, {"torso_follow": 1.0}, {"entropy_rate": 0.0}):
        mine, theirs = _wiring(STEPS, **overrides)
        apart = deviations(mine, theirs, 8.0)
        worst = max((bits for bits, _ in deviations(mine, theirs, 0.0)), default=0.0)
        print(
            f"{'ok  ' if not apart else 'FAIL'} {STEPS} steps {overrides or '{}'}: "
            f"worst {worst:.1f} last bits"
        )

    print("\n=== the cell, judged by the published BPTT ===")
    ours, rtrl, bptt = _cell()
    print(f"  {'leaf':12s} {'published RTRL':>16s} {'memorax':>12s}")
    for name in sorted(bptt):
        truth = jnp.asarray(bptt[name])
        scale = float(jnp.max(jnp.abs(truth))) or 1.0

        def apart_from(other):
            return float(jnp.max(jnp.abs(jnp.asarray(other) - truth))) / scale

        note = "  (vacuous)" if name in ("nu_log", "theta_log") else ""
        print(
            f"  {name:12s} {apart_from(rtrl[name]):16.3e} "
            f"{apart_from(ours[name]):12.3e}{note}"
        )


if __name__ == "__main__":
    # Under pytest the path is set up by ``pytest.ini`` and by conftest's
    # location. Run as a script, only this directory is on it.
    import pathlib as _pathlib
    import sys as _sys

    _here = _pathlib.Path(__file__).resolve().parent
    for _root in (_here, _here.parent):
        if str(_root) not in _sys.path:
            _sys.path.insert(0, str(_root))

    main()
