"""``StreamAC.py`` against ``stream_ac.py``, which it is a rebuild of.

The layered kernel was written to be the flat one in four layers rather than a
new algorithm, so the flat one is its reference: same networks, same
configuration, same seed, and every leaf of the state and of the readings
compared to the last bit. Nothing else has to say what a number should be.

The cases matter more than the count. A training run with normalization off
reaches neither estimator and never opens an evaluation, and that is exactly
where the layering did most of its rearranging -- two components out of one, an
evaluation opener deleted, a second forward pass folded into the first. Both
bugs this file found were found by the cases and not by the comparison:

- ``stream_ac.evaluate`` scanned ``num_steps`` rounds while its own ``train``
  scanned ``num_steps // num_envs``, so an evaluation budget meant environment
  steps in one method and scan rounds in the other, and every shipped evaluation
  ran ``num_envs`` times longer than it was asked for. Both mean environment
  steps now.
- The layered kernel returned no ``forward`` or ``update`` readings at all,
  while the shipped entry declares eighteen of them and the driver drops a
  declared name it cannot find without saying so.

This file expires when the flat kernel does, which is the point: it guards a
migration, and a migration that has happened needs no guard. Until then it is
what makes "the refactor changed nothing" a thing anyone can check.

Run it for a verdict::

    pytest tests/test_layered_parity.py

Run it for the numbers::

    python tests/test_layered_parity.py
"""

from __future__ import annotations

import importlib

import jax
import pytest
from conftest import TinyContinuousEnv, assert_within, deviations, flattened

from reference import stream_ac as flat
from memorax.algorithms.contract import EvaluationConfig
from memorax.networks import heads
from memorax.networks.components import FFN, LayerNorm, Readout, Tanh
from memorax.networks.sequence import Sequence
from memorax.networks.sequence_models import RNN, RTUCell, RTUConfig
from memorax.rl import NormalizationConfig
from memorax.rl.updates import ObBound, Sgd

# Asked for by path: the package's lazy ``__getattr__`` hands back the *class*
# under this name, and what is wanted is the module beside it.
layered = importlib.import_module("memorax.algorithms.stream_ac")

ENVS = 3
FEATURES = 4
HIDDEN = 2
ROUNDS = 5

CASES = {
    "training, no normalization": (False, None, False),
    "training, both estimators": (True, None, False),
    "evaluation on inherited scales": (
        True,
        EvaluationConfig(reset_on_start=False, update_during_eval=True),
        True,
    ),
    "evaluation on fresh scales": (
        True,
        EvaluationConfig(reset_on_start=True, update_during_eval=True),
        True,
    ),
}


def build(module, *, normalize, evaluation):
    env = TinyContinuousEnv()
    action_dim = int(env.action_space(env.default_params).shape[0])

    def network(head):
        return Sequence(
            components=(
                FFN(features=FEATURES),
                LayerNorm(),
                Tanh(),
                RNN(
                    cell=RTUCell(config=RTUConfig(features=FEATURES, hidden_dim=HIDDEN))
                ),
                Readout(module=head),
            )
        )

    return module.StreamAC(
        module.StreamACConfig(
            num_envs=ENVS,
            gamma=0.9,
            trace_lambda=0.8,
            actor_bound=ObBound(kappa=2.0),
            actor_base=Sgd(lr=0.1),
            critic_bound=ObBound(kappa=2.0),
            critic_base=Sgd(lr=0.1),
            credit="rtrl",
            entropy_coefficient=0.01,
        ),
        env,
        env.default_params,
        network(heads.Gaussian(action_dim=action_dim)),
        network(heads.VNetwork()),
        observation_normalization=(
            NormalizationConfig(center=True, cold_start="seeded") if normalize else None
        ),
        reward_normalization=(
            NormalizationConfig(center=False, discount=0.9, reset_on_done=True)
            if normalize
            else None
        ),
        evaluation=evaluation,
    )


def _driven(module, *, normalize, evaluation, run_eval):
    agent = build(module, normalize=normalize, evaluation=evaluation)
    state = agent.init(jax.random.key(0))
    state, metrics = agent.train(jax.random.key(1), state, ROUNDS * ENVS)
    readings = None
    if run_eval:
        evaluated = agent.evaluate(jax.random.key(2), state, ROUNDS * ENVS)
        # The layered kernel hands back readings alone, which is what stops an
        # evaluation state being carried into training by accident.
        readings = evaluated[1] if module is flat else evaluated
    return state, metrics, readings


def compared(module, state, metrics, readings):
    """Everything both kernels carry, under one spelling.

    The flat kernel keeps a field per quantity and the layered one groups them
    by what owns them, so the correspondence is written out. There is nothing to
    infer here: it is a map between two layouts and it is the whole of what this
    file knows that a general comparison could not.
    """

    scales = (
        (state.observation_statistics, state.reward_statistics)
        if module is flat
        else (state.scales.observation, state.scales.reward)
    )
    found = {
        "interaction": metrics.interaction,
        "actor/params": state.actor_params if module is flat else state.actor.params,
        "critic/params": state.critic_params if module is flat else state.critic.params,
        "actor/traces": (
            state.actor_traces if module is flat else state.actor.rule.traces
        ),
        "critic/traces": (
            state.critic_traces if module is flat else state.critic.rule.traces
        ),
        "actor/carry": (
            state.actor_carry if module is flat else state.actor.recurrence.carry
        ),
        "timestep": state.timestep,
        "env_state": state.env_state,
        "scales/observation": scales[0],
        "scales/reward": scales[1],
        "forward/value": (
            metrics.forward.value if module is flat else metrics.forward.critic.value
        ),
        "forward/next_value": (
            metrics.forward.next_value
            if module is flat
            else metrics.forward.critic.next_value
        ),
        "forward/log_prob": (
            metrics.forward.log_prob
            if module is flat
            else metrics.forward.actor.log_prob
        ),
        "update/td_error": metrics.update.td_error,
    }
    for role in ("actor", "critic"):
        for reading in ("step_size", "grad_norm", "trace_norm"):
            found[f"update/{role}/{reading}"] = (
                getattr(metrics.update, f"{role}_{reading}")
                if module is flat
                else getattr(getattr(metrics.update, role), reading)
            )
    if readings is not None:
        found["evaluation"] = readings.interaction
    return flattened(found)


def _both(case):
    normalize, evaluation, run_eval = CASES[case]
    return {
        module: compared(
            module,
            *_driven(
                module, normalize=normalize, evaluation=evaluation, run_eval=run_eval
            ),
        )
        for module in (flat, layered)
    }


@pytest.mark.parametrize("case", list(CASES))
def test_the_layered_kernel_is_the_flat_one(case):
    """To the last bit, in every case the flat kernel can also be driven in."""

    found = _both(case)
    assert_within(found[layered], found[flat], case, allowed=0.0)


def main() -> None:
    """The same comparison, printed rather than asserted."""

    for case in CASES:
        found = _both(case)
        mine, theirs = found[layered], found[flat]
        apart = deviations(mine, theirs, 0.0)
        print(f"{'ok  ' if not apart else 'FAIL'} {case}: {len(theirs)} leaves")
        for bits, path in apart[:10]:
            print(f"       {bits:8.1f} last bits  {path}")


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
