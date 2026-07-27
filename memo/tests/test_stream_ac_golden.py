"""StreamAC still computes what it computed when the recorded score was set.

The kernel has been rewritten twice since a masked Hopper run scored 1043, and
the reproduction attempt reached 850. Neither rewrite was checked against a
number: the first one deleted this snapshot rather than answer it, on the
grounds that the builder which produced it no longer existed. The snapshot does
not describe a builder. It describes an initialisation, one transition, three
transitions, an evaluation rollout, and the eight intermediate quantities a
transition passes through, all bit-exact, on an environment small enough to
state in full and a seed that is written down.

So it is back, and this file answers it. A failure names the quantity that
differs by the most, which is where the reproduction went.
"""

from __future__ import annotations

import json
from pathlib import Path

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import TinyDiscreteEnv, assert_within, deviations, flattened

from memorax.algorithms.stream_ac_rtrl import StreamACRTRL, StreamACRTRLConfig
from memorax.networks import (
    RNN,
    FeatureExtractor,
    Network,
    RTUCell,
    RTUConfig,
    heads,
)

GOLDEN = Path(__file__).with_name("golden") / "stream_ac_rtu"

# Named by the snapshot, not chosen here: the quantities one transition passes
# through, and where the kernel reports each of them now.
OBSERVED = {
    "value": lambda metrics: metrics.value,
    "next_value": lambda metrics: metrics.next_value,
    "td": lambda metrics: metrics.td_error,
    "logprob": lambda metrics: metrics.log_prob,
    "sampled_action": lambda metrics: metrics.action_decision.sampled_action,
    "logprob_action": lambda metrics: metrics.action_decision.logprob_action,
    "env_action": lambda metrics: metrics.action_decision.env_action,
    "feedback_action": (
        lambda metrics: metrics.action_decision.bootstrap_feedback_action
    ),
}

SECTIONS = ("init", "one_step", "train", "evaluate")

# Where a section of a variant is allowed to have moved from the recording, in
# float32 last bits. Everywhere not named here: nowhere at all.
#
# One place is named, and this is what is known about it. The bounded step used
# to weight the trace by the TD error before the step size reached it, which
# rounds a bit away from the order the recording multiplies in; the rule now
# multiplies in the recorded order. That made the non-adaptive variant exact in
# every section and left exactly this much: after one transition of the adaptive
# variant, four bias vectors move -- the critic's two by two last bits, the
# actor's two by one -- while every weight matrix, the other two sections, and
# the whole non-adaptive variant are exact.
#
# It is not the bounded step. The rule that recorded the snapshot was lifted out
# of commit 5f7ff4e and asked directly, on seeded inputs in both branches, and it
# answers bit-for-bit what our rule answers. So something reaching that block
# differs by less than the non-adaptive path can resolve, and the adaptive path
# resolves it: its step size divides the trace norm by sqrt(v_hat) + 1e-6, so a
# second moment near zero turns a difference far below one bit into a bit of step
# size, and a shifted step size shifts every leaf by the same relative amount.
# That is visible on vectors whose elements are all one size and hidden on
# matrices with a large element to be measured against, which is why the biases
# are named and no kernel is.
ALLOWED = {("obgd/adaptive", "one_step"): 2.0}


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads(GOLDEN.with_suffix(".json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def arrays() -> dict:
    with np.load(GOLDEN.with_suffix(".npz")) as payload:
        return {name: payload[name] for name in payload.files}


def recorded(arrays: dict, variant: str, section: str) -> dict:
    prefix = f"{variant}/{section}/"
    return {
        name[len(prefix) :]: value
        for name, value in arrays.items()
        if name.startswith(prefix)
    }


def agent_for(manifest: dict, variant: str) -> StreamACRTRL:
    """Build exactly what the snapshot says was built."""

    network_spec = manifest["config"]["network"]
    torso = network_spec["torso"]
    features = int(torso["features"])
    env = TinyDiscreteEnv()

    def network(head):
        return Network(
            feature_extractor=FeatureExtractor(
                observation_extractor=nn.Sequential((nn.Dense(features), nn.tanh))
            ),
            torso=RNN(
                cell=RTUCell(
                    config=RTUConfig(
                        features=features, hidden_dim=int(torso["hidden_dim"])
                    )
                )
            ),
            head=head,
        )

    # The snapshot was recorded when the bounded rule was a boolean. Its two
    # values are the two rules that existed then, under the names they have now.
    recorded_config = dict(manifest["config"]["algorithm"]["variants"][variant])
    was_adaptive = bool(recorded_config.pop("adaptive"))
    recorded_config["bounded_rule"] = "adaptive_obgd" if was_adaptive else "obgd"

    return StreamACRTRL(
        StreamACRTRLConfig(**recorded_config),
        env,
        env.default_params,
        network(
            heads.Categorical(
                action_dim=int(network_spec["actor"]["head"]["action_dim"])
            )
        ),
        network(heads.VNetwork()),
    )


@pytest.fixture(scope="module")
def replayed(manifest: dict) -> dict:
    """Rerun the recorded protocol, once per variant, exactly as written."""

    steps = int(manifest["steps"])
    base_key = jax.random.key(int(manifest["seed"]))
    runs = {}
    for variant in manifest["snapshots"]:
        agent = agent_for(manifest, variant)
        initial = agent.init(jax.random.fold_in(base_key, 0))
        stepped, metrics = agent._step(initial, jax.random.fold_in(base_key, 1))
        trained, _ = agent.train(
            jax.random.fold_in(base_key, 2), initial, num_steps=steps
        )
        evaluated, _ = agent.evaluate(
            jax.random.fold_in(base_key, 3), trained, num_steps=steps
        )
        runs[variant] = {
            "init": initial,
            "one_step": stepped,
            "train": trained,
            "evaluate": evaluated,
            "observables": {
                **{name: read(metrics) for name, read in OBSERVED.items()},
                # The carry and sensitivity the acting forward produced are
                # what the kernel carries away, so the stepped state answers
                # for them.
                "actor_carry_after_action": stepped.actor_carry,
                "actor_sensitivity_after_action": stepped.actor_sensitivity,
            },
        }
    return runs


@pytest.mark.parametrize("section", SECTIONS)
def test_every_carried_leaf_is_what_was_recorded(manifest, arrays, replayed, section):
    for variant in manifest["snapshots"]:
        compared = assert_within(
            flattened(replayed[variant][section]),
            recorded(arrays, variant, section),
            f"{variant}/{section}",
            allowed=ALLOWED.get((variant, section), 0.0),
        )
        assert compared >= 60, f"{variant}/{section}: only {compared} leaves"


def test_the_quantities_one_transition_passes_through_are_unchanged(
    manifest, arrays, replayed
):
    """The transition itself, exactly.

    The action taken, its log probability, both value estimates, the TD error
    between them, and the carry and sensitivity the acting forward produced.
    A rewrite that changed what the algorithm computes rather than the order
    it multiplies in would differ here, and none of these is allowed to.
    """

    for variant in manifest["snapshots"]:
        assert_within(
            flattened(replayed[variant]["observables"]),
            recorded(arrays, variant, "observables"),
            f"{variant}/observables",
        )


def test_the_snapshot_is_the_one_this_file_was_written_for(manifest):
    assert manifest["algorithm"] == "stream_ac_rtu"
    assert manifest["config"]["algorithm"]["type"] == "StreamACRtrl"
    assert manifest["config"]["environment"]["type"] == "TinyDiscreteEnv"
    assert manifest["config"]["evaluation"]["policy"] == "deterministic_argmax"
    # Bit-exactness is a statement about a library as much as about a kernel,
    # and uv.lock still pins the one that produced these numbers.
    assert manifest["jax"] == jax.__version__, "the recording used another jax"


def test_a_changed_number_would_be_reported():
    """The comparison above is only worth running if it can fail."""

    expected = {"a": np.full((2, 3), 0.5, np.float32)}
    nudged = np.full((2, 3), 0.5, np.float32)
    nudged[1, 2] = np.nextafter(nudged[1, 2], np.float32(1), dtype=np.float32)

    # One last bit is a difference every place without an allowance refuses.
    with pytest.raises(AssertionError, match="more than 0 last bits"):
        assert_within({"a": nudged}, expected, "sanity")
    # And the one allowance there is holds to its size: a bit passes, five do not.
    assert not deviations({"a": nudged}, expected, allowed=4.0)
    for _ in range(4):
        nudged[1, 2] = np.nextafter(nudged[1, 2], np.float32(1), dtype=np.float32)
    assert deviations({"a": nudged}, expected, allowed=4.0)
    assert not deviations({"a": jnp.full((2, 3), 0.5)}, expected)
