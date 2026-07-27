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
from conftest import TinyDiscreteEnv

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


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads(GOLDEN.with_suffix(".json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def arrays() -> dict:
    with np.load(GOLDEN.with_suffix(".npz")) as payload:
        return {name: payload[name] for name in payload.files}


def flattened(tree) -> dict:
    """Leaf paths spelled the way the snapshot spells them."""

    pairs, _ = jax.tree_util.tree_flatten_with_path(tree)
    return {
        "/".join(str(getattr(key, "key", getattr(key, "idx", key))) for key in path): (
            np.asarray(leaf)
        )
        for path, leaf in pairs
    }


def recorded(arrays: dict, variant: str, section: str) -> dict:
    prefix = f"{variant}/{section}/"
    return {
        name[len(prefix) :]: value
        for name, value in arrays.items()
        if name.startswith(prefix)
    }


def deviations(actual: dict, expected: dict) -> list:
    """Every leaf that differs, worst first, with what it differed by."""

    missing = sorted(set(expected) - set(actual))
    assert not missing, f"the kernel no longer reports {missing}"

    found = []
    for path, wanted in expected.items():
        wanted = np.asarray(wanted)
        got = np.asarray(actual[path])
        assert got.shape == wanted.shape, f"{path}: {got.shape} not {wanted.shape}"
        if wanted.dtype.kind in "biufc" and got.dtype.kind in "biufc":
            gap = float(
                np.max(np.abs(got.astype(np.float64) - wanted.astype(np.float64)))
            )
            if gap:
                found.append((gap, path, wanted, got))
        elif not np.array_equal(got, wanted):
            found.append((float("inf"), path, wanted, got))
    return sorted(found, reverse=True, key=lambda entry: entry[0])


def assert_recorded(actual: dict, expected: dict, what: str) -> int:
    assert expected, f"{what}: the snapshot holds nothing to answer"
    found = deviations(actual, expected)
    if found:
        gap, path, wanted, got = found[0]
        raise AssertionError(
            f"{what}: {len(found)} of {len(expected)} leaves differ; worst is "
            f"{path} by {gap:g}, recorded {wanted.reshape(-1)[:4]} "
            f"against {got.reshape(-1)[:4]}"
        )
    return len(expected)


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

    return StreamACRTRL(
        StreamACRTRLConfig(**manifest["config"]["algorithm"]["variants"][variant]),
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
        compared = assert_recorded(
            flattened(replayed[variant][section]),
            recorded(arrays, variant, section),
            f"{variant}/{section}",
        )
        assert compared >= 60, f"{variant}/{section}: only {compared} leaves"


def test_the_quantities_one_transition_passes_through_are_unchanged(
    manifest, arrays, replayed
):
    """Where a carried leaf differs, this says which quantity it came from."""

    for variant in manifest["snapshots"]:
        assert_recorded(
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

    expected = {"a": np.zeros((2, 3), np.float32)}
    nudged = np.zeros((2, 3), np.float32)
    nudged[1, 2] = np.float32(1e-7)
    with pytest.raises(AssertionError, match="worst is a"):
        assert_recorded({"a": nudged}, expected, "sanity")
    assert not deviations({"a": jnp.zeros((2, 3))}, expected)
