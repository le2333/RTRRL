import json

import jax
import jaxlib
import numpy as np
import pytest
from golden import (
    GOLDEN_DIR,
    assert_tree_allclose,
    flatten_with_paths,
    load_golden,
    snapshot_rtrrl,
    snapshot_stream_ac,
)


def test_tiny_continuous_env_is_deterministic_and_terminates_on_third_step(
    tiny_continuous_env,
):
    env, params = tiny_continuous_env
    key = jax.random.key(0)
    obs, state = env.reset(key, params)
    np.testing.assert_array_equal(obs, np.array([0.25, -0.5], np.float32))

    expected = (
        ([0.6, -0.15], 0.75, False),
        ([0.95, 0.2], 1.1, False),
        ([1.3, 0.55], 1.45, True),
    )
    for expected_obs, expected_reward, expected_done in expected:
        obs, state, reward, done, info = env.step(
            key, state, np.array([0.2, -0.1], np.float32), params
        )
        np.testing.assert_allclose(obs, expected_obs, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(reward, expected_reward, rtol=1e-6, atol=1e-7)
        assert bool(done) is expected_done
        assert set(info) == {"step_count"}


def test_tiny_discrete_env_has_two_actions_and_terminates_on_third_step(
    tiny_discrete_env,
):
    env, params = tiny_discrete_env
    key = jax.random.key(0)
    obs, state = env.reset(key, params)
    assert env.action_space(params).n == 2
    np.testing.assert_array_equal(obs, np.array([-0.25, 0.5], np.float32))

    for index, action in enumerate((0, 1, 1), start=1):
        obs, state, reward, done, _ = env.step(key, state, action, params)
        assert int(state.step_count) == index
        assert bool(done) is (index == 3)


def test_flatten_paths_and_tree_assertion_are_leaf_exact():
    tree = {"b": (np.array([2.0]),), "a": {"x": np.array([1.0, 3.0])}}
    leaves, treedef = flatten_with_paths(tree)
    assert list(leaves) == ["a/x", "b/0"]
    assert treedef.num_leaves == 2
    assert_tree_allclose(tree, tree, rtol=0.0, atol=0.0)

    changed = {"b": (np.array([2.0]),), "a": {"x": np.array([1.0, 4.0])}}
    with pytest.raises(AssertionError, match="a/x"):
        assert_tree_allclose(tree, changed, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("fresh_trace", [False, True], ids=["incoming", "fresh"])
def test_rtrrl_lru_matches_versioned_legacy_oracle(rtrrl_agent_factory, fresh_trace):
    agent = rtrrl_agent_factory(fresh_trace=fresh_trace)
    actual = snapshot_rtrrl(agent, seed=7, steps=3)
    golden, manifest = load_golden("rtrrl_lru")
    assert manifest["algorithm"] == "rtrrl_lru"
    assert manifest["seed"] == 7
    assert manifest["steps"] == 3
    assert_tree_allclose(
        actual, golden[f"trace_timing/{'fresh' if fresh_trace else 'incoming'}"]
    )


@pytest.mark.parametrize("adaptive", [False, True], ids=["non_adaptive", "adaptive"])
def test_stream_ac_rtu_matches_versioned_legacy_oracle(
    stream_ac_agent_factory, adaptive
):
    agent = stream_ac_agent_factory(adaptive=adaptive)
    actual = snapshot_stream_ac(agent, seed=7, steps=3)
    golden, manifest = load_golden("stream_ac_rtu")
    assert manifest["algorithm"] == "stream_ac_rtu"
    assert manifest["seed"] == 7
    assert manifest["steps"] == 3
    assert_tree_allclose(
        actual, golden[f"obgd/{'adaptive' if adaptive else 'non_adaptive'}"]
    )


@pytest.mark.parametrize("name", ["rtrrl_lru", "stream_ac_rtu"])
def test_golden_manifest_matches_active_runtime(name):
    _, manifest = load_golden(name)
    assert manifest["jax"] == jax.__version__
    assert (
        manifest["jaxlib"]
        == jaxlib.__version__  # pyright: ignore[reportPrivateImportUsage]
    )
    assert manifest["backend"] == "cpu"
    assert manifest["leaf_paths"] == sorted(manifest["leaf_paths"])

    on_disk = json.loads((GOLDEN_DIR / f"{name}.json").read_text())
    assert on_disk == manifest


def test_rtrrl_manifest_records_complete_effective_config():
    manifest = json.loads((GOLDEN_DIR / "rtrrl_lru.json").read_text())
    config = manifest["config"]
    incoming = config["algorithm"]["variants"]["trace_timing/incoming"]
    fresh = config["algorithm"]["variants"]["trace_timing/fresh"]

    assert (
        set(incoming)
        == set(fresh)
        == {
            "num_envs",
            "gamma",
            "lambda_pi",
            "lambda_v",
            "lambda_rnn",
            "td_lr",
            "rnn_lr",
            "eta_pi",
            "eta_f",
            "entropy_rate",
            "update_period",
            "b1",
            "b2",
            "eps",
            "rnn_grad_clip",
            "act_clip",
            "freeze_gamma",
            "update_trace_before_td",
            "logprob_scale",
            "pred_obs",
            "pred_coeff",
        }
    )
    assert incoming["update_trace_before_td"] is False
    assert fresh["update_trace_before_td"] is True
    assert incoming["gamma"] == fresh["gamma"] == 0.91
    assert incoming["lambda_rnn"] == 0.61
    assert config["network"]["torso"] == {
        "type": "Memoroid",
        "cell": "LRUCell",
        "features": 9,
        "hidden_dim": 2,
        "output_dim": 3,
        "expose_hidden": False,
        "r_min": 0.0,
        "r_max": 1.0,
        "max_phase": 6.28,
        "dtype": None,
        "param_dtype": "float32",
    }
    assert config["action"]["shape"] == [2]
    assert config["evaluation"]["policy"] == "deterministic_mode"


def test_stream_ac_manifest_records_complete_effective_config():
    manifest = json.loads((GOLDEN_DIR / "stream_ac_rtu.json").read_text())
    config = manifest["config"]
    non_adaptive = config["algorithm"]["variants"]["obgd/non_adaptive"]
    adaptive = config["algorithm"]["variants"]["obgd/adaptive"]

    assert (
        set(non_adaptive)
        == set(adaptive)
        == {
            "num_envs",
            "gamma",
            "trace_lambda",
            "actor_lr",
            "critic_lr",
            "actor_kappa",
            "critic_kappa",
            "entropy_coefficient",
            "adaptive",
            "beta2",
            "eps",
        }
    )
    assert non_adaptive["adaptive"] is False
    assert adaptive["adaptive"] is True
    assert non_adaptive["gamma"] == adaptive["gamma"] == 0.89
    assert adaptive["beta2"] == 0.95
    assert config["network"]["actor"]["head"] == {
        "type": "Categorical",
        "action_dim": 2,
    }
    assert config["network"]["torso"]["cell"] == "RTUCell"
    assert config["action"]["count"] == 2
    assert config["evaluation"]["policy"] == "deterministic_argmax"


@pytest.mark.parametrize("name", ["rtrrl_lru", "stream_ac_rtu"])
def test_golden_payload_keys_match_manifest(name):
    manifest = json.loads((GOLDEN_DIR / f"{name}.json").read_text())
    with np.load(GOLDEN_DIR / f"{name}.npz", allow_pickle=False) as payload:
        assert sorted(payload.files) == manifest["leaf_paths"]
