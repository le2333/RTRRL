from dataclasses import replace

import jax
import numpy as np
import pytest
from golden import assert_tree_allclose

from memorax.algorithms.rtrrl import RTRRL
from memorax.algorithms.rtrrl.compatibility import normalize_legacy_config
from memorax.algorithms.rtrrl.components import (
    RecurrentComponent,
    select_memorax_components,
)
from memorax.networks import Memoroid, RNN, heads
from memorax.networks.sequence_models.lru import LRUCell
from memorax.networks.sequence_models.rtu import RTUCell


def _tree_at(tree, index):
    return jax.tree.map(lambda leaf: leaf[index], tree)


@pytest.mark.parametrize(
    ("overrides", "recurrent_type", "cell_type", "expected"),
    [
        (
            {},
            Memoroid,
            LRUCell,
            {
                "recurrent_component": "memorax_memoroid_lru",
                "feature_component": "encoder",
                "actor_component": "unbounded_gaussian",
                "action_clipping": "none",
                "input_gain": "trainable",
                "prediction_component": "none",
                "trace_timing": "fresh",
                "logprob_reduction": "sum",
                "topology": "shared",
                "normalize_observation": False,
                "normalize_reward": False,
            },
        ),
        (
            {
                "backbone": "rtu",
                "use_encoder": False,
                "bound_actor": True,
                "act_clip": 0.5,
                "freeze_gamma": True,
                "pred_obs": True,
                "update_trace_before_td": False,
                "logprob_reduction": "mean",
                "normalize_obs": True,
                "normalize_reward": True,
            },
            RNN,
            RTUCell,
            {
                "recurrent_component": "memorax_rnn_rtu",
                "feature_component": "raw",
                "actor_component": "bounded_gaussian",
                "action_clipping": "env",
                "input_gain": "frozen",
                "prediction_component": "observation_reward",
                "trace_timing": "incoming",
                "logprob_reduction": "mean",
                "topology": "shared",
                "normalize_observation": True,
                "normalize_reward": True,
            },
        ),
    ],
)
def test_memorax_component_selection_records_every_retained_shared_branch(
    monkeypatch, overrides, recurrent_type, cell_type, expected
):
    legacy = normalize_legacy_config(
        {"profile": "memo_experimental", **overrides}
    )
    monkeypatch.setattr(
        jax,
        "jit",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("component selection reached JIT")
        ),
    )

    selected = select_memorax_components(
        legacy,
        observation_dim=3,
        action_dim=2,
        topology="shared",
    )

    assert isinstance(selected.recurrent, recurrent_type)
    assert isinstance(selected.recurrent.cell, cell_type)
    assert isinstance(selected.recurrent_adapter, RecurrentComponent)
    assert {
        key: getattr(selected.effective_config, key) for key in expected
    } == expected
    assert (selected.prediction_head is not None) == bool(
        overrides.get("pred_obs")
    )


def test_experimental_facade_exposes_the_one_composed_program(
    rtrrl_agent_factory,
):
    agent = rtrrl_agent_factory(fresh_trace=False)

    assert isinstance(agent, RTRRL)
    assert agent.profile == "memo_experimental"
    assert agent.as_legacy_program() is agent
    assert agent.program is agent._delegate.program
    assert agent.program_config.static_config is agent.cfg


@pytest.mark.parametrize("fresh_trace", [False, True], ids=["incoming", "fresh"])
def test_composed_program_selects_trace_timing_before_scan(
    rtrrl_agent_factory, fresh_trace
):
    agent = rtrrl_agent_factory(fresh_trace=fresh_trace)
    state = agent.program.init_fn(jax.random.key(7))

    final_state, metrics = agent.program.train_epoch_fn(
        jax.random.key(11), state, 1
    )

    selected = (
        _tree_at(metrics.carried_traces, 0)
        if fresh_trace
        else _tree_at(metrics.incoming_traces, 0)
    )
    assert_tree_allclose(_tree_at(metrics.update_traces, 0), selected)
    assert jax.tree.structure(final_state) == jax.tree.structure(state)


def test_composed_program_clips_only_environment_action(rtrrl_agent_factory):
    base = rtrrl_agent_factory(fresh_trace=False)
    agent = replace(base, cfg=replace(base.cfg, act_clip=0.05))
    state = agent.program.init_fn(jax.random.key(13))

    _, metrics = agent.program.train_epoch_fn(jax.random.key(17), state, 1)

    sampled = metrics.action_decision.sampled_action[0]
    env_action = metrics.action_decision.env_action[0]
    assert np.max(np.abs(env_action)) <= 0.05
    assert not np.allclose(sampled, env_action)
    assert_tree_allclose(metrics.action_decision.logprob_action[0], sampled)


def test_composed_program_routes_prediction_head_direct_gradient(
    rtrrl_agent_factory,
):
    base = rtrrl_agent_factory(fresh_trace=False)
    agent = replace(
        base,
        cfg=replace(base.cfg, pred_obs=True, pred_coeff=0.7),
        pred_head=heads.Regressor(out_dim=3),
    )
    state = agent.program.init_fn(jax.random.key(19))

    _, metrics = agent.program.train_epoch_fn(jax.random.key(23), state, 1)

    prediction = metrics.prediction_direct_grads
    assert any(
        np.any(np.asarray(leaf) != 0) for leaf in jax.tree.leaves(prediction)
    )
