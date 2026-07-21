from __future__ import annotations

from pathlib import Path

import pytest

from trainer_infra.image_catalog import decode_catalog, encode_catalog, load_catalog_index

REPOSITORY_ROOT = Path(__file__).parents[4]
MEMO_ROOT = REPOSITORY_ROOT / "memo"
CATALOG_INDEX = MEMO_ROOT / "infra" / "scripts" / "index.yaml"


def _catalog():
    return load_catalog_index(CATALOG_INDEX)


def test_memo_catalog_protocol_v1_has_exactly_two_launchers() -> None:
    catalog = _catalog()

    assert catalog.protocol_version == "1"
    assert tuple(catalog.scripts) == ("memo_stream_ac", "memo_rtrrl")
    assert decode_catalog(encode_catalog(catalog)) == catalog
    assert encode_catalog(catalog)


@pytest.mark.parametrize(
    ("name", "argv", "environments", "fixed_field", "fixed_choice"),
    [
        (
            "memo_stream_ac",
            (
                "python",
                "/app/experiments/memo_stream_ac/run.py",
                "--config",
                "{config_path}",
            ),
            ("memory_chain", "kmemory_chain", "mujoco_masked"),
            "agent_type",
            ("rtu_rtrl",),
        ),
        (
            "memo_rtrrl",
            (
                "python",
                "/app/experiments/memo_rtrrl/run.py",
                "--config",
                "{config_path}",
            ),
            ("hopper",),
            "rtrrl_topology",
            ("shared",),
        ),
    ],
)
def test_memo_descriptors_have_exact_launcher_scope_and_real_objective(
    name: str,
    argv: tuple[str, ...],
    environments: tuple[str, ...],
    fixed_field: str,
    fixed_choice: tuple[str, ...],
) -> None:
    descriptor = _catalog().scripts[name]

    assert descriptor.argv == argv
    assert descriptor.sdk_protocol_version == "1"
    assert descriptor.environments == environments
    assert descriptor.fields[fixed_field].choices == fixed_choice
    assert descriptor.objective.model_dump() == {
        "metric": "eval/rewards",
        "direction": "maximize",
        "reduction": "last",
    }


def test_stream_descriptor_fields_are_exact_and_target_real_config_namespaces() -> None:
    fields = _catalog().scripts["memo_stream_ac"].fields

    assert {
        name: (field.path, field.default, field.choices)
        for name, field in fields.items()
    } == {
        "agent_type": ("algorithm.agent_type", "rtu_rtrl", ("rtu_rtrl",)),
        "seed": ("runtime.seed", 0, None),
        "hidden_dim": ("network.hidden_dim", 192, None),
        "encoder_dim": ("network.encoder_dim", 64, None),
        "gamma": ("algorithm.gamma", 0.99, None),
        "trace_lambda": ("algorithm.trace_lambda", 0.9, None),
        "actor_lr": ("algorithm.actor_lr", 1.0, None),
        "critic_lr": ("algorithm.critic_lr", 1.0, None),
        "entropy_coefficient": ("algorithm.entropy_coefficient", 0.01, None),
        "num_envs": ("runtime.num_envs", 16, None),
    }


def test_rtrrl_descriptor_fields_are_exact_and_target_real_config_namespaces() -> None:
    fields = _catalog().scripts["memo_rtrrl"].fields

    assert {
        name: (field.path, field.default, field.choices)
        for name, field in fields.items()
    } == {
        "rtrrl_topology": ("algorithm.rtrrl_topology", "shared", ("shared",)),
        "seed": ("runtime.seed", 0, None),
        "backbone": ("network.backbone", "lru", ("lru", "rtu")),
        "hidden_dim": ("network.hidden_dim", 32, None),
        "gamma": ("algorithm.gamma", 0.95, None),
        "lambda_pi": ("algorithm.lambda_pi", 0.97, None),
        "lambda_v": ("algorithm.lambda_v", 0.9, None),
        "lambda_rnn": ("algorithm.lambda_rnn", 0.945, None),
        "td_lr": ("algorithm.td_lr", 0.00003, None),
        "rnn_lr": ("algorithm.rnn_lr", 0.000002, None),
        "eta_pi": ("algorithm.eta_pi", 0.38, None),
        "eta_f": ("algorithm.eta_f", 0.5, None),
        "entropy_rate": ("algorithm.entropy_rate", 0.00003, None),
    }


