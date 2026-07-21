from __future__ import annotations

import base64
import gzip
import json
from pathlib import Path
import subprocess
import sys

import pytest

from trainer_infra.image_catalog import decode_catalog, encode_catalog, load_catalog_index

REPOSITORY_ROOT = Path(__file__).parents[4]
MEMO_ROOT = REPOSITORY_ROOT / "memo"
CATALOG_INDEX = MEMO_ROOT / "infra" / "scripts" / "index.yaml"


def _catalog():
    return load_catalog_index(CATALOG_INDEX)


def _field(
    *,
    path: str,
    type_: str,
    default,
    searchable: bool = False,
    constraints: dict[str, float | None] | None = None,
    default_search: dict | None = None,
    choices: list | None = None,
) -> dict:
    return {
        "path": path,
        "type": type_,
        "default": default,
        "searchable": searchable,
        "constraints": constraints
        or {"gt": None, "ge": None, "lt": None, "le": None},
        "default_search": default_search,
        "choices": choices,
    }


def test_catalog_encoder_emits_nonempty_bounded_deterministic_codec_payload() -> None:
    catalog = _catalog()
    command = [sys.executable, "-m", "trainer_infra.image_catalog", str(CATALOG_INDEX)]
    first = subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()
    second = subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()
    compressed = base64.b64decode(first, validate=True)
    raw = gzip.decompress(compressed)

    assert first == second
    assert 100 < len(first) < 65_536
    assert json.loads(raw) == catalog.model_dump(mode="json", exclude_none=True)
    assert decode_catalog(first) == catalog
    assert catalog.protocol_version == "1"
    assert tuple(catalog.scripts) == ("memo_stream_ac", "memo_rtrrl")
    assert decode_catalog(encode_catalog(catalog)) == catalog


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

    assert {name: field.model_dump(mode="json") for name, field in fields.items()} == {
        "agent_type": _field(
            path="algorithm.agent_type",
            type_="str",
            default="rtu_rtrl",
            choices=["rtu_rtrl"],
        ),
        "seed": _field(
            path="runtime.seed",
            type_="int",
            default=0,
            constraints={"gt": None, "ge": 0, "lt": None, "le": None},
        ),
        "hidden_dim": _field(
            path="network.hidden_dim",
            type_="int",
            default=192,
            searchable=True,
            constraints={"gt": 0, "ge": None, "lt": None, "le": None},
            default_search={"values": [64, 128, 192]},
        ),
        "encoder_dim": _field(
            path="network.encoder_dim",
            type_="int",
            default=64,
            searchable=True,
            constraints={"gt": 0, "ge": None, "lt": None, "le": None},
            default_search={"values": [32, 64, 128]},
        ),
        "gamma": _field(
            path="algorithm.gamma",
            type_="float",
            default=0.99,
            searchable=True,
            constraints={"gt": 0, "ge": None, "lt": None, "le": 1},
            default_search={"min": 0.9, "max": 1.0, "scale": "linear"},
        ),
        "trace_lambda": _field(
            path="algorithm.trace_lambda",
            type_="float",
            default=0.9,
            searchable=True,
            constraints={"gt": None, "ge": 0, "lt": None, "le": 1},
            default_search={"min": 0.7, "max": 1.0, "scale": "linear"},
        ),
        "actor_lr": _field(
            path="algorithm.actor_lr",
            type_="float",
            default=1.0,
            searchable=True,
            constraints={"gt": 0, "ge": None, "lt": None, "le": None},
            default_search={"min": 0.01, "max": 2.0, "scale": "log"},
        ),
        "critic_lr": _field(
            path="algorithm.critic_lr",
            type_="float",
            default=1.0,
            searchable=True,
            constraints={"gt": 0, "ge": None, "lt": None, "le": None},
            default_search={"min": 0.01, "max": 2.0, "scale": "log"},
        ),
        "entropy_coefficient": _field(
            path="algorithm.entropy_coefficient",
            type_="float",
            default=0.01,
            searchable=True,
            constraints={"gt": None, "ge": 0, "lt": None, "le": None},
            default_search={"min": 0.00001, "max": 0.1, "scale": "log"},
        ),
        "num_envs": _field(
            path="runtime.num_envs",
            type_="int",
            default=16,
            constraints={"gt": 0, "ge": None, "lt": None, "le": None},
        ),
    }


def test_rtrrl_descriptor_fields_are_exact_and_target_real_config_namespaces() -> None:
    fields = _catalog().scripts["memo_rtrrl"].fields

    assert {name: field.model_dump(mode="json") for name, field in fields.items()} == {
        "rtrrl_topology": _field(
            path="algorithm.rtrrl_topology",
            type_="str",
            default="shared",
            choices=["shared"],
        ),
        "seed": _field(
            path="runtime.seed",
            type_="int",
            default=0,
            constraints={"gt": None, "ge": 0, "lt": None, "le": None},
        ),
        "backbone": _field(
            path="network.backbone",
            type_="str",
            default="lru",
            choices=["lru", "rtu"],
        ),
        "hidden_dim": _field(
            path="network.hidden_dim",
            type_="int",
            default=32,
            searchable=True,
            constraints={"gt": 0, "ge": None, "lt": None, "le": None},
            default_search={"values": [16, 32, 64]},
        ),
        "gamma": _field(
            path="algorithm.gamma",
            type_="float",
            default=0.95,
            searchable=True,
            constraints={"gt": 0, "ge": None, "lt": None, "le": 1},
            default_search={"min": 0.9, "max": 1.0, "scale": "linear"},
        ),
        "lambda_pi": _field(
            path="algorithm.lambda_pi",
            type_="float",
            default=0.97,
            searchable=True,
            constraints={"gt": None, "ge": 0, "lt": None, "le": 1},
            default_search={"min": 0.8, "max": 1.0, "scale": "linear"},
        ),
        "lambda_v": _field(
            path="algorithm.lambda_v",
            type_="float",
            default=0.9,
            searchable=True,
            constraints={"gt": None, "ge": 0, "lt": None, "le": 1},
            default_search={"min": 0.8, "max": 1.0, "scale": "linear"},
        ),
        "lambda_rnn": _field(
            path="algorithm.lambda_rnn",
            type_="float",
            default=0.945,
            searchable=True,
            constraints={"gt": None, "ge": 0, "lt": None, "le": 1},
            default_search={"min": 0.8, "max": 1.0, "scale": "linear"},
        ),
        "td_lr": _field(
            path="algorithm.td_lr",
            type_="float",
            default=0.00003,
            searchable=True,
            constraints={"gt": 0, "ge": None, "lt": None, "le": None},
            default_search={"min": 0.000001, "max": 0.001, "scale": "log"},
        ),
        "rnn_lr": _field(
            path="algorithm.rnn_lr",
            type_="float",
            default=0.000002,
            searchable=True,
            constraints={"gt": 0, "ge": None, "lt": None, "le": None},
            default_search={"min": 0.0000001, "max": 0.0001, "scale": "log"},
        ),
        "eta_pi": _field(
            path="algorithm.eta_pi",
            type_="float",
            default=0.38,
            searchable=True,
            constraints={"gt": 0, "ge": None, "lt": None, "le": None},
            default_search={"min": 0.05, "max": 1.0, "scale": "log"},
        ),
        "eta_f": _field(
            path="algorithm.eta_f",
            type_="float",
            default=0.5,
            searchable=True,
            constraints={"gt": 0, "ge": None, "lt": None, "le": None},
            default_search={"min": 0.05, "max": 1.0, "scale": "log"},
        ),
        "entropy_rate": _field(
            path="algorithm.entropy_rate",
            type_="float",
            default=0.00003,
            searchable=True,
            constraints={"gt": None, "ge": 0, "lt": None, "le": None},
            default_search={"min": 0.000001, "max": 0.001, "scale": "log"},
        ),
    }


