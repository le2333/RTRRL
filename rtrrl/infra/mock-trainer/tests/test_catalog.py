from pathlib import Path

from trainer_infra.image_catalog import (
    decode_catalog,
    encode_catalog,
    load_catalog_index,
)

SCRIPTS = Path(__file__).parents[1] / "scripts"


def test_acceptance_catalog_round_trips_through_image_label_codec() -> None:
    catalog = load_catalog_index(SCRIPTS / "index.yaml")

    decoded = decode_catalog(encode_catalog(catalog))

    assert decoded == catalog
    assert decoded.protocol_version == "1"
    assert set(decoded.scripts) == {"brax_ppo_acceptance"}


def test_brax_ppo_descriptor_matches_the_runtime_contract() -> None:
    descriptor = load_catalog_index(SCRIPTS / "index.yaml").scripts["brax_ppo_acceptance"]

    assert descriptor.argv == (
        "python",
        "-m",
        "brax_ppo_acceptance",
        "--config",
        "{config_path}",
    )
    assert descriptor.sdk_protocol_version == "1"
    assert descriptor.objective.metric == "eval/episode_return"
    assert descriptor.objective.direction == "maximize"
    assert descriptor.environments == ("inverted_pendulum",)
    assert descriptor.fields["failure_mode"].choices == ("none",)
