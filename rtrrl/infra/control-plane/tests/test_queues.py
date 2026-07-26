import pytest

from trainer_infra.preflight import PreflightError
from trainer_infra.queues import QUEUES, binding, job_definition_name

DIGEST = "sha256:" + "1" * 64


def test_every_instance_type_maps_to_both_tiers() -> None:
    assert set(QUEUES) == {"c7a.medium", "c7a.large", "c7a.xlarge", "g6.xlarge"}
    for instance_type, entry in QUEUES.items():
        assert entry.queue("run").startswith("run-"), instance_type
        assert entry.queue("dev").startswith("dev-"), instance_type


def test_unknown_queue_tier_is_rejected() -> None:
    with pytest.raises(PreflightError, match="tier"):
        binding("c7a.medium").queue("prod")


def test_concurrency_follows_compute_environment_capacity() -> None:
    assert binding("g6.xlarge").concurrency == 8
    assert binding("c7a.medium").concurrency == 16
    assert binding("c7a.xlarge").concurrency == 4


def test_job_definition_name_embeds_the_digest() -> None:
    assert job_definition_name(binding("c7a.medium"), DIGEST) == f"trainer-c7am-{'1' * 64}"


def test_unknown_instance_type_is_rejected_with_the_available_list() -> None:
    with pytest.raises(PreflightError, match="c7a.medium"):
        binding("p5.48xlarge")
