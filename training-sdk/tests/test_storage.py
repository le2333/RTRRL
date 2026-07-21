import pytest

from training_sdk.storage import ExperimentS3Namespace, parse_s3_uri


def test_experiment_namespace_builds_and_validates_only_exact_keys() -> None:
    namespace = ExperimentS3Namespace.from_prefix(
        "s3://bucket-name/experiments/exp-1/"
    )
    assert namespace.uri("jobs/job-1/bundle.json") == (
        "s3://bucket-name/experiments/exp-1/jobs/job-1/bundle.json"
    )
    assert namespace.require_uri(
        "s3://bucket-name/experiments/exp-1/jobs/job-1/bundle.json"
    ).key.endswith("bundle.json")


@pytest.mark.parametrize(
    "uri",
    [
        "s3://bucket-name/experiments/exp-1/../other",
        "s3://bucket-name/experiments/exp-1//object",
        "s3://bucket-name/experiments/exp-1/object?version=x",
        "s3://bucket-name/experiments/exp-1/object#fragment",
        "s3://user@bucket-name/experiments/exp-1/object",
        "s3://bucket-name/experiments/exp-1/%2e%2e/object",
        "s3://bucket-name/experiments/exp-1\\object",
    ],
)
def test_strict_parser_rejects_ambiguous_or_escaping_uris(uri: str) -> None:
    with pytest.raises(ValueError, match="S3 URI|experiment"):
        parse_s3_uri(uri)
    with pytest.raises(ValueError, match="S3 URI|experiment"):
        ExperimentS3Namespace.from_prefix(uri)


def test_experiment_prefix_must_start_at_key_root() -> None:
    with pytest.raises(ValueError, match="experiment"):
        ExperimentS3Namespace.from_prefix(
            "s3://bucket-name/other/experiments/exp-1/"
        )
