import hashlib
import json
from pathlib import Path

import pytest
from botocore.exceptions import ClientError
from training_sdk.contract import Catalog

from trainer_infra.experiment import load_experiment
from trainer_infra.images import CATALOG_LABEL, encode_catalog, resolve_image
from trainer_infra.preflight import PreflightError, check_aws, check_offline
from tests.helpers import EXAMPLE
from tests.test_preflight_offline import CATALOG

ECR_BATCH_GET_IMAGE_FIXTURE = (
    Path(__file__).parent / "data" / "ecr-batch-get-image.json"
)

DIGEST = "sha256:1111111111111111111111111111111111111111111111111111111111111111"
ACCOUNT_ID = "007122174918"


def _require_registry_id(kwargs: object, *, method: str, account_id: str) -> None:
    assert isinstance(kwargs, dict)
    registry_id = kwargs.get("registryId")
    assert registry_id == account_id, (
        f"{method} requires registryId={account_id!r}, got {registry_id!r}"
    )


def _config_blob(catalog: Catalog = CATALOG) -> bytes:
    return json.dumps(
        {"config": {"Labels": {CATALOG_LABEL: encode_catalog(catalog)}}}
    ).encode()


CONFIG_BLOB = _config_blob()
CONFIG_DIGEST = "sha256:" + hashlib.sha256(CONFIG_BLOB).hexdigest()


class FakeEcr:
    def __init__(
        self,
        digest: str = DIGEST,
        *,
        config_blob: bytes = CONFIG_BLOB,
        account_id: str = ACCOUNT_ID,
    ) -> None:
        self.digest = digest
        self.config_blob = config_blob
        self.config_digest = "sha256:" + hashlib.sha256(config_blob).hexdigest()
        self.account_id = account_id

    def describe_images(self, **kwargs: object) -> dict:
        """Refuse, exactly as the control plane's instance role does.

        That role is granted BatchGetImage and GetDownloadUrlForLayer but not
        DescribeImages, so calling this in production fails with AccessDenied
        after preflight has already been reported as passing.
        """
        del kwargs
        raise ClientError(
            {
                "Error": {
                    "Code": "AccessDeniedException",
                    "Message": "not authorized to perform: ecr:DescribeImages",
                }
            },
            "DescribeImages",
        )

    def batch_get_image(self, **kwargs: object) -> dict:
        _require_registry_id(kwargs, method="batch_get_image", account_id=self.account_id)
        fixture = json.loads(ECR_BATCH_GET_IMAGE_FIXTURE.read_text(encoding="utf-8"))
        response = json.loads(json.dumps(fixture))
        manifest = json.loads(response["images"][0]["imageManifest"])
        manifest["config"]["digest"] = self.config_digest
        response["images"][0]["imageId"]["imageDigest"] = self.digest
        response["images"][0]["imageManifest"] = json.dumps(manifest)
        return response

    def get_download_url_for_layer(self, **kwargs: object) -> dict:
        _require_registry_id(
            kwargs, method="get_download_url_for_layer", account_id=self.account_id
        )
        return {"downloadUrl": "https://example.invalid/config"}


def read_url(url: str) -> bytes:
    assert url == "https://example.invalid/config"
    return CONFIG_BLOB


DEFINITION = f"trainer-c7am-{DIGEST.removeprefix('sha256:')}"


class FakeBatch:
    def __init__(
        self,
        queues=("run-cpu-c7am-queue",),
        definitions=(DEFINITION,),
        *,
        queue_state: str = "ENABLED",
        queue_status: str = "VALID",
    ) -> None:
        self.queues, self.definitions = queues, definitions
        self.queue_state = queue_state
        self.queue_status = queue_status

    def describe_job_queues(self, **kwargs: object) -> dict:
        return {
            "jobQueues": [
                {
                    "jobQueueName": name,
                    "state": self.queue_state,
                    "status": self.queue_status,
                }
                for name in self.queues
            ]
        }

    def describe_job_definitions(self, jobDefinitionName: str, **kwargs: object) -> dict:
        if jobDefinitionName not in self.definitions:
            return {"jobDefinitions": []}
        return {
            "jobDefinitions": [
                {
                    "jobDefinitionName": jobDefinitionName,
                    "revision": 1,
                    "status": "ACTIVE",
                }
            ]
        }


class FakeS3:
    def __init__(self, *, head_bucket_error: ClientError | None = None) -> None:
        self.head_bucket_error = head_bucket_error

    def head_bucket(self, **kwargs: object) -> dict:
        if self.head_bucket_error is not None:
            raise self.head_bucket_error
        return {}


def plan_arguments():
    experiment = load_experiment(EXAMPLE)
    return experiment, CATALOG, check_offline(experiment, CATALOG)


def check(ecr=None, batch=None, s3=None, connect=lambda host, port: None):
    experiment, catalog, space = plan_arguments()
    return check_aws(
        experiment,
        catalog,
        space,
        ecr_client=ecr or FakeEcr(),
        batch_client=batch or FakeBatch(),
        s3_client=s3 or FakeS3(),
        read_url=read_url,
        connect=connect,
    )


def test_plan_carries_digest_queue_and_job_definition() -> None:
    plan = check()
    assert plan.digest == DIGEST
    assert plan.queue == "run-cpu-c7am-queue"
    assert plan.job_definition == DEFINITION


def test_a_tagged_image_resolves_to_its_digest() -> None:
    """The example experiment names a tag, so this is the production path.

    It must resolve using BatchGetImage alone. The fake's DescribeImages refuses
    the way the real instance role does, so reaching for it fails here instead of
    after preflight has told the operator everything was fine.
    """
    resolved = resolve_image(
        "007122174918.dkr.ecr.eu-north-1.amazonaws.com/rtrrl:some-tag",
        FakeEcr(),
        read_url,
    )

    assert resolved.digest == DIGEST
    assert resolved.reference == (
        f"007122174918.dkr.ecr.eu-north-1.amazonaws.com/rtrrl@{DIGEST}"
    )
    assert resolved.repository == "007122174918.dkr.ecr.eu-north-1.amazonaws.com/rtrrl"


def test_a_digest_image_resolves_to_the_same_reference() -> None:
    reference = f"007122174918.dkr.ecr.eu-north-1.amazonaws.com/rtrrl@{DIGEST}"

    resolved = resolve_image(reference, FakeEcr(), read_url)

    assert resolved.reference == reference
    assert resolved.digest == DIGEST


def test_dev_tier_selects_the_dev_queue() -> None:
    experiment, catalog, space = plan_arguments()
    plan = check_aws(
        experiment,
        catalog,
        space,
        ecr_client=FakeEcr(),
        batch_client=FakeBatch(queues=("dev-cpu-c7am-queue",)),
        s3_client=FakeS3(),
        read_url=read_url,
        connect=lambda host, port: None,
        tier="dev",
    )
    assert plan.queue == "dev-cpu-c7am-queue"


@pytest.mark.parametrize("host", ["127.0.0.1", "127.0.1.5", "localhost", "::1"])
def test_loopback_aim_endpoint_is_rejected(host: str) -> None:
    """Connecting from here proves nothing about what the job can reach.

    The endpoint is copied verbatim into every run config, so a loopback address
    sends the job to itself while preflight, running on the control plane, connects
    happily to the real Aim server. Reading the address is the only way to catch it.
    """
    experiment, catalog, space = plan_arguments()
    experiment = experiment.model_copy(
        update={"logging": experiment.logging.model_copy(update={"aim": f"aim://{host}:53801"})}
    )

    with pytest.raises(PreflightError, match="loopback"):
        check_aws(
            experiment,
            catalog,
            space,
            ecr_client=FakeEcr(),
            batch_client=FakeBatch(),
            s3_client=FakeS3(),
            read_url=read_url,
            connect=lambda host, port: None,
        )


def test_a_routable_aim_endpoint_is_accepted() -> None:
    experiment, catalog, space = plan_arguments()
    experiment = experiment.model_copy(
        update={
            "logging": experiment.logging.model_copy(update={"aim": "aim://10.1.2.3:53801"})
        }
    )

    plan = check_aws(
        experiment,
        catalog,
        space,
        ecr_client=FakeEcr(),
        batch_client=FakeBatch(),
        s3_client=FakeS3(),
        read_url=read_url,
        connect=lambda host, port: None,
    )

    assert plan.experiment.logging.aim == "aim://10.1.2.3:53801"


def test_unreachable_aim_endpoint_is_rejected() -> None:
    def refuse(host: str, port: int) -> None:
        raise OSError("connection refused")

    with pytest.raises(PreflightError, match="aim"):
        check(connect=refuse)


def test_missing_queue_is_rejected() -> None:
    with pytest.raises(PreflightError, match="run-cpu-c7am-queue"):
        check(batch=FakeBatch(queues=()))


def test_disabled_queue_is_rejected() -> None:
    with pytest.raises(
        PreflightError,
        match=r"queue 'run-cpu-c7am-queue' is not ready \(state='DISABLED'",
    ):
        check(batch=FakeBatch(queue_state="DISABLED"))


def test_invalid_queue_is_rejected() -> None:
    with pytest.raises(
        PreflightError,
        match=r"queue 'run-cpu-c7am-queue' is not ready .*status='INVALID'",
    ):
        check(batch=FakeBatch(queue_status="INVALID"))


def test_missing_s3_bucket_is_rejected() -> None:
    error = ClientError(
        {"Error": {"Code": "404", "Message": "Not Found"}},
        "HeadBucket",
    )
    with pytest.raises(PreflightError, match=r"S3 bucket 'rtrrl-training-data' is not reachable"):
        check(s3=FakeS3(head_bucket_error=error))


def test_forbidden_s3_bucket_is_rejected() -> None:
    error = ClientError(
        {"Error": {"Code": "403", "Message": "Forbidden"}},
        "HeadBucket",
    )
    with pytest.raises(PreflightError, match=r"S3 bucket 'rtrrl-training-data' is not reachable"):
        check(s3=FakeS3(head_bucket_error=error))


def test_image_without_a_registered_job_definition_is_rejected() -> None:
    other = "sha256:" + "2" * 64
    with pytest.raises(PreflightError, match=f"trainer-c7am-{'2' * 64}"):
        check(ecr=FakeEcr(digest=other))


def test_image_catalog_disagreeing_with_offline_catalog_is_rejected() -> None:
    wrong = Catalog.model_validate(CATALOG.model_dump() | {"contract": 99})
    wrong_blob = _config_blob(wrong)

    def read_wrong(url: str) -> bytes:
        assert url == "https://example.invalid/config"
        return wrong_blob

    experiment, catalog, space = plan_arguments()
    with pytest.raises(PreflightError, match=r"contract differs \(image 99, offline 2\)"):
        check_aws(
            experiment,
            catalog,
            space,
            ecr_client=FakeEcr(config_blob=wrong_blob),
            batch_client=FakeBatch(),
            s3_client=FakeS3(),
            read_url=read_wrong,
            connect=lambda host, port: None,
        )


def _image_catalog_check(drifted: Catalog) -> None:
    drifted_blob = _config_blob(drifted)

    def read_drifted(url: str) -> bytes:
        assert url == "https://example.invalid/config"
        return drifted_blob

    experiment, catalog, space = plan_arguments()
    with pytest.raises(PreflightError) as error:
        check_aws(
            experiment,
            catalog,
            space,
            ecr_client=FakeEcr(config_blob=drifted_blob),
            batch_client=FakeBatch(),
            s3_client=FakeS3(),
            read_url=read_drifted,
            connect=lambda host, port: None,
        )
    return error.value


def test_image_source_hash_drift_is_rejected() -> None:
    entry = CATALOG.entries["brax_ppo_acceptance"].model_dump() | {
        "source_hash": "sha256:deadbeef"
    }
    drifted = Catalog.model_validate(
        CATALOG.model_dump() | {"entries": {"brax_ppo_acceptance": entry}}
    )
    error = _image_catalog_check(drifted)
    message = str(error)
    assert "brax_ppo_acceptance" in message
    assert "source_hash" in message
    assert "sha256:deadbeef" in message
    assert CATALOG.entries["brax_ppo_acceptance"].source_hash in message


def test_image_parameter_space_drift_is_rejected() -> None:
    space = dict(CATALOG.entries["brax_ppo_acceptance"].space)
    space["total_steps"] = {"type": "int", "low": 1, "high": 50000}
    entry = CATALOG.entries["brax_ppo_acceptance"].model_dump() | {"space": space}
    drifted = Catalog.model_validate(
        CATALOG.model_dump() | {"entries": {"brax_ppo_acceptance": entry}}
    )
    error = _image_catalog_check(drifted)
    message = str(error)
    assert "brax_ppo_acceptance" in message
    assert "space.total_steps" in message


def test_non_ecr_image_reference_is_rejected() -> None:
    experiment, catalog, space = plan_arguments()
    bad_reference = "registry.example/repo:tag"
    experiment = experiment.model_copy(update={"image": bad_reference})
    with pytest.raises(PreflightError, match=bad_reference):
        check_aws(
            experiment,
            catalog,
            space,
            ecr_client=FakeEcr(),
            batch_client=FakeBatch(),
            s3_client=FakeS3(),
            read_url=read_url,
            connect=lambda host, port: None,
        )
