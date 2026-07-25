import hashlib
import json

import pytest
from training_sdk.contract import Catalog

from trainer_infra.experiment import load_experiment
from trainer_infra.images import CATALOG_LABEL, encode_catalog
from trainer_infra.preflight import PreflightError, check_aws, check_offline
from tests.helpers import EXAMPLE
from tests.test_preflight_offline import CATALOG

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
        _require_registry_id(kwargs, method="describe_images", account_id=self.account_id)
        return {"imageDetails": [{"imageDigest": self.digest}]}

    def batch_get_image(self, **kwargs: object) -> dict:
        _require_registry_id(kwargs, method="batch_get_image", account_id=self.account_id)
        manifest = json.dumps({"config": {"digest": self.config_digest}})
        return {
            "images": [
                {
                    "imageId": {"imageDigest": self.digest},
                    "imageManifest": manifest,
                }
            ]
        }

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
    def __init__(self, queues=("run-cpu-c7am-queue",), definitions=(DEFINITION,)) -> None:
        self.queues, self.definitions = queues, definitions

    def describe_job_queues(self, **kwargs: object) -> dict:
        return {
            "jobQueues": [
                {"jobQueueName": name, "state": "ENABLED", "status": "VALID"}
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
    def head_bucket(self, **kwargs: object) -> dict:
        return {}


def plan_arguments():
    experiment = load_experiment(EXAMPLE)
    return experiment, CATALOG, check_offline(experiment, CATALOG)


def check(ecr=None, batch=None, connect=lambda host, port: None):
    experiment, catalog, space = plan_arguments()
    return check_aws(
        experiment,
        catalog,
        space,
        ecr_client=ecr or FakeEcr(),
        batch_client=batch or FakeBatch(),
        s3_client=FakeS3(),
        read_url=read_url,
        connect=connect,
    )


def test_plan_carries_digest_queue_and_job_definition() -> None:
    plan = check()
    assert plan.digest == DIGEST
    assert plan.queue == "run-cpu-c7am-queue"
    assert plan.job_definition == DEFINITION


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


def test_unreachable_aim_endpoint_is_rejected() -> None:
    def refuse(host: str, port: int) -> None:
        raise OSError("connection refused")

    with pytest.raises(PreflightError, match="aim"):
        check(connect=refuse)


def test_missing_queue_is_rejected() -> None:
    with pytest.raises(PreflightError, match="run-cpu-c7am-queue"):
        check(batch=FakeBatch(queues=()))


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
    with pytest.raises(PreflightError, match="contract"):
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
