from copy import deepcopy
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from trainer_infra.batch_topology import (
    ACCOUNT_ID,
    REGION,
    AwsNetworkSettings,
    BatchTopologyValidator,
    ExecutionPurpose,
    ProfileDriftError,
    expected_topology,
    queue_for,
)


NETWORK = AwsNetworkSettings(
    subnets=(
        "subnet-08127d1c5d4de6ac2",
        "subnet-0b8c68ea0a9784758",
        "subnet-01a2aa195678f8411",
    ),
    security_group_ids=("sg-0c0ed6b927c5113dc",),
    instance_role=(
        "arn:aws:iam::007122174918:instance-profile/rtrrl-ecs-instance-role"
    ),
)


class FakeSts:
    def __init__(self, account: str = ACCOUNT_ID) -> None:
        self.account = account
        self.calls = 0

    def get_caller_identity(self) -> dict[str, str]:
        self.calls += 1
        return {"Account": self.account}


class FakeBatch:
    def __init__(self, region: str = REGION) -> None:
        topology = expected_topology()
        self.meta = SimpleNamespace(region_name=region)
        self.compute_environments = {
            key: {
                "computeEnvironmentName": spec.name,
                "computeEnvironmentArn": (
                    f"arn:aws:batch:{REGION}:{ACCOUNT_ID}:"
                    f"compute-environment/{spec.name}"
                ),
                "type": "MANAGED",
                "state": "ENABLED",
                "status": "VALID",
                "computeResources": {
                    "type": "EC2",
                    "minvCpus": 0,
                    "maxvCpus": spec.max_vcpus,
                    "desiredvCpus": spec.max_vcpus // 2,
                    "instanceTypes": [spec.instance_type],
                    "subnets": list(reversed(NETWORK.subnets)),
                    "securityGroupIds": list(NETWORK.security_group_ids),
                    "instanceRole": NETWORK.instance_role,
                    "ec2Configuration": [{"imageType": spec.ami_family}],
                },
                "updateStatus": "UPDATE_COMPLETE",
            }
            for key, spec in topology.compute_environments.items()
        }
        self.queues = {}
        for key, spec in topology.queues.items():
            environment_key = spec.compute_environments[0]
            self.queues[key] = {
                "jobQueueName": spec.name,
                "jobQueueArn": (
                    f"arn:aws:batch:{REGION}:{ACCOUNT_ID}:job-queue/{spec.name}"
                ),
                "state": "ENABLED",
                "status": "VALID",
                "priority": spec.priority,
                "computeEnvironmentOrder": [
                    self.binding(environment_key, order=1)
                ],
            }
        self.describe_compute_environment_calls: list[list[str]] = []
        self.describe_job_queue_calls: list[list[str]] = []
        self.mutation_calls: list[tuple[str, dict[str, object]]] = []

    def binding(self, key: str, *, order: int) -> dict[str, object]:
        return {
            "order": order,
            "computeEnvironment": self.compute_environments[key][
                "computeEnvironmentArn"
            ],
        }

    def describe_compute_environments(
        self, *, computeEnvironments: list[str]
    ) -> dict[str, object]:
        self.describe_compute_environment_calls.append(computeEnvironments)
        found = [
            environment
            for environment in self.compute_environments.values()
            if environment["computeEnvironmentName"] in computeEnvironments
        ]
        return {"computeEnvironments": found}

    def describe_job_queues(self, *, jobQueues: list[str]) -> dict[str, object]:
        self.describe_job_queue_calls.append(jobQueues)
        found = [
            queue
            for queue in self.queues.values()
            if queue["jobQueueName"] in jobQueues
        ]
        return {"jobQueues": found}

    def mutate_compute_environment(self, key: str, field: str, value: object) -> None:
        environment = self.compute_environments[key]
        resources = environment["computeResources"]
        assert isinstance(resources, dict)
        target = resources if field in resources else environment
        target[field] = value

    def update_compute_environment(self, **kwargs: object) -> None:
        self.mutation_calls.append(("update_compute_environment", kwargs))

    def update_job_queue(self, **kwargs: object) -> None:
        self.mutation_calls.append(("update_job_queue", kwargs))

    def create_compute_environment(self, **kwargs: object) -> None:
        self.mutation_calls.append(("create_compute_environment", kwargs))

    def create_job_queue(self, **kwargs: object) -> None:
        self.mutation_calls.append(("create_job_queue", kwargs))

    def delete_compute_environment(self, **kwargs: object) -> None:
        self.mutation_calls.append(("delete_compute_environment", kwargs))

    def delete_job_queue(self, **kwargs: object) -> None:
        self.mutation_calls.append(("delete_job_queue", kwargs))

    def tag_resource(self, **kwargs: object) -> None:
        self.mutation_calls.append(("tag_resource", kwargs))

    def untag_resource(self, **kwargs: object) -> None:
        self.mutation_calls.append(("untag_resource", kwargs))


@pytest.fixture
def fake_services() -> SimpleNamespace:
    return SimpleNamespace(batch=FakeBatch(), sts=FakeSts())


def validator(fake_services: SimpleNamespace) -> BatchTopologyValidator:
    return BatchTopologyValidator(
        fake_services.batch,
        fake_services.sts,
        expected_topology(),
        NETWORK,
    )


def test_expected_topology_is_exact_and_shared() -> None:
    topology = expected_topology()
    assert expected_topology() is topology
    assert tuple(topology.profiles) == ("c7am", "c7al", "c7ax", "g6x")
    assert topology.profiles["c7am"].resource_requirements == (
        ("VCPU", "1"),
        ("MEMORY", "1600"),
    )
    assert topology.profiles["c7al"].resource_requirements == (
        ("VCPU", "2"),
        ("MEMORY", "3200"),
    )
    assert topology.profiles["c7ax"].resource_requirements == (
        ("VCPU", "4"),
        ("MEMORY", "7168"),
    )
    assert topology.profiles["g6x"].resource_requirements == (
        ("VCPU", "4"),
        ("MEMORY", "12000"),
        ("GPU", "1"),
    )
    assert {
        key: (spec.instance_type, spec.max_vcpus, spec.ami_family)
        for key, spec in topology.compute_environments.items()
    } == {
        "c7am": ("c7a.medium", 16, "ECS_AL2023"),
        "c7al": ("c7a.large", 32, "ECS_AL2023"),
        "c7ax": ("c7a.xlarge", 16, "ECS_AL2023"),
        "g6x": ("g6.xlarge", 32, "ECS_AL2023_NVIDIA"),
    }
    assert {
        key: (spec.name, spec.priority, spec.compute_environments)
        for key, spec in topology.queues.items()
    } == {
        "dev-c7am": ("dev-cpu-c7am-queue", 10, ("c7am",)),
        "dev-c7al": ("dev-cpu-c7al-queue", 10, ("c7al",)),
        "dev-c7ax": ("dev-cpu-c7ax-queue", 10, ("c7ax",)),
        "run-c7am": ("run-cpu-c7am-queue", 100, ("c7am",)),
        "run-c7al": ("run-cpu-c7al-queue", 100, ("c7al",)),
        "run-c7ax": ("run-cpu-c7ax-queue", 100, ("c7ax",)),
        "dev-g6x": ("dev-gpu-queue", 10, ("g6x",)),
        "run-g6x": ("run-gpu-queue", 100, ("g6x",)),
    }
    assert queue_for(ExecutionPurpose.DEV, "g6x").name == "dev-gpu-queue"
    assert queue_for(ExecutionPurpose.RUN, "g6x").name == "run-gpu-queue"


def test_topology_is_recursively_immutable() -> None:
    topology = expected_topology()
    with pytest.raises(TypeError):
        topology.profiles["other"] = topology.profiles["c7am"]
    with pytest.raises(FrozenInstanceError):
        topology.profiles["c7am"].vcpus = 2


def test_run_has_higher_nonpreemptive_queue_priority() -> None:
    topology = expected_topology()
    assert topology.queues["dev-c7ax"].priority == 10
    assert topology.queues["run-c7ax"].priority == 100


def test_queue_lookup_rejects_unknown_profile() -> None:
    with pytest.raises(ValueError, match="unknown Batch resource profile"):
        queue_for(ExecutionPurpose.RUN, "gpu")


def test_validator_accepts_exact_topology_using_only_read_calls(
    fake_services: SimpleNamespace,
) -> None:
    original_compute_environments = deepcopy(
        fake_services.batch.compute_environments
    )
    original_queues = deepcopy(fake_services.batch.queues)

    validated = validator(fake_services).validate()

    assert tuple(validated.compute_environment_arns) == (
        "c7am",
        "c7al",
        "c7ax",
        "g6x",
    )
    assert tuple(validated.queue_arns) == (
        "dev-c7am",
        "dev-c7al",
        "dev-c7ax",
        "run-c7am",
        "run-c7al",
        "run-c7ax",
        "dev-g6x",
        "run-g6x",
    )
    assert len(fake_services.batch.describe_compute_environment_calls) == 4
    assert len(fake_services.batch.describe_job_queue_calls) == 8
    assert fake_services.sts.calls == 1
    assert fake_services.batch.mutation_calls == []
    assert fake_services.batch.compute_environments == original_compute_environments
    assert fake_services.batch.queues == original_queues
    with pytest.raises(TypeError):
        validated.queue_arns["other"] = "arn"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("instanceTypes", ["c7a.2xlarge"]),
        ("maxvCpus", 64),
        ("state", "DISABLED"),
        ("status", "INVALID"),
        ("type", "UNMANAGED"),
    ],
)
def test_compute_environment_drift_fails_without_mutation(
    fake_services: SimpleNamespace,
    field: str,
    value: object,
) -> None:
    fake_services.batch.mutate_compute_environment("c7al", field, value)

    with pytest.raises(ProfileDriftError, match=field):
        validator(fake_services).validate()

    assert fake_services.batch.mutation_calls == []


@pytest.mark.parametrize(
    "arn",
    [
        (
            "arn:aws-us-gov:batch:eu-north-1:007122174918:"
            "compute-environment/rtrrl-cpu-c7am-ce"
        ),
        (
            "arn:aws:s3:eu-north-1:007122174918:"
            "compute-environment/rtrrl-cpu-c7am-ce"
        ),
        (
            "arn:aws:batch:us-east-1:007122174918:"
            "compute-environment/rtrrl-cpu-c7am-ce"
        ),
        (
            "arn:aws:batch:eu-north-1:123456789012:"
            "compute-environment/rtrrl-cpu-c7am-ce"
        ),
        (
            "arn:aws:batch:eu-north-1:007122174918:"
            "job-queue/rtrrl-cpu-c7am-ce"
        ),
        (
            "arn:aws:batch:eu-north-1:007122174918:"
            "compute-environment/rtrrl-cpu-c7ax-ce"
        ),
        "not-an-arn",
    ],
)
def test_compute_environment_arn_must_be_exact_even_when_queue_binding_agrees(
    fake_services: SimpleNamespace, arn: str
) -> None:
    fake_services.batch.compute_environments["c7am"]["computeEnvironmentArn"] = arn
    for key in ("dev-c7am", "run-c7am"):
        fake_services.batch.queues[key]["computeEnvironmentOrder"] = [
            {"order": 1, "computeEnvironment": arn}
        ]

    with pytest.raises(ProfileDriftError, match="computeEnvironmentArn"):
        validator(fake_services).validate()


@pytest.mark.parametrize(
    "arn",
    [
        "arn:aws-us-gov:batch:eu-north-1:007122174918:job-queue/dev-cpu-c7am-queue",
        "arn:aws:s3:eu-north-1:007122174918:job-queue/dev-cpu-c7am-queue",
        "arn:aws:batch:us-east-1:007122174918:job-queue/dev-cpu-c7am-queue",
        "arn:aws:batch:eu-north-1:123456789012:job-queue/dev-cpu-c7am-queue",
        (
            "arn:aws:batch:eu-north-1:007122174918:"
            "compute-environment/dev-cpu-c7am-queue"
        ),
        "arn:aws:batch:eu-north-1:007122174918:job-queue/run-cpu-c7am-queue",
        "not-an-arn",
    ],
)
def test_job_queue_arn_must_match_exact_queue(fake_services: SimpleNamespace, arn: str) -> None:
    fake_services.batch.queues["dev-c7am"]["jobQueueArn"] = arn

    with pytest.raises(ProfileDriftError, match="jobQueueArn"):
        validator(fake_services).validate()


@pytest.mark.parametrize(
    "value",
    [
        NETWORK.subnets[1:],
        (*NETWORK.subnets, "subnet-extra"),
        (*NETWORK.subnets, NETWORK.subnets[0]),
        (*NETWORK.subnets[:-1], 7),
    ],
    ids=["missing", "extra", "duplicate", "non-string"],
)
def test_network_subnets_fail_closed_for_non_exact_sets(
    fake_services: SimpleNamespace, value: tuple[object, ...]
) -> None:
    resources = fake_services.batch.compute_environments["c7am"]["computeResources"]
    assert isinstance(resources, dict)
    resources["subnets"] = list(value)

    with pytest.raises(ProfileDriftError, match="computeResources.subnets"):
        validator(fake_services).validate()


@pytest.mark.parametrize("field", ["subnets", "securityGroupIds"])
def test_network_rejects_duplicate_values(
    fake_services: SimpleNamespace, field: str
) -> None:
    resources = fake_services.batch.compute_environments["c7am"]["computeResources"]
    assert isinstance(resources, dict)
    values = resources[field]
    assert isinstance(values, list)
    resources[field] = [*values, values[0]]

    with pytest.raises(ProfileDriftError, match=field):
        validator(fake_services).validate()


def test_desired_vcpus_and_update_available_are_not_drift(
    fake_services: SimpleNamespace,
) -> None:
    environment = fake_services.batch.compute_environments["g6x"]
    resources = environment["computeResources"]
    assert isinstance(resources, dict)
    resources["desiredvCpus"] = 31
    environment["updateStatus"] = "UPDATE_AVAILABLE"

    validator(fake_services).validate()


def test_ami_validation_allows_unrelated_metadata_and_empty_override(
    fake_services: SimpleNamespace,
) -> None:
    resources = fake_services.batch.compute_environments["c7al"]["computeResources"]
    assert isinstance(resources, dict)
    resources["ec2Configuration"] = [
        {
            "imageType": "ECS_AL2023",
            "imageIdOverride": "",
            "imageKubernetesVersion": "ignored",
        }
    ]
    validator(fake_services).validate()


def test_ami_validation_rejects_wrong_image_type(fake_services: SimpleNamespace) -> None:
    resources = fake_services.batch.compute_environments["c7al"]["computeResources"]
    assert isinstance(resources, dict)
    resources["ec2Configuration"] = [{"imageType": "ECS_AL2"}]
    with pytest.raises(ProfileDriftError, match="imageType"):
        validator(fake_services).validate()


@pytest.mark.parametrize(
    ("field", "value", "path"),
    [
        (
            "ec2Configuration",
            [{"imageType": "ECS_AL2023", "imageIdOverride": "ami-override"}],
            "imageIdOverride",
        ),
        ("imageId", "ami-legacy", "computeResources.imageId"),
        ("launchTemplate", {"launchTemplateName": "custom"}, "launchTemplate"),
    ],
)
def test_ami_overrides_fail_closed(
    fake_services: SimpleNamespace,
    field: str,
    value: object,
    path: str,
) -> None:
    resources = fake_services.batch.compute_environments["c7al"]["computeResources"]
    assert isinstance(resources, dict)
    resources[field] = value

    with pytest.raises(ProfileDriftError, match=path):
        validator(fake_services).validate()


@pytest.mark.parametrize(
    ("location", "field", "value"),
    [
        ("resources", "minvCpus", False),
        ("resources", "minvCpus", 0.0),
        ("resources", "maxvCpus", True),
        ("resources", "maxvCpus", 32.0),
        ("resources", "instanceTypes", ("c7a.large",)),
        (
            "resources",
            "ec2Configuration",
            ({"imageType": "ECS_AL2023"},),
        ),
        ("queue", "priority", True),
        ("queue", "priority", 10.0),
        (
            "queue",
            "computeEnvironmentOrder",
            (
                {
                    "order": 1,
                    "computeEnvironment": (
                        "arn:aws:batch:eu-north-1:007122174918:"
                        "compute-environment/rtrrl-cpu-c7am-ce"
                    ),
                },
            ),
        ),
        (
            "queue",
            "computeEnvironmentOrder",
            [
                {
                    "order": True,
                    "computeEnvironment": (
                        "arn:aws:batch:eu-north-1:007122174918:"
                        "compute-environment/rtrrl-cpu-c7am-ce"
                    ),
                }
            ],
        ),
    ],
)
def test_aws_fields_require_exact_python_types(
    fake_services: SimpleNamespace,
    location: str,
    field: str,
    value: object,
) -> None:
    if location == "resources":
        resources = fake_services.batch.compute_environments["c7al"][
            "computeResources"
        ]
        assert isinstance(resources, dict)
        resources[field] = value
    else:
        fake_services.batch.queues["dev-c7am"][field] = value

    with pytest.raises(ProfileDriftError, match=field):
        validator(fake_services).validate()


def test_cpu_queue_rejects_wrong_single_environment(
    fake_services: SimpleNamespace,
) -> None:
    fake_services.batch.queues["run-c7al"]["computeEnvironmentOrder"] = [
        fake_services.batch.binding("c7ax", order=1)
    ]

    with pytest.raises(ProfileDriftError, match="computeEnvironmentOrder"):
        validator(fake_services).validate()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("state", "DISABLED"),
        ("status", "INVALID"),
        ("priority", 1),
        ("computeEnvironmentOrder", []),
    ],
)
def test_queue_drift_fails_closed(
    fake_services: SimpleNamespace, field: str, value: object
) -> None:
    fake_services.batch.queues["dev-c7ax"][field] = value

    with pytest.raises(ProfileDriftError, match=field):
        validator(fake_services).validate()


@pytest.mark.parametrize(
    ("account", "region"),
    [
        ("123456789012", REGION),
        (ACCOUNT_ID, "us-east-1"),
    ],
)
def test_validator_rejects_wrong_account_or_region(
    fake_services: SimpleNamespace, account: str, region: str
) -> None:
    fake_services.sts.account = account
    fake_services.batch.meta.region_name = region

    with pytest.raises(ProfileDriftError, match=f"{ACCOUNT_ID}/{REGION}"):
        validator(fake_services).validate()


@pytest.mark.parametrize(
    ("response", "path"),
    [
        (None, "sts.get_caller_identity"),
        ([], "sts.get_caller_identity"),
        ({}, "sts.Account"),
        ({"Account": 7122174918}, "sts.Account"),
        ({"Account": True}, "sts.Account"),
    ],
)
def test_sts_response_failures_are_profile_drift(
    fake_services: SimpleNamespace, response: object, path: str
) -> None:
    fake_services.sts.get_caller_identity = lambda: response

    with pytest.raises(ProfileDriftError, match=path):
        validator(fake_services).validate()


@pytest.mark.parametrize(
    ("response", "path"),
    [
        (None, "describe_compute_environments"),
        ([], "describe_compute_environments"),
        ({}, "computeEnvironments"),
        ({"computeEnvironments": ()}, "computeEnvironments"),
        ({"computeEnvironments": [None]}, r"computeEnvironments\[0\]"),
    ],
)
def test_compute_environment_response_containers_fail_with_paths(
    fake_services: SimpleNamespace, response: object, path: str
) -> None:
    fake_services.batch.describe_compute_environments = (
        lambda *, computeEnvironments: response
    )

    with pytest.raises(ProfileDriftError, match=path):
        validator(fake_services).validate()


@pytest.mark.parametrize(
    ("response", "path"),
    [
        (None, "describe_job_queues"),
        ([], "describe_job_queues"),
        ({}, "jobQueues"),
        ({"jobQueues": ()}, "jobQueues"),
        ({"jobQueues": [None]}, r"jobQueues\[0\]"),
    ],
)
def test_job_queue_response_containers_fail_with_paths(
    fake_services: SimpleNamespace, response: object, path: str
) -> None:
    fake_services.batch.describe_job_queues = lambda *, jobQueues: response

    with pytest.raises(ProfileDriftError, match=path):
        validator(fake_services).validate()


@pytest.mark.parametrize(
    ("target", "field", "path"),
    [
        ("compute", "computeResources", "computeResources"),
        ("compute", "computeEnvironmentArn", "computeEnvironmentArn"),
        ("queue", "computeEnvironmentOrder", "computeEnvironmentOrder"),
        ("queue", "jobQueueArn", "jobQueueArn"),
    ],
)
def test_missing_aws_fields_raise_profile_drift_with_path(
    fake_services: SimpleNamespace,
    target: str,
    field: str,
    path: str,
) -> None:
    resource = (
        fake_services.batch.compute_environments["c7am"]
        if target == "compute"
        else fake_services.batch.queues["dev-c7am"]
    )
    del resource[field]

    with pytest.raises(ProfileDriftError, match=path):
        validator(fake_services).validate()


def test_missing_batch_region_metadata_is_profile_drift(
    fake_services: SimpleNamespace,
) -> None:
    fake_services.batch.meta = None

    with pytest.raises(ProfileDriftError, match="batch.meta.region_name"):
        validator(fake_services).validate()


@pytest.mark.parametrize("duplicate", [False, True])
def test_validator_requires_exactly_one_named_resource(
    fake_services: SimpleNamespace, duplicate: bool
) -> None:
    original = fake_services.batch.describe_compute_environments

    def return_wrong_cardinality(
        *, computeEnvironments: list[str]
    ) -> dict[str, object]:
        response = original(computeEnvironments=computeEnvironments)
        found = response["computeEnvironments"]
        assert isinstance(found, list)
        return {"computeEnvironments": found * 2 if duplicate else []}

    fake_services.batch.describe_compute_environments = return_wrong_cardinality

    with pytest.raises(ProfileDriftError, match="compute environment"):
        validator(fake_services).validate()
