from __future__ import annotations

import ipaddress
import socket
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from training_sdk.contract import CONTRACT_VERSION, Catalog, ChoiceSpec, EntryDescriptor
from training_sdk.contract import SpaceEntry

from trainer_infra.experiment import Experiment
from trainer_infra.space import distributions, resolve_space
from trainer_infra.study import check_sampler


class PreflightError(ValueError):
    """A launch precondition failed before anything was spent."""


@dataclass(frozen=True)
class LaunchPlan:
    experiment: Experiment
    entry_name: str
    entry: EntryDescriptor
    space: dict[str, SpaceEntry]
    digest: str
    queue: str
    job_definition: str


def check_offline(experiment: Experiment, catalog: Catalog) -> dict[str, SpaceEntry]:
    if catalog.contract != CONTRACT_VERSION:
        raise PreflightError(
            f"image declares contract {catalog.contract}; "
            f"this control plane implements contract {CONTRACT_VERSION}"
        )
    entry = catalog.entries.get(experiment.entry)
    if entry is None:
        available = ", ".join(sorted(catalog.entries))
        raise PreflightError(
            f"image does not declare entry {experiment.entry!r}; available: {available}"
        )
    if experiment.score.metric not in entry.metrics:
        reported = ", ".join(entry.metrics)
        raise PreflightError(
            f"entry {experiment.entry} does not report metric "
            f"{experiment.score.metric!r}; it reports: {reported}"
        )
    space = resolve_space(entry, experiment.space)
    check_sampler(experiment.hpo.sampler, distributions(space))
    if experiment.score.window_steps[1] > experiment.training.total_steps:
        raise PreflightError(
            f"score window upper bound {experiment.score.window_steps[1]} exceeds "
            f"the training total_steps ({experiment.training.total_steps})"
        )
    return space


def connect(host: str, port: int) -> None:
    try:
        with socket.create_connection((host, port), timeout=5):
            return
    except OSError as error:
        raise PreflightError(f"aim endpoint aim://{host}:{port} is not reachable: {error}") from error


def _parse_s3_bucket(storage: str) -> str:
    if not storage.startswith("s3://"):
        raise PreflightError(f"storage {storage!r} is not an s3:// URI")
    bucket = storage.removeprefix("s3://").split("/", 1)[0]
    if not bucket:
        raise PreflightError(f"storage {storage!r} has no bucket name")
    return bucket


def _parse_aim_endpoint(aim: str) -> tuple[str, int]:
    if not aim.startswith("aim://"):
        raise PreflightError(f"logging aim {aim!r} is not an aim:// URI")
    host_port = aim.removeprefix("aim://")
    if ":" not in host_port:
        raise PreflightError(f"logging aim {aim!r} has no port")
    host, port_text = host_port.rsplit(":", 1)
    if not host:
        raise PreflightError(f"logging aim {aim!r} has no host")
    try:
        port = int(port_text)
    except ValueError as error:
        raise PreflightError(f"logging aim {aim!r} has invalid port {port_text!r}") from error
    return host, port


def _reject_loopback_aim(host: str, aim: str) -> None:
    """A Batch host's loopback is its own, not the control plane's.

    This endpoint is copied verbatim into every run config, so a loopback address
    reaches the Aim server here and nothing at all from the job. Preflight cannot
    catch that by connecting, because connecting from here succeeds — which is the
    whole trap. It has to be read.
    """
    if host == "localhost":
        loopback = True
    else:
        try:
            loopback = ipaddress.ip_address(host).is_loopback
        except ValueError:
            loopback = False
    if loopback:
        raise PreflightError(
            f"logging aim {aim!r} is a loopback address, which a Batch job resolves "
            "to itself; name the address the job can reach, such as this machine's "
            "private IP"
        )


def _describe_catalog_mismatch(image_catalog: Catalog, offline_catalog: Catalog) -> str:
    parts: list[str] = []
    if image_catalog.contract != offline_catalog.contract:
        parts.append(
            f"contract differs (image {image_catalog.contract}, "
            f"offline {offline_catalog.contract})"
        )

    image_entries = set(image_catalog.entries)
    offline_entries = set(offline_catalog.entries)
    for name in sorted(image_entries - offline_entries):
        parts.append(f"entry {name!r} only in image catalog")
    for name in sorted(offline_entries - image_entries):
        parts.append(f"entry {name!r} only in offline catalog")

    for name in sorted(image_entries & offline_entries):
        image_entry = image_catalog.entries[name]
        offline_entry = offline_catalog.entries[name]
        if image_entry.command != offline_entry.command:
            parts.append(f"entry {name!r} command differs")
        if image_entry.metrics != offline_entry.metrics:
            parts.append(f"entry {name!r} metrics differs")

        image_space = image_entry.space
        offline_space = offline_entry.space
        for key in sorted(set(image_space) | set(offline_space)):
            field = f"space.{key}"
            if key not in image_space:
                parts.append(f"entry {name!r} {field} only in offline catalog")
            elif key not in offline_space:
                parts.append(f"entry {name!r} {field} only in image catalog")
            elif image_space[key] != offline_space[key]:
                parts.append(f"entry {name!r} {field} differs")

    if not parts:
        parts.append("catalogs differ with no detected field changes")
    return "image catalog does not match the offline-validated catalog: " + "; ".join(parts)


def check_aws(
    experiment: Experiment,
    catalog: Catalog,
    space: dict[str, SpaceEntry],
    *,
    ecr_client: Any,
    batch_client: Any,
    s3_client: Any,
    read_url: Callable[[str], bytes],
    connect: Callable[[str, int], None] = connect,
    tier: str = "run",
) -> LaunchPlan:
    from trainer_infra.images import resolve_image
    from trainer_infra.queues import binding, job_definition_name

    resolved = resolve_image(experiment.image, ecr_client, read_url)
    if resolved.catalog != catalog:
        raise PreflightError(_describe_catalog_mismatch(resolved.catalog, catalog))

    queue_binding = binding(experiment.compute.instance_type)
    queue_name = queue_binding.queue(tier)

    queue_response = batch_client.describe_job_queues(jobQueues=[queue_name])
    queues = queue_response.get("jobQueues", [])
    if not queues:
        raise PreflightError(f"queue {queue_name!r} is not registered")
    queue_entry = queues[0]
    if queue_entry.get("state") != "ENABLED" or queue_entry.get("status") != "VALID":
        raise PreflightError(
            f"queue {queue_name!r} is not ready "
            f"(state={queue_entry.get('state')!r}, status={queue_entry.get('status')!r})"
        )

    definition_name = job_definition_name(queue_binding, resolved.digest)
    definition_response = batch_client.describe_job_definitions(
        jobDefinitionName=definition_name,
        status="ACTIVE",
    )
    if not definition_response.get("jobDefinitions"):
        raise PreflightError(
            f"job definition {definition_name!r} is not registered; "
            "run scripts/deploy_facility.py --register "
            "--cpu-digest <digest> --gpu-digest <digest>"
        )

    bucket = _parse_s3_bucket(experiment.storage)
    try:
        s3_client.head_bucket(Bucket=bucket)
    except Exception as error:
        raise PreflightError(f"S3 bucket {bucket!r} is not reachable: {error}") from error

    aim_host, aim_port = _parse_aim_endpoint(experiment.logging.aim)
    _reject_loopback_aim(aim_host, experiment.logging.aim)
    try:
        connect(aim_host, aim_port)
    except PreflightError:
        raise
    except OSError as error:
        raise PreflightError(
            f"aim endpoint {experiment.logging.aim!r} is not reachable: {error}"
        ) from error

    return LaunchPlan(
        experiment=experiment,
        entry_name=experiment.entry,
        entry=catalog.entries[experiment.entry],
        space=space,
        digest=resolved.digest,
        queue=queue_name,
        job_definition=definition_name,
    )


def format_space(space: dict[str, SpaceEntry]) -> str:
    lines = []
    for key in sorted(space):
        spec = space[key]
        if isinstance(spec, ChoiceSpec):
            rendered = " | ".join(repr(choice) for choice in spec.choices)
        else:
            rendered = spec.model_dump_json()
        lines.append(f"  {key}: {rendered}")
    return "resolved search space:\n" + "\n".join(lines)
