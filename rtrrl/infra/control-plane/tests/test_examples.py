"""Guards on the experiment files shipped under `examples/`.

The suite runs against `tests/data/experiment.yaml` so that rebuilding an image does
not redden CI. That leaves the shipped examples unexercised, and they are the files a
person copies to launch a paid run — an unloadable or loopback-addressed example only
shows up once jobs are already running. These checks cover exactly that.
"""

from pathlib import Path

import pytest

from trainer_infra.experiment import load_experiment
from trainer_infra.preflight import (
    PreflightError,
    _parse_aim_endpoint,
    _reject_loopback_aim,
    check_offline,
)
from tests.helpers import CATALOG

EXAMPLES = sorted(Path("examples").glob("*.yaml"))


def test_examples_exist() -> None:
    assert EXAMPLES, "examples/ has no experiment files"


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda path: path.name)
def test_example_loads_and_passes_offline_checks(path: Path) -> None:
    check_offline(load_experiment(path), CATALOG)


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda path: path.name)
def test_example_pins_an_image_digest(path: Path) -> None:
    """A tag can be repointed after preflight resolves it; a digest cannot."""
    image = load_experiment(path).image
    assert "@sha256:" in image, f"{path.name} names a mutable tag: {image}"


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda path: path.name)
def test_example_aim_endpoint_is_reachable_from_batch(path: Path) -> None:
    """A Batch worker resolves this itself, so a loopback host reaches nothing."""
    aim = load_experiment(path).logging.aim
    host, _ = _parse_aim_endpoint(aim)
    try:
        _reject_loopback_aim(host, aim)
    except PreflightError as error:  # pragma: no cover - only on a bad example
        pytest.fail(str(error))
