from dataclasses import replace
from pathlib import Path
import subprocess
import warnings

import numpy as np
import pytest
import training_sdk

from training_sdk import Episode, MemorySpool, RunContext, TrainingRun
from training_sdk.rerun_adapter import RerunAdapter


def make_context(tmp_path, **overrides):
    values = {
        "experiment_name": "hopper",
        "experiment_id": "exp-1",
        "group": "dual",
        "script": "train.py",
        "run_id": "run-1",
        "run_number": 12,
        "trial_number": 3,
        "seed": 4,
        "metadata": {},
        "environment": {"name": "hopper"},
        "training_budget": {"env_steps": 1_000},
        "fixed_parameters": {},
        "sampled_parameters": {},
        "final_parameters": {},
        "image_digest": "sha256:abc",
        "resource_profile": "cpu-small",
        "artifact_directory": Path(tmp_path) / "artifacts",
    }
    values.update(overrides)
    return RunContext(**values)


def complete_episode(**overrides):
    values = {
        "number": 100,
        "phase": "eval",
        "start_env_steps": 100,
        "end_env_steps": 102,
        "observations": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        "actions": [[0.1], [0.2]],
        "rewards": [1.0, 2.0],
        "terminals": [False, True],
        "truncations": [False, False],
        "environment_states": [[10.0], [11.0], [12.0]],
    }
    values.update(overrides)
    return Episode(**values)


class FakeRecording:
    def __init__(self, path, *, fail_flush=False):
        self.path = path
        self.fail_flush = fail_flush
        self.properties = {}
        self.series = {}
        self.flushed = False
        self.closed = False

    def log_properties(self, properties):
        self.properties.update(properties)

    def log_series(self, name, values, times):
        self.series[name] = (values.copy(), tuple(times))

    def flush(self):
        if self.fail_flush:
            raise OSError("flush failed")
        self.path.write_bytes(b"fake rrd")
        self.flushed = True

    def close(self):
        self.closed = True


class FakeFactory:
    def __init__(self, *, fail_flush=False):
        self.fail_flush = fail_flush
        self.recordings = []

    def __call__(self, path):
        recording = FakeRecording(path, fail_flush=self.fail_flush)
        self.recordings.append(recording)
        return recording


def test_only_selected_complete_episode_is_recorded(tmp_path):
    factory = FakeFactory()
    adapter = RerunAdapter(
        make_context(tmp_path),
        every_episodes=100,
        root=tmp_path,
        factory=factory,
    )

    assert adapter.log_episode(complete_episode(number=99)) is None
    path = adapter.log_episode(complete_episode(number=100))

    assert path == tmp_path / "hopper" / "dual-0012" / "episode-000100.rrd"
    assert list(tmp_path.rglob("*.rrd")) == [path]
    assert factory.recordings[0].flushed
    assert factory.recordings[0].closed


@pytest.mark.parametrize("every_episodes", [0, -1, 1.5, True])
def test_every_episodes_must_be_a_positive_integer(tmp_path, every_episodes):
    with pytest.raises((TypeError, ValueError), match="every_episodes"):
        RerunAdapter(
            make_context(tmp_path),
            every_episodes=every_episodes,
            root=tmp_path,
            factory=FakeFactory(),
        )


def test_artifact_components_are_safe_and_collision_free(tmp_path):
    first_context = make_context(
        tmp_path,
        experiment_name="../hopper/%2F",
        group="../dual/%",
    )
    second_context = replace(first_context, experiment_name="%2E%2E/hopper/%252F")
    first = RerunAdapter(first_context, root=tmp_path, factory=FakeFactory())
    second = RerunAdapter(second_context, root=tmp_path, factory=FakeFactory())

    first_path = first.log_episode(complete_episode(number=1))
    second_path = second.log_episode(complete_episode(number=1))

    assert first_context.experiment_name == "../hopper/%2F"
    assert first_path != second_path
    assert first_path.parent.parent.name == "%2E%2E%2Fhopper%2F%252F"
    assert first_path.parent.name == "%2E%2E%2Fdual%2F%25-0012"
    assert first_path.resolve().is_relative_to(tmp_path.resolve())
    assert second_path.resolve().is_relative_to(tmp_path.resolve())


def test_metadata_and_all_episode_arrays_preserve_timeline_meaning(tmp_path):
    factory = FakeFactory()
    adapter = RerunAdapter(
        make_context(tmp_path), root=tmp_path, factory=factory
    )

    adapter.log_episode(complete_episode(number=7))

    recording = factory.recordings[0]
    assert recording.properties == {
        "experiment": "hopper",
        "group": "dual",
        "script": "train.py",
        "run_number": 12,
        "trial_number": 3,
        "episode_number": 7,
        "phase": "eval",
        "start_env_steps": 100,
        "end_env_steps": 102,
    }
    assert set(recording.series) == {
        "observations",
        "actions",
        "rewards",
        "terminals",
        "truncations",
        "environment_states",
    }
    assert recording.series["observations"][1] == (0, 1, 2)
    assert recording.series["environment_states"][1] == (0, 1, 2)
    for name in ("actions", "rewards", "terminals", "truncations"):
        assert recording.series[name][1] == (0, 1)
    np.testing.assert_array_equal(
        recording.series["observations"][0],
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
    )


def test_empty_environment_states_are_not_logged(tmp_path):
    factory = FakeFactory()
    adapter = RerunAdapter(
        make_context(tmp_path), root=tmp_path, factory=factory
    )

    adapter.log_episode(complete_episode(environment_states=[]))

    assert "environment_states" not in factory.recordings[0].series


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("observations", [[1.0], [2.0, 3.0], [4.0]]),
        ("actions", [object(), object()]),
        ("rewards", ["one", "two"]),
        ("environment_states", [[1.0], [object()], [3.0]]),
    ],
)
def test_adapter_rejects_ragged_object_and_non_numeric_arrays(
    tmp_path, field, value
):
    adapter = RerunAdapter(
        make_context(tmp_path), root=tmp_path, factory=FakeFactory()
    )

    with pytest.raises((TypeError, ValueError), match=field):
        adapter.log_episode(complete_episode(**{field: value}))

    assert list(tmp_path.rglob("*.rrd")) == []


def test_failed_flush_leaves_no_completed_or_temporary_artifact(tmp_path):
    adapter = RerunAdapter(
        make_context(tmp_path),
        root=tmp_path,
        factory=FakeFactory(fail_flush=True),
    )

    with pytest.raises(OSError, match="flush failed"):
        adapter.log_episode(complete_episode())

    assert list(tmp_path.rglob("*")) == [
        tmp_path / "hopper",
        tmp_path / "hopper" / "dual-0012",
    ]


def test_existing_episode_artifact_is_never_overwritten(tmp_path):
    target = tmp_path / "hopper" / "dual-0012" / "episode-000100.rrd"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"existing episode")
    adapter = RerunAdapter(
        make_context(tmp_path), root=tmp_path, factory=FakeFactory()
    )

    with pytest.raises(FileExistsError, match="episode-000100"):
        adapter.log_episode(complete_episode())

    assert target.read_bytes() == b"existing episode"
    assert list(target.parent.iterdir()) == [target]


def test_training_run_returns_delegated_episode_path(tmp_path):
    expected = tmp_path / "episode-000100.rrd"

    class ReturningRerun:
        def log_episode(self, episode):
            assert episode.number == 100
            return expected

    class NoopAim:
        def start(self, context):
            del context

        def send(self, event):
            del event

    run = TrainingRun(
        make_context(tmp_path),
        aim=NoopAim(),
        rerun=ReturningRerun(),
        spool=MemorySpool(),
    )

    assert run.log_episode(complete_episode()) == expected


def test_public_sdk_does_not_export_rerun_backend_types():
    assert not hasattr(training_sdk, "RerunAdapter")


def test_real_rerun_writes_nonempty_readable_rrd(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        path = RerunAdapter(make_context(tmp_path), root=tmp_path).log_episode(
            complete_episode(number=7)
        )

    assert path is not None
    assert path.stat().st_size > 0
    verified = subprocess.run(
        ["rerun", "rrd", "verify", str(path)],
        capture_output=True,
        check=False,
        text=True,
    )
    assert verified.returncode == 0, verified.stderr
