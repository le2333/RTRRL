from pathlib import Path
from unittest.mock import patch

import pytest
from rerun.experimental import RrdReader

from training_sdk import objects
from training_sdk.episode import Episode
from training_sdk.reporter import build_default_sinks
from training_sdk.sinks.rerun import RerunSink
from tests.test_reporter import make_config


def make_episode(
    number: int,
    phase: str = "eval",
    *,
    span: tuple[int, int] = (0, 2),
    stream: int = 0,
) -> Episode:
    return Episode(
        number=number,
        phase=phase,
        stream=stream,
        start_env_steps=span[0],
        end_env_steps=span[1],
        observations=[[0.0], [1.0], [2.0]],
        actions=[[0.0], [1.0]],
        rewards=[1.0, 2.0],
        terminals=[False, True],
        truncations=[False, False],
        series={"td_error": [0.5, 1.5]},
    )


def recording(s3_base: str, tmp_path: Path, *, where: str, every: int, num_envs: int = 1):
    """A sink whose sampling is the only thing under test."""

    config = make_config()
    config = config.model_copy(
        update={
            "logging": config.logging.model_copy(
                update={
                    "rerun_s3": f"{s3_base}/{where}/",
                    "rerun_every_steps": every,
                }
            ),
            "training": config.training.model_copy(update={"num_envs": num_envs}),
        }
    )
    return RerunSink(config, tmp_path)


def kept(sink, episodes) -> list[int]:
    for episode in episodes:
        sink.log_episode(episode)
    sink.close()
    return [
        episode.number
        for episode in episodes
        if objects.exists(f"{sink._prefix}/{episode.phase}-{episode.number:06d}.rrd")
    ]


def test_episode_is_uploaded_and_local_copy_removed(s3_base: str, tmp_path: Path) -> None:
    config = make_config().model_copy(
        update={
            "logging": make_config().logging.model_copy(
                update={"rerun_s3": f"{s3_base}/episodes/", "rerun_every_steps": 1}
            )
        }
    )
    sink = RerunSink(config, tmp_path)
    sink.log_episode(make_episode(1))
    sink.close()

    uri = f"{s3_base}/episodes/eval-000001.rrd"
    payload = objects.get_bytes(uri)
    assert payload.startswith(b"RRF2")
    assert list(tmp_path.glob("*.rrd")) == []


def test_an_episode_is_kept_when_a_sample_step_falls_inside_it(
    s3_base: str, tmp_path: Path
) -> None:
    """The stride is over environment steps, not over episode numbers.

    Numbering counts each stream's episodes in turn, so a stride over numbers
    spreads recordings over an enumeration order rather than over training.
    """

    sink = recording(s3_base, tmp_path, where="by-step", every=100)
    episodes = [
        make_episode(1, "train", span=(0, 90)),     # holds step 0
        make_episode(2, "train", span=(110, 190)),  # holds nothing
        make_episode(3, "train", span=(190, 260)),  # holds step 200
    ]

    assert kept(sink, episodes) == [1, 3]


def test_the_stream_a_sample_step_belongs_to_is_the_one_recorded(
    s3_base: str, tmp_path: Path
) -> None:
    """The step counter numbers every stream's every step, so a step names one.

    Step S is row S // num_envs of stream S % num_envs. Two streams whose
    episodes span the same steps are not both the answer for one sample point.
    """

    sink = recording(s3_base, tmp_path, where="by-stream", every=10, num_envs=4)
    episodes = [
        make_episode(1, "train", span=(0, 40), stream=0),  # step 0 and 20 are its
        make_episode(2, "train", span=(0, 40), stream=1),  # nothing lands on it
        make_episode(3, "train", span=(0, 40), stream=2),  # step 10 and 30 are its
    ]

    assert kept(sink, episodes) == [1, 3]


def test_an_episode_with_no_span_is_never_sampled(
    s3_base: str, tmp_path: Path
) -> None:
    """Evaluation episodes are dated at one step and cover no interval.

    They measure the policy at an epoch boundary rather than spending training
    steps, so no sample point is inside them and none is recorded. That falls
    out of the rule rather than being a case beside it.
    """

    sink = recording(s3_base, tmp_path, where="no-span", every=1)
    episodes = [make_episode(1, "eval", span=(64, 64))]

    assert kept(sink, episodes) == []


def test_recording_is_readable_by_rerun(s3_base: str, tmp_path: Path) -> None:
    config = make_config().model_copy(
        update={
            "logging": make_config().logging.model_copy(
                update={"rerun_s3": f"{s3_base}/readable/", "rerun_every_steps": 1}
            )
        }
    )
    sink = RerunSink(config, tmp_path)
    sink.log_episode(make_episode(1))
    sink.close()
    downloaded = tmp_path / "downloaded.rrd"
    downloaded.write_bytes(objects.get_bytes(f"{s3_base}/readable/eval-000001.rrd"))
    store = RrdReader(downloaded).store()
    summary = store.summary()
    assert "/episode/rewards" in summary
    assert "/episode/series/td_error" in summary


def _config_with_local_aim(tmp_path: Path, **logging_updates: object):
    aim_repo = str(tmp_path / "aim")
    logging = make_config().logging.model_copy(update={"aim": aim_repo, **logging_updates})
    return make_config().model_copy(update={"logging": logging})


def test_rerun_sink_requires_rerun_config(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="rerun sink requires rerun_s3 and rerun_every_steps"):
        RerunSink(make_config(), tmp_path)


def test_local_file_retained_when_upload_fails(s3_base: str, tmp_path: Path) -> None:
    config = make_config().model_copy(
        update={
            "logging": make_config().logging.model_copy(
                update={"rerun_s3": f"{s3_base}/fail/", "rerun_every_steps": 1}
            )
        }
    )
    sink = RerunSink(config, tmp_path)
    with patch("training_sdk.objects.put_file", side_effect=RuntimeError("upload failed")):
        with pytest.raises(RuntimeError, match="upload failed"):
            sink.log_episode(make_episode(1))
    assert list(tmp_path.glob("*.rrd")) == [tmp_path / "eval-000001.rrd"]


def test_build_default_sinks_omits_rerun_when_disabled(tmp_path: Path) -> None:
    from aim import Repo

    Repo.from_path(str(tmp_path / "aim"), init=True)
    sinks = build_default_sinks(_config_with_local_aim(tmp_path), tmp_path)
    assert all(not isinstance(sink, RerunSink) for sink in sinks)


def test_build_default_sinks_includes_rerun_when_enabled(
    s3_base: str, tmp_path: Path
) -> None:
    """Two sinks, two switches. Aim is not turned on by rerun's destination."""

    from aim import Repo

    Repo.from_path(str(tmp_path / "aim"), init=True)
    config = _config_with_local_aim(
        tmp_path,
        enable_rerun=True,
        rerun_s3=f"{s3_base}/episodes/",
        rerun_every_steps=1,
    )
    sinks = build_default_sinks(config, tmp_path)
    assert any(isinstance(sink, RerunSink) for sink in sinks)


def test_a_destination_without_the_switch_does_not_turn_rerun_on(
    s3_base: str, tmp_path: Path
) -> None:
    from aim import Repo

    Repo.from_path(str(tmp_path / "aim"), init=True)
    config = _config_with_local_aim(
        tmp_path,
        rerun_s3=f"{s3_base}/episodes/",
        rerun_every_steps=1,
    )
    sinks = build_default_sinks(config, tmp_path)
    assert all(not isinstance(sink, RerunSink) for sink in sinks)


def test_a_training_and_an_evaluation_episode_do_not_collide(
    s3_base: str, tmp_path: Path
) -> None:
    """Each phase counts its own episodes, so the number alone is not a name."""

    config = make_config().model_copy(
        update={
            "logging": make_config().logging.model_copy(
                update={"rerun_s3": f"{s3_base}/phases/", "rerun_every_steps": 1}
            )
        }
    )
    sink = RerunSink(config, tmp_path)
    sink.log_episode(make_episode(1, phase="train"))
    sink.log_episode(make_episode(1, phase="eval"))
    sink.close()

    assert objects.exists(f"{s3_base}/phases/train-000001.rrd") is True
    assert objects.exists(f"{s3_base}/phases/eval-000001.rrd") is True
