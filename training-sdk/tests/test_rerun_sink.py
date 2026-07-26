from pathlib import Path
from unittest.mock import patch

import pytest
from rerun.experimental import RrdReader

from training_sdk import objects
from training_sdk.episode import Episode
from training_sdk.reporter import build_default_sinks
from training_sdk.sinks.rerun import RerunSink
from tests.test_reporter import make_config


def make_episode(number: int) -> Episode:
    return Episode(
        number=number,
        phase="evaluation",
        start_env_steps=0,
        end_env_steps=2,
        observations=[[0.0], [1.0], [2.0]],
        actions=[[0.0], [1.0]],
        rewards=[1.0, 2.0],
        terminals=[False, True],
        truncations=[False, False],
    )


def test_episode_is_uploaded_and_local_copy_removed(s3_base: str, tmp_path: Path) -> None:
    config = make_config().model_copy(
        update={
            "logging": make_config().logging.model_copy(
                update={"rerun_s3": f"{s3_base}/episodes/", "rerun_every_episodes": 1}
            )
        }
    )
    sink = RerunSink(config, tmp_path)
    sink.log_episode(make_episode(1))
    sink.close()

    uri = f"{s3_base}/episodes/episode-000001.rrd"
    payload = objects.get_bytes(uri)
    assert payload.startswith(b"RRF2")
    assert list(tmp_path.glob("*.rrd")) == []


def test_only_every_nth_episode_is_recorded(s3_base: str, tmp_path: Path) -> None:
    config = make_config().model_copy(
        update={
            "logging": make_config().logging.model_copy(
                update={"rerun_s3": f"{s3_base}/every2/", "rerun_every_episodes": 2}
            )
        }
    )
    sink = RerunSink(config, tmp_path)
    sink.log_episode(make_episode(1))
    sink.log_episode(make_episode(2))
    sink.close()
    assert objects.exists(f"{s3_base}/every2/episode-000001.rrd") is False
    assert objects.exists(f"{s3_base}/every2/episode-000002.rrd") is True


def test_recording_is_readable_by_rerun(s3_base: str, tmp_path: Path) -> None:
    config = make_config().model_copy(
        update={
            "logging": make_config().logging.model_copy(
                update={"rerun_s3": f"{s3_base}/readable/", "rerun_every_episodes": 1}
            )
        }
    )
    sink = RerunSink(config, tmp_path)
    sink.log_episode(make_episode(1))
    sink.close()
    downloaded = tmp_path / "downloaded.rrd"
    downloaded.write_bytes(objects.get_bytes(f"{s3_base}/readable/episode-000001.rrd"))
    store = RrdReader(downloaded).store()
    assert "/episode/rewards" in store.summary()


def _config_with_local_aim(tmp_path: Path, **logging_updates: object):
    aim_repo = str(tmp_path / "aim")
    logging = make_config().logging.model_copy(update={"aim": aim_repo, **logging_updates})
    return make_config().model_copy(update={"logging": logging})


def test_rerun_sink_requires_rerun_config(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="rerun sink requires rerun_s3 and rerun_every_episodes"):
        RerunSink(make_config(), tmp_path)


def test_local_file_retained_when_upload_fails(s3_base: str, tmp_path: Path) -> None:
    config = make_config().model_copy(
        update={
            "logging": make_config().logging.model_copy(
                update={"rerun_s3": f"{s3_base}/fail/", "rerun_every_episodes": 1}
            )
        }
    )
    sink = RerunSink(config, tmp_path)
    with patch("training_sdk.objects.put_file", side_effect=RuntimeError("upload failed")):
        with pytest.raises(RuntimeError, match="upload failed"):
            sink.log_episode(make_episode(1))
    assert list(tmp_path.glob("*.rrd")) == [tmp_path / "episode-000001.rrd"]


def test_build_default_sinks_omits_rerun_when_disabled(tmp_path: Path) -> None:
    from aim import Repo

    Repo.from_path(str(tmp_path / "aim"), init=True)
    sinks = build_default_sinks(_config_with_local_aim(tmp_path), tmp_path)
    assert all(not isinstance(sink, RerunSink) for sink in sinks)


def test_build_default_sinks_includes_rerun_when_configured(
    s3_base: str, tmp_path: Path
) -> None:
    from aim import Repo

    Repo.from_path(str(tmp_path / "aim"), init=True)
    config = _config_with_local_aim(
        tmp_path,
        rerun_s3=f"{s3_base}/episodes/",
        rerun_every_episodes=1,
    )
    sinks = build_default_sinks(config, tmp_path)
    assert any(isinstance(sink, RerunSink) for sink in sinks)
