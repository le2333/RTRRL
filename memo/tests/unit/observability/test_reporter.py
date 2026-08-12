from memorax.observability import Reporter
from tests.support.observability import completed_episode


class ScalarRecorder:
    def __init__(self):
        self.records = []
        self.closed = False

    def report(self, step, metrics):
        self.records.append((step, dict(metrics)))

    def close(self):
        self.closed = True


class EpisodeRecorder:
    def __init__(self):
        self.episodes = []
        self.closed = False

    def log_episode(self, episode):
        self.episodes.append(episode)

    def close(self):
        self.closed = True


def test_reporter_aggregates_once_and_fans_out_through_narrow_protocols():
    scalar = ScalarRecorder()
    trajectory = EpisodeRecorder()
    episode = completed_episode()

    with Reporter(scalar_sinks=[scalar], episode_sinks=[trajectory]) as reporter:
        reporter.log_episode(episode)

    assert scalar.records == [
        (
            8,
            {
                "train/episode/length": 2.0,
                "train/episode/return": 4.0,
                "train/episode/return_per_step": 2.0,
                "train/episode/return_per_step_variance": 1.0,
                "train/episode/td_error": 1.0,
                "train/episode/td_error_variance": 1.0,
            },
        )
    ]
    assert trajectory.episodes == [episode]
    assert scalar.closed and trajectory.closed


def test_reporter_fans_out_scalars_without_knowing_their_origin():
    first = ScalarRecorder()
    second = ScalarRecorder()

    with Reporter(scalar_sinks=[first, second]) as reporter:
        reporter.report(3, {"loss": 2.0})

    assert first.records == second.records == [(3, {"loss": 2.0})]
