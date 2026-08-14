from memorax.observability import Reporter
from tests.support.observability import completed_episode, completed_trajectory


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


class TrajectoryRecorder:
    def __init__(self):
        self.trajectories = []
        self.closed = False

    def log_trajectory(self, trajectory):
        self.trajectories.append(trajectory)

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


def test_reporter_reduces_every_episode_but_routes_only_sampled_trajectories():
    scalar = ScalarRecorder()
    trajectories = TrajectoryRecorder()
    sampled = completed_trajectory(sample_step=8)

    with Reporter(scalar_sinks=[scalar], trajectory_sinks=[trajectories]) as reporter:
        reporter.log_episode(completed_episode())
        reporter.log_trajectory(sampled)

    # A sampled trajectory is a second view of an episode already reduced, so it
    # must not reduce again.
    assert len(scalar.records) == 1
    assert scalar.records[0][1]["train/episode/return"] == 4.0
    assert trajectories.trajectories == [sampled]
    assert scalar.closed and trajectories.closed


def test_reporter_fans_out_scalars_without_knowing_their_origin():
    first = ScalarRecorder()
    second = ScalarRecorder()

    with Reporter(scalar_sinks=[first, second]) as reporter:
        reporter.report(3, {"loss": 2.0})

    assert first.records == second.records == [(3, {"loss": 2.0})]
