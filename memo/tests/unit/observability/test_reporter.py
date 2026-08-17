from memorax.observability import EpisodeScope, Reporter, StepScope, WindowScope
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


def steps_reaching(sink):
    return [step for step, _ in sink.records]


def test_the_record_keeps_every_episode_and_the_dashboard_keeps_a_few():
    """A run ends an episode every few dozen steps before it is any good.

    All of them belong in the record, because a question asked afterwards is
    answered from it. Sending all of them to a dashboard is millions of points
    that draw as one band, at a cost the run cannot pay.
    """

    record, dashboard = ScalarRecorder(), ScalarRecorder()
    reporter = Reporter(
        scalar_sinks=[record],
        sampled_sinks=[dashboard],
        training_scopes=[EpisodeScope(every_episodes=5)],
    )

    for number, end in enumerate(range(20, 421, 20), start=1):
        reporter.log_episode(completed_episode(number, span=(end - 20, end)))

    assert steps_reaching(record) == list(range(20, 421, 20))
    assert steps_reaching(dashboard) == [100, 200, 300, 400]


def test_the_record_names_the_episode_scope_whatever_the_dashboard_was_asked():
    """The record is one reduction per episode, so it is one scope's names.

    A scope is what the dashboard is configured with. What the run is scored
    on afterwards does not move because a dashboard was asked for a window.
    """

    record, dashboard = ScalarRecorder(), ScalarRecorder()
    reporter = Reporter(
        scalar_sinks=[record],
        sampled_sinks=[dashboard],
        training_scopes=[WindowScope(every_steps=40)],
    )

    reporter.log_episode(completed_episode(1, span=(0, 8)))
    reporter.close()

    assert list(record.records[0][1]) == [
        "train/episode/length",
        "train/episode/return",
        "train/episode/return_per_step",
        "train/episode/return_per_step_variance",
        "train/episode/td_error",
        "train/episode/td_error_variance",
    ]
    assert list(dashboard.records[0][1]) == [
        "train/window/length",
        "train/window/return",
        "train/window/return_per_step",
        "train/window/return_per_step_variance",
        "train/window/td_error",
        "train/window/td_error_variance",
    ]


def test_the_dashboard_receives_every_scope_the_run_asked_for():
    dashboard = ScalarRecorder()
    reporter = Reporter(
        sampled_sinks=[dashboard],
        training_scopes=[StepScope(every_steps=4), EpisodeScope(every_episodes=1)],
    )

    reporter.log_episode(completed_episode(1, span=(0, 8)))

    assert steps_reaching(dashboard) == [4, 8]
    assert set(dashboard.records[0][1]) == {
        "train/step/return_per_step",
        "train/step/td_error",
    }


def test_an_evaluation_reaches_the_dashboard_whatever_the_scopes():
    record, dashboard = ScalarRecorder(), ScalarRecorder()
    reporter = Reporter(
        scalar_sinks=[record],
        sampled_sinks=[dashboard],
        training_scopes=[EpisodeScope(every_episodes=1000)],
    )

    reporter.log_episode(completed_episode(1, phase="eval", span=(64, 64)))
    reporter.log_episode(completed_episode(2, span=(0, 8)))

    # The evaluation is what the run is scored on, so no schedule withholds it.
    assert steps_reaching(dashboard) == [64]
    assert steps_reaching(record) == [64, 8]


def test_no_scope_at_all_sends_the_dashboard_no_training():
    record, dashboard = ScalarRecorder(), ScalarRecorder()
    reporter = Reporter(scalar_sinks=[record], sampled_sinks=[dashboard])

    reporter.log_episode(completed_episode(1, span=(0, 8)))
    reporter.log_episode(completed_episode(2, phase="eval", span=(8, 8)))

    assert steps_reaching(dashboard) == [8]
    assert steps_reaching(record) == [8, 8]


def test_the_end_of_the_run_reaches_the_dashboard_before_it_is_closed():
    """A window the budget cut short is still a stretch, and still reported."""

    dashboard = ScalarRecorder()
    with Reporter(
        sampled_sinks=[dashboard], training_scopes=[WindowScope(every_steps=100)]
    ) as reporter:
        reporter.log_episode(completed_episode(1, span=(0, 8)))
        assert steps_reaching(dashboard) == []

    assert steps_reaching(dashboard) == [100]
    assert dashboard.closed


def test_reporter_fans_out_scalars_without_knowing_their_origin():
    first = ScalarRecorder()
    second = ScalarRecorder()

    with Reporter(scalar_sinks=[first, second]) as reporter:
        reporter.report(3, {"loss": 2.0})

    assert first.records == second.records == [(3, {"loss": 2.0})]
