"""The logger their loop calls, checked against what their loop does with it.

The calls asserted here are the ones `rtrrl.py` actually makes, in the order it
makes them: seed the best, log a training dictionary, log an evaluation, read
the best back, finalize twice. A logger that satisfies the protocol on paper and
raises on the third of those costs whatever the run had spent by then.
"""

from __future__ import annotations

from collections.abc import Mapping

from entries.reporting import ReporterLogger


class Recorder:
    """A reporter that keeps what it was told."""

    def __init__(self) -> None:
        self.reports: list[tuple[int, dict[str, float]]] = []

    def report(self, step: int, metrics: Mapping[str, float]) -> None:
        self.reports.append((step, dict(metrics)))


def test_the_best_is_readable_before_anything_has_been_logged() -> None:
    """Their loop compares against this key at the first evaluation."""

    logger = ReporterLogger(Recorder())

    assert logger["best_eval_reward"] == float("-inf")


def test_their_evaluation_scalar_arrives_under_the_scored_name() -> None:
    recorder = Recorder()
    logger = ReporterLogger(recorder)

    logger.log({"steps": 100_000, "eval/rewards": 42.0}, step=100_000)

    assert recorder.reports == [(100_000, {"eval/episode_return": 42.0})]


def test_their_training_scalars_are_kept_apart_from_the_score() -> None:
    recorder = Recorder()
    logger = ReporterLogger(recorder)

    logger.log(
        {"steps": 1000, "mean_reward": 1.5, "mean_v": -0.25, "total_td_loss": 3.0},
        step=1000,
    )

    assert recorder.reports == [
        (
            1000,
            {
                "train/mean_reward": 1.5,
                "train/mean_v": -0.25,
                "train/total_td_loss": 3.0,
            },
        )
    ]


def test_the_step_counts_environment_transitions_rather_than_iterations() -> None:
    """Their `steps` is the argument multiplied by the environment batch size.

    At the batch size of one this comparison runs at they are the same number.
    At any other, scoring on the argument would place the run's whole curve at a
    fraction of the steps it actually cost.
    """

    recorder = Recorder()
    logger = ReporterLogger(recorder)

    logger.log({"steps": 8000, "mean_reward": 1.0}, step=1000)

    assert recorder.reports[0][0] == 8000


def test_a_report_without_their_count_falls_back_to_the_argument() -> None:
    """Their loop omits the payload's count on iterations it does not log."""

    recorder = Recorder()
    logger = ReporterLogger(recorder)

    logger.log({"eval/rewards": 1.0}, step=5000)

    assert recorder.reports[0][0] == 5000


def test_an_empty_report_is_not_a_report() -> None:
    """Their loop calls `log` unconditionally, with `{}` on quiet iterations."""

    recorder = Recorder()
    logger = ReporterLogger(recorder)

    logger.log({}, step=1000)

    assert recorder.reports == []


def test_the_calls_their_loop_makes_and_ignores_do_not_raise() -> None:
    logger = ReporterLogger(Recorder())

    logger.log_params({"seed": 1})
    logger["avg_reward"] = 0.0
    logger.finalize([])
    logger.finalize()
    logger.log_video("env/video", frames=None, fps=30, caption="")

    assert logger["avg_reward"] == 0.0
