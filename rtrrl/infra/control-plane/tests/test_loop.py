from trainer_infra.loop import select_best
from trainer_infra.report import TrialRecord


def test_select_best_picks_the_lowest_value_when_minimizing() -> None:
    records = [
        TrialRecord(trial=0, params={"x": 1}, value=3.0),
        TrialRecord(trial=1, params={"x": 2}, value=1.0),
        TrialRecord(trial=2, params={"x": 3}, value=2.0),
    ]

    best = select_best(records, maximize=False)

    assert best is not None
    assert best.trial == 1
    assert best.value == 1.0


def test_select_best_picks_the_highest_value_when_maximizing() -> None:
    records = [
        TrialRecord(trial=0, params={"x": 1}, value=3.0),
        TrialRecord(trial=1, params={"x": 2}, value=1.0),
        TrialRecord(trial=2, params={"x": 3}, value=2.0),
    ]

    best = select_best(records, maximize=True)

    assert best is not None
    assert best.trial == 0
    assert best.value == 3.0
