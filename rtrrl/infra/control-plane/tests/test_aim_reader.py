from __future__ import annotations

import hashlib

import pytest

from trainer_infra.aim_reader import AimResultError, AimResultTimeout, AimReader

pytestmark = pytest.mark.filterwarnings(
    "ignore:.*declarative_base.*:sqlalchemy.exc.MovedIn20Warning"
)


class MetricData:
    def __init__(self, values: list[object]) -> None:
        self._values = values

    def values_list(self) -> tuple[list[object]]:
        return (self._values,)


class Metric:
    def __init__(self, values: list[object]) -> None:
        self.data = MetricData(values)


class FakeRun:
    def __init__(
        self,
        run_id: str,
        *,
        finalized: object = True,
        failed: object = False,
        objective: object = "eval/reward",
        values: list[object] | None = None,
    ) -> None:
        self.values = {
            "hparams": {"identity": {"run_id": run_id}},
            "sdk/finalized": finalized,
            "sdk/failed": failed,
            "sdk/objective_metric": objective,
        }
        self.metric = Metric([4.25] if values is None else values)

    def get(self, key: str, default: object = None) -> object:
        return self.values.get(key, default)

    def get_metric(self, name: str, context: object) -> Metric | None:
        del context
        return self.metric if name == "eval/reward" else None

    def close(self) -> None:
        return None


def test_reader_replays_spool_and_opens_only_the_hash_of_exact_run_id() -> None:
    calls: list[object] = []

    def replay(run_id: str) -> None:
        calls.append(("replay", run_id))

    def factory(**kwargs: object) -> FakeRun:
        calls.append(("open", kwargs))
        return FakeRun("experiment:group:0001")

    result = AimReader(
        repo="/aim",
        run_factory=factory,
        replay_spool=replay,
        sleep=lambda _: None,
    ).wait_for_result("experiment:group:0001", "eval/reward", 1)

    assert result == 4.25
    assert calls[0] == ("replay", "experiment:group:0001")
    assert calls[1] == (
        "open",
        {
            "run_hash": hashlib.sha256(b"experiment:group:0001").hexdigest()[:24],
            "repo": "/aim",
            "read_only": True,
        },
    )


@pytest.mark.parametrize(
    ("run", "message"),
    [
        (FakeRun("wrong-run"), "exact run_id"),
        (FakeRun("run", finalized=False), "timed out"),
        (FakeRun("run", objective="other"), "objective"),
        (FakeRun("run", values=[]), "objective"),
        (FakeRun("run", values=[True]), "finite numeric"),
        (FakeRun("run", values=[float("nan")]), "finite numeric"),
    ],
)
def test_reader_rejects_wrong_identity_unfinalized_or_invalid_objective(
    run: FakeRun, message: str
) -> None:
    ticks = iter((0.0, 2.0))
    reader = AimReader(
        run_factory=lambda **_: run,
        clock=lambda: next(ticks),
        sleep=lambda _: None,
    )

    error = AimResultTimeout if message == "timed out" else AimResultError
    with pytest.raises(error, match=message):
        reader.wait_for_result("run", "eval/reward", 1)


def test_reader_fails_immediately_on_sdk_failed_without_sleeping() -> None:
    sleeps: list[float] = []
    reader = AimReader(
        run_factory=lambda **_: FakeRun("run", failed=True, finalized=False),
        clock=lambda: 0.0,
        sleep=sleeps.append,
    )

    with pytest.raises(AimResultError, match="sdk/failed"):
        reader.wait_for_result("run", "eval/reward", 60)
    assert sleeps == []


def test_timeout_uses_injected_monotonic_clock_and_sleep() -> None:
    ticks = iter((10.0, 10.5, 11.0))
    sleeps: list[float] = []
    reader = AimReader(
        run_factory=lambda **_: FakeRun("run", finalized=False),
        clock=lambda: next(ticks),
        sleep=sleeps.append,
        poll_interval=0.5,
    )

    with pytest.raises(AimResultTimeout, match="run"):
        reader.wait_for_result("run", "eval/reward", 1)
    assert sleeps == [0.5]
