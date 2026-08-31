from pathlib import Path

INFRA = Path(__file__).parents[1]


def test_the_persistent_scheduler_service_is_experiment_neutral() -> None:
    service = INFRA / "systemd" / "study-scheduler.service"
    text = service.read_text()

    assert "r1" not in text.lower()
    assert "r2" not in text.lower()
    assert "runs/study-scheduler.sqlite" in text
    assert "trainer_infra.scheduler_cli" in text
