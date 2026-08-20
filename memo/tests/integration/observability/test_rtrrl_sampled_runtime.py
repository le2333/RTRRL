"""One fixed RTRRL run, end to end, through the path a deployment uses.

The schedule is the acceptance schedule scaled down: a 50M run sampling every
10M becomes a 50-step run sampling every 10, on an environment whose episodes
are three transitions long. The last sample therefore cannot end inside the
budget, which is the case the continuation exists for.
"""

import json

import pytest
from rerun.experimental import RrdReader

from memorax.algorithms import rtrrl_aaai as rtrrl
from memorax.assembly import BuildRequest, EnvironmentSpec, assemble
from memorax.observability import Reporter, RunMetadata
from memorax.observability.sinks import METRICS_FILENAME, MetricsSink, RerunSink
from memorax.parameters import expand
from memorax.runtime import Runtime, RuntimeConfig
from tests.support.environments import TinyContinuousEnv

pytestmark = [pytest.mark.integration, pytest.mark.service]

HORIZON = 3
TOTAL_STEPS = 50
EVALUATE_EVERY = 10
SAMPLE_STEPS = (10, 20, 30, 40, 50)


def parameters():
    return expand(
        rtrrl.PARAMETERS,
        {
            "torso.backbone.kind": "lru",
            "torso.backbone.lru.feature_dim": 4,
            "torso.backbone.lru.hidden_dim": 2,
            "torso.backbone.lru.differentiation.kind": "exact_rtrl",
            "torso.optimizer.kind": "adam",
            "torso.optimizer.adam.lr": 1e-3,
            "torso.grad_clip": 1.0,
            "torso.follow": 1.0,
            "actor.optimizer.kind": "adam",
            "actor.optimizer.adam.lr": 5e-4,
            "critic.optimizer.kind": "adam",
            "critic.optimizer.adam.lr": 5e-4,
            "actor.head.kind": "state_std",
            "critic.head.kind": "value",
            "normalization.observation.kind": "none",
            "normalization.reward.kind": "none",
            "meta_rl": False,
        },
    )


def tiny_environment(identifier, **options):
    del identifier, options
    environment = TinyContinuousEnv()
    return environment, environment.default_params


def sampled_runtime(*, trajectory_at_steps):
    record = (
        rtrrl.OBSERVATIONS.trajectory_fields if trajectory_at_steps else frozenset()
    )
    algorithm = assemble(
        rtrrl.RTRRL,
        BuildRequest(
            parameters=parameters(),
            environment=EnvironmentSpec(
                id="tiny", backend=None, observed=None, episode_length=HORIZON
            ),
            num_envs=1,
            record=record,
        ),
        environment_factory=tiny_environment,
    )
    return Runtime(
        algorithm=algorithm,
        config=RuntimeConfig(
            total_steps=TOTAL_STEPS,
            chunk_steps=HORIZON,
            max_episode_steps=HORIZON,
            evaluate_every_steps=EVALUATE_EVERY,
            evaluation_episodes=1,
            evaluation_chunk_steps=HORIZON,
            evaluation_seed=1000,
            num_envs=1,
            seed=0,
            trajectory_at_steps=trajectory_at_steps,
        ),
    )


def metadata():
    return RunMetadata(
        run_id="run-t0",
        experiment="experiment",
        launch_id="launch",
        trial=0,
        seed=0,
        role="tuning",
        entry="rtrrl",
        digest="local@sha256:" + "a" * 64,
    )


def reporter_for(directory, *, rerun: bool):
    scalars = [MetricsSink(directory / METRICS_FILENAME)]
    walks = [RerunSink(directory / "rerun", metadata=metadata())] if rerun else []
    return Reporter(scalar_sinks=scalars, trajectory_sinks=walks)


def read_metrics(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_a_fixed_run_writes_one_walk_and_one_evaluation_per_sample(tmp_path):
    with reporter_for(tmp_path, rerun=True) as reporter:
        sampled_runtime(trajectory_at_steps=SAMPLE_STEPS).run(reporter)

    written = sorted(path.name for path in (tmp_path / "rerun").glob("*.rrd"))
    assert written == [f"train-sample-{step:012d}.rrd" for step in SAMPLE_STEPS]

    records = read_metrics(tmp_path / METRICS_FILENAME)
    evaluations = [
        record for record in records if "eval/episode/return" in record["metrics"]
    ]
    assert [record["step"] for record in evaluations] == list(SAMPLE_STEPS)


def test_the_last_sample_is_carried_past_the_budget_to_its_ending(tmp_path):
    with reporter_for(tmp_path, rerun=True) as reporter:
        sampled_runtime(trajectory_at_steps=SAMPLE_STEPS).run(reporter)

    # The episode holding step 50 has two transitions inside the budget, so
    # exactly one of its three was taken after it.
    path = tmp_path / "rerun" / "train-sample-000000000050.rrd"
    summary = RrdReader(path).store().summary()
    assert "/episode/post_budget" in summary
    assert budget_side(path) == [0.0, 0.0, 1.0]

    # An earlier sample ended inside the budget and is marked accordingly.
    assert budget_side(tmp_path / "rerun" / "train-sample-000000000010.rrd") == [
        0.0,
        0.0,
        0.0,
    ]


def budget_side(path) -> list[float]:
    """The post-budget mark of every transition, in the order it was walked."""

    marks: dict[int, float] = {}
    for chunk in RrdReader(path).store().stream().to_chunks():
        if chunk.entity_path != "/episode/post_budget":
            continue
        batch = chunk.to_record_batch()
        steps = batch.column("episode_step").to_pylist()
        for step, cell in zip(steps, batch.column("Tensor:data").to_pylist()):
            marks[int(step)] = float(cell[0]["buffer"][0])
    return [marks[step] for step in sorted(marks)]


def test_a_run_without_sample_points_reports_scalars_and_writes_no_walk(tmp_path):
    with reporter_for(tmp_path, rerun=False) as reporter:
        sampled_runtime(trajectory_at_steps=()).run(reporter)

    records = read_metrics(tmp_path / METRICS_FILENAME)
    assert any("eval/episode/return" in record["metrics"] for record in records)
    assert any("train/episode/return" in record["metrics"] for record in records)
    assert not (tmp_path / "rerun").exists()
