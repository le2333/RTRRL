"""R2.2's shape: one ensemble job, four SGD trials, a different ``grad_clip``.

The study this reproduces is stage one of R2.2 -- masked HalfCheetah, SGD at
the torso and both heads, ``torso.grad_clip`` drawn from
``[0.0, 0.3, 1.0, 3.0, 10.0]``, one Batch job carrying four vmapped
configurations. Every trial in it failed before training, during ensemble
program construction, because the chain a rate is assembled from decided its
own length from the clip and a swept clip has no length to give. No checkpoint
was produced, so the failure was invisible in the metrics and visible only in
the job log.

What is exercised here is the whole path a job takes below the worker: the
entry's agreement check, its reading of which leaves the group varies, the
build repeated inside ``vmap`` once per member, training to a boundary, and the
evaluation at it. The reporters are the one thing replaced -- publishing to S3
and Aim is a facility this says nothing about, and the smoke tests under
``tests/integration/worker`` already own that boundary.

The budget is a test's, not the study's: eight environment steps, a short
episode, and a torso small enough to compile quickly. The *shape* is R2.2's,
and the shape is what broke.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import pytest

import entries._ensemble as ensemble
from deployment.contract import CONTRACT_VERSION
from entries._contract import RunSpec
from entries.rtrrl import build_request, runtime_config
from memorax.algorithms.rtrrl_aaai import PARAMETERS, RTRRL
from memorax.parameters import expand
from tests.support.fakes import EpisodeRecorder

pytestmark = pytest.mark.integration

# Four of the five values the study searches, one per vmapped configuration.
# Zero is the one that has to be there: it is the value that means "no clip",
# and the reason the chain could not be assembled from a number it could not
# read.
CLIPS = (0.0, 0.3, 1.0, 3.0)

# Half of HalfCheetah's readings withheld, which is the masking R2.2 runs
# under. Which half is the study's to choose; what matters here is that the
# observation is partial, because that is what RTRRL is being asked about.
OBSERVED = [0, 2, 4, 6, 8, 10, 12, 14, 16]

TOTAL_STEPS = 8
EPISODE_LENGTH = 8


def parameters(clip: float) -> dict:
    """One trial's parameters: R2.2's SGD arm at one value of the clip."""

    return expand(
        PARAMETERS,
        {
            "torso.backbone.kind": "lru",
            "torso.backbone.lru.feature_dim": 4,
            "torso.backbone.lru.hidden_dim": 4,
            "torso.backbone.lru.differentiation.kind": "exact_rtrl",
            "torso.optimizer.kind": "sgd",
            "torso.optimizer.sgd.lr": 1e-3,
            "torso.grad_clip": clip,
            "torso.follow": 1.0,
            "actor.optimizer.kind": "sgd",
            "actor.optimizer.sgd.lr": 1e-3,
            "actor.head.kind": "state_std",
            "critic.optimizer.kind": "sgd",
            "critic.optimizer.sgd.lr": 1e-3,
            "critic.head.kind": "value",
            "normalization.observation.kind": "none",
            "normalization.reward.kind": "none",
            "gamma": 0.99,
            "lambda_pi": 0.9,
            "lambda_v": 0.9,
            "lambda_rnn": 0.9,
            "eta_pi": 1.0,
            "eta_f": 1.0,
            "entropy_rate": 1e-5,
            "meta_rl": False,
        },
    )


def member(trial: int, clip: float) -> RunSpec:
    """One run document, as the control plane emits a swept round's members.

    A trial is one combination of the swept values, so the trial number and the
    clip move together and the seed is the same in all four -- which is what
    makes these four members rather than four repetitions.
    """

    return RunSpec.model_validate(
        {
            "contract": CONTRACT_VERSION,
            "identity": {
                "run_id": f"r2-2-halfcheetah-sgd-t{trial}-s0",
                "experiment": "r2-2-halfcheetah-sgd-stage1",
                "launch_id": "20260831-000000",
                "trial": trial,
                "seed": 0,
                "role": "tuning",
                "digest": "local@sha256:" + "a" * 64,
            },
            "entry": "rtrrl_ensemble",
            "artifacts": {"root": f"s3://bucket/r2-2/t{trial}-seed-0"},
            "algorithm": {
                "environment": {
                    "id": "brax::halfcheetah",
                    "backend": "spring",
                    "observed": OBSERVED,
                    "episode_length": EPISODE_LENGTH,
                },
                "num_envs": 1,
                "parameters": parameters(clip),
            },
            "training": {
                "seed": 0,
                "total_steps": TOTAL_STEPS,
                "chunk_steps": TOTAL_STEPS,
            },
            "evaluation": {
                "every_steps": TOTAL_STEPS,
                "episodes": 1,
                "chunk_steps": EPISODE_LENGTH,
                "seed": 1000,
            },
            "logging": {"aim": {"url": "aim://127.0.0.1:1"}},
        }
    )


@pytest.fixture
def recorders(monkeypatch: pytest.MonkeyPatch) -> list[EpisodeRecorder]:
    """The four destinations, in place of the S3 and Aim sinks."""

    kept: list[EpisodeRecorder] = []

    @contextmanager
    def build_reporter(spec: RunSpec, scratch: Path):
        del spec, scratch
        recorder = EpisodeRecorder()
        kept.append(recorder)
        yield recorder

    monkeypatch.setattr(ensemble, "build_reporter", build_reporter)
    return kept


def test_the_four_trial_sgd_round_reaches_its_first_evaluation(
    recorders: list[EpisodeRecorder], tmp_path: Path
) -> None:
    members = tuple((member(trial, clip), tmp_path) for trial, clip in enumerate(CLIPS))

    ensemble.run_group(
        RTRRL,
        members,
        build_request=build_request,
        runtime_config=runtime_config,
        declared=PARAMETERS,
    )

    assert len(recorders) == len(CLIPS)
    for clip, recorder in zip(CLIPS, recorders):
        # The evaluation at the boundary is the checkpoint the study never
        # produced. One episode was asked for and one has to come back, under
        # the phase that says it was measured rather than trained.
        measured = recorder.of("eval")
        assert len(measured) == 1, f"the trial at clip={clip} was never evaluated"


def test_the_round_varies_the_clip_and_nothing_else(tmp_path: Path) -> None:
    """The sweep the entry reads out of the group is the one that was written.

    Stated separately because it is what makes the run above the R2.2 shape
    rather than four copies of one configuration: had the members agreed about
    the clip, the entry would have built one graph outside the map and the
    tracer this issue is about would never have been created.
    """

    members = tuple((member(trial, clip), tmp_path) for trial, clip in enumerate(CLIPS))

    swept = ensemble.swept_parameters(members, PARAMETERS)

    assert swept == {"torso.grad_clip": list(CLIPS)}
