"""The complete run document validated at the Entry composition boundary."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from deployment.contract import ContractVersion
from memorax.parameters import Scalar


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class RunIdentity(_Frozen):
    """Which run this is, in terms that survive into the artifact.

    ``trial`` names the configuration and ``seed`` names the repetition of it,
    because a configuration is run on a list of seeds and the two together are
    what is unique. ``role`` says which protocol the run belongs to: a tuning
    run chose the configuration, a formal run measures a configuration that
    was already chosen, and only the second may be reported. Both are carried
    here rather than inferred later, so that a result found on its own can
    still say what it is allowed to be used for.
    """

    run_id: str
    experiment: str
    launch_id: str
    trial: int
    seed: int
    role: Literal["tuning", "formal"]
    digest: str

    @model_validator(mode="after")
    def _usable(self) -> "RunIdentity":
        if self.seed < 0:
            raise ValueError("seed must not be negative")
        return self


class Artifacts(_Frozen):
    root: str


class EnvironmentSpec(_Frozen):
    """What a run says about the environment its graph is built against.

    ``backend`` is null wherever the namespace has only one implementation to
    choose between. Brax names a physics backend; Gymnax has none, and saying
    so is not the same as omitting a field that means something.

    ``kwargs`` is what the environment is constructed with, as opposed to what
    the deployment does to it afterwards. Some tasks are defined by a
    constructor argument rather than by their name -- an UmbrellaChain of
    length 10 and one of length 40 are different tasks, and bsuite's own sweep
    is over exactly that number -- and without this a run document could name
    the family but never the member. It is passed through untouched: every
    namespace adapter already forwards ``**kwargs`` to the library it wraps, so
    what is legal here is whatever that library's constructor accepts, and a
    key it does not is that library's error to raise.
    """

    id: str
    backend: str | None = None
    observed: tuple[int, ...] | None = None
    episode_length: int = 1000
    kwargs: dict[str, Scalar] = {}

    @model_validator(mode="after")
    def _usable(self) -> "EnvironmentSpec":
        if self.episode_length < 1:
            raise ValueError("episode_length must be positive")
        if self.observed is not None and (
            not self.observed
            or len(set(self.observed)) != len(self.observed)
            or any(index < 0 for index in self.observed)
        ):
            raise ValueError("observed must contain unique non-negative indices")
        return self


class AlgorithmSpec(_Frozen):
    environment: EnvironmentSpec
    num_envs: int
    parameters: dict[str, Scalar]

    @model_validator(mode="after")
    def _usable(self) -> "AlgorithmSpec":
        if self.num_envs < 1:
            raise ValueError("num_envs must be positive")
        return self


class TrainingSpec(_Frozen):
    """The budget and the shape of the run, and nothing about reporting.

    ``chunk_steps`` is how much one training call may hold, which is the
    memory a run costs. It is deliberately unrelated to the environment's
    episode limit: how long an episode runs is decided while the run is going
    and changes as the policy improves, so sizing a buffer from it would tie a
    memory budget to a number that has nothing to do with memory.

    ``snapshot_every_steps`` is how often the run is written somewhere that
    outlives the machine, so that a job which was preempted, timed out or lost
    its host is continued rather than started again. Zero writes nothing down,
    which is the right answer for a run short enough that losing it costs less
    than carrying it: every snapshot is the whole state of the run over the
    network, and taking one every interval of a long run is a real share of
    what the run costs.
    """

    seed: int
    total_steps: int
    chunk_steps: int
    snapshot_every_steps: int = 0

    @model_validator(mode="after")
    def _usable(self) -> "TrainingSpec":
        if self.seed < 0:
            raise ValueError("seed must not be negative")
        if self.total_steps < 1 or self.chunk_steps < 1:
            raise ValueError("training step budgets must be positive")
        if self.snapshot_every_steps < 0:
            raise ValueError("snapshot_every_steps must not be negative")
        return self


class EvaluationSpec(_Frozen):
    """How often the policy is measured, on how much, and under whose keys.

    ``episodes`` is the measurement: a checkpoint is scored on exactly this
    many complete episodes, and zero measures nothing. It is not a step budget
    -- how long those episodes run is what the policy decides, and asking for
    steps instead is what makes the count vary with the task and the policy,
    which is the thing a formal protocol may not let vary.

    ``chunk_steps`` is only how much of that rollout one call may hold, the
    same kind of memory bound as the training block's, and it says nothing
    about how much is run.

    ``seed`` opens the evaluation's own key stream. Declaring it is what lets
    two methods be measured on the same evaluation episodes, and what keeps
    the training stream from moving because a measurement was taken.
    """

    every_steps: int
    episodes: int
    chunk_steps: int
    seed: int

    @model_validator(mode="after")
    def _usable(self) -> "EvaluationSpec":
        if self.every_steps < 1:
            raise ValueError("evaluation every_steps must be positive")
        if self.episodes < 0:
            raise ValueError("evaluation episodes must not be negative")
        if self.chunk_steps < 1:
            raise ValueError("evaluation chunk_steps must be positive")
        if self.seed < 0:
            raise ValueError("evaluation seed must not be negative")
        return self


class AimStepSpec(_Frozen):
    """What a reading looks like at a typical moment."""

    every_steps: int

    @model_validator(mode="after")
    def _usable(self) -> "AimStepSpec":
        if self.every_steps < 1:
            raise ValueError("aim training step every_steps must be positive")
        return self


class AimEpisodeSpec(_Frozen):
    """What a typical episode's statistic is, counted in episodes."""

    every_episodes: int

    @model_validator(mode="after")
    def _usable(self) -> "AimEpisodeSpec":
        if self.every_episodes < 1:
            raise ValueError("aim training episode every_episodes must be positive")
        return self


class AimWindowSpec(_Frozen):
    """What every episode in a stretch averaged.

    ``length_steps`` is null when the stretch is the whole interval, which
    tiles the axis and uses every episode.
    """

    every_steps: int
    length_steps: int | None = None

    @model_validator(mode="after")
    def _usable(self) -> "AimWindowSpec":
        if self.every_steps < 1:
            raise ValueError("aim training window every_steps must be positive")
        if self.length_steps is not None and not (
            1 <= self.length_steps <= self.every_steps
        ):
            raise ValueError(
                "aim training window length_steps must be positive and no longer "
                "than every_steps"
            )
        return self


class AimTrainingSpec(_Frozen):
    """Which scopes of training reach the dashboard, each in its own unit.

    A scope's interval cannot be stated in another scope's unit, so the
    combination that sampled an episode by a step mark -- and thereby favoured
    the long episodes, which are worth more -- has no spelling here.
    """

    step: AimStepSpec | None = None
    episode: AimEpisodeSpec | None = None
    window: AimWindowSpec | None = None

    @model_validator(mode="after")
    def _usable(self) -> "AimTrainingSpec":
        if self.step is None and self.episode is None and self.window is None:
            raise ValueError(
                "aim training names no scope; omit the block to record "
                "evaluation only"
            )
        return self


class AimSpec(_Frozen):
    """Evaluation always reaches Aim; training reaches it only if asked.

    A run ends one episode every few dozen steps before its policy is any
    good, so recording every one of them is tens of millions of points nobody
    can read. The complete record is the metrics artifact, which is not
    optional, not configurable, and keeps every episode's own reduction.
    """

    url: str
    training: AimTrainingSpec | None = None


class RerunSpec(_Frozen):
    log_every_steps: int

    @model_validator(mode="after")
    def _usable(self) -> "RerunSpec":
        if self.log_every_steps < 1:
            raise ValueError("rerun log_every_steps must be positive")
        return self


class LoggingSpec(_Frozen):
    aim: AimSpec
    rerun: RerunSpec | None = None


class RunSpec(_Frozen):
    contract: ContractVersion
    identity: RunIdentity
    entry: str
    artifacts: Artifacts
    algorithm: AlgorithmSpec
    training: TrainingSpec
    evaluation: EvaluationSpec
    logging: LoggingSpec

    @model_validator(mode="after")
    def _graph_width_matches_schedule(self) -> "RunSpec":
        streams = self.algorithm.num_envs
        if self.training.chunk_steps % streams:
            raise ValueError("chunk_steps must contain whole environment steps")
        if self.evaluation.every_steps % streams:
            raise ValueError(
                "evaluation every_steps must contain whole environment steps"
            )
        if self.evaluation.chunk_steps % streams:
            raise ValueError(
                "evaluation chunk_steps must contain whole environment steps"
            )
        if self.training.total_steps % self.evaluation.every_steps:
            raise ValueError("total_steps must consist of whole evaluation intervals")
        # A snapshot is taken at an evaluation boundary, which is the only
        # moment a run is quiet enough to be restarted from. An interval that
        # is not whole boundaries names moments that will never arrive, and a
        # run configured that way would write nothing down while appearing to.
        snapshot = self.training.snapshot_every_steps
        if snapshot and snapshot % self.evaluation.every_steps:
            raise ValueError(
                "snapshot_every_steps must consist of whole evaluation intervals"
            )
        return self
