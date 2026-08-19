"""Schedule a closed algorithm and report the complete episodes it produces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import jax

from memorax.runtime.checkpoint import Checkpoint, CheckpointDirectory
from memorax.runtime.episode import Episode, SampledTrajectory
from memorax.runtime.rollout import complete_episodes
from memorax.runtime.tracker import EpisodeTracker, TrackingResult

from .program import BuiltAlgorithm


class Destination(Protocol):
    """All of the reporter Runtime touches."""

    def log_episode(self, episode: Episode) -> None: ...
    def log_trajectory(self, trajectory: SampledTrajectory) -> None: ...


def evaluation_boundaries(
    *, total_steps: int, every_steps: int, num_envs: int
) -> range:
    """Return every step count at which the policy is measured.

    A boundary has to fall between two vectorized rows and the budget has to
    end on one, or the last interval would be a different length from the rest
    and the evaluations would no longer be comparable.
    """

    if every_steps % num_envs:
        raise ValueError(f"every_steps {every_steps} is not {num_envs} streams' worth")
    if total_steps % every_steps:
        raise ValueError(
            f"total_steps {total_steps} is not whole intervals of {every_steps}"
        )
    return range(every_steps, total_steps + 1, every_steps)


def checkpointed(*, every_steps: int, evaluate_every_steps: int) -> int:
    """How often a run files its whole state, checked against when it is measured.

    A checkpoint that did not land on an evaluation boundary would be a moment
    the run has no measurement of, and the branch taken from it could not be
    dated against the curve that decided to take it. So the interval is a whole
    number of evaluation intervals or it is refused; there is no rounding.
    """

    if every_steps and every_steps % evaluate_every_steps:
        raise ValueError(
            f"checkpoint every_steps {every_steps} is not whole intervals of the "
            f"evaluation's {evaluate_every_steps}; a checkpoint that is not also "
            "an evaluation cannot be placed on the curve"
        )
    return every_steps


@dataclass(frozen=True)
class RuntimeConfig:
    """Only the scheduling values Runtime consumes.

    Three schedules that used to be one. ``chunk_steps`` is how much a single
    training call may hold, which is what a run costs in memory.
    ``evaluate_every_steps`` is when the policy is measured. Neither derives
    from the other, and neither derives from ``max_episode_steps``, which is
    the environment's own limit and only a guard: an episode longer than that
    is refused rather than reported.

    ``trajectory_at_steps`` names the environment steps whose training episode
    is to be kept whole.

    ``checkpoint_every_steps`` is how often the run's whole state is filed, and
    zero is a run that files none. It is a fourth schedule and not a fifth: it
    is stated in evaluation intervals, so that every checkpoint is a boundary
    the run also measured.
    """

    total_steps: int
    chunk_steps: int
    max_episode_steps: int
    evaluate_every_steps: int
    rollout_steps: int
    num_envs: int
    seed: int
    trajectory_at_steps: tuple[int, ...] = ()
    checkpoint_every_steps: int = 0


@dataclass(frozen=True)
class Runtime:
    """One run: a closed algorithm, an execution budget, and episode output."""

    algorithm: BuiltAlgorithm
    config: RuntimeConfig
    # Where whole-state checkpoints are filed, when the run files any. A run
    # that does not is the ordinary case and passes nothing.
    checkpoints: CheckpointDirectory | None = None

    def run(self, reporter: Destination, *, resume: Checkpoint | None = None) -> None:
        """Run the algorithm to its budget, reporting every complete episode.

        ``resume`` is a checkpoint this run continues from: its state and its
        PRNG stand in for the ones a fresh run would have built, and the budget
        is measured on the same axis the parent used, so a branch's evaluation
        at 750k is comparable with its parent's. The evaluation boundaries
        already behind the checkpoint are not re-run -- they happened, and the
        parent reported them.

        What a resumed run does not inherit is the episode record: the tracker
        starts empty, so the episode open across the checkpoint is reported by
        neither run rather than by both with different halves. That costs one
        episode per stream and keeps every reported return a whole one.
        """

        config = self.config
        checkpoint_every = checkpointed(
            every_steps=config.checkpoint_every_steps,
            evaluate_every_steps=config.evaluate_every_steps,
        )
        program = self.algorithm.program
        train = jax.jit(program.train, static_argnums=2)
        evaluate = jax.jit(program.evaluate, static_argnums=2)
        interact = jax.jit(program.interact)

        if resume is None:
            key = jax.random.key(config.seed)
            key, init_key = jax.random.split(key)
            state = jax.jit(program.init)(init_key)
            trained_steps = 0
        else:
            key, state = resume.key, resume.state
            trained_steps = resume.env_steps

        if trained_steps % config.evaluate_every_steps:
            raise ValueError(
                f"the checkpoint is at {trained_steps} steps, which is not an "
                f"evaluation boundary of {config.evaluate_every_steps}; a branch "
                "taken from between two measurements cannot be placed on the curve"
            )
        boundaries = [
            boundary
            for boundary in evaluation_boundaries(
                total_steps=config.total_steps,
                every_steps=config.evaluate_every_steps,
                num_envs=config.num_envs,
            )
            if boundary > trained_steps
        ]

        tracker = EpisodeTracker(
            observations=self.algorithm.observations,
            num_envs=config.num_envs,
            max_episode_steps=config.max_episode_steps,
            sample_steps=tuple(config.trajectory_at_steps),
        )
        eval_number = 1

        for boundary in boundaries:
            while trained_steps < boundary:
                # The last call before a boundary is short if the interval is
                # not a whole number of chunks. Nothing else varies its size.
                steps = min(config.chunk_steps, boundary - trained_steps)
                key, chunk_key = jax.random.split(key)
                state, chunk = train(chunk_key, state, steps)
                self._publish(
                    reporter, tracker.consume(chunk, start_env_steps=trained_steps)
                )
                trained_steps += steps

            if config.rollout_steps:
                key, eval_key = jax.random.split(key)
                observations = evaluate(eval_key, state, config.rollout_steps)
                eval_number = self._report(
                    reporter,
                    observations,
                    phase="eval",
                    start_env_steps=boundary,
                    stride=0,
                    first_number=eval_number,
                )

            # Filed after the evaluation's own draw, so that the key stored is
            # the one the next training chunk would have used. A branch taken
            # from here therefore draws exactly what the parent drew next, and
            # a same-rule branch is the parent's own continuation rather than
            # merely a run that started from the same numbers.
            if (
                checkpoint_every
                and self.checkpoints
                and not boundary % checkpoint_every
            ):
                self.checkpoints.save(env_steps=boundary, state=state, key=key)

        key, tail_key = jax.random.split(key)
        self._finish_samples(reporter, tracker, interact, tail_key, state)

    def _finish_samples(
        self,
        reporter: Destination,
        tracker: EpisodeTracker,
        interact,
        key,
        state,
    ) -> None:
        """Carry the last requested episode to its end without learning from it.

        The budget is spent, so these transitions are taken from a copy of the
        final state and marked post-budget.  Nothing here reaches the training
        state, the scalar statistics, or the step count.
        """

        continued = state
        continued_steps = self.config.total_steps
        for _ in range(self.config.max_episode_steps):
            if not tracker.pending_sample_steps:
                return
            key, step_key = jax.random.split(key)
            continued, metrics = interact(step_key, continued)
            self._publish(
                reporter,
                tracker.consume(
                    jax.tree.map(lambda leaf: leaf[None], metrics),
                    start_env_steps=continued_steps,
                    post_budget=True,
                    report_completed=False,
                ),
            )
            continued_steps += self.config.num_envs

        if tracker.pending_sample_steps:
            raise ValueError("sampled episode exceeded maximum episode length")

    def _publish(self, reporter: Destination, result: TrackingResult) -> None:
        for episode in result.completed:
            reporter.log_episode(episode)
        for trajectory in result.sampled:
            reporter.log_trajectory(trajectory)

    def _report(self, reporter: Destination, chunk, **cut) -> int:
        number = cut["first_number"]
        for episode in complete_episodes(
            chunk,
            observations=self.algorithm.observations,
            num_envs=self.config.num_envs,
            **cut,
        ):
            reporter.log_episode(episode)
            number = episode.number + 1
        return number
