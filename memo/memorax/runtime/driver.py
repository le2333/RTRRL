"""Schedule a closed algorithm and report the complete episodes it produces."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol

import jax

from memorax.runtime.episode import Episode, SampledTrajectory
from memorax.runtime.snapshot import RunSnapshot, SnapshotStore
from memorax.runtime.snapshot import resume as resume_destination
from memorax.runtime.snapshot import suspend as suspend_destination
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


def evaluation_quota(*, episodes: int, num_envs: int) -> tuple[int, ...]:
    """How many episodes each stream contributes to one checkpoint's score.

    A checkpoint is scored on an exact number of episodes, and which ones
    those are must not depend on which finished first: an episode that ends
    early is not thereby more representative, and taking the first to arrive
    would score short episodes preferentially. So a slot is named before the
    rollout runs -- stream ``i``'s ``j``-th episode fills slot ``j * n + i`` --
    and the scored episodes are the ones whose slot number is below the
    requested count. The rollout then runs until every named slot is filled.

    This asks nothing of ``episodes`` and ``num_envs`` beyond both being
    positive: the streams that hold a lower index simply contribute one more.
    """

    return tuple(len(range(stream, episodes, num_envs)) for stream in range(num_envs))


@dataclass(frozen=True)
class RuntimeConfig:
    """Only the scheduling values Runtime consumes.

    Four schedules that used to be two. ``chunk_steps`` is how much a single
    training call may hold, which is what a run costs in memory.
    ``evaluate_every_steps`` is when the policy is measured. Neither derives
    from the other, and neither derives from ``max_episode_steps``, which is
    the environment's own limit and here does double duty: it refuses an
    episode longer than itself, and it is what bounds an evaluation that is
    counted in episodes rather than in steps.

    ``evaluation_episodes`` is how many complete episodes each checkpoint is
    scored on, exactly; zero measures nothing. ``evaluation_chunk_steps`` is
    the memory bound on one evaluation call, the same kind of number as
    ``chunk_steps`` and unrelated to how long the episodes turn out to be.

    ``evaluation_seed`` opens a key stream of its own. Evaluation must not
    move the training one, or whether a run was measured would change what it
    then learned; deriving each checkpoint's keys from this seed and the
    boundary also makes the evaluation of two runs reproducible on its own
    terms, and pairable across methods that declare the same seed.

    ``trajectory_at_steps`` names the environment steps whose training episode
    is to be kept whole.

    ``snapshot_every_evaluations`` is how often the run is written down so
    that a second process can carry it on, counted in the boundaries it is
    taken at; zero never writes it down. Counted in evaluations and not in
    steps because a boundary is the only moment a run is quiet enough to be
    restarted from -- between two of them a chunk is in flight -- and an
    interval stated in steps can name a moment that will never arrive, which
    is a run that writes nothing down while appearing to.
    """

    total_steps: int
    chunk_steps: int
    max_episode_steps: int
    evaluate_every_steps: int
    evaluation_episodes: int
    evaluation_chunk_steps: int
    evaluation_seed: int
    num_envs: int
    seed: int
    trajectory_at_steps: tuple[int, ...] = ()
    snapshot_every_evaluations: int = 0


@dataclass(frozen=True)
class Runtime:
    """One run: a closed algorithm, an execution budget, and episode output.

    ``snapshots`` is where the run is written down and where a second process
    looks for it. With none the run is what it always was: one process, from
    step zero to the budget, and an interruption costs all of it.
    """

    algorithm: BuiltAlgorithm
    config: RuntimeConfig
    snapshots: SnapshotStore | None = None

    def run(self, reporter: Destination) -> None:
        """Run the algorithm to its budget, reporting every complete episode."""

        config = self.config
        boundaries = evaluation_boundaries(
            total_steps=config.total_steps,
            every_steps=config.evaluate_every_steps,
            num_envs=config.num_envs,
        )
        every = config.snapshot_every_evaluations
        if every < 0:
            raise ValueError("snapshot_every_evaluations must not be negative")
        if every and self.snapshots is None:
            raise ValueError(
                f"this run asks to be written down every {every} evaluations "
                "and was given nowhere to write it; a run that says it can be "
                "resumed and cannot is worse than one that never claimed to"
            )

        program = self.algorithm.program
        train = jax.jit(program.train, static_argnums=2)
        open_evaluation = jax.jit(program.open_evaluation)
        evaluate = jax.jit(program.evaluate, static_argnums=2)
        interact = jax.jit(program.interact)

        # Never split from ``key``: the training stream must run identically
        # whether or not the policy was measured along the way.
        evaluation_key = jax.random.key(config.evaluation_seed)

        tracker = EpisodeTracker(
            observations=self.algorithm.observations,
            num_envs=config.num_envs,
            max_episode_steps=config.max_episode_steps,
            sample_steps=tuple(config.trajectory_at_steps),
        )
        key, state, trained_steps, eval_number = self._begin(program, reporter, tracker)
        # Where this process took the run over, so the boundary it resumed at
        # is not immediately written back out. Zero for a run that started at
        # the beginning, which names no boundary.
        resumed_at = trained_steps

        for measurement, boundary in enumerate(boundaries, start=1):
            # An interval whose training and whose evaluation a snapshot both
            # already hold. The boundary the run was resumed *at* is not one
            # of them: its training is in the snapshot and its evaluation is
            # not, so the loop below runs no chunks and the measurement is
            # taken again.
            if boundary < trained_steps:
                continue
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

            # Before the measurement rather than after it. Most of what a
            # boundary costs is its evaluation, and a snapshot taken after one
            # means a machine that dies measuring loses the whole training
            # interval behind it as well. Taken before, the evaluation is the
            # only thing a resume redoes -- and redoing it is exact, because
            # ``_measure`` reads the state without touching it and draws its
            # keys from the evaluation seed and the boundary, so one boundary
            # measured twice reports the same episodes both times.
            if every and not measurement % every and boundary != resumed_at:
                self._suspend(
                    reporter,
                    tracker,
                    state=state,
                    key=key,
                    step=boundary,
                    eval_number=eval_number,
                )
            if config.evaluation_episodes:
                eval_number = self._measure(
                    reporter,
                    open_evaluation,
                    evaluate,
                    jax.random.fold_in(evaluation_key, boundary),
                    state,
                    boundary=boundary,
                    first_number=eval_number,
                )

        key, tail_key = jax.random.split(key)
        self._finish_samples(reporter, tracker, interact, tail_key, state)

    def _begin(self, program, reporter: Destination, tracker: EpisodeTracker):
        """Start the run, or take back the one that was interrupted.

        Resuming reaches past ``program.init`` entirely: the key stream is the
        one the stopped process had arrived at rather than one rebuilt from
        the seed, and the tracker and the reporter are handed back what they
        were in the middle of. Rebuilding any of the three from the run
        document would replay transitions that have already been taken.
        """

        resumed = None if self.snapshots is None else self.snapshots.latest()
        if resumed is None:
            key = jax.random.key(self.config.seed)
            key, init_key = jax.random.split(key)
            return key, jax.jit(program.init)(init_key), 0, 1

        tracker.resume(_single(resumed.trackers, "trackers"))
        resume_destination(reporter, _single(resumed.destinations, "destinations"))
        return (
            resumed.key(),
            resumed.algorithm_state(),
            resumed.step,
            resumed.eval_number,
        )

    def _suspend(
        self,
        reporter: Destination,
        tracker: EpisodeTracker,
        *,
        state,
        key,
        step: int,
        eval_number: int,
    ) -> None:
        """Write the run down at a boundary, where it is briefly quiet."""

        if self.snapshots is None:  # pragma: no cover - the caller checks first
            return
        self.snapshots.save(
            RunSnapshot.taken(
                step=step,
                eval_number=eval_number,
                state=state,
                key=key,
                trackers=(tracker.suspend(),),
                destinations=(suspend_destination(reporter),),
            )
        )

    def _measure(
        self,
        reporter: Destination,
        open_evaluation,
        evaluate,
        key,
        state,
        *,
        boundary: int,
        first_number: int,
    ) -> int:
        """Score one checkpoint on exactly the episodes it was asked for.

        The rollout is opened once and advanced in bounded calls until every
        named slot holds a complete episode, which is why it cannot be a
        single scan: how many steps that takes is what the policy decides.
        Nothing here reaches the training state -- ``state`` is only ever read,
        and what the algorithm hands back is the evaluation's own.
        """

        config = self.config
        quota = evaluation_quota(
            episodes=config.evaluation_episodes, num_envs=config.num_envs
        )
        tracker = EpisodeTracker(
            observations=self.algorithm.observations,
            num_envs=config.num_envs,
            max_episode_steps=config.max_episode_steps,
            phase="eval",
            stride=0,
            require_series=False,
        )
        collected: list[list[Episode]] = [[] for _ in quota]

        # Every stream ends an episode at least once per ``max_episode_steps``
        # rows, because that is where the environment truncates, so this many
        # steps cannot fail to fill the quota. Reaching it means the episodes
        # are not ending, and a measurement that never finishes is worse than
        # one that says so.
        budget = max(quota) * config.max_episode_steps * config.num_envs
        key, open_key = jax.random.split(key)
        evaluation = open_evaluation(open_key, state)
        spent = 0
        while any(len(found) < wanted for found, wanted in zip(collected, quota)):
            if spent >= budget:
                raise ValueError(
                    f"evaluation at {boundary} environment steps did not complete "
                    f"{config.evaluation_episodes} episodes within {budget} steps"
                )
            steps = min(config.evaluation_chunk_steps, budget - spent)
            key, chunk_key = jax.random.split(key)
            evaluation, chunk = evaluate(chunk_key, evaluation, steps)
            for episode in tracker.consume(chunk, start_env_steps=boundary).completed:
                collected[episode.stream].append(episode)
            spent += steps

        for slot, episode in sorted(self._slots(collected, quota)):
            reporter.log_episode(replace(episode, number=first_number + slot))
        return first_number + config.evaluation_episodes

    def _slots(self, collected, quota) -> list[tuple[int, Episode]]:
        """The scored episodes, each under the slot number that named it.

        A stream that ran ahead of its quota has episodes here that no slot
        asked for. They are not scored: the checkpoint was defined as an exact
        number of episodes, and an extra one is as much a change to that
        number as a missing one.
        """

        num_envs = self.config.num_envs
        return [
            (index * num_envs + stream, episode)
            for stream, wanted in enumerate(quota)
            for index, episode in enumerate(collected[stream][:wanted])
        ]

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


def _single(values: tuple, what: str):
    """The one member's worth of a snapshot this driver knows how to take.

    A snapshot written by the ensemble scheduler holds one entry per member,
    and handing a member's open episodes to a single-member run would report
    another run's episodes under this one's name.
    """

    if len(values) != 1:
        raise ValueError(
            f"the snapshot holds {len(values)} members' {what} and this run has one"
        )
    return values[0]
