"""Schedule many members of one algorithm as a single graph.

A sweep's members differ only in what they were seeded with, so their graph is
one graph and a device can be filled by the sweep rather than by enlarging any
single run. This is :mod:`~memorax.runtime.driver` with a member axis: the same
five arrows, the same episode tracking, the same reported episodes, mapped.

Seeds only. The arrows are vmapped over their key argument and the algorithm is
built exactly once, because members that differ only in a seed share their
graph down to the constant. Members that differ in a *value* -- a learning rate,
a discount -- need the build itself inside the map, which is a larger change and
is not this one. Nothing here forecloses it: the member axis and the per-member
reporting it needs are the same either way.

Members are not streams. ``num_envs`` streams inside a member share a policy and
a set of parameters; two members share nothing but their shapes, and folding one
axis into the other would let an episode from one member be scored against
another's quota. So each member keeps its own tracker and its own destination,
and the only place they meet is the compiled graph.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import jax

from memorax.runtime.driver import (
    Destination,
    RuntimeConfig,
    evaluation_boundaries,
    evaluation_quota,
)
from memorax.runtime.episode import Episode
from memorax.runtime.program import BuiltAlgorithm
from memorax.runtime.tracker import EpisodeTracker, TrackingResult


def _split(keys):
    """Split every member's key, keeping the member axis outermost."""

    return jax.vmap(jax.random.split, out_axes=1)(keys)


@dataclass(frozen=True)
class EnsembleRuntime:
    """One round: a closed algorithm, a budget, and a seed per member.

    ``seeds`` replaces ``RuntimeConfig.seed``, which is ignored: a round has no
    single training seed, and leaving the field to be read by accident is worth
    less than the field's absence would cost. ``evaluation_seed`` is *not*
    replaced -- every member is measured on the same evaluation stream, which is
    what makes their scores paired rather than merely comparable.
    """

    algorithm: BuiltAlgorithm
    config: RuntimeConfig
    seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.seeds:
            raise ValueError("an ensemble needs at least one member")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError(f"the seeds repeat: {self.seeds}")

    @property
    def members(self) -> int:
        return len(self.seeds)

    def run(self, reporters: Sequence[Destination]) -> None:
        """Run every member to the budget, reporting each to its own destination."""

        if len(reporters) != self.members:
            raise ValueError(
                f"{len(reporters)} destinations for {self.members} members"
            )

        config = self.config
        boundaries = evaluation_boundaries(
            total_steps=config.total_steps,
            every_steps=config.evaluate_every_steps,
            num_envs=config.num_envs,
        )

        program = self.algorithm.program
        # `steps` stays static, as it is in the driver: it is a scan length.
        # Every other argument carries the member axis.
        train = jax.jit(jax.vmap(program.train, in_axes=(0, 0, None)), static_argnums=2)
        open_evaluation = jax.jit(jax.vmap(program.open_evaluation))
        evaluate = jax.jit(
            jax.vmap(program.evaluate, in_axes=(0, 0, None)), static_argnums=2
        )

        keys = jax.numpy.stack([jax.random.key(seed) for seed in self.seeds])
        keys, init_keys = _split(keys)
        state = jax.jit(jax.vmap(program.init))(init_keys)
        # Never split from `keys`: whether a member was measured must not change
        # what it then learned, exactly as in the single-member driver.
        evaluation_key = jax.random.key(config.evaluation_seed)

        trackers = [self._tracker() for _ in range(self.members)]
        trained_steps = 0
        eval_number = 1

        for boundary in boundaries:
            while trained_steps < boundary:
                steps = min(config.chunk_steps, boundary - trained_steps)
                keys, chunk_keys = _split(keys)
                state, chunk = train(chunk_keys, state, steps)
                for member, (tracker, reporter) in enumerate(
                    zip(trackers, reporters)
                ):
                    _publish(
                        reporter,
                        tracker.consume(
                            _member(chunk, member), start_env_steps=trained_steps
                        ),
                    )
                trained_steps += steps

            if not config.evaluation_episodes:
                continue
            eval_number = self._measure(
                reporters,
                open_evaluation,
                evaluate,
                jax.random.fold_in(evaluation_key, boundary),
                state,
                boundary=boundary,
                first_number=eval_number,
            )

    def _tracker(self) -> EpisodeTracker:
        config = self.config
        return EpisodeTracker(
            observations=self.algorithm.observations,
            num_envs=config.num_envs,
            max_episode_steps=config.max_episode_steps,
            sample_steps=tuple(config.trajectory_at_steps),
        )

    def _measure(
        self,
        reporters: Sequence[Destination],
        open_evaluation,
        evaluate,
        key,
        state,
        *,
        boundary: int,
        first_number: int,
    ) -> int:
        """Score every member's checkpoint on the episodes it was asked for.

        The rollout is advanced for the whole ensemble at once and stops when
        the *last* member has filled its quota. A member that finished earlier
        keeps stepping, which costs a few compiled steps it does not need and
        buys the alternative's absence: masking finished members would make what
        a member computed depend on when its neighbours ended.
        """

        config = self.config
        quota = evaluation_quota(
            episodes=config.evaluation_episodes, num_envs=config.num_envs
        )
        trackers = [self._evaluation_tracker() for _ in range(self.members)]
        collected: list[list[list[Episode]]] = [
            [[] for _ in quota] for _ in range(self.members)
        ]

        budget = max(quota) * config.max_episode_steps * config.num_envs
        key, open_key = jax.random.split(key)
        # One key, repeated. Every member is measured on the same evaluation
        # stream, so what separates their scores is the policy each learned and
        # nothing else -- the pairing that makes two members comparable rather
        # than merely both measured. It is also what makes a member's evaluation
        # identical to the one the single-member driver would have given it.
        member_keys = jax.numpy.stack([open_key] * self.members)
        evaluation = open_evaluation(member_keys, state)
        spent = 0
        while not all(
            len(found) >= wanted
            for member in collected
            for found, wanted in zip(member, quota)
        ):
            if spent >= budget:
                short = [
                    index
                    for index, member in enumerate(collected)
                    if any(len(f) < w for f, w in zip(member, quota))
                ]
                raise ValueError(
                    f"evaluation at {boundary} environment steps did not complete "
                    f"{config.evaluation_episodes} episodes within {budget} steps "
                    f"for member(s) {short}"
                )
            steps = min(config.evaluation_chunk_steps, budget - spent)
            member_keys, chunk_keys = _split(member_keys)
            evaluation, chunk = evaluate(chunk_keys, evaluation, steps)
            for member in range(self.members):
                result = trackers[member].consume(
                    _member(chunk, member), start_env_steps=boundary
                )
                for episode in result.completed:
                    collected[member][episode.stream].append(episode)
            spent += steps

        for member, reporter in enumerate(reporters):
            for slot, episode in sorted(self._slots(collected[member], quota)):
                reporter.log_episode(replace(episode, number=first_number + slot))
        return first_number + config.evaluation_episodes

    def _evaluation_tracker(self) -> EpisodeTracker:
        config = self.config
        return EpisodeTracker(
            observations=self.algorithm.observations,
            num_envs=config.num_envs,
            max_episode_steps=config.max_episode_steps,
            phase="eval",
            stride=0,
            require_series=False,
        )

    def _slots(self, collected, quota) -> list[tuple[int, Episode]]:
        num_envs = self.config.num_envs
        return [
            (index * num_envs + stream, episode)
            for stream, wanted in enumerate(quota)
            for index, episode in enumerate(collected[stream][:wanted])
        ]


def _member(chunk, member: int):
    """One member's slice of a chunk, in the layout a tracker already reads."""

    return jax.tree.map(lambda leaf: leaf[member], chunk)


def _publish(reporter: Destination, result: TrackingResult) -> None:
    for episode in result.completed:
        reporter.log_episode(episode)
    for trajectory in result.sampled:
        reporter.log_trajectory(trajectory)
