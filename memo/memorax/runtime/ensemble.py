"""Schedule many members of one algorithm as a single graph.

A sweep's members differ only in what they were seeded with, so their graph is
one graph and a device can be filled by the sweep rather than by enlarging any
single run. This is :mod:`~memorax.runtime.driver` with a member axis: the same
five arrows, the same episode tracking, the same reported episodes, mapped.

Two ways to differ. Members that differ only in a seed share their graph down
to the constant, so it is built once and the arrows are vmapped over their key.
Members that differ in a *value* -- a learning rate, a discount -- need the
build inside the map, because vmap maps over arguments and a value closed into
a graph is not one.

Which of the two is in use is the only branch, and it is at the arrows. The
schedule below them is one loop either way, so a round of seeds runs exactly the
compiled graph it ran before this could sweep anything.

A swept leaf has to be one the graph reads arithmetically. One that sizes an
array or selects a branch cannot vary across members, who share both, and those
are declared ``static`` where they are declared -- the entry refuses them before
a job compiles anything.

Members are not streams. ``num_envs`` streams inside a member share a policy and
a set of parameters; two members share nothing but their shapes, and folding one
axis into the other would let an episode from one member be scored against
another's quota. So each member keeps its own tracker and its own destination,
and the only place they meet is the compiled graph.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Callable

import jax
import jax.numpy as jnp

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
    # Set both or neither. `build` turns one member's parameters into its graph
    # and `swept` gives each member its own values; `algorithm` is still
    # required, because the observation schema is read from it and a schema does
    # not depend on the values a sweep varies.
    build: Callable[[Mapping[str, Any]], BuiltAlgorithm] | None = None
    parameters: Mapping[str, Any] = field(default_factory=dict)
    swept: Mapping[str, Sequence[Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.seeds:
            raise ValueError("an ensemble needs at least one member")
        if bool(self.swept) != (self.build is not None):
            raise ValueError(
                "a swept ensemble needs both a build and the values to sweep; "
                f"got build={self.build is not None} swept={sorted(self.swept)}"
            )
        for path, values in self.swept.items():
            if len(values) != len(self.seeds):
                raise ValueError(
                    f"{path} has {len(values)} values for "
                    f"{len(self.seeds)} members"
                )

        # A member is its seed *and* the values it was handed. Two members on
        # one seed are distinct runs when a sweep gave them different values,
        # which is what a round of several trials over several seeds is made of.
        # With nothing swept this is the seeds being distinct, as it was.
        members = [
            (seed, *(tuple(values)[index] for values in self.swept.values()))
            for index, seed in enumerate(self.seeds)
        ]
        if len(set(members)) != len(members):
            raise ValueError(
                f"two members are the same run: {sorted(map(str, members))}"
            )

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

        init, train, open_evaluation, evaluate = self._arrows()
        varied = self._varied()

        keys = jax.numpy.stack([jax.random.key(seed) for seed in self.seeds])
        keys, init_keys = _split(keys)
        state = init(*varied, init_keys)
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
                state, chunk = train(*varied, chunk_keys, state, steps)
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
                varied,
                jax.random.fold_in(evaluation_key, boundary),
                state,
                boundary=boundary,
                first_number=eval_number,
            )

    def _varied(self) -> tuple:
        """The swept values as one mapped argument, or nothing to map."""

        if not self.swept:
            return ()
        return ({path: jnp.asarray(values) for path, values in self.swept.items()},)

    def _arrows(self):
        """The four arrows, mapped, and built where the members require.

        Seeds alone: one graph, built here, and the arrows are its own methods.
        A sweep: the graph is built inside each arrow from that member's leaves,
        which costs a build per trace -- four of them for a job -- and buys the
        only thing that makes a value mappable at all.
        """

        if not self.swept:
            program = self.algorithm.program
            # `steps` stays static, as in the driver: it is a scan length.
            return (
                jax.jit(jax.vmap(program.init)),
                jax.jit(
                    jax.vmap(program.train, in_axes=(0, 0, None)), static_argnums=2
                ),
                jax.jit(jax.vmap(program.open_evaluation)),
                jax.jit(
                    jax.vmap(program.evaluate, in_axes=(0, 0, None)),
                    static_argnums=2,
                ),
            )

        build = self.build
        shared = dict(self.parameters)

        def program(overrides):
            return build({**shared, **overrides}).program

        return (
            jax.jit(jax.vmap(lambda o, key: program(o).init(key))),
            jax.jit(
                jax.vmap(
                    lambda o, key, state, steps: program(o).train(key, state, steps),
                    in_axes=(0, 0, 0, None),
                ),
                static_argnums=3,
            ),
            jax.jit(
                jax.vmap(
                    lambda o, key, state: program(o).open_evaluation(key, state)
                )
            ),
            jax.jit(
                jax.vmap(
                    lambda o, key, run, steps: program(o).evaluate(key, run, steps),
                    in_axes=(0, 0, 0, None),
                ),
                static_argnums=3,
            ),
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
        varied: tuple,
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
        evaluation = open_evaluation(*varied, member_keys, state)
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
            evaluation, chunk = evaluate(*varied, chunk_keys, evaluation, steps)
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
