"""Cut a stacked rollout into the complete episodes that are worth reporting.

A kernel runs a fixed number of steps across a fixed number of streams and never
stops at an episode boundary, so a chunk holds whole episodes, partial ones at
both ends, and nothing marking which is which except the done flag. Only the
whole ones are reported; a partial episode would either invent a terminal that
did not happen or claim a return that is not one.

An episode belongs to one stream, which is why the stream axis survives here and
is never averaged: two streams that ended at different steps are two answers.

This is arithmetic on an array of that shape and nothing about any one trainer,
so it lives beside the episode it produces rather than inside whichever trainer
needed it first.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping

import numpy as np

from training_sdk.episode import Episode


def complete_episodes(
    summary,
    *,
    phase: str,
    start_env_steps: int,
    num_envs: int,
    stride: int | None = None,
    first_number: int = 1,
    reward: str = "reward",
    terminal: str = "terminal",
    series: Iterable[str] = (),
) -> Iterator[Episode]:
    """Yield every episode that both starts and ends inside the chunk.

    ``reward`` names where the environment's own reward sits, because what an
    episode is worth is a statement about the task and an algorithm is entitled
    to learn on another scale. A kernel that normalises internally keeps the
    untouched one under its ordinary name; one that normalises in an environment
    wrapper has had it overwritten, and the entry -- which is what knows the
    wrapper is there -- names where the original went.

    ``stride`` is how many environment steps one row of the chunk advanced the
    axis. Evaluation advances it by nothing: a rollout run at an epoch boundary
    measures the policy as it stood there, and spreading its episodes forward
    would date them to training that has not happened.

    ``terminal`` names the failure ending; what is done without being terminal
    was truncated. A kernel that does not tell the two apart reports every
    ending as a termination, which is what it knew.
    """

    stride = num_envs if stride is None else stride
    rewards = np.asarray(read(summary, reward))
    dones = np.asarray(read(summary, "done")).astype(bool)
    found = read(summary, terminal)
    terminals = dones if found is None else np.asarray(found).astype(bool)
    steps = dones.shape[0]
    traces = {
        name: _streamed(found, steps, num_envs)
        for name in series
        if (found := read(summary, name)) is not None
    }
    walked = [
        read(summary, name) for name in ("observation", "next_observation", "action")
    ]
    trajectory = all(one is not None for one in walked)
    if trajectory:
        before, after, actions = (np.asarray(one) for one in walked)

    number = first_number
    for env in range(num_envs):
        start = 0
        for end in (step for step in range(steps) if dones[step, env]):
            span = range(start, end + 1)
            walk = {}
            if trajectory:
                walk = {
                    "observations": [before[step, env].tolist() for step in span]
                    + [after[end, env].tolist()],
                    "actions": [actions[step, env].tolist() for step in span],
                }
            yield Episode(
                number=number,
                phase=phase,
                stream=env,
                start_env_steps=start_env_steps + start * stride,
                end_env_steps=start_env_steps + (end + 1) * stride,
                rewards=[float(rewards[step, env]) for step in span],
                terminals=[bool(terminals[step, env]) for step in span],
                truncations=[
                    bool(dones[step, env] and not terminals[step, env])
                    for step in span
                ],
                series={
                    name: [float(column[step, env]) for step in span]
                    for name, column in traces.items()
                },
                **walk,
            )
            number += 1
            start = end + 1


def read(source, path: str):
    """A named field, or a named member of one, which is how a family is read.

    A kernel that reports one number per part of a network cannot know what the
    parts are called, so it hands back a mapping and the entry names the parts
    it built. The separator is the one a parameter inside a structure uses, so
    that a family stays inside a single segment of a metric's name.
    """

    found = source
    for name in path.split("."):
        found = (
            found.get(name)
            if isinstance(found, Mapping)
            else getattr(found, name, None)
        )
        if found is None:
            return None
    return found


def _streamed(values, steps: int, num_envs: int):
    """One column per stream, from a reading that may have measured them all.

    A per-stream quantity arrives with the stream axis already there; one the
    kernel reduced over the batch arrives without it and belongs to every stream
    alike.
    """

    return np.broadcast_to(np.asarray(values).reshape(steps, -1), (steps, num_envs))
