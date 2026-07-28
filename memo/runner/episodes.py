"""Cut a stacked evaluation rollout into the complete episodes a sink accepts.

The kernels run a fixed number of steps across a fixed number of environments
and never stop at an episode boundary, so a rollout holds whole episodes,
partial ones at both ends, and nothing marking which is which except the done
flag. Only the whole ones are reported; a partial episode would either invent
a terminal that did not happen or claim a return that is not one.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from training_sdk.episode import Episode


def complete_episodes(
    summary,
    *,
    phase: str,
    start_env_steps: int,
    num_envs: int,
    first_number: int = 1,
    rewards=None,
) -> Iterator[Episode]:
    """Yield every episode that both starts and ends inside the rollout.

    The rewards may be given rather than read off the summary, because what an
    episode is worth is a statement about the task and an algorithm is entitled
    to learn on another scale. A kernel that normalises internally keeps the
    environment's own reward for the summary and needs nothing here; one that
    normalises in an environment wrapper has had it overwritten, and its entry --
    which is what knows the wrapper is there -- passes the untouched one in.
    """

    before = np.asarray(summary.observation)
    after = np.asarray(summary.next_observation)
    actions = np.asarray(summary.action)
    rewards = np.asarray(summary.reward if rewards is None else rewards)
    dones = np.asarray(summary.done).astype(bool)
    steps = dones.shape[0]

    number = first_number
    for env in range(num_envs):
        start = 0
        for end in (step for step in range(steps) if dones[step, env]):
            span = range(start, end + 1)
            yield Episode(
                number=number,
                phase=phase,
                start_env_steps=start_env_steps + start * num_envs,
                end_env_steps=start_env_steps + (end + 1) * num_envs,
                observations=[before[step, env].tolist() for step in span]
                + [after[end, env].tolist()],
                actions=[actions[step, env].tolist() for step in span],
                rewards=[float(rewards[step, env]) for step in span],
                terminals=[bool(dones[step, env]) for step in span],
                truncations=[False] * (end - start + 1),
            )
            number += 1
            start = end + 1
