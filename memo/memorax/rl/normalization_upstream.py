"""Our normaliser computing what the StreamAC authors' wrappers compute.

``normalization.py`` beside this is ours, and it is not theirs. The two were
never compared until the LRU work asked the same question of a different module,
because the test that claimed to check the normalisation compared our kernel
against our own environment wrapper -- two copies of the same arithmetic, which
agree however wrong they both are. Driven against
``normalization_wrappers.py`` from mohmdelsayed/streaming-drl, three things
differ, and none of them is a rounding.

The first is where the statistics start. Their ``SampleMeanStd.update`` opens
with ``if self.count == 0: self.mean = x; self.p = 0``, so the first sample
becomes the mean exactly and the variance is exactly one, and the first
normalised observation is exactly zero. Ours seeds a pseudo-observation -- mean
zero, second moment one, count one -- and folds the real one in beside it, so
the first observation normalises to something near but not at zero.

The second is the divisor. Theirs is ``p / (count - 1)``, the unbiased sample
variance, held at exactly one while ``count < 2``. Ours is ``M2 / count``, over a
count the pseudo-observation has already advanced by one.

Those two fade. Both estimators converge on the variance, and after a few
thousand observations the difference between the divisors is the difference
between ``n - 1`` and ``n + 1``. The third does not fade.

The third is the episode boundary in the discounted-return trace the reward
scale is measured from. Theirs is ``G = G * gamma * (1 - done) + r``, and the
``G`` it carries into the next step is that, unmasked -- so on a terminal step
the trace becomes the terminal reward, and the next episode starts with that
reward still in it, decayed once. Ours masks twice, reading ``G * (1 - done)``
and storing ``G * (1 - done)`` again, so our trace restarts at zero. Theirs
leaks a reward across every episode boundary in the run, and ours does not.

The third is a defect, and it is one gymnasium's own ``NormalizeReward`` carried
for years, which is the likely provenance. That is not a reason to leave it out
of this file. Comparing our normalisation against theirs and finding a gap would
attribute to nothing: it could be our framework, or it could be these three. An
arm that reproduces them separates those, and what the defects cost is then
measurable against our normaliser rather than mixed into it.

Not a thing to train with. It exists to be compared against.
"""

from __future__ import annotations

from dataclasses import replace

import jax.numpy as jnp

from .normalization import (
    NormalizationConfig,
    Normalizer,
    NormalizerState,
    RewardStatistics,
    RunningStatistics,
    _expand_for,
)


class UpstreamNormalizer(Normalizer):
    """Their three, put back: the cold start, the divisor, the unmasked trace.

    Everything else is inherited. The batch handling, the eval switches and the
    episode-return channel are ours and are not what this reproduces.
    """

    def _initial_state(self, observation) -> NormalizerState:
        """Their ``SampleMeanStd.__init__``, before any sample has arrived.

        Count zero is what makes the first ``_welford`` take their opening
        branch, so it is the count and not the mean that carries the cold start.
        Their ``p`` starts at one and is overwritten with zero by that branch
        before it is ever read, so it starts at zero here.
        """

        num_envs = observation.shape[0]
        zeros = jnp.zeros((num_envs,), dtype=jnp.float32)
        return NormalizerState(
            observation=(
                RunningStatistics(
                    mean=jnp.zeros_like(observation),
                    M2=jnp.zeros_like(observation),
                    count=zeros,
                )
                if self.config.normalize_observation
                else None
            ),
            reward=(
                RewardStatistics(mean=zeros, M2=zeros, count=zeros, G=zeros)
                if self.config.normalize_reward
                else None
            ),
            episode_return=zeros,
        )

    def _welford(self, mean, M2, count, sample):
        """Their ``update`` and ``update_mean_var_count_from_moments``, in order.

        The opening branch is a ``where`` rather than a branch because every
        stream is stepped together and they need not have started together. It
        replaces the mean with the sample and the second moment with zero, which
        is what makes their first variance exactly one rather than nearly one.
        """

        first = count == 0
        mean = jnp.where(_expand_for(first, mean), sample, mean)
        M2 = jnp.where(_expand_for(first, M2), jnp.zeros_like(M2), M2)

        count = count + 1
        expanded = _expand_for(count, mean)
        delta = sample - mean
        new_mean = mean + delta / expanded
        return new_mean, M2 + delta * (sample - new_mean), count

    def _variance(self, M2, count):
        """``p / (count - 1)``, and exactly one until there are two of them."""

        expanded = _expand_for(count, M2)
        return jnp.where(
            expanded < 2,
            jnp.ones_like(M2),
            M2 / jnp.maximum(expanded - 1, 1.0),
        )

    def _update_observation(self, state, observation):
        mean, M2, count = self._welford(state.mean, state.M2, state.count, observation)
        return replace(state, mean=mean, M2=M2, count=count)

    def _normalize_observation(self, state, observation):
        variance = self._variance(state.M2, state.count)
        return (observation - state.mean) / jnp.sqrt(variance + self.config.eps)

    def _update_reward(self, state, reward, done):
        G = state.G * self.config.reward_gamma * (1 - done) + reward
        mean, M2, count = self._welford(state.mean, state.M2, state.count, G)
        # Carried as it stands. The mask above is the only one they apply, and
        # applying it again here is the difference this arm exists to undo.
        return replace(state, mean=mean, M2=M2, count=count, G=G)

    def _scale_reward(self, state, reward):
        return reward / jnp.sqrt(
            self._variance(state.M2, state.count) + self.config.eps
        )


def make_upstream_normalizer(config: NormalizationConfig) -> UpstreamNormalizer:
    """The arm, built from the same config ours is built from."""

    return UpstreamNormalizer(config)
