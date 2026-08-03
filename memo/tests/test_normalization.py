"""One estimator, instantiated twice, knowing neither of its two callers.

Normalising an observation and normalising a reward are the same running
estimate of a mean and a spread. What differed was never the estimator: an
observation is centred and a reward is only scaled, and a reward is fed the
discounted accumulation of itself rather than its own value. Both are settings.

Written as two named halves, the component said where it was called from, and a
kernel that wanted a third stream had nowhere to put it.
"""

from __future__ import annotations

import dataclasses
import inspect

import jax.numpy as jnp
import pytest

from memorax.rl.normalization import (
    DiscountedNormalization,
    NormalizationConfig,
    Normalizer,
    RunningNormalization,
    make_normalizer,
)

STREAMS = 2
# A vector per stream, as an observation is, and a scalar per stream, as
# anything the discount accumulates is.
VALUES = jnp.asarray([[1.0, 3.0], [5.0, 7.0]], dtype=jnp.float32)
SCALARS = jnp.asarray([1.0, 2.0], dtype=jnp.float32)
LIVE = jnp.zeros((STREAMS,), dtype=jnp.bool_)


def estimator(**settings) -> Normalizer:
    return make_normalizer(NormalizationConfig(**settings))


def test_the_same_estimator_serves_both_streams():
    """Two instances of one class, differing only in what they were told."""

    centred = estimator(center=True)
    scaled = estimator(center=False, discount=0.9)

    assert type(centred) is type(scaled)
    assert centred is not scaled


def test_centred_subtracts_the_mean_and_uncentred_only_scales():
    """The one thing an observation wants that a reward does not.

    A reward divided by the spread of its own discounted return keeps its sign
    and its zero; subtracting a mean from it would move the point at which the
    agent is indifferent.
    """

    values = jnp.asarray([[4.0, 4.0], [4.0, 4.0]], dtype=jnp.float32)
    # Opened on its first sample, so the mean is that sample and centring it
    # leaves exactly nothing. Under a seeded start the mean is halfway there.
    centred = estimator(center=True, cold_start="first_sample")
    scaled = estimator(center=False, cold_start="first_sample")
    centred, _ = centred.observe(centred.initial(values), values, done=LIVE)
    scaled, _ = scaled.observe(scaled.initial(values), values, done=LIVE)

    assert jnp.allclose(centred, 0.0)
    assert not jnp.allclose(scaled, 0.0)
    assert jnp.all(jnp.sign(scaled) == jnp.sign(values))


def test_a_discount_feeds_it_the_accumulation_instead_of_the_value():
    """Which is the whole of what made the reward path a different path.

    Without a discount the estimator sees each value; with one it sees the
    running sum the values discount into, so its spread is the spread of a
    return rather than of a single step.
    """

    plain, discounted = estimator(center=False), estimator(center=False, discount=0.9)
    plain_state = plain.initial(SCALARS)
    discounted_state = discounted.initial(SCALARS)
    for _ in range(3):
        _, plain_state = plain.observe(plain_state, SCALARS, done=LIVE)
        _, discounted_state = discounted.observe(discounted_state, SCALARS, done=LIVE)

    assert discounted_state.trace is not None
    assert plain_state.trace is None
    assert not jnp.allclose(plain_state.mean, discounted_state.mean)


def test_the_accumulation_restarts_with_the_episode_when_asked():
    over = jnp.ones((STREAMS,), dtype=jnp.bool_)
    keeping = estimator(center=False, discount=0.9, reset_on_done=False)
    dropping = estimator(center=False, discount=0.9, reset_on_done=True)

    _, kept = keeping.observe(keeping.initial(SCALARS), SCALARS, done=over)
    _, dropped = dropping.observe(dropping.initial(SCALARS), SCALARS, done=over)

    assert jnp.allclose(dropped.trace, 0.0)
    assert not jnp.allclose(kept.trace, 0.0)


def test_a_stream_that_ended_does_not_drop_another_stream_s_accumulation():
    """One estimate per stream, so one ending is one stream's business."""

    ending = jnp.asarray([True, False])
    one = estimator(center=False, discount=0.9)
    _, after = one.observe(one.initial(SCALARS), SCALARS, done=ending)

    assert after.trace[0] == 0.0
    assert after.trace[1] != 0.0


NAMED = ("observation", "reward")


@pytest.mark.parametrize("named", NAMED)
def test_nothing_here_names_a_stream_it_might_be_called_on(named):
    """A component that names its call sites is a component with two callers.

    The kernel names its streams -- it is the one holding them -- and asking the
    estimator to would mean a third stream had nowhere to go.
    """

    surface = [name for name, _ in inspect.getmembers(Normalizer)]
    surface += [field.name for field in dataclasses.fields(NormalizationConfig)]
    for component in (RunningNormalization, DiscountedNormalization):
        surface += [field.name for field in dataclasses.fields(component)]

    offending = [name for name in surface if named in name]
    assert not offending, f"{named} is named by {offending}"
