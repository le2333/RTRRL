"""A network is a sequence of components, not three named slots.

``Network`` took ``observation, done, action, reward, initial_carry`` and handed
every one of them to all three of its slots, so a slot that wanted none of them
still had to accept them, and a slot that wanted ``done`` got it whether or not
it had anything to reset. The step here is ``(carries, x) -> (carries, y)`` and
nothing else crosses it: what a component needs beyond ``x`` it declares, and the
sequence supplies only what was declared.

The carry is one entry per component. A stateless component hands back the entry
it was given and a recurrent one hands back a new one, which is what makes "a
stateless component ignores the carry" something to check rather than something
to promise.
"""

from __future__ import annotations

import dataclasses
import inspect

import flax.linen as nn
import jax
import jax.numpy as jnp
import pytest
from conftest import TinyContinuousEnv, deviations, flattened

from memorax.algorithms.stream_ac import StreamAC, StreamACConfig
from memorax.networks import heads
from memorax.networks.components import FFN, LayerNorm, Readout, Tanh
from memorax.networks.sequence import Sequence
from memorax.networks.sequence_models import RNN, RTUCell, RTUConfig
from memorax.rl import make_credit
from memorax.rl.updates import ObBound, Sgd

ENVS = 3
FEATURES = 4
HIDDEN = 2
SHAPE = (ENVS, FEATURES)


def recurrent(features: int = FEATURES):
    return RNN(cell=RTUCell(config=RTUConfig(features=features, hidden_dim=HIDDEN)))


def stateless() -> tuple:
    """The three components a backbone puts in front of its recurrence."""

    return (FFN(features=FEATURES), LayerNorm(), Tanh())


def built(*components) -> Sequence:
    return Sequence(components=tuple(components))


def driven(sequence, *, done=False, key=jax.random.key(0), carries=None):
    """Initialise a sequence and take one step through it."""

    x = jnp.reshape(
        jnp.linspace(-1.0, 1.0, ENVS * FEATURES, dtype=jnp.float32), (ENVS, 1, FEATURES)
    )
    ended = jnp.full((ENVS, 1), done, dtype=jnp.bool_)
    if carries is None:
        carries = sequence.initialize_carry(key, SHAPE)
    params = sequence.init(key, x, done=ended, initial_carry=carries)
    return carries, sequence.apply(params, x, done=ended, initial_carry=carries)


def test_the_carry_is_one_entry_per_component():
    """Not one carry belonging to whichever slot happened to be recurrent."""

    sequence = built(*stateless(), recurrent(), Readout(module=heads.VNetwork()))
    carries = sequence.initialize_carry(jax.random.key(0), SHAPE)

    assert len(carries) == len(sequence.components)

    _, (next_carries, _) = driven(sequence, carries=carries)

    assert len(next_carries) == len(sequence.components)


def test_a_stateless_component_hands_back_the_carry_it_was_given():
    """Which is why the entries are per component rather than one for the lot.

    A sentinel is put where a stateless component's entry goes: it has no use
    for one, so the only thing it can do with it is hand it back.
    """

    sequence = built(FFN(features=FEATURES), recurrent())
    carries = sequence.initialize_carry(jax.random.key(0), SHAPE)
    sentinel = jnp.asarray([7.0, 8.0, 9.0], dtype=jnp.float32)
    carries = [sentinel, *carries[1:]]

    _, (next_carries, _) = driven(sequence, carries=carries)

    assert next_carries[0] is sentinel


def test_only_the_recurrent_component_contributes_to_the_carry():
    sequence = built(*stateless(), recurrent(), Readout(module=heads.VNetwork()))

    carries, (next_carries, _) = driven(sequence)

    moved = [
        index
        for index, (was, now) in enumerate(zip(carries, next_carries))
        if deviations(flattened(now), flattened(was))
    ]
    assert moved == [sequence.recurrent]


def test_done_reaches_the_component_that_declared_it_and_no_other():
    """A component that has nothing to reset never sees an ending.

    Declared as an input the component asks for, so the check is on the
    signature rather than on a component quietly swallowing what it was pushed.
    """

    for component in stateless():
        assert component.reads == ()
        taken = inspect.signature(type(component).__call__).parameters
        assert "done" not in taken, f"{type(component).__name__} takes done"
        assert not any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in taken.values()
        ), f"{type(component).__name__} would swallow anything pushed at it"

    assert recurrent().reads == ("done",)


def test_an_ending_reaches_the_recurrence_through_that_declaration():
    """And the declaration is wired, not merely written down.

    Driven from a carry that holds something, since resetting one already at its
    initial value is a step an ending could never be seen in.
    """

    sequence = built(*stateless(), recurrent())
    carries = jax.tree.map(
        jnp.ones_like, sequence.initialize_carry(jax.random.key(0), SHAPE)
    )

    _, (_, live) = driven(sequence, done=False, carries=carries)
    _, (_, ended) = driven(sequence, done=True, carries=carries)

    assert deviations(flattened({"y": live}), flattened({"y": ended}))


def test_two_recurrent_components_are_refused():
    """Exact credit across two of them needs a dense cross-layer sensitivity.

    Refusing says so where the pair was asked for, rather than crediting the
    first one for what the second did.
    """

    with pytest.raises(ValueError, match="recurrent"):
        built(recurrent(), LayerNorm(), recurrent())


def test_the_sequence_names_its_recurrent_component_so_a_credit_can_wrap_it():
    """The credit wraps the recurrence, not the network it sits in.

    ``make_credit`` carries a sensitivity through one Jacobian; handed a whole
    sequence there would be nothing whose Jacobian that is.
    """

    core = recurrent()
    sequence = built(*stateless(), core, Readout(module=heads.VNetwork()))

    assert sequence.components[sequence.recurrent] is core
    assert (
        make_credit("rtrl", sequence.core).initialize(jax.random.key(0), SHAPE)
        is not None
    )


def test_walking_with_truncated_credit_is_walking_plainly():
    """The credit is where a parameter's effect on the past enters, and under
    ``tbptt`` nothing enters, so the two ways of driving the sequence agree."""

    sequence = built(*stateless(), recurrent(), Readout(module=heads.VNetwork()))
    carries, (plain_carries, (plain, _)) = driven(sequence)

    x = jnp.reshape(
        jnp.linspace(-1.0, 1.0, ENVS * FEATURES, dtype=jnp.float32), (ENVS, 1, FEATURES)
    )
    ended = jnp.zeros((ENVS, 1), dtype=jnp.bool_)
    params = sequence.init(jax.random.key(0), x, done=ended, initial_carry=carries)
    (walked, sensitivity), (walk_output, _) = sequence.walk(
        params,
        x,
        done=ended,
        carries=carries,
        sensitivity=None,
        credit=make_credit("tbptt", sequence.core),
    )

    assert sensitivity is None
    assert not deviations(flattened({"y": walk_output}), flattened({"y": plain}))
    assert not deviations(flattened(walked), flattened(plain_carries))


def test_a_component_knows_its_own_name_on_both_ways_through():
    """The name is the sequence's, and both traversals have to hand it over.

    ``__call__`` composes through flax, so a component is bound under its name
    and can read it. ``walk`` applies each component against its own slice of
    the parameter tree, which re-roots it, and a re-rooted module has no name
    unless it is given one. A component that cannot say which position it is
    cannot label anything it produces, and the two traversals disagreeing about
    that would make the label depend on which one ran.
    """

    seen: dict[str, list] = {"call": [], "walk": []}

    class Reporting(FFN):
        """A component that reads its own name, and nothing else."""

        where: str = "call"

        @nn.compact
        def __call__(self, x, initial_carry=None):
            seen[self.where].append(self.name)
            return initial_carry, nn.Dense(self.features)(x)

    sequence = built(
        Reporting(features=FEATURES),
        Tanh(),
        Reporting(features=FEATURES),
        recurrent(),
        Readout(module=heads.VNetwork()),
    )
    x = jnp.reshape(
        jnp.linspace(-1.0, 1.0, ENVS * FEATURES, dtype=jnp.float32), (ENVS, 1, FEATURES)
    )
    ended = jnp.zeros((ENVS, 1), dtype=jnp.bool_)
    carries = sequence.initialize_carry(jax.random.key(0), SHAPE)
    params = sequence.init(jax.random.key(0), x, done=ended, initial_carry=carries)

    seen["call"].clear()
    sequence.apply(params, x, done=ended, initial_carry=carries)

    walking = dataclasses.replace(
        sequence,
        components=tuple(
            (
                dataclasses.replace(component, where="walk")
                if isinstance(component, Reporting)
                else component
            )
            for component in sequence.components
        ),
    )
    walking.walk(
        params,
        x,
        done=ended,
        carries=carries,
        sensitivity=None,
        credit=make_credit("tbptt", sequence.core),
    )

    assert seen["call"] == ["components_0", "components_2"]
    assert seen["walk"] == seen["call"]


NAMED = ("observation", "action", "reward", "embedding")


@pytest.mark.parametrize("named", NAMED)
def test_nothing_here_names_what_the_kernel_feeds_it(named):
    """A network that names its inputs is a network its caller cannot rearrange.

    Putting the previous action and reward beside the observation is input
    composition; it happens where those values already are, and the sequence
    sees one vector.
    """

    surface = [name for name, _ in inspect.getmembers(Sequence)]
    surface += [field.name for field in dataclasses.fields(Sequence)]
    for component in (FFN, LayerNorm, Tanh, Readout):
        surface += [name for name, _ in inspect.getmembers(component)]
        surface += [field.name for field in dataclasses.fields(component)]

    offending = sorted({name for name in surface if named in name})
    assert not offending, f"{named} is named by {offending}"


def kernel(**overrides) -> StreamAC:
    """The kernel, built on sequences rather than on three slots."""

    env = TinyContinuousEnv()
    action_dim = int(env.action_space(env.default_params).shape[0])

    def network(head):
        return built(
            FFN(features=FEATURES),
            LayerNorm(),
            Tanh(),
            recurrent(),
            Readout(module=head),
        )

    settings = {
        "num_envs": ENVS,
        "gamma": 0.9,
        "trace_lambda": 0.8,
        "actor_bound": ObBound(kappa=2.0),
        "actor_base": Sgd(lr=0.1),
        "critic_bound": ObBound(kappa=2.0),
        "critic_base": Sgd(lr=0.1),
        "credit": "tbptt",
    }
    return StreamAC(
        StreamACConfig(**{**settings, **overrides}),
        env,
        env.default_params,
        network(heads.Gaussian(action_dim=action_dim)),
        network(heads.VNetwork()),
    )


def test_the_kernel_steps_on_a_sequence():
    """The functional half: a sequence is a thing StreamAC can actually run."""

    agent = kernel()
    state = agent.init(jax.random.key(0))
    state, metrics = agent.train(jax.random.key(1), state, 2 * ENVS)

    assert metrics.interaction.reward.shape == (2, ENVS)
    assert len(state.actor.recurrence.carry) == len(
        agent.core.actor.block.network.components
    )


def test_composing_the_input_widens_the_first_layer_and_nothing_else():
    """``meta_rl`` is the kernel's doing, so it shows up at the kernel's edge.

    The previous action and reward arrive beside the observation, so the first
    layer's fan-in grows by exactly their width and no component learns what
    they were.
    """

    env = TinyContinuousEnv()
    width = int(env.observation_space(env.default_params).shape[0])
    action_dim = int(env.action_space(env.default_params).shape[0])

    plain = kernel().init(jax.random.key(0))
    composed = kernel(meta_rl=True).init(jax.random.key(0))

    def fan_in(params):
        tree = params["params"]
        return tree["components_0"]["Dense_0"]["kernel"].shape[0]

    assert fan_in(plain.actor.params) == width
    assert fan_in(composed.actor.params) == width + action_dim + 1
