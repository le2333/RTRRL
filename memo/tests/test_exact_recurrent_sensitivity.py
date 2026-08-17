"""Does ``exact_rtrl`` carry the recurrent sensitivity it claims to carry?

The structured cores keep one forward sensitivity per recurrent parameter and
inject it back into the carry as a phantom, so that one step of ordinary
autodiff produces the gradient a full unroll would have produced. That is the
whole claim behind calling the method exact, and the R1 comparison rests on it,
so it is checked here against something that shares none of its arithmetic:
autodiff through the truncation-free walk of the same frozen sequence.

The oracle is deliberately not a second sensitivity recurrence. ``TruncatedBPTT``
carries no derivative state, so chaining it over a prefix and differentiating the
chain is plain BPTT over the whole prefix -- the quantity RTRL is supposed to
equal. Nothing on that path reads ``local_jacobian``, ``compute_phantom`` or
``initialize_sensitivity``.

What the claim covers: the sensitivity is kept for the recurrent kernel's own
parameters, and on this torso those are all of them. Nothing precedes the cell
-- the observation reaches it directly -- and the ``LayerNorm`` behind it is
affine-free, so it holds no parameters to credit. "Exact online recurrent
sensitivity" is therefore true of the whole torso here, and a test below says so
by asserting the torso has nothing in front of its recurrence rather than by
leaving it to be read off ``_construct_torso``.

The distinction is not academic. A learned projection ahead of the cell would
reach its own past only through a carry the phantom injection cuts with
``stop_gradient``, so it would take the one-step gradient under ``exact_rtrl``
exactly as under ``tbptt`` -- and the torso's gradient could then no longer be
reported as exact. That is checked too, on a sequence built with such a
projection, so the claim is bounded by something run rather than by a comment.

Streams: the readout objectives are written for the single stream
``Core._per_stream`` vmaps them over, so the gradient tests drive one stream and
are the vmapped unit exactly. The sensitivity itself has no such restriction and
is driven over several streams at once, which is where a recurrence that leaked
across them would show.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from memorax.algorithms import rtrrl_aaai as rtrrl
from memorax.algorithms.rtrrl_aaai import RTRRLConfig
from memorax.networks import heads
from memorax.networks.backbones import backbone
from memorax.networks.components import FFN, LayerNorm, Tanh
from memorax.networks.differentiation import TruncatedBPTT
from memorax.networks.sequence import Sequence
from memorax.networks.sequence_models.lru import LRUStructuredRTRL
from memorax.networks.sequence_models.rtu import RTUStructuredRTRL
from memorax.utils import Timestep
from tests.support.numerics import assert_within, deviations, flattened

CORES = ("lru", "rtu")
LENGTHS = (1, 2, 3, 5, 8)

STREAMS = 3
OBSERVED = 3
ACTIONS = 2
HIDDEN = 2
READOUT = 4
# What the cell reads, the way ``RTRRL.graph`` counts it: the observation, and
# under meta-RL the previous action and reward beside it.
INPUT = OBSERVED + ACTIONS + 1

# Both quantities are float32 sums over a prefix, so the gap is allowed to grow
# with the prefix rather than to be zero. Set from what was measured over these
# parametrisations -- 2.5 last bits for the sensitivity, 9 for the gradient, and
# the carry bit-identical -- with room for another platform's summation order,
# and five orders of magnitude below the distance to the truncated gradient that
# ``test_the_oracle_can_tell_the_two_methods_apart`` holds open.
SENSITIVITY_BITS = 64.0
GRADIENT_BITS = 128.0

EXACT = {"lru": LRUStructuredRTRL, "rtu": RTUStructuredRTRL}


# ------------------------------------------------------------------- the graph
def recurrent_components(kind: str) -> tuple:
    """The cell, sized the way ``_construct_torso`` sizes it.

    Only the LRU has a readout width to set; the RTU's output is its two carries
    concatenated, so ``hidden_dim`` fixes it and ``output_dim`` is not its to
    take.
    """

    return backbone(
        kind,
        features=INPUT,
        hidden_dim=HIDDEN,
        output_dim=READOUT if kind == "lru" else None,
    )


def torso_network(kind: str) -> Sequence:
    """RTRRL's torso, built the way ``_construct_torso`` builds it.

    Nothing in front of the cell, and the ``LayerNorm`` behind it is affine-free,
    so the torso's whole parameter tree is the recurrent kernel's.
    """

    return Sequence(
        components=(
            *recurrent_components(kind),
            LayerNorm(use_scale=False, use_bias=False),
        )
    )


def projected_network(kind: str) -> Sequence:
    """The same torso with a learned projection put back in front of the cell.

    Not what RTRRL builds. It exists so that the boundary of the exactness claim
    is stated against a graph where it bites: a parameter reachable only through
    the cut carry, whose gradient exact credit therefore does not extend to.
    """

    return Sequence(
        components=(
            FFN(features=INPUT),
            Tanh(),
            *recurrent_components(kind),
            LayerNorm(use_scale=False, use_bias=False),
        )
    )


def settings(**overrides) -> RTRRLConfig:
    return RTRRLConfig(
        **{
            "num_envs": STREAMS,
            "gamma": 0.9,
            "lambda_pi": 0.8,
            "lambda_v": 0.7,
            "lambda_rnn": 0.6,
            "eta_pi": 0.5,
            "entropy_rate": 1e-3,
            **overrides,
        }
    )


def nontrivial(params, key):
    """Initialisation draws small numbers; a sensitivity needs larger ones."""

    leaves, structure = jax.tree.flatten(params)
    return jax.tree.unflatten(
        structure,
        [
            leaf + 0.3 * jax.random.normal(jax.random.fold_in(key, index), leaf.shape)
            for index, leaf in enumerate(leaves)
        ],
    )


class Harness:
    """One torso, two readouts, and a frozen sequence to drive them along.

    The blocks are RTRRL's own. Two torsos are built over the same network: one
    differentiating with the structured sensitivity and one with no derivative
    state at all, which is what makes the second an oracle for the first.
    """

    def __init__(
        self,
        kind: str,
        length: int,
        *,
        streams: int = 1,
        seed: int = 0,
        network=torso_network,
    ):
        keys = jax.random.split(jax.random.key(seed), 8)
        cfg = settings(num_envs=streams)
        self.kind = kind
        self.length = length
        self.streams = streams
        self.network = network(kind)
        self.exact = rtrrl.Torso(cfg, self.network, EXACT[kind](self.network.core))
        self.plain = rtrrl.Torso(cfg, self.network, TruncatedBPTT(self.network.core))
        self.actor = rtrrl.Actor(cfg, heads.StateStdGaussian(action_dim=ACTIONS))
        self.critic = rtrrl.Critic(cfg, heads.VNetwork())

        self.timesteps = [self._timestep(keys, step) for step in range(length)]
        self.chosen = [
            jax.random.normal(jax.random.fold_in(keys[4], step), (streams, 1, ACTIONS))
            for step in range(length)
        ]

        torso = self.exact.init((keys[5], keys[6], keys[7]), self.timesteps[0])
        self.params = nontrivial(torso.params, keys[0])
        self.start = torso.recurrence
        _, hidden = self.exact.apply(self.params, self.timesteps[0], self.start)
        self.actor_params = self.actor.init(keys[1], hidden, self.timesteps[0]).params
        self.critic_params = self.critic.init(keys[2], hidden, self.timesteps[0]).params

    def _timestep(self, keys, step: int) -> Timestep:
        shape = (self.streams,)
        return Timestep(
            obs=jax.random.normal(
                jax.random.fold_in(keys[0], step), (*shape, OBSERVED)
            ),
            action=jax.random.normal(
                jax.random.fold_in(keys[1], step), (*shape, ACTIONS)
            ),
            reward=jax.random.normal(jax.random.fold_in(keys[2], step), shape),
            done=(
                jnp.zeros(shape, dtype=jnp.bool_)
                if step == 0
                else jax.random.bernoulli(
                    jax.random.fold_in(keys[3], step), 0.25, shape
                )
            ),
        ).to_sequence()

    def ending_at(self, step: int) -> None:
        """Put an ending on one step of the frozen sequence, for every stream."""

        self.timesteps[step] = self.timesteps[step].replace(
            done=jnp.ones((self.streams, 1), dtype=jnp.bool_)
        )

    # -------------------------------------------------------------- driving it
    def carried(self, upto: int | None = None):
        """The online recurrence after ``upto`` steps of the exact walk."""

        recurrence = self.start
        for step in range(self.length if upto is None else upto):
            recurrence, _ = self.exact.apply(
                self.params, self.timesteps[step], recurrence
            )
        return recurrence

    def walked(self, params):
        """Where a truncation-free walk of the whole sequence arrives."""

        recurrence = self.start
        hidden = None
        for step in range(self.length):
            recurrence, hidden = self.plain.apply(
                params, self.timesteps[step], recurrence
            )
        return recurrence, hidden

    # ------------------------------------------------- the hidden state's past
    def online(self):
        """One exact step from the carried sensitivity: what RTRRL does."""

        recurrence = jax.lax.stop_gradient(self.carried(self.length - 1))
        last = self.timesteps[self.length - 1]

        def hidden_of(params):
            return self.exact.apply(params, last, recurrence)[1]

        return hidden_of

    def truncated(self):
        """The same step with no sensitivity behind it: what tbptt does."""

        recurrence = jax.lax.stop_gradient(self.carried(self.length - 1))
        last = self.timesteps[self.length - 1]

        def hidden_of(params):
            return self.plain.apply(params, last, recurrence)[1]

        return hidden_of

    def unrolled(self, since: int = 0, timesteps=None):
        """Every step from ``since`` chained, and differentiated as one."""

        walking = self.timesteps if timesteps is None else timesteps

        def hidden_of(params):
            recurrence = self.start
            hidden = None
            for step in range(since, self.length):
                recurrence, hidden = self.plain.apply(params, walking[step], recurrence)
            return hidden

        return hidden_of

    def live(self) -> list:
        """The same frozen sequence with nothing ending anywhere in it."""

        return [
            timestep.replace(done=jnp.zeros((self.streams, 1), dtype=jnp.bool_))
            for timestep in self.timesteps
        ]

    # ---------------------------------------------------- the gradient sources
    def sources(self, hidden_of) -> dict:
        """Both recurrent gradient sources and their sum, as RTRRL routes them.

        ``Core._per_stream``'s shape: one vector-Jacobian product out of the
        torso's output, driven by whichever readout's ascent direction is being
        asked about.
        """

        last = self.length - 1
        hidden, upstream = jax.vjp(hidden_of, self.params)
        actor_upward = jax.grad(self.actor.traced_objective, argnums=1)(
            self.actor_params, hidden, self.timesteps[last], self.chosen[last]
        )
        critic_upward = jax.grad(self.critic.traced_objective, argnums=1)(
            self.critic_params, hidden, self.timesteps[last]
        )
        return {
            "actor": upstream(actor_upward)[0],
            "critic": upstream(critic_upward)[0],
            "combined": upstream(actor_upward + critic_upward)[0],
        }

    # ------------------------------------------------------------- projections
    def recurrent(self, tree):
        """The recurrent kernel's own parameters, which is what is claimed."""

        return self.network.split(tree)["recurrence"]

    def projection(self, tree):
        """Everything the sequence puts in front of the recurrence."""

        return self.network.split(tree)["before"]


# ------------------------------------------------------------- the hidden state
@pytest.mark.parametrize("kind", CORES)
@pytest.mark.parametrize("length", LENGTHS)
def test_the_online_walk_arrives_where_the_unrolled_one_does(kind, length):
    """Before any derivative is compared, the two must be walking one sequence.

    The structured walk is a different traversal, not a wrapper: the RTU
    recomputes its step inside ``local_jacobian`` and the LRU injects a phantom
    into the carry ahead of its scan. Either could drift from the plain forward
    while every sensitivity stayed self-consistent, and then the comparisons
    below would be two different sequences agreeing about neither.
    """

    driven = Harness(kind, length, streams=STREAMS)
    walked, _ = driven.walked(driven.params)

    assert_within(
        flattened(driven.carried().carry),
        flattened(walked.carry),
        f"{kind}: carry after {length} steps",
        allowed=SENSITIVITY_BITS,
    )


# ------------------------------------------------------------ the sensitivities
@pytest.mark.parametrize("kind", CORES)
@pytest.mark.parametrize("length", LENGTHS)
def test_the_carried_sensitivity_is_the_unrolled_jacobian(kind, length):
    """Every recurrent parameter block, against ``jacrev`` of the whole prefix.

    The carried sensitivity holds one entry per hidden unit per parameter
    coordinate, because the recurrence is structured: unit ``h``'s state is a
    function of unit ``h``'s own previous state. The unrolled Jacobian has a
    hidden axis for the state and another for the unit whose parameter is being
    varied, so the claim is two things at once -- the unit-matched part is what
    is carried, and everything off it is zero. Checking only the first would
    pass for an implementation that had silently dropped real credit.
    """

    driven = Harness(kind, length, streams=STREAMS)
    carried = driven.carried().differentiation_state
    assert carried, f"{kind}: exact credit carries no sensitivity"

    jacobian = jax.jacrev(
        lambda params: _state_vector(driven, driven.walked(params)[0].carry)
    )(driven.params)

    wanted, got = {}, {}
    for path, leaf in jax.tree_util.tree_leaves_with_path(driven.recurrent(jacobian)):
        name = str(path[-1].key)
        if name not in carried:
            # The readout matrices see the state, they do not carry it, so no
            # sensitivity is kept for them and their gradient is exact anyway.
            continue
        # (streams, 2 * hidden, hidden, *parameter): the state's two real
        # components, then the unit whose parameter moved.
        block = leaf.reshape(driven.streams, 2, HIDDEN, *leaf.shape[2:])
        wanted[name] = jnp.moveaxis(jnp.diagonal(block, axis1=2, axis2=3), -1, 2)
        got[name] = _unit_axis(kind, carried[name])

        across = block * jnp.reshape(
            1.0 - jnp.eye(HIDDEN), (1, 1, HIDDEN, HIDDEN, *(1,) * (block.ndim - 4))
        )
        assert jnp.count_nonzero(across) == 0, (
            f"{kind}: {name} moves a hidden unit the structured recurrence says "
            "it cannot reach, so the carried part is not the whole Jacobian"
        )

    assert wanted, f"{kind}: no recurrent parameter block was compared"
    assert_within(
        flattened(got),
        flattened(wanted),
        f"{kind}: sensitivity after {length} steps",
        allowed=SENSITIVITY_BITS,
    )


def _state_vector(driven: Harness, carry):
    """The recurrent state as one real vector, however the core spells it."""

    cell = carry[driven.network.recurrent]
    if driven.kind == "rtu":
        return jnp.concatenate([cell.real, cell.imaginary], axis=-1)
    state = cell.state[:, 0]
    return jnp.concatenate([state.real, state.imag], axis=-1)


def _unit_axis(kind: str, sensitivity):
    """The carried sensitivity with its two real components on one axis.

    The RTU keeps them as an explicit ``(real, imaginary)`` axis; the LRU keeps
    one complex number behind a length-one time axis. Both mean the same pair.
    """

    if kind == "rtu":
        return sensitivity
    value = sensitivity[:, 0]
    return jnp.stack([value.real, value.imag], axis=1)


# -------------------------------------------------------- the gradient sources
@pytest.mark.parametrize("kind", CORES)
@pytest.mark.parametrize("length", LENGTHS)
@pytest.mark.parametrize("source", ("actor", "critic", "combined"))
def test_the_recurrent_gradient_from_each_source_is_the_unrolled_one(
    kind, length, source
):
    """Actor, critic, and the sum RTRRL actually ascends.

    Each is a different cotangent entering the same vector-Jacobian product, and
    a sensitivity that was right for one and wrong for another would be one that
    had been fitted to a particular readout. The combined source is what
    ``Core._per_stream`` forms; the two singles are what it is formed from.
    """

    driven = Harness(kind, length)
    got = driven.sources(driven.online())[source]
    wanted = driven.sources(driven.unrolled())[source]

    assert_within(
        flattened(driven.recurrent(got)),
        flattened(driven.recurrent(wanted)),
        f"{kind}: {source}-source recurrent gradient after {length} steps",
        allowed=GRADIENT_BITS,
    )


@pytest.mark.parametrize("kind", CORES)
def test_the_combined_source_is_the_two_sources_added(kind):
    """Which is what makes testing the three of them testing the routing too.

    ``Core._per_stream`` adds the two ascent directions before the single
    product it takes. The comment there says a gate would be a factor on one of
    the terms; this pins the ungated case it is a comment about.
    """

    driven = Harness(kind, 5)
    got = driven.sources(driven.online())

    assert_within(
        flattened(got["combined"]),
        flattened(jax.tree.map(jnp.add, got["actor"], got["critic"])),
        f"{kind}: combined source against its two parts",
        allowed=GRADIENT_BITS,
    )


@pytest.mark.parametrize("kind", CORES)
def test_exact_credit_reaches_every_parameter_the_torso_has(kind):
    """Which is a fact about the torso's shape, so it is checked on the shape.

    Exact credit is the recurrent kernel's. It covers the whole torso only
    because the torso is the kernel: nothing precedes the cell, and the
    ``LayerNorm`` behind it is affine-free and so contributes no parameter to
    credit. Asserting the split directly means the claim in this module's
    docstring cannot quietly stop being true if something is put back in front.
    """

    driven = Harness(kind, 5)
    places = driven.network.split(driven.params)

    assert places["before"] == {}, (
        f"{kind}: something precedes the recurrence, so the torso's gradient is "
        "no longer exact throughout -- see the projection test below"
    )
    assert places["after"] == {}, (
        f"{kind}: the LayerNorm behind the cell grew parameters, which RTRRL "
        "would then carry an eligibility trace for and the published one has not"
    )
    assert set(flattened(driven.recurrent(driven.params))) == set(
        flattened(driven.params)
    ), f"{kind}: the recurrent block is not the whole torso after all"


@pytest.mark.parametrize("kind", CORES)
@pytest.mark.parametrize("length", (3, 8))
def test_a_projection_in_front_of_the_cell_would_be_credited_one_step_only(
    kind, length
):
    """The boundary of the claim, run on a graph where it bites.

    RTRRL's torso has nothing in front of its cell, so on RTRRL's own graph this
    has nothing to catch. Put a learned projection there and the phantom, which
    is injected into a carry the same expression cut with ``stop_gradient``,
    leaves that projection unable to reach its own past: it takes exactly what
    ``tbptt`` gives it. That is why "exact" has to name the recurrent kernel
    rather than whatever graph the kernel is sitting in.
    """

    driven = Harness(kind, length, network=projected_network)
    exact = driven.sources(driven.online())["combined"]
    truncated = driven.sources(driven.truncated())["combined"]
    unrolled = driven.sources(driven.unrolled())["combined"]

    assert driven.projection(exact), f"{kind}: the projection contributed nothing"
    assert_within(
        flattened(driven.projection(exact)),
        flattened(driven.projection(truncated)),
        f"{kind}: projection gradient under exact credit",
        allowed=GRADIENT_BITS,
    )
    assert deviations(
        flattened(driven.projection(exact)),
        flattened(driven.projection(unrolled)),
        GRADIENT_BITS,
    ), (
        f"{kind}: the projection's truncated and unrolled gradients agree, so "
        "this sequence cannot say where exact credit stops"
    )


@pytest.mark.parametrize("kind", CORES)
@pytest.mark.parametrize("length", (2, 5, 8))
def test_the_oracle_can_tell_the_two_methods_apart(kind, length):
    """Otherwise every comparison above would pass on a sequence with no past.

    At the first step of a run the sensitivity is still zero and exact credit
    and truncation genuinely agree. A test that never left that state would be
    asserting nothing, so the distance between them is held open here, on the
    sequences the exactness tests use.
    """

    driven = Harness(kind, length)
    truncated = driven.sources(driven.truncated())["combined"]
    unrolled = driven.sources(driven.unrolled())["combined"]

    assert deviations(
        flattened(driven.recurrent(truncated)),
        flattened(driven.recurrent(unrolled)),
        GRADIENT_BITS,
    ), (
        f"{kind}: truncated and unrolled recurrent gradients agree after "
        f"{length} steps, so the exactness comparison has nothing to catch"
    )


# ---------------------------------------------------------------- episode reset
@pytest.mark.parametrize("kind", CORES)
def test_an_ending_begins_the_sensitivity_again(kind):
    """A stream that ended owes nothing to what it did before it ended.

    The sequence is put live everywhere and then given one ending, so that the
    only difference between the two references is the ending itself. The
    gradient the exact walk arrives at is held to the unroll that begins at the
    ending; the unroll of the same six steps with the ending taken out is what
    says that agreement is about the reset rather than about a prefix that never
    mattered.
    """

    driven = Harness(kind, 6)
    ending = 3
    driven.timesteps = driven.live()
    driven.ending_at(ending)

    got = driven.sources(driven.online())["combined"]
    since_reset = driven.sources(driven.unrolled(since=ending))["combined"]
    uninterrupted = driven.sources(driven.unrolled(timesteps=driven.live()))["combined"]

    assert_within(
        flattened(driven.recurrent(got)),
        flattened(driven.recurrent(since_reset)),
        f"{kind}: recurrent gradient after an ending",
        allowed=GRADIENT_BITS,
    )
    assert deviations(
        flattened(driven.recurrent(uninterrupted)),
        flattened(driven.recurrent(since_reset)),
        GRADIENT_BITS,
    ), f"{kind}: the ending changed nothing, so nothing was reset"


@pytest.mark.parametrize("kind", CORES)
def test_a_live_step_does_not_begin_the_sensitivity_again(kind):
    """The other half: clearing on every step would pass the test above."""

    driven = Harness(kind, 4, streams=STREAMS)
    driven.timesteps = driven.live()
    carried = driven.carried().differentiation_state

    assert any(
        jnp.count_nonzero(leaf) for leaf in jax.tree.leaves(carried)
    ), f"{kind}: four live steps left the sensitivity at zero"


# ----------------------------------------------------------- eligibility traces
def trace_blocks(cfg: RTRRLConfig) -> dict:
    network = torso_network("lru")
    return {
        "torso": rtrrl.Torso(cfg, network, TruncatedBPTT(network.core)),
        "actor": rtrrl.Actor(cfg, heads.StateStdGaussian(action_dim=ACTIONS)),
        "critic": rtrrl.Critic(cfg, heads.VNetwork()),
    }


@pytest.mark.parametrize(
    ("block", "declared"),
    (("torso", "lambda_rnn"), ("actor", "lambda_pi"), ("critic", "lambda_v")),
)
def test_the_trace_recursion_is_the_one_each_block_declares(block, declared):
    """``z <- gamma * lambda * (1 - reset) * z + F * grad``, per block and stream.

    Each block decays at its own rate, the reset is read before the decay rather
    than after it, and the emphasis multiplies the incoming gradient rather than
    the retained trace. Written out step by step, because the three are easy to
    permute into something that still trains.
    """

    cfg = settings()
    rate = cfg.gamma * getattr(cfg, declared)
    keys = jax.random.split(jax.random.key(4), 3)

    trace = {"kernel": jnp.zeros((STREAMS, 2, 3))}
    wanted = trace
    for step in range(5):
        gradient = {
            "kernel": jax.random.normal(
                jax.random.fold_in(keys[0], step), (STREAMS, 2, 3)
            )
        }
        reset = jax.random.bernoulli(
            jax.random.fold_in(keys[1], step), 0.4, (STREAMS,)
        ).astype(jnp.float32)
        emphasis = jax.random.uniform(jax.random.fold_in(keys[2], step), (STREAMS,))

        trace = trace_blocks(cfg)[block].advance_trace(
            trace, gradient, reset_before=reset, emphasis=emphasis
        )
        wanted = {
            "kernel": rate * (1.0 - reset)[:, None, None] * wanted["kernel"]
            + emphasis[:, None, None] * gradient["kernel"]
        }

    assert_within(flattened(trace), flattened(wanted), f"{block} trace recursion")


class _Emphasised:
    """``_advance_traces`` reads one field of the state, and this is it."""

    def __init__(self, emphasis):
        self.emphasis = emphasis


def test_the_emphasis_recursion_discounts_until_a_stream_ends():
    """``F <- gamma * F * (1 - reset) + reset``, which is what scales the traces.

    Kept apart from the recursion above because the emphasis is formed once in
    ``_advance_traces`` and handed to all three blocks: getting it wrong moves
    every trace by the same factor and leaves their ratios intact.
    """

    cfg = settings()
    network = torso_network("lru")
    core = rtrrl.Core(
        cfg,
        network,
        TruncatedBPTT(network.core),
        heads.StateStdGaussian(action_dim=ACTIONS),
        heads.VNetwork(),
    )
    zeros = {"kernel": jnp.zeros((STREAMS, 2, 3))}
    traces = {
        rtrrl.TORSO_GROUP: {"torso": zeros},
        rtrrl.HEAD_GROUP: {"actor": zeros, "critic": zeros},
    }
    gradients = {"torso": zeros, "actor": zeros, "critic": zeros}

    emphasis = jnp.ones((STREAMS,), dtype=jnp.float32)
    wanted = emphasis
    for reset in (
        jnp.asarray([0.0, 1.0, 0.0]),
        jnp.asarray([0.0, 0.0, 1.0]),
        jnp.asarray([0.0, 0.0, 0.0]),
    ):
        emphasis, _ = core._advance_traces(
            _Emphasised(emphasis), traces, gradients, reset
        )
        wanted = cfg.gamma * wanted * (1.0 - reset) + reset

    assert_within(
        flattened({"emphasis": emphasis}),
        flattened({"emphasis": wanted}),
        "emphasis recursion",
    )
