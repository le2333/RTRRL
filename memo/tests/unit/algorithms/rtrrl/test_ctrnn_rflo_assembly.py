"""RTRRL-CTRNN-RFLO as a whole: what it selects, what it steps, what it refuses.

``tests/test_ctrnn_rflo.py`` holds the cell and its sensitivity against the
equations, and ``tests/test_ctrnn_rflo_parity.py`` holds both against the
published implementation. Neither says the algorithm is wired to them. These do:
every claim below is about the graph an entry builds and the state one
transition leaves behind.

The claims are the ones the algorithm is answerable for beyond its torso -- the
transition inside a real step, the sensitivity the torso carries, each block's
eligibility trace at its own decay, the TD error, the constrained parameter,
what an ending restarts, and what a member of a vmapped round is owed. The
update *order* is checked where it has content: the first step of a run with no
immediate objective moves nothing, because the step is taken along the trace as
it stood before this transition's derivative joined it.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import jax
import jax.numpy as jnp
import pytest
import yaml

from entries._ensemble import GroupError, swept_parameters
from memorax.algorithms import rtrrl_ctrnn_rflo as ctrnn_rflo
from memorax.algorithms.rtrrl_aaai import Recurrence
from memorax.algorithms.rtrrl_ctrnn_rflo import RTRRLCtrnnRflo
from memorax.assembly import BuildRequest, EnvironmentSpec, assemble
from memorax.networks.sequence import Sequence
from memorax.networks.sequence_models import RNN, RTUCell, RTUConfig
from memorax.networks.sequence_models.ctrnn import (
    CTRNN_DIFFERENTIATION_FAMILY,
    CTRNNCell,
    CTRNNConfig,
    CTRNNRflo,
)
from memorax.parameters import expand, flatten
from tests.support.builders import graph_of
from tests.support.environments import TinyContinuousEnv
from tests.support.parameters import kinds

ENVS = 3
OBSERVATION = 2  # TinyContinuousEnv's, and its action is as wide again
ACTION = 2
HIDDEN = 4
CELL = "components_0"

# Every decay live, so a claim about one of them is not a claim about zero.
LIVE = {
    "gamma": 0.9,
    "lambda_pi": 0.8,
    "lambda_v": 0.7,
    "lambda_rnn": 0.6,
    "eta_pi": 0.5,
    "eta_f": 0.5,
    "entropy_rate": 1e-3,
}


def parameters(**overrides):
    return expand(
        ctrnn_rflo.PARAMETERS,
        {
            "torso.hidden_dim": HIDDEN,
            "torso.dt": 1.0,
            "torso.tau_floor": 1.0,
            "torso.wiring": "fully_connected",
            "torso.layer_norm": False,
            "torso.differentiation.kind": "rflo",
            "torso.optimizer.kind": "adam",
            "torso.optimizer.adam.lr": 1e-3,
            "torso.grad_clip": 1.0,
            "torso.follow": 0.25,
            "actor.optimizer.kind": "adam",
            "actor.optimizer.adam.lr": 1e-3,
            "critic.optimizer.kind": "adam",
            "critic.optimizer.adam.lr": 1e-3,
            "actor.head.kind": "state_std",
            "critic.head.kind": "value",
            "normalization.observation.kind": "none",
            "normalization.reward.kind": "none",
            "meta_rl": True,
            **LIVE,
            **overrides,
        },
    )


def environment(identifier, **options):
    del identifier, options
    env = TinyContinuousEnv()
    return env, env.default_params


def build(**overrides) -> RTRRLCtrnnRflo:
    """The graph an entry builds, on an environment small enough to read."""

    built = assemble(
        RTRRLCtrnnRflo,
        BuildRequest(
            parameters=parameters(**overrides),
            environment=EnvironmentSpec(
                id="tiny", backend=None, observed=None, episode_length=8
            ),
            num_envs=ENVS,
        ),
        environment_factory=environment,
    )
    return graph_of(built)


def run(agent, rounds, key=1):
    state = agent.init(jax.random.key(0))
    return agent.train(jax.random.key(key), state, rounds * agent.cfg.num_envs)


def apart(a, b) -> float:
    return max(
        float(jnp.max(jnp.abs(x - y)))
        for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b))
    )


def cell_of(agent) -> CTRNNCell:
    return agent.core.torso._network.components[0].cell


def torso_params(state):
    return state.core.torso.params[CELL]["cell"]


# ------------------------------------------------------------- what it selects
def test_the_torso_is_a_ctrnn_and_holds_exactly_its_two_parameters():
    """One matrix and one time constant, and nothing before or behind them.

    The width says the layout: ``[observation, previous action, previous
    reward, hidden, bias]``, one column each, in one array. A projection in
    front of the cell would show up here as a second component, and the
    affine-free normalization behind it holds nothing to credit -- so the whole
    torso is the recurrence, which is what makes RFLO's credit the torso's.
    """

    state = build().init(jax.random.key(0))
    tree = state.core.torso.params

    assert set(tree) == {CELL}
    assert set(tree[CELL]["cell"]) == {"W", "tau"}
    features = OBSERVATION + ACTION + 1
    assert tree[CELL]["cell"]["W"].shape == (HIDDEN, features + HIDDEN + 1)
    assert tree[CELL]["cell"]["tau"].shape == (HIDDEN,)


def test_the_selected_differentiation_is_rflo_and_carries_its_sensitivity():
    agent = build()
    assert isinstance(agent.core.torso._differentiation, CTRNNRflo)

    after, _ = run(agent, 3)
    sensitivity = after.core.torso.recurrence.differentiation_state
    assert set(sensitivity) == {"W", "tau"}
    assert sensitivity["W"].shape[0] == ENVS
    assert float(jnp.abs(sensitivity["W"]).max()) > 0.0


def test_this_entry_may_not_select_anything_but_rflo():
    """The entry's name is a claim about which online gradient produced a run.

    The family carries ``tbptt`` -- it is the exact judge the cell's tests
    measure the approximation against -- and the entry does not offer it, so a
    result filed under ``rtrrl_ctrnn_rflo`` cannot have been produced by
    anything else. An experiment identity a ``kind:`` line can quietly falsify
    is not an identity.
    """

    assert set(kinds(ctrnn_rflo.PARAMETERS, "torso.differentiation")) == {"rflo"}
    assert set(CTRNN_DIFFERENTIATION_FAMILY.branches) == {"rflo", "tbptt"}

    with pytest.raises(KeyError):
        build(**{"torso.differentiation.kind": "tbptt"})


def test_the_normalization_behind_the_cell_is_a_choice_that_holds_no_parameters():
    """Selecting it changes what the heads read and not what is credited."""

    plain = build().init(jax.random.key(0))
    normalized = build(**{"torso.layer_norm": True}).init(jax.random.key(0))

    assert jax.tree.structure(plain.core.torso.params) == jax.tree.structure(
        normalized.core.torso.params
    )
    assert apart(torso_params(plain), torso_params(normalized)) == 0.0
    moved = run(build(**{"torso.layer_norm": True}), 2)[0]
    assert apart(torso_params(moved), torso_params(normalized)) > 0.0


def test_the_configuration_the_issue_names_builds():
    """``hidden_dim: 32``, which is the published width, under both entries."""

    agent = build(**{"torso.hidden_dim": 32})
    state = agent.init(jax.random.key(0))
    assert state.core.torso.params[CELL]["cell"]["tau"].shape == (32,)


# --------------------------------------------------------- what one step is
def test_one_transition_is_the_euler_step_of_the_ctrnn_ode():
    """The transition inside a real step, recomputed from the parameters alone.

    The reading copy is what acts and what the torso is walked with, so this is
    that copy rather than the stepped one -- getting those two the wrong way
    round is a mistake the equation would not catch on the first transition,
    when they are still the same tree.
    """

    agent = build(**{"torso.follow": 1.0})
    state, _ = run(agent, 4)
    torso = agent.core.torso
    timestep = state.timestep.to_sequence()
    weights = state.core.torso.slow_params[CELL]["cell"]

    advanced, _ = torso.apply(
        state.core.torso.slow_params, timestep, state.core.torso.recurrence
    )

    carry = state.core.torso.recurrence.carry[0]
    row = jnp.concatenate(
        [torso._input(timestep)[:, 0], carry, jnp.ones((ENVS, 1))], -1
    )
    activated = jnp.tanh(row @ weights["W"].T)
    wanted = carry + (1.0 / weights["tau"]) * (activated - carry)
    assert apart(advanced.carry[0], wanted) < 1e-5


def test_the_sensitivity_the_algorithm_carries_is_the_cell_s_own_recurrence():
    """The torso advances the trace the differentiation component defines.

    Run standalone over the same input, from the same carry and the same
    sensitivity, and the two have to be one number -- otherwise the algorithm
    is carrying something that resembles RFLO rather than RFLO.
    """

    agent = build()
    state, _ = run(agent, 4)
    torso = agent.core.torso
    timestep = state.timestep.to_sequence()

    advanced, _ = torso.apply(
        state.core.torso.slow_params, timestep, state.core.torso.recurrence
    )
    standalone = CTRNNRflo(torso._network.components[0])
    _, _, wanted = standalone(
        state.core.torso.slow_params[CELL],
        torso._input(timestep),
        timestep.done,
        state.core.torso.recurrence.carry[0],
        state.core.torso.recurrence.differentiation_state,
    )
    assert apart(advanced.differentiation_state, wanted) == 0.0


def test_the_step_is_taken_along_the_trace_as_it_stood():
    """Update order: the first transition of a traced-only run moves nothing.

    The trace the step reads has not been written yet, so a run whose whole
    objective is traced has no first update. A step taken along the advanced
    trace instead would move every block on transition one.
    """

    agent = build(entropy_rate=0.0)
    start = agent.init(jax.random.key(0))
    after, _ = agent.train(jax.random.key(1), start, agent.cfg.num_envs)

    for name in ("torso", "actor", "critic"):
        block = getattr(after.core, name)
        assert apart(block.params, getattr(start.core, name).params) == 0.0
    assert float(jnp.abs(after.core.torso.traces[CELL]["cell"]["W"]).max()) > 0.0


def test_the_td_error_is_the_one_step_return_against_the_carried_value():
    agent = build()
    state, metrics = run(agent, 6)

    value = metrics.forward.critic.value
    reward = metrics.interaction.reward
    terminal = metrics.interaction.terminal
    # A transition's error is read one row later than the reward that closed
    # it, because the value it is measured against is the one this step
    # carried in from the last.
    wanted = reward[:-1] + 0.9 * value[1:] * (1 - terminal[:-1]) - value[:-1]

    assert apart(metrics.update.td_error[1:], wanted) < 1e-5
    assert float(jnp.abs(metrics.update.td_error).max()) > 1e-6


@pytest.mark.parametrize(
    "decay,owner",
    (("lambda_rnn", "torso"), ("lambda_pi", "actor"), ("lambda_v", "critic")),
)
def test_each_decay_moves_one_block_s_trace_and_no_other(decay, owner):
    """Which trace forgets at which rate, one block at a time.

    Two transitions, because the first writes the traces and the second is the
    first at which a decay has been applied to anything.
    """

    reference, _ = run(build(entropy_rate=0.0), 2)
    changed, _ = run(build(entropy_rate=0.0, **{decay: 0.1}), 2)

    for name in ("torso", "actor", "critic"):
        moved = apart(
            getattr(reference.core, name).traces, getattr(changed.core, name).traces
        )
        if name == owner:
            assert moved > 0.0, f"{decay} did not reach {name}"
        else:
            assert moved == 0.0, f"{decay} reached {name}"


def test_a_zero_decay_leaves_each_trace_this_step_s_emphasised_derivative():
    """The eligibility recurrence, read off the algorithm's own reports.

    ``z_t = gamma*lambda*(1 - reset)*z_{t-1} + F_t * p_t``, so at
    ``lambda = 0`` the carry is gone and what is left is the derivative
    weighted by the emphasis -- one exact identity per block, in the norms the
    graph files. All three decays are set, because a claim about the actor's
    trace is not a claim about the torso's.
    """

    agent = build(entropy_rate=0.0, lambda_pi=0.0, lambda_v=0.0, lambda_rnn=0.0)
    _, metrics = run(agent, 5)
    emphasis = metrics.update.emphasis

    torso = metrics.update.torso
    assert (
        apart(torso.trace_norm["recurrence"], emphasis * torso.grad_norm["recurrence"])
        < 1e-5
    )
    assert float(jnp.abs(torso.grad_norm["recurrence"]).max()) > 1e-6

    for name in ("actor", "critic"):
        block = getattr(metrics.update, name)
        assert apart(block.trace_norm, emphasis * block.grad_norm) < 1e-5
        assert float(jnp.abs(block.grad_norm).max()) > 1e-6


def test_eta_f_scales_the_torso_s_error_and_no_other_group_s():
    """Which parameter group each scalar the update forms belongs to.

    Two transitions is the horizon at which this has content: the first moves
    nothing, and the second is the first update -- taken along a trace both
    runs wrote identically, from heads that read a torso neither had moved yet.
    So the only thing that can differ at the end of it is what ``eta_f`` scales.
    """

    reference, _ = run(build(entropy_rate=0.0, eta_f=0.5), 2)
    changed, _ = run(build(entropy_rate=0.0, eta_f=2.0), 2)

    assert apart(torso_params(reference), torso_params(changed)) > 0.0
    for name in ("actor", "critic"):
        moved = apart(
            getattr(reference.core, name).params, getattr(changed.core, name).params
        )
        assert moved == 0.0, f"eta_f reached the {name}"


def test_an_ending_restarts_the_emphasis_the_trace_and_the_sensitivity():
    """Everything the episode owned is cleared together, and only then.

    ``TinyContinuousEnv`` terminates on its own, so this reads a real ending
    rather than one the test arranged: the emphasis returns to one exactly
    where a transition began after a ``done``.
    """

    agent = build(entropy_rate=0.0)
    _, metrics = run(agent, 12)

    emphasis = metrics.update.emphasis
    ended = metrics.interaction.done
    assert bool(jnp.any(ended)), "nothing ended, so this asserts nothing"

    wanted = jnp.where(ended[:-1], 1.0, 0.9 * emphasis[:-1])
    assert apart(emphasis[1:], wanted) < 1e-6
    assert float(jnp.min(emphasis)) < 1.0, "nothing decayed, so nothing restarted"


def test_a_stream_that_ended_walks_the_torso_from_an_empty_sensitivity():
    """The cell's reset reaching the algorithm, on the stream that ended.

    The emphasis above is the algorithm's own restart; this is the torso's.
    One stream is told its episode ended and the sensitivity it comes out with
    has to be the one a sequence that has not run would have -- while its
    neighbours, told nothing, keep theirs.
    """

    agent = build()
    state, _ = run(agent, 5)
    torso = agent.core.torso
    ended = state.timestep.replace(done=jnp.arange(ENVS) == 0).to_sequence()

    advanced, _ = torso.apply(
        state.core.torso.slow_params, ended, state.core.torso.recurrence
    )
    fresh, _ = torso.apply(
        state.core.torso.slow_params,
        ended,
        Recurrence(
            carry=torso._network.initialize_carry(jax.random.key(0), (ENVS, None)),
            differentiation_state=torso._differentiation.initialize(
                jax.random.key(0), (ENVS, None)
            ),
        ),
    )
    first = jax.tree.map(lambda leaf: leaf[:1], advanced.differentiation_state)
    assert (
        apart(first, jax.tree.map(lambda leaf: leaf[:1], fresh.differentiation_state))
        == 0.0
    )

    rest = jax.tree.map(lambda leaf: leaf[1:], advanced.differentiation_state)
    assert (
        apart(rest, jax.tree.map(lambda leaf: leaf[1:], fresh.differentiation_state))
        > 0.0
    )


def test_every_recurrent_parameter_of_the_cell_learns():
    """Both leaves move, which RFLO's two recurrences are what make possible."""

    start = build().init(jax.random.key(0))
    after, _ = run(build(), 6)

    for leaf in ("W", "tau"):
        assert apart(torso_params(after)[leaf], torso_params(start)[leaf]) > 0.0


# --------------------------------------------------------- the constrained tau
def test_tau_is_projected_back_onto_its_floor_after_a_step():
    """A step that would carry ``tau`` under the floor leaves it at the floor.

    The floor is raised above every drawn value, so the whole vector has to be
    projected and the same run without a floor is there to say what would
    otherwise have happened. Below ``dt`` the leak ``1 - dt/tau`` changes sign,
    so this is a bound on the parameter's domain rather than a preference.
    """

    floor = 6.0  # above the initial draw, which is uniform on [1, 5]
    settings = {"torso.optimizer.adam.lr": 1.0}
    held, _ = run(build(**settings, **{"torso.tau_floor": floor}), 4)
    # The published floor, and the lowest one this entry will build: a floor of
    # zero is refused, so the contrast has to come from a legal run rather than
    # from an unconstrained one.
    low, _ = run(build(**settings, **{"torso.tau_floor": 1.0}), 4)

    assert float(jnp.min(torso_params(held)["tau"])) >= floor - 1e-6
    assert float(jnp.min(torso_params(low)["tau"])) < floor, (
        "the run at the published floor stayed above the raised one anyway, "
        "so the raised one says nothing"
    )


def test_the_reading_copy_is_inside_the_constraint_too():
    """Both copies are points of the set, so acting never divides by a bad tau.

    With a partial follow the reading copy is a convex combination of two
    points that are already inside, but it starts from the drawn parameters --
    which a raised floor puts outside -- so the projection has to reach it.
    """

    floor = 6.0
    agent = build(
        **{
            "torso.tau_floor": floor,
            "torso.optimizer.adam.lr": 1.0,
            "torso.follow": 0.25,
        }
    )
    state, _ = run(agent, 4)

    slow = state.core.torso.slow_params[CELL]["cell"]["tau"]
    assert float(jnp.min(slow)) >= floor - 1e-6


def test_the_floor_is_the_kernel_s_to_name_and_a_kernel_may_name_none():
    """The projection belongs to the component whose parameter it bounds.

    The algorithm applies whatever the cell states and knows only where that
    cell's parameters sit in the sequence's tree. A core that states nothing --
    which is every other core in the repository -- gets no projection at all,
    so this is not something a kernel has to implement to be usable here.
    """

    assert (
        float(cell_of(build()).constrain({"tau": jnp.asarray([-1.0])})["tau"][0]) == 1.0
    )
    raised = build(**{"torso.tau_floor": 3.0, "torso.dt": 1.0})
    assert (
        float(cell_of(raised).constrain({"tau": jnp.asarray([-1.0])})["tau"][0]) == 3.0
    )

    silent = Sequence(
        components=(RNN(cell=RTUCell(config=RTUConfig(features=2, hidden_dim=2))),)
    )
    assert ctrnn_rflo._constraint(silent) is None


@pytest.mark.parametrize(
    "settings,reason",
    (
        ({"torso.tau_floor": 0.0}, "tau_floor"),
        ({"torso.tau_floor": 0.5, "torso.dt": 1.0}, "tau_floor"),
        ({"torso.dt": 0.0, "torso.tau_floor": 1.0}, "torso.dt"),
    ),
)
def test_a_pair_of_settings_the_recurrence_is_undefined_for_is_refused(
    settings, reason
):
    """Each value is inside its own domain and the pair is not.

    A parameter tree conditions on branches, not on a sibling's value, so this
    is a build-time refusal rather than a declaration -- the same shape as
    ``rtrrl_aaai``'s refusal of an outer clip over the intentional step. What
    it buys is measured in the test below: without it these configurations do
    not fail, they produce numbers.
    """

    with pytest.raises(ValueError, match=reason):
        build(**settings)


def test_the_refused_settings_are_the_ones_that_would_have_produced_nan():
    """Why the refusal is a refusal and not a warning.

    ``tau`` is what the step divides by and the projection lands on the floor
    exactly, so a floor of zero is a division by zero on the first update that
    pushes ``tau`` down -- not a small number, ``NaN``. A floor under ``dt``
    inverts the leak, and past ``dt/2`` the Euler step diverges. Both are driven
    here through the cell directly, which is the only way left to reach them.
    """

    def walked(dt, tau):
        core = RNN(
            cell=CTRNNCell(
                config=CTRNNConfig(features=2, hidden_dim=3, dt=dt, tau_floor=tau)
            )
        )
        x = jax.random.normal(jax.random.key(1), (1, 12, 2))
        done = jnp.zeros((1, 12), dtype=bool)
        empty = jnp.zeros((1, 3))
        params = core.init(jax.random.key(2), x, done=done, initial_carry=empty)[
            "params"
        ]
        # The parameters a run reaches once the projection has landed on the
        # floor, which is where an update that overshot leaves them.
        landed = {"cell": {**params["cell"], "tau": jnp.full((3,), tau)}}
        _, output = cast(
            "tuple[Any, Any]",
            core.apply({"params": landed}, x, done=done, initial_carry=empty),
        )
        return jnp.asarray(output)

    assert not bool(jnp.all(jnp.isfinite(walked(1.0, 0.0)))), "a zero floor is finite"
    assert float(jnp.abs(walked(1.0, 0.25)).max()) > 1e4, "a floor under dt/2 is calm"
    assert float(jnp.abs(walked(1.0, 1.0)).max()) < 10.0, "the published domain is not"


# ------------------------------------------------------- what a sweep may vary
STRUCTURAL = (
    "torso.hidden_dim",
    "torso.dt",
    "torso.tau_floor",
    "torso.wiring",
    "torso.layer_norm",
    "torso.optimizer.kind",
    "meta_rl",
)
SWEEPABLE = (
    "gamma",
    "lambda_rnn",
    "lambda_pi",
    "lambda_v",
    "eta_f",
    "eta_pi",
    "entropy_rate",
    "torso.optimizer.adam.lr",
    "torso.grad_clip",
    "torso.follow",
)


@pytest.mark.parametrize("name", STRUCTURAL)
def test_a_structural_leaf_is_declared_static(name):
    assert flatten(ctrnn_rflo.PARAMETERS)[name].static, f"{name} may be swept"


@pytest.mark.parametrize("name", SWEEPABLE)
def test_a_leaf_the_graph_reads_arithmetically_may_be_swept(name):
    assert not flatten(ctrnn_rflo.PARAMETERS)[name].static, f"{name} may not be swept"


def members(**varying) -> Any:
    """Two run documents differing in exactly what is passed.

    ``swept_parameters`` reads a member's identity and its parameters and
    nothing else, so that is all a member has to be here.
    """

    base = parameters()
    return cast(
        "Any",
        [
            (
                SimpleNamespace(
                    identity=SimpleNamespace(run_id=name),
                    algorithm=SimpleNamespace(parameters=values),
                ),
                None,
            )
            for name, values in (("a", dict(base)), ("b", dict(base) | dict(varying)))
        ],
    )


def test_a_group_may_not_vary_the_shape_of_the_recurrence():
    with pytest.raises(GroupError, match="torso.hidden_dim"):
        swept_parameters(members(**{"torso.hidden_dim": 8}), ctrnn_rflo.PARAMETERS)


def test_a_group_may_vary_a_decay():
    swept = swept_parameters(members(lambda_rnn=0.1), ctrnn_rflo.PARAMETERS)
    assert set(swept) == {"lambda_rnn"}


# --------------------------------------------------------- one graph, N members
def test_a_member_of_a_vmapped_round_is_the_run_it_would_have_been_alone():
    """The ensemble entry's arithmetic, against the same seeds run one at a time.

    Not bit-for-bit, and it cannot be: ``jax.vmap`` rewrites the computation
    into batched operations and XLA reduces them in a different order, which
    ``tests/unit/runtime/test_ensemble.py`` measures on a one-member round. What
    must hold is that a member is a function of its seed -- so the comparison is
    to a budget, and the budget is small enough that a member picking up its
    neighbour's parameters could not fit inside it.
    """

    agent = build()
    seeds = (0, 1, 2)
    keys = jnp.stack([jax.random.key(seed) for seed in seeds])
    steps = 4 * agent.cfg.num_envs

    def alone(key):
        return agent.train(jax.random.key(9), agent.init(key), steps)[0]

    grouped = jax.vmap(alone)(keys)
    for index, seed in enumerate(seeds):
        one = alone(jax.random.key(seed))
        member = jax.tree.map(lambda leaf: leaf[index], grouped)  # noqa: B023
        assert apart(member.core.torso.params, one.core.torso.params) < 1e-5
        assert apart(member.core.actor.params, one.core.actor.params) < 1e-5
        assert apart(member.core.critic.params, one.core.critic.params) < 1e-5

    # And the members are not copies of each other, by a margin wider than the
    # budget above -- so the agreement is about seeds rather than about every
    # member having been the same run.
    first = jax.tree.map(lambda leaf: leaf[0], grouped)
    second = jax.tree.map(lambda leaf: leaf[1], grouped)
    assert apart(first.core.torso.params, second.core.torso.params) > 1e-3


# ------------------------------------------------------------- the declarations
def test_the_algorithm_declares_its_own_surface_beside_the_shared_one():
    """Its parameters are its own; its metrics and its schema are RTRRL's.

    A reader comparing an LRU run against a CTRNN one is asking the same
    question of both, so the series they file have to have the same names.
    """

    from memorax.algorithms import rtrrl_aaai

    assert ctrnn_rflo.METRICS == rtrrl_aaai.METRICS
    assert ctrnn_rflo.OBSERVATIONS == rtrrl_aaai.OBSERVATIONS
    assert ctrnn_rflo.PARAMETERS != rtrrl_aaai.PARAMETERS
    assert "torso.hidden_dim" in flatten(ctrnn_rflo.PARAMETERS)
    assert "torso.backbone.kind" not in flatten(ctrnn_rflo.PARAMETERS)
    assert "torso.backbone.kind" in flatten(rtrrl_aaai.PARAMETERS)


def test_the_two_entries_declare_one_schema_and_one_set_of_metrics():
    from entries import rtrrl_ctrnn_rflo, rtrrl_ctrnn_rflo_ensemble

    assert rtrrl_ctrnn_rflo.PARAMETERS == rtrrl_ctrnn_rflo_ensemble.PARAMETERS
    assert rtrrl_ctrnn_rflo.METRICS == rtrrl_ctrnn_rflo_ensemble.METRICS
    assert rtrrl_ctrnn_rflo_ensemble.GROUPED is True
    assert getattr(rtrrl_ctrnn_rflo, "GROUPED", False) is False


# ---------------------------------------------------- the launch documents
# The launch documents live beside `memo/` rather than inside it: they are the
# repository's, and the image's build context is this directory.
EXPERIMENTS = Path(__file__).resolve().parents[5] / "experiments"
ALONE = EXPERIMENTS / "rtrrl issue65 ctrnn rflo.yaml"
GROUPED = EXPERIMENTS / "rtrrl issue65 ctrnn rflo ensemble.yaml"


def document(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def paths(space, prefix=""):
    """A search space as the flat leaf names an entry reads."""

    found = {}
    for name, value in space.items():
        key = f"{prefix}{name}"
        if isinstance(value, dict):
            found |= paths(value, prefix=f"{key}.")
        else:
            found[key] = value
    return found


def test_the_two_launch_documents_differ_only_in_the_entry_and_what_it_runs():
    """The ensemble configuration is the other one with a different entry.

    That is what "the same parameter schema" means where it is read rather than
    where it is declared: a run document written for either entry is a run
    document for the other, and a divergence in `space` would be two graphs
    being compared under one issue number.
    """

    alone, grouped = document(ALONE), document(GROUPED)

    assert alone["entry"] == "rtrrl_ctrnn_rflo"
    assert grouped["entry"] == "rtrrl_ctrnn_rflo_ensemble"
    assert alone["space"] == grouped["space"]
    assert grouped["environment"]["seeds"] == [0, 1, 2]

    differing = {
        key for key in set(alone) | set(grouped) if alone.get(key) != grouped.get(key)
    }
    assert differing == {"entry", "name", "description", "environment"}


def test_the_published_configuration_resolves_against_what_the_entry_declares():
    """Every leaf the document names is one the entry reads, and it builds.

    The issue asks for `torso.hidden_dim: 32` under this entry; this is that
    request taken from the file it would arrive in, resolved against the
    declared tree, and built into a graph.
    """

    declared = flatten(ctrnn_rflo.PARAMETERS)
    space = paths(document(ALONE)["space"])

    unknown = sorted(name for name in space if name not in declared)
    assert not unknown, f"the document names {unknown}, which the entry does not"

    chosen = {name: values[0] for name, values in space.items()}
    assert chosen["torso.hidden_dim"] == 32
    assert chosen["torso.differentiation.kind"] == "rflo"

    agent = build(**chosen)
    state, _ = run(agent, 2)
    assert torso_params(state)["W"].shape[0] == 32
