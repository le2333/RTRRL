"""RTRRL over the published CTRNN torso, differentiated online by RFLO.

    RTRRL             the order things happen in, and the scan
    Environment       where every stream is
    Normalization     the scales the environment's numbers are read through
    Core              a torso, two heads, and everything that couples them
      Torso           a CTRNN, RFLO, and the floor tau is projected onto
      Actor / Critic  a readout, and the objectives it names

This is the ``RTRRL-CTRNN-RFLO`` of the paper's own experiments, which is a
different network and a different online-gradient method from the LRU and RTU
torsos ``rtrrl_aaai`` offers -- not a fourth name for them. Three things here
have no counterpart there:

- **The recurrence is not diagonal.** A CTRNN unit reads every other unit's
  previous state, so the cross-unit block of ``dh/dh`` is real rather than
  identically zero. Exact forward sensitivity would cost a factor of the hidden
  width; RFLO is what the published algorithm spends instead, and what it drops
  is a term that is genuinely there. On the declared LRU and RTU cores the same
  approximation is a no-op, which is why they do not offer it.
- **A parameter is constrained.** ``tau`` is a divisor, and a step that carries
  it under ``dt`` inverts the sign of the leak. The published implementation
  clips it after every update; here the cell names the set and the torso
  projects onto it, so the constraint is part of the algorithm rather than a
  fix-up beside it.
- **The torso is one matrix.** Input weights, recurrent weights and the bias
  live in one array, optionally masked by a wiring, and the mask has to be the
  same in the forward and in the sensitivity.

Everything downstream of the torso is the same algorithm and the same code:
the TD error, the three eligibility traces, the emphasis, the entropy term, the
per-block optimizers, the followed reading copy and the order the transition is
processed in are ``rtrrl_aaai``'s, because they are the published algorithm's
and do not change with the recurrent kernel. This module owns the torso graph,
the parameter surface, and the projection; it inherits the flow.

The parameters of the recurrence sit directly under ``torso`` rather than under
a backbone branch. There is one recurrence here and naming it twice would put a
constant in every run document.

Rebuilt from ``RTRRL-AAAI25/rtrrl.py`` and ``RTRRL-AAAI25/models/ctrnn.py``.
The defects found in that implementation and what was done about each are in
``docs/rtrrl-ctrnn-rflo-corrections.md``. ``tests/test_ctrnn_rflo.py`` holds the
cell against the equations, ``tests/test_ctrnn_rflo_parity.py`` holds it against
the published implementation in both directions, and
``tests/unit/algorithms/rtrrl/test_ctrnn_rflo_assembly.py`` holds this graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from memorax.building import BuildContext, ComponentBuilder
from memorax.networks.components import LayerNorm
from memorax.networks.readouts import ACTOR_HEAD_FAMILY, CRITIC_HEAD_FAMILY
from memorax.networks.sequence import Sequence
from memorax.networks.sequence_models import RNN
from memorax.networks.sequence_models.ctrnn import (
    CTRNN_DIFFERENTIATION_FAMILY,
    WIRINGS,
    CTRNNCell,
    CTRNNConfig,
)
from memorax.parameters import describe_parameters, group, numeric, param, structure
from memorax.rl import action_classes, action_dim
from memorax.rl.normalization import (
    DISCOUNTED_NORMALIZATION_FAMILY,
    NORMALIZATION_FAMILY,
)

from .rtrrl_aaai import AVAILABLE_REPORTS as AVAILABLE_REPORTS
from .rtrrl_aaai import METRICS as METRICS
from .rtrrl_aaai import OBSERVATIONS as OBSERVATIONS
from .rtrrl_aaai import PARTS as PARTS
from .rtrrl_aaai import (
    RTRRL,
    RTRRL_OPTIMIZERS,
    RTRRL_TORSO_OPTIMIZERS,
)
from .rtrrl_aaai import TRAINING_METRICS as TRAINING_METRICS
from .rtrrl_aaai import (
    ActorParameters,
    CriticParameters,
    NormalizationParameters,
    RTRRLConfig,
    kernel_constraint,
    reports_for,
)

#: What this entry may select, which is not everything the family carries.
#: ``tbptt`` is the exact judge the cell's tests measure RFLO against, and this
#: entry's name is a claim about which online gradient produced a result.
CTRNN_DIFFERENTIATION = CTRNN_DIFFERENTIATION_FAMILY.restricted("rflo")


# --------------------------------------------------------------- declarations
@dataclass(frozen=True)
class TorsoParameters:
    """The shared block: one CTRNN, how it is differentiated, how it is stepped.

    ``dt``, ``tau_floor`` and the two structural choices are static because they
    are built into the cell rather than read arithmetically by the running
    graph, so the members of one vmapped round cannot disagree about them. The
    published run is ``dt: 1``, ``tau_floor: 1``, ``wiring: fully_connected``,
    ``layer_norm: false``.

    ``grad_clip`` carries ``rtrrl_aaai``'s condition unchanged: the two rules
    that size or bound their own step refuse a second bound over it, so the
    four branches selecting one -- ``input_iu``, ``output_iu``, ``input_obgd``
    and ``output_obgd`` -- require ``grad_clip: 0``. ``adam``, ``sgd`` and
    ``d_rtrrl`` keep it.
    """

    hidden_dim: int = param(valid=(1, 4096), search=(32, 512), static=True)
    # The Euler step. One is the published value and the only one every
    # reported result used; the recurrences are written with it throughout, so
    # a run that moves it moves the sensitivity with it.
    dt: float = param(valid=(0.0, 1.0), search=(0.1, 1.0), static=True)
    # The floor `tau` is projected back onto after every update, and it is a
    # bound on the parameter's domain rather than a preference. At zero the
    # projection lands `tau` exactly on a value the forward divides by. Between
    # zero and `dt` the leak `1 - dt/tau` is negative, and past `dt/2` the Euler
    # step diverges outright -- measured at 3.6e5 after twelve transitions with
    # `dt = 1, tau_floor = 0.25`. `_refuse_an_unstable_step` refuses both.
    tau_floor: float = param(valid=(0.0, 100.0), search=(1.0, 2.0), static=True)
    wiring: str = param(valid=list(WIRINGS), search=["fully_connected"])
    # The affine-free normalization behind the cell. The published default is
    # off, which is what every reported CTRNN result ran with; `rtrrl_aaai`'s
    # LRU torso has it on unconditionally, which is one more reason these are
    # two graphs rather than one graph with a backbone setting.
    layer_norm: bool = param(valid=[False, True], search=[False])
    # RFLO alone. The family carries `tbptt` as well -- it is the exact judge
    # the cell's tests measure the approximation against -- but this entry's
    # name is a claim about which online gradient produced a result, and an
    # experiment identity that a `kind:` line can quietly falsify is not one.
    # A CTRNN-TBPTT arm, if one is ever wanted, gets an entry that says so.
    differentiation: str = structure(
        branches=CTRNN_DIFFERENTIATION.branches, search=("rflo",)
    )
    optimizer: str = structure(branches=RTRRL_TORSO_OPTIMIZERS.branches)
    grad_clip: float = param(valid=(0.0, 100.0), search=(0.0, 10.0))
    follow: float = param(valid=(0.0, 1.0), search=(0.0, 1.0))


@dataclass(frozen=True)
class RTRRLCtrnnRfloParameters:
    torso: TorsoParameters = group(of=TorsoParameters)
    actor: ActorParameters = group(of=ActorParameters)
    critic: CriticParameters = group(of=CriticParameters)
    normalization: NormalizationParameters = group(of=NormalizationParameters)
    gamma: float = param(valid=(0.5, 0.9999), search=(0.9, 0.9999))
    lambda_pi: float = param(valid=(0.0, 1.0), search=(0.0, 1.0))
    lambda_v: float = param(valid=(0.0, 1.0), search=(0.0, 1.0))
    lambda_rnn: float = param(valid=(0.0, 1.0), search=(0.0, 1.0))
    eta_pi: float = param(valid=(0.0, 100.0), search=(0.0, 10.0))
    eta_f: float = param(valid=(0.0, 100.0), search=(0.0, 10.0))
    entropy_rate: float = param(valid=(0.0, 1.0), search=(1e-8, 1e-2), log=True)
    meta_rl: bool = param(valid=[False, True], search=[True])


PARAMETERS = describe_parameters(RTRRLCtrnnRfloParameters)


# ------------------------------------------------------------------ the torso
def _refuse_an_unstable_step(dt: float, tau_floor: float) -> None:
    """Refuse a pair of settings the recurrence is not defined for.

    Each is inside its own declared domain and the pair is not, which is a
    condition a parameter tree cannot state -- it conditions on branches, not
    on a sibling's value -- so the build refuses it with the reason rather than
    accepting a run that produces `NaN` on the transition that reaches it. The
    same shape as ``rtrrl_aaai``'s refusal of a clip over the intentional step.

    ``tau`` is a divisor and the projection lands on the floor exactly, so a
    floor of zero is a division by zero rather than a value near one: the
    forward returns `NaN` from the first step at which an update pushes `tau`
    down. Above zero but under ``dt`` the leak ``1 - dt/tau`` is negative,
    which past ``dt/2`` is an Euler step that diverges -- 3.6e5 after twelve
    transitions at ``dt = 1, tau_floor = 0.25``. ``tau_floor >= dt`` is the
    published domain, the one the recurrences are written for, and the one the
    integration is stable on.
    """

    if dt <= 0:
        raise ValueError(
            f"torso.dt is {dt}: a CTRNN with a non-positive integration step "
            "never advances its state, and its sensitivity is zero for every "
            "parameter"
        )
    if tau_floor < dt:
        raise ValueError(
            f"torso.tau_floor is {tau_floor} and torso.dt is {dt}: tau is what "
            "the step divides by and the projection lands on the floor exactly,"
            f" so this run reaches a leak of 1 - {dt}/{tau_floor} on the first "
            "update that pushes tau down. Hold the floor at or above dt, which "
            "is where the published run puts it"
        )


def _torso(parameters, components: ComponentBuilder, *, features: int):
    """The CTRNN, whatever follows it, and the online method that credits it."""

    dt = float(parameters["torso.dt"])
    tau_floor = float(parameters["torso.tau_floor"])
    _refuse_an_unstable_step(dt, tau_floor)
    cell = CTRNNCell(
        config=CTRNNConfig(
            features=features,
            hidden_dim=int(parameters["torso.hidden_dim"]),
            dt=dt,
            wiring=str(parameters["torso.wiring"]),
            tau_floor=tau_floor,
        )
    )
    following = (
        (LayerNorm(use_scale=False, use_bias=False),)
        if bool(parameters["torso.layer_norm"])
        else ()
    )
    network = Sequence(components=(RNN(cell=cell), *following))
    differentiation = components.build(
        CTRNN_DIFFERENTIATION,
        "torso.differentiation",
        core=network.core,
    )
    return network, differentiation, kernel_constraint(network)


# -------------------------------------------------------------- the algorithm
class RTRRLCtrnnRflo(RTRRL):
    """RTRRL's flow, over the CTRNN torso and its RFLO credit.

    Only the graph differs, and it differs in the three ways the module
    docstring names. Sharing the flow is not a convenience: the update order,
    the traces and the emphasis *are* the published algorithm's, they are the
    same for either torso, and a second copy of them would be a second place
    for the two to drift apart under a change meant for both.
    """

    observations = OBSERVATIONS

    @classmethod
    def graph(
        cls,
        parameters: dict[str, Any],
        components: ComponentBuilder,
        context: BuildContext,
        *,
        record=(),
    ) -> RTRRLCtrnnRflo:
        """Declare the CTRNN torso, the two readouts, and a rule for each."""

        gamma = numeric(parameters["gamma"])
        meta_rl = bool(parameters["meta_rl"])
        classes = action_classes(context.action_space)
        # What the cell reads: the observation, and under meta-RL the previous
        # action and reward beside it. Nothing widens it first -- the published
        # CTRNN's one weight matrix is the projection.
        features = int(context.observation_space.shape[0])
        if meta_rl:
            features += action_dim(context.action_space) + 1
        torso_optimizer = components.build(RTRRL_TORSO_OPTIMIZERS, "torso.optimizer")
        actor_optimizer = components.build(RTRRL_OPTIMIZERS, "actor.optimizer")
        critic_optimizer = components.build(RTRRL_OPTIMIZERS, "critic.optimizer")
        network, differentiation, constraint = _torso(
            parameters, components, features=features
        )
        return cls(
            RTRRLConfig(
                num_envs=context.num_envs,
                gamma=gamma,
                lambda_pi=numeric(parameters["lambda_pi"]),
                lambda_v=numeric(parameters["lambda_v"]),
                lambda_rnn=numeric(parameters["lambda_rnn"]),
                eta_pi=numeric(parameters["eta_pi"]),
                eta_f=numeric(parameters["eta_f"]),
                entropy_rate=numeric(parameters["entropy_rate"]),
                torso_grad_clip=numeric(parameters["torso.grad_clip"]),
                torso_follow=numeric(parameters["torso.follow"]),
                meta_rl=meta_rl,
                action_classes=classes,
                torso_optimizer=torso_optimizer,
                actor_optimizer=actor_optimizer,
                critic_optimizer=critic_optimizer,
            ),
            context.environment,
            context.environment_parameters,
            network,
            differentiation,
            components.build(
                ACTOR_HEAD_FAMILY,
                "actor.head",
                action_dim=action_dim(context.action_space),
                discrete=classes is not None,
            ),
            components.build(CRITIC_HEAD_FAMILY, "critic.head"),
            observation_normalization=components.build(
                NORMALIZATION_FAMILY, "normalization.observation"
            ),
            reward_normalization=components.build(
                DISCOUNTED_NORMALIZATION_FAMILY,
                "normalization.reward",
                discount=gamma,
            ),
            record=record,
            reports=reports_for(
                torso=torso_optimizer,
                actor=actor_optimizer,
                critic=critic_optimizer,
            ),
            torso_constraint=constraint,
        )
