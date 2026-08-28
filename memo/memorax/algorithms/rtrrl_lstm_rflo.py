"""RTRRL over an LSTM torso, differentiated online by RFLO.

    RTRRL             the order things happen in, and the scan
    Environment       where every stream is
    Normalization     the scales the environment's numbers are read through
    Core              a torso, two heads, and everything that couples them
      Torso           an LSTM and the RFLO trace of its cell state
      Actor / Critic  a readout, and the objectives it names

This is a fourth RTRRL torso and not a configuration of any of the three that
came before it. Against ``rtrrl_aaai``'s LRU and RTU: the recurrence is dense,
so the cross-unit block RFLO drops is genuinely there and exact forward
sensitivity would cost a factor of the hidden width -- which is why those two
declare ``exact_rtrl`` and this one declares ``rflo``. Against
``rtrrl_ctrnn_rflo``: the same word names a different recurrence. The CTRNN's
leak is the constant ``1 - dt/tau`` and its trace is one matrix; the LSTM's
leak is the forget gate, which is itself a learned function of the state, and
its trace is three. Against ``drqn``, which is where an LSTM appears in this
repository today: that one differentiates by truncated backpropagation through
a replayed window, and this one never unrolls.

So there is no shared abstraction over the CTRNN's RFLO and this one, and the
two cells state their own equations. What they do share is the *shape* of the
method -- a sensitivity carried as state, a phantom on the leak path, a reset
at an episode boundary -- and that shape is the differentiation boundary the
sequence already owns rather than a base class either cell inherits.

Everything downstream of the torso is the same algorithm and the same code: the
TD error, the three eligibility traces, the emphasis, the entropy term, the
per-block optimizers, the followed reading copy and the order the transition is
processed in are ``rtrrl_aaai``'s, because they are the published algorithm's
and do not change with the recurrent kernel. This module owns the torso graph
and the parameter surface; it inherits the flow.

The parameters of the recurrence sit directly under ``torso`` rather than under
a backbone branch, for the reason ``rtrrl_ctrnn_rflo`` gives: there is one
recurrence here and naming it twice would put a constant in every run document.

Nothing here projects its parameters onto a set. The LSTM has no divisor and no
parameter whose sign changes the character of the step -- the gates are bounded
by construction and ``c`` grows at most linearly in the number of transitions
-- so the kernel names no constraint and the torso applies none. That the
mechanism exists and is not used here is the point of it being the kernel's to
name: see ``rtrrl_ctrnn_rflo``, where ``tau`` needs it.

``docs/rtrrl-lstm-rflo.md`` derives the trace and states what the approximation
drops. ``tests/test_lstm_rflo.py`` holds the cell against the equations and
against autodiff, and ``tests/unit/algorithms/rtrrl/test_lstm_rflo_assembly.py``
holds this graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from memorax.building import BuildContext, ComponentBuilder
from memorax.networks.components import LayerNorm
from memorax.networks.readouts import ACTOR_HEAD_FAMILY, CRITIC_HEAD_FAMILY
from memorax.networks.sequence import Sequence
from memorax.networks.sequence_models import RNN
from memorax.networks.sequence_models.lstm import (
    LSTM_DIFFERENTIATION_FAMILY,
    LSTMCell,
    LSTMConfig,
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
    reports_for,
)

#: What this entry may select, which is not everything the family carries.
#: ``tbptt`` is the exact judge the cell's tests measure RFLO against, and this
#: entry's name is a claim about which online gradient produced a result.
LSTM_DIFFERENTIATION = LSTM_DIFFERENTIATION_FAMILY.restricted("rflo")


# --------------------------------------------------------------- declarations
@dataclass(frozen=True)
class TorsoParameters:
    """The shared block: one LSTM, how it is differentiated, how it is stepped.

    ``hidden_dim``, ``forget_bias`` and ``layer_norm`` are static because they
    are built into the cell rather than read arithmetically by the running
    graph, so the members of one vmapped round cannot disagree about them.

    ``grad_clip`` carries ``rtrrl_aaai``'s condition unchanged: the intentional
    update derives its own step size and refuses a second bound over it, so
    ``optimizer.kind: iu`` requires ``grad_clip: 0``.
    """

    hidden_dim: int = param(valid=(1, 4096), search=(32, 512), static=True)
    # The bias column the forget gate is drawn with, and the one initialisation
    # choice this cell's own recurrence gives a reason for: `f_t` is what the
    # trace is multiplied by every transition, so a gate that starts near a
    # half starts the trace at a half-life of one step. One is the standard
    # value and the one every configuration under `config/` runs.
    forget_bias: float = param(valid=(-5.0, 5.0), search=(0.0, 2.0), static=True)
    # The affine-free normalization behind the cell, off by default as on the
    # CTRNN torso. It holds no parameters either way, so it changes what the
    # heads read and not what RFLO credits.
    layer_norm: bool = param(valid=[False, True], search=[False])
    # RFLO alone. The family carries `tbptt` as well -- it is the exact judge
    # the cell's tests measure the approximation against -- but this entry's
    # name is a claim about which online gradient produced a result, and an
    # experiment identity that a `kind:` line can quietly falsify is not one.
    # An LSTM torso trained by truncated backpropagation already has a name in
    # this repository, and it is `drqn`.
    differentiation: str = structure(
        branches=LSTM_DIFFERENTIATION.branches, search=("rflo",)
    )
    optimizer: str = structure(branches=RTRRL_TORSO_OPTIMIZERS.branches)
    grad_clip: float = param(valid=(0.0, 100.0), search=(0.0, 10.0))
    follow: float = param(valid=(0.0, 1.0), search=(0.0, 1.0))


@dataclass(frozen=True)
class RTRRLLstmRfloParameters:
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


PARAMETERS = describe_parameters(RTRRLLstmRfloParameters)


# ------------------------------------------------------------------ the torso
def _torso(parameters, components: ComponentBuilder, *, features: int):
    """The LSTM, whatever follows it, and the online method that credits it."""

    cell = LSTMCell(
        config=LSTMConfig(
            features=features,
            hidden_dim=int(parameters["torso.hidden_dim"]),
            forget_bias=float(parameters["torso.forget_bias"]),
        )
    )
    following = (
        (LayerNorm(use_scale=False, use_bias=False),)
        if bool(parameters["torso.layer_norm"])
        else ()
    )
    network = Sequence(components=(RNN(cell=cell), *following))
    differentiation = components.build(
        LSTM_DIFFERENTIATION,
        "torso.differentiation",
        core=network.core,
    )
    return network, differentiation


# -------------------------------------------------------------- the algorithm
class RTRRLLstmRflo(RTRRL):
    """RTRRL's flow, over the LSTM torso and its RFLO credit.

    Only the graph differs. Sharing the flow is not a convenience: the update
    order, the traces and the emphasis *are* the published algorithm's, they are
    the same for any torso, and a second copy of them would be a second place
    for them to drift apart under a change meant for both.
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
    ) -> RTRRLLstmRflo:
        """Declare the LSTM torso, the two readouts, and a rule for each."""

        gamma = numeric(parameters["gamma"])
        meta_rl = bool(parameters["meta_rl"])
        classes = action_classes(context.action_space)
        # What the cell reads: the observation, and under meta-RL the previous
        # action and reward beside it. Nothing widens it first -- each gate's
        # one weight matrix is the projection.
        features = int(context.observation_space.shape[0])
        if meta_rl:
            features += action_dim(context.action_space) + 1
        torso_optimizer = components.build(RTRRL_TORSO_OPTIMIZERS, "torso.optimizer")
        actor_optimizer = components.build(RTRRL_OPTIMIZERS, "actor.optimizer")
        critic_optimizer = components.build(RTRRL_OPTIMIZERS, "critic.optimizer")
        network, differentiation = _torso(parameters, components, features=features)
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
        )
