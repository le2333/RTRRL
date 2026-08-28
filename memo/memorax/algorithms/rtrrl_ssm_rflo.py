"""RTRRL over a dense linear state-space torso, differentiated by RFLO.

    RTRRL             the order things happen in, and the scan
    Environment       where every stream is
    Normalization     the scales the environment's numbers are read through
    Core              a torso, two heads, and everything that couples them
      Torso           a dense SSM, RFLO, and the norm ball A is projected onto
      Actor / Critic  a readout, and the objectives it names

This entry exists to answer a question the other RTRRL entries raise and cannot
settle. ``rtrrl``'s LRU and RTU are state-space recurrences with ``A``
constrained to be diagonal, and on them RFLO and exact RTRL are the same
recurrence -- the cross-unit block is identically zero, so there is nothing to
drop. ``docs/exact-recurrent-sensitivity.md`` makes that argument structurally
and offers no arm that varies the structure. This is that arm: the same linear
step with ``A`` dense, where the block is present, RFLO is an approximation
again, and the cost of exact credit is a factor of the hidden width.

So the comparison it supports is not "SSM against LSTM". It is *the diagonal
structure*, priced twice -- in what it saves per transition, and in what the
approximation that replaces it costs in return. Setting the off-diagonal of
``A`` to zero recovers the LRU's case as a limit, and
``tests/test_dense_ssm_rflo.py`` runs it as a measurement rather than leaving
it as a claim.

Two things here have no counterpart in ``rtrrl_ctrnn_rflo`` or
``rtrrl_lstm_rflo``:

- **The recurrence is linear.** There is no activation between ``h_{t-1}`` and
  ``h_t``, so the term RFLO drops is the off-diagonal of ``A`` itself rather
  than a Jacobian factor that happens to contain it. The two recurrences differ
  by a matrix product and nothing else, which is why this is the cell the
  approximation is easiest to state on.
- **``A`` has a domain and can leave it.** A free matrix with a spectral radius
  above one diverges over an episode, and unlike the LRU -- whose
  parameterisation puts ``|lambda| < 1`` in the exponent -- nothing stops an
  ordinary gradient step from reaching it. The cell names a bound on the
  induced infinity-norm and the torso projects onto it, which is
  ``rtrrl_ctrnn_rflo``'s mechanism for ``tau`` used a second time and the
  reason that helper now lives in ``rtrrl_aaai``.

Everything downstream of the torso is the same algorithm and the same code: the
TD error, the three eligibility traces, the emphasis, the entropy term, the
per-block optimizers, the followed reading copy and the order the transition is
processed in are ``rtrrl_aaai``'s, because they are the published algorithm's
and do not change with the recurrent kernel.

``docs/rtrrl-dense-ssm-rflo.md`` derives the trace and states the bound.
``tests/unit/algorithms/rtrrl/test_ssm_rflo_assembly.py`` holds this graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from memorax.building import BuildContext, ComponentBuilder
from memorax.networks.components import LayerNorm
from memorax.networks.readouts import ACTOR_HEAD_FAMILY, CRITIC_HEAD_FAMILY
from memorax.networks.sequence import Sequence
from memorax.networks.sequence_models import RNN
from memorax.networks.sequence_models.dense_ssm import (
    DENSE_SSM_DIFFERENTIATION_FAMILY,
    DenseSSMCell,
    DenseSSMConfig,
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
DENSE_SSM_DIFFERENTIATION = DENSE_SSM_DIFFERENTIATION_FAMILY.restricted("rflo")


# --------------------------------------------------------------- declarations
@dataclass(frozen=True)
class TorsoParameters:
    """The shared block: one dense SSM, how it is differentiated, how it steps.

    ``hidden_dim``, ``spectral_bound`` and ``layer_norm`` are static because
    they are built into the cell rather than read arithmetically by the running
    graph, so the members of one vmapped round cannot disagree about them.

    ``grad_clip`` carries ``rtrrl_aaai``'s condition unchanged: the intentional
    update derives its own step size and refuses a second bound over it, so
    ``optimizer.kind: iu`` requires ``grad_clip: 0``.
    """

    hidden_dim: int = param(valid=(1, 4096), search=(32, 512), static=True)
    # The norm ball `A` is projected back onto after every update, and it is a
    # bound on the parameter's domain rather than a preference. `max_i sum_j
    # |A_ij|` bounds the spectral radius, so a value under one is a recurrence
    # that decays and a value at or above one is an episode whose state can
    # grow without bound -- which is why the declared range stops short of it
    # rather than accepting a run that produces `inf` on the transition that
    # reaches it. The horizon the state can carry is about `1/(1 - bound)`
    # transitions, so this is also the memory the arm is being given.
    spectral_bound: float = param(valid=(0.01, 0.99), search=(0.5, 0.99), static=True)
    # The affine-free normalization behind the cell, off by default as on the
    # other two RFLO torsos. It holds no parameters either way, so it changes
    # what the heads read and not what is credited.
    layer_norm: bool = param(valid=[False, True], search=[False])
    # RFLO alone. The family carries `tbptt` as well -- it is the exact judge
    # the cell's tests measure the approximation against -- but this entry's
    # name is a claim about which online gradient produced a result, and an
    # experiment identity that a `kind:` line can quietly falsify is not one.
    differentiation: str = structure(
        branches=DENSE_SSM_DIFFERENTIATION.branches, search=("rflo",)
    )
    optimizer: str = structure(branches=RTRRL_TORSO_OPTIMIZERS.branches)
    grad_clip: float = param(valid=(0.0, 100.0), search=(0.0, 10.0))
    follow: float = param(valid=(0.0, 1.0), search=(0.0, 1.0))


@dataclass(frozen=True)
class RTRRLSsmRfloParameters:
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


PARAMETERS = describe_parameters(RTRRLSsmRfloParameters)


# ------------------------------------------------------------------ the torso
def _torso(parameters, components: ComponentBuilder, *, features: int):
    """The SSM, whatever follows it, and the online method that credits it."""

    cell = DenseSSMCell(
        config=DenseSSMConfig(
            features=features,
            hidden_dim=int(parameters["torso.hidden_dim"]),
            spectral_bound=float(parameters["torso.spectral_bound"]),
        )
    )
    following = (
        (LayerNorm(use_scale=False, use_bias=False),)
        if bool(parameters["torso.layer_norm"])
        else ()
    )
    network = Sequence(components=(RNN(cell=cell), *following))
    differentiation = components.build(
        DENSE_SSM_DIFFERENTIATION,
        "torso.differentiation",
        core=network.core,
    )
    return network, differentiation, kernel_constraint(network)


# -------------------------------------------------------------- the algorithm
class RTRRLSsmRflo(RTRRL):
    """RTRRL's flow, over the dense state-space torso and its RFLO credit."""

    observations = OBSERVATIONS

    @classmethod
    def graph(
        cls,
        parameters: dict[str, Any],
        components: ComponentBuilder,
        context: BuildContext,
        *,
        record=(),
    ) -> RTRRLSsmRflo:
        """Declare the state-space torso, the two readouts, and a rule for each."""

        gamma = numeric(parameters["gamma"])
        meta_rl = bool(parameters["meta_rl"])
        classes = action_classes(context.action_space)
        # What the cell reads: the observation, and under meta-RL the previous
        # action and reward beside it. Nothing widens it first -- `B` is the
        # projection.
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
