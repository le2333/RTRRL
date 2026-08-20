"""Small assembly adapters shared by tests that start a real algorithm graph."""

from types import MethodType
from typing import Any

from memorax.algorithms import rtrrl_aaai as rtrrl
from memorax.algorithms.r2d2 import R2D2
from memorax.algorithms.stream_ac import StreamAC
from memorax.assembly import BuildRequest, EnvironmentSpec, assemble
from memorax.parameters import expand
from memorax.runtime import BuiltAlgorithm
from tests.support.environments import TinyContinuousEnv


def graph_of(built: BuiltAlgorithm) -> Any:
    """The algorithm object an assembled program's arrows are bound to.

    ``Program`` is four callables and says nothing about where they came from.
    A test that wants to see what assembly selected reaches back through one of
    them, and this is the one place that says the arrows are bound methods.
    """

    init = built.program.init
    assert isinstance(init, MethodType), "program.init is not a bound method"
    return init.__self__


def assemble_r2d2(parameters, environment, *, num_envs):
    return assemble(
        R2D2,
        BuildRequest(
            parameters=parameters,
            environment=EnvironmentSpec(
                id=environment.id,
                backend=environment.backend,
                observed=environment.observed,
                episode_length=environment.episode_length,
            ),
            num_envs=num_envs,
        ),
    ).program


def assemble_stream_ac(parameters, environment, *, num_envs):
    return assemble(
        StreamAC,
        BuildRequest(
            parameters=parameters,
            environment=EnvironmentSpec(
                id=environment.id,
                backend=environment.backend,
                observed=environment.observed,
                episode_length=environment.episode_length,
            ),
            num_envs=num_envs,
        ),
    ).program


# ------------------------------------------------------- RTRRL on a tiny graph
# The smallest RTRRL that still has every piece the algorithm is about: a
# differentiated recurrent torso, two readouts stepping as one group, and both
# rule families reachable from a parameter dictionary. Shared because three
# suites now start one -- assembly, the update-scale telemetry, and the
# checkpoint forks -- and a second copy of the dictionary would let them drift
# into testing three different algorithms.

# The threshold both D-RTRRL arms are written over, and the outer clip these
# fixtures use, so a test can say `C` and mean the number the arm is defined by.
C = 1.0

# `expand` fills anything unset from the low end of its search domain, which
# leaves `eta_f`, `eta_pi` and every `lambda` at zero. That is harmless for
# structural assertions and fatal for anything asserting a step was taken:
# `eta_f == 0` makes the torso's TD error zero, so `sign` of it is zero and the
# torso never takes a traced step at all.
LIVE = {
    "eta_f": 1.0,
    "eta_pi": 1.0,
    "lambda_pi": 0.9,
    "lambda_v": 0.9,
    "lambda_rnn": 0.9,
    "entropy_rate": 1e-5,
}

D_RTRRL = {
    **LIVE,
    "torso.optimizer.kind": "d_rtrrl",
    "torso.optimizer.d_rtrrl.c": C,
    "torso.optimizer.d_rtrrl.magnitude": "sign",
    "torso.optimizer.d_rtrrl.scope": "block",
    "torso.optimizer.d_rtrrl.eps": 1e-8,
    "heads.optimizer.kind": "d_rtrrl",
    "heads.optimizer.d_rtrrl.c": C,
    "heads.optimizer.d_rtrrl.magnitude": "sign",
    "heads.optimizer.d_rtrrl.scope": "block",
    "heads.optimizer.d_rtrrl.eps": 1e-8,
}


def tiny_environment(identifier, **options):
    del identifier, options
    environment = TinyContinuousEnv()
    return environment, environment.default_params


def rtrrl_parameters(backbone="lru", differentiation="exact_rtrl", optimizer=None):
    branch = f"torso.backbone.{backbone}"
    return expand(
        rtrrl.PARAMETERS,
        {
            "torso.backbone.kind": backbone,
            **({f"{branch}.feature_dim": 4} if backbone == "lru" else {}),
            f"{branch}.hidden_dim": 2,
            f"{branch}.differentiation.kind": differentiation,
            "torso.optimizer.kind": "adam",
            "torso.optimizer.adam.lr": 1e-3,
            "torso.grad_clip": 1.0,
            "torso.follow": 0.25,
            "heads.optimizer.kind": "adam",
            "heads.optimizer.adam.lr": 5e-4,
            **(optimizer or {}),
            "actor.head.kind": "state_std",
            "critic.head.kind": "value",
            "normalization.observation.kind": "none",
            "normalization.reward.kind": "none",
            "meta_rl": False,
        },
    )


def assemble_rtrrl(
    backbone="lru",
    differentiation="exact_rtrl",
    record=None,
    optimizer=None,
    *,
    num_envs=1,
):
    return assemble(
        rtrrl.RTRRL,
        BuildRequest(
            parameters=rtrrl_parameters(backbone, differentiation, optimizer),
            environment=EnvironmentSpec(
                id="tiny",
                backend=None,
                observed=None,
                episode_length=8,
            ),
            num_envs=num_envs,
            record=(rtrrl.OBSERVATIONS.trajectory_fields if record is None else record),
        ),
        environment_factory=tiny_environment,
    )
