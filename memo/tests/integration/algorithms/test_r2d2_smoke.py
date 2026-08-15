from functools import partial

import jax
import optax
import pytest

from memorax.algorithms.r2d2 import (
    R2D2,
    Core,
    QFunction,
    R2D2Config,
    signed_hyperbolic,
    signed_parabolic,
    tbptt_starts,
)
from memorax.buffers import make_prioritised_episode_buffer
from tests.support.environments import TinyDiscreteEnv


def _algorithm(num_envs):
    env = TinyDiscreteEnv()
    core = Core(
        q_function=QFunction(
            action_dim=2,
            feature_dim=4,
            hidden_dim=3,
            backbone_kind="rtu",
            head_kind="linear",
        ),
        optimizer=optax.adam(0.01),
        gamma=0.9,
        n_step=1,
        burn_in_length=0,
        unroll_length=2,
        importance_sampling_exponent=0.4,
        max_priority_weight=0.9,
        target_update_period=2,
        transform=signed_hyperbolic,
        inverse_transform=signed_parabolic,
    )
    buffer = make_prioritised_episode_buffer(
        max_length=16,
        min_length=2,
        sample_batch_size=1,
        sample_sequence_length=2,
        get_start_flags=partial(tbptt_starts, burn_in_length=0),
        add_sequences=False,
        add_batch_size=num_envs,
    )
    return R2D2(
        cfg=R2D2Config(
            num_envs=num_envs,
            epsilon_start=0.2,
            epsilon_end=0.05,
            epsilon_decay_steps=4,
            evaluation_epsilon=0.0,
        ),
        env=env,
        env_params=env.default_params,
        core=core,
        buffer=buffer,
    )


@pytest.mark.parametrize(
    ("num_envs", "train_steps", "eval_steps", "train_shape", "eval_shape"),
    [
        (1, 4, 3, (4, 1), (3, 1)),
        (2, 4, 4, (2, 2), (2, 2)),
    ],
)
def test_r2d2_train_and_evaluate_follow_program_scan_shape(
    num_envs, train_steps, eval_steps, train_shape, eval_shape
):
    algorithm = _algorithm(num_envs)
    state = algorithm.init(jax.random.key(0))

    trained, train_metrics = algorithm.train(
        jax.random.key(1), state, num_steps=train_steps
    )
    eval_metrics = algorithm.evaluate(jax.random.key(2), trained, num_steps=eval_steps)

    assert int(trained.step) == train_steps
    assert train_metrics.interaction.reward.shape == train_shape
    assert eval_metrics.interaction.reward.shape == eval_shape
