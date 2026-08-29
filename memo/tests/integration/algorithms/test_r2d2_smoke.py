import jax
import numpy as np
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
from tests.support.replay import start_flags


def _algorithm(num_envs, *, backbone_kind="rtu", hidden_dim=3):
    env = TinyDiscreteEnv()
    core = Core(
        q_function=QFunction(
            action_dim=2,
            feature_dim=4,
            hidden_dim=hidden_dim,
            backbone_kind=backbone_kind,
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
        get_start_flags=start_flags(tbptt_starts, burn_in_length=0),
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
    opened = algorithm.open_evaluation(jax.random.key(2), trained)
    _, eval_metrics = algorithm.evaluate(
        jax.random.key(3), opened, num_steps=eval_steps
    )

    assert int(trained.step) == train_steps
    assert train_metrics.interaction.reward.shape == train_shape
    assert eval_metrics.interaction.reward.shape == eval_shape


# The width R1.1.2 pins on both sides of its recurrent comparison: DRQN-LSTM
# runs the published cell at 32, so an R2D2-LSTM read against it runs at 32.
PROTOCOL_HIDDEN = 32


def test_an_lstm_at_the_protocol_width_trains_evaluates_and_stays_finite():
    """The LSTM core through a real train and eval scan, not a hand-driven loop.

    ``TinyDiscreteEnv`` is two observations wide and nothing here is a result.
    What the run answers is that the branch a manifest can now name completes
    the whole path -- acting, replay, an update, evaluation -- and that every
    number it left behind is a number.
    """

    algorithm = _algorithm(1, backbone_kind="lstm", hidden_dim=PROTOCOL_HIDDEN)
    state = algorithm.init(jax.random.key(0))

    trained, train_metrics = algorithm.train(jax.random.key(1), state, num_steps=8)
    opened = algorithm.open_evaluation(jax.random.key(2), trained)
    _, eval_metrics = algorithm.evaluate(jax.random.key(3), opened, num_steps=4)

    # An update ran, so the update readings below are a real loss rather than
    # the zeros a run that never sampled would report.
    assert int(trained.core.update_step) > 0
    cell = trained.core.params["params"]["OptimizedLSTMCell_0"]
    assert cell["hi"]["kernel"].shape == (PROTOCOL_HIDDEN, PROTOCOL_HIDDEN)

    for tree in (
        trained.core.params,
        trained.core.target_params,
        train_metrics,
        eval_metrics,
    ):
        for leaf in jax.tree.leaves(tree):
            values = np.asarray(leaf)
            if np.issubdtype(values.dtype, np.floating):
                assert np.all(np.isfinite(values))
