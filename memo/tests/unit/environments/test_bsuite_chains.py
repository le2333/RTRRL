"""The two bsuite chains, against what a run document can say about them.

Both are diagnostic tasks whose whole content is a number a run has to be able
to name -- how long the chain is, which action is the paying one -- and both
end their episodes in a way the framework has to be able to tell apart. What
is under test is that a run can name those numbers and that the endings are
reported for what they are; how well anything learns either task is not.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from memorax.environments import make

DISCOUNTING = "gymnax::DiscountingChain-bsuite"
UMBRELLA = "gymnax::UmbrellaChain-bsuite"

# DiscountingChain pays its last action at t=100, so this is the shortest
# horizon at which the task is still the one it is named after.
FULL_CHAIN = 100


def build(env_id, *, episode_length, **kwargs):
    return make(
        env_id, observed=None, backend=None, episode_length=episode_length, **kwargs
    )


def episode(env_id, *, action, episode_length, **kwargs):
    """Run one episode of a constant policy and report how it ended."""

    env, params = build(env_id, episode_length=episode_length, **kwargs)
    observation, state = env.reset(jax.random.key(0), params)
    opening = observation
    total = 0.0
    for step in range(1, episode_length + 2):
        chosen = action(opening) if callable(action) else action
        observation, state, reward, done, info = env.step(
            jax.random.key(step), state, jnp.int32(chosen), params
        )
        total += float(reward)
        if done:
            return {
                "steps": step,
                "return": total,
                "reward_dtype": reward.dtype,
                "terminal": bool(info["terminal"]),
                "truncation": bool(info["truncation"]),
                "regret": float(info["returned_episode_regret"]),
            }
    raise AssertionError(f"{env_id} did not end within {episode_length} steps")


# --------------------------------------------------------------------------
# What a run document can name


def test_a_constructor_argument_reaches_the_environment():
    """``n_distractor`` is UmbrellaChain's, and it widens the observation.

    It is a constructor argument rather than a parameter, so this is the half
    of the split that has to be applied before the environment exists.
    """

    plain, plain_params = build(UMBRELLA, episode_length=FULL_CHAIN)
    padded, padded_params = build(UMBRELLA, episode_length=FULL_CHAIN, n_distractor=5)

    assert plain.observation_space(plain_params).shape == (3,)
    assert padded.observation_space(padded_params).shape == (8,)


def test_a_parameter_argument_reaches_the_environment_parameters():
    """``chain_length`` is UmbrellaChain's, and it is what ends the episode.

    It lives in ``EnvParams`` rather than in the constructor, which is the
    other half of the split -- and it is the number bsuite's own sweep varies,
    so a run that could not name it could not run the task's sweep at all.
    """

    assert episode(UMBRELLA, action=0, episode_length=FULL_CHAIN)["steps"] == 10
    assert (
        episode(UMBRELLA, action=0, episode_length=FULL_CHAIN, chain_length=40)["steps"]
        == 40
    )


def test_a_constructor_argument_can_move_which_action_is_the_paying_one():
    """``mapping_seed`` is what DiscountingChain's sweep varies."""

    def best_action(**kwargs):
        returns = [
            episode(DISCOUNTING, action=a, episode_length=FULL_CHAIN, **kwargs)[
                "return"
            ]
            for a in range(5)
        ]
        return max(range(5), key=returns.__getitem__)

    assert best_action(mapping_seed=0) == 0
    assert best_action(mapping_seed=3) == 3


def test_the_horizon_is_named_once_and_not_twice():
    """``max_steps_in_episode`` is ``episode_length`` under another name."""

    with pytest.raises(ValueError, match="episode_length"):
        build(UMBRELLA, episode_length=FULL_CHAIN, max_steps_in_episode=50)


def test_a_horizon_that_would_delete_actions_is_refused():
    """DiscountingChain's horizon is the task, not a budget applied to it.

    Its five actions pay at t=1, 3, 10, 30 and 100 and the task is that a
    discount rate has to choose between them. At a horizon of 20 the last two
    payments never arrive and two actions become worth nothing -- a different
    task under this one's name, which is quieter to refuse than to read out of
    a curve later.
    """

    with pytest.raises(ValueError, match="last reward"):
        build(DISCOUNTING, episode_length=20)

    build(DISCOUNTING, episode_length=FULL_CHAIN)


# --------------------------------------------------------------------------
# Which of the two ways an episode ended


def test_discounting_chain_ends_only_on_its_clock():
    """Nothing in the task ends an episode, so no ending is a termination.

    ``td0`` gates the bootstrap on ``terminal``. A critic told that t=100 is
    terminal is being taught that the far reward is worth nothing to reach,
    which is the exact judgement this task exists to measure.
    """

    ended = episode(DISCOUNTING, action=0, episode_length=FULL_CHAIN)

    assert ended["steps"] == FULL_CHAIN
    assert ended["truncation"]
    assert not ended["terminal"]


def test_umbrella_chain_reaching_the_end_of_its_chain_is_a_termination():
    ended = episode(UMBRELLA, action=0, episode_length=FULL_CHAIN, chain_length=40)

    assert ended["steps"] == 40
    assert ended["terminal"]
    assert not ended["truncation"]


def test_umbrella_chain_cut_off_before_its_chain_is_a_truncation():
    """The distinction is drawn from the chain length, not from gymnax's flag.

    Gymnax raises one ``done`` for both, and the state it hands back on that
    step has already been reset -- so a wrapper reading the environment's own
    ``time`` here would see zero and call every ending a truncation. This is
    the case that tells the two apart.
    """

    ended = episode(UMBRELLA, action=0, episode_length=5, chain_length=40)

    assert ended["steps"] == 5
    assert ended["truncation"]
    assert not ended["terminal"]


# --------------------------------------------------------------------------
# What the episode cost


def test_discounting_chain_regret_is_the_return_it_gave_up():
    """Its regret is never a negative reward, because it never pays one."""

    optimal = episode(DISCOUNTING, action=0, episode_length=FULL_CHAIN)
    other = episode(DISCOUNTING, action=1, episode_length=FULL_CHAIN)

    assert optimal["regret"] == pytest.approx(0.0, abs=1e-6)
    assert other["regret"] == pytest.approx(0.1, abs=1e-6)
    assert other["return"] < optimal["return"]


def test_umbrella_chain_regret_is_paid_only_for_the_decision_it_scores():
    """The umbrella is right or wrong; the coins in between are neither.

    ``need_umbrella`` is the opening observation's first component, so the
    policy that carries it and the policy that contradicts it are both
    writable here without knowing the key that drew it.
    """

    right = episode(
        UMBRELLA, action=lambda opening: int(opening[0]), episode_length=FULL_CHAIN
    )
    wrong = episode(
        UMBRELLA, action=lambda opening: 1 - int(opening[0]), episode_length=FULL_CHAIN
    )

    assert right["terminal"] and wrong["terminal"]
    assert right["regret"] == pytest.approx(0.0)
    assert wrong["regret"] == pytest.approx(2.0)


def test_umbrella_chain_that_was_cut_off_reports_no_regret():
    """An episode whose decision was never scored has none to report.

    Its last reward is a distractor coin, and reading that coin's sign as a
    wrong umbrella would report regret for a decision the episode never
    reached.
    """

    cut_off = episode(UMBRELLA, action=0, episode_length=5, chain_length=40)

    assert cut_off["truncation"]
    assert cut_off["regret"] == pytest.approx(0.0)


def test_umbrella_chain_hands_back_a_float_reward():
    """It accumulates its reward from integer terms and would hand back int32."""

    assert (
        episode(UMBRELLA, action=0, episode_length=FULL_CHAIN)["reward_dtype"]
        == jnp.float32
    )


# --------------------------------------------------------------------------
# What did not change


def test_an_environment_that_names_no_arguments_is_built_as_before():
    environment, parameters = build("gymnax::CartPole-v1", episode_length=32)

    assert parameters.max_steps_in_episode == 32
    assert environment.action_space(parameters).n == 2
