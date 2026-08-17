"""POPJym behind the deployment's environment vocabulary.

An environment is named the way POPJym registers it -- ``MinesweeperEasy``, not
``Minesweeper`` plus a difficulty -- because that is the one spelling both this
side and the authors' fork resolve, and a comparison whose two arms name the
task differently is a comparison over the naming.

Of the three things assembly always supplies, only ``observed`` means anything
here. POPJym ships one implementation per environment, so there is no backend
to choose between; and its ``EnvParams`` is empty, with each environment
carrying its own ``max_episode_length`` as a class attribute, so a run document
has nothing to move an episode limit through.
"""

from typing import Any

from flax import struct
from gymnax.environments import spaces

from memorax.environments.wrappers import GymnaxWrapper, SelectObservationWrapper
from memorax.utils.typing import Array, Key

max_steps_in_episode = {
    "AutoencodeEasy": 105,
    "AutoencodeMedium": 209,
    "AutoencodeHard": 313,
    "BattleshipEasy": 64,
    "BattleshipMedium": 100,
    "BattleshipHard": 144,
    "StatelessCartPoleEasy": 200,
    "StatelessCartPoleMedium": 400,
    "StatelessCartPoleHard": 600,
    "NoisyStatelessCartPoleEasy": 200,
    "NoisyStatelessCartPoleMedium": 200,
    "NoisyStatelessCartPoleHard": 200,
    "ConcentrationEasy": 104,
    "ConcentrationMedium": 208,
    "ConcentrationHard": 104,
    "CountRecallEasy": 52,
    "CountRecallMedium": 104,
    "CountRecallHard": 208,
    "HigherLowerEasy": 52,
    "HigherLowerMedium": 104,
    "HigherLowerHard": 156,
    "RepeatFirstEasy": 52,
    "RepeatFirstMedium": 416,
    "RepeatFirstHard": 832,
    "RepeatPreviousEasy": 52,
    "RepeatPreviousMedium": 104,
    "RepeatPreviousHard": 156,
    "MinesweeperEasy": 14,
    "MinesweeperMedium": 30,
    "MinesweeperHard": 54,
    "MultiArmedBanditEasy": 200,
    "MultiArmedBanditMedium": 400,
    "MultiArmedBanditHard": 600,
    "StatelessPendulumEasy": 200,
    "StatelessPendulumMedium": 150,
    "StatelessPendulumHard": 100,
    "NoisyStatelessPendulumEasy": 200,
    "NoisyStatelessPendulumMedium": 200,
    "NoisyStatelessPendulumHard": 200,
    "NoisyStatelessMetaCartPole": 3200,
}


@struct.dataclass(frozen=True)
class EnvParams:
    env_params: Any
    max_steps_in_episode: int


class PopJymWrapper(GymnaxWrapper):
    def reset(self, key: Key, params) -> tuple[Array, Any]:
        return self._env.reset(key, params.env_params)

    def step(self, key: Key, state, action: Array, params) -> tuple[Array, Any, Array, Array, dict]:
        obs, new_state, reward, done, info = self._env.step(
            key, state, action, params.env_params
        )
        return obs, new_state, reward, done, info

    def observation_space(self, params) -> spaces.Space:
        return self._env.observation_space(params.env_params)

    def action_space(self, params) -> spaces.Space:
        return self._env.action_space(params.env_params)

    def state_space(self, params) -> spaces.Space:
        return self._env.state_space(params.env_params)


def make(
    env_id: str,
    observed=None,
    backend=None,
    episode_length: int | None = None,
    **kwargs,
) -> tuple[PopJymWrapper, EnvParams]:
    import popjym

    # Neither is a choice this environment offers; see the module docstring.
    del backend, episode_length

    env, env_params = popjym.make(env_id, **kwargs)
    env = PopJymWrapper(env)
    if observed is not None:
        env = SelectObservationWrapper(env, observed)
    return env, EnvParams(
        env_params=env_params, max_steps_in_episode=max_steps_in_episode[env_id]
    )
