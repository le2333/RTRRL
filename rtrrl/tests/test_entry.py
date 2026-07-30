"""The translation from one vocabulary to another, which nothing else checks.

Everything between `train_rtrrl` and the environment is the authors' and is
covered by their paper and by the parity tests in `memo`. What is ours is the
mapping from the names the control plane samples onto the fields their dataclass
reads, and a mistake in it does not raise: it produces a run that trains
perfectly well under a hyperparameter nobody chose. So the mapping is tested by
value, and the budget arithmetic -- the one piece of it that computes rather
than renames -- is tested at its edges.

None of this imports their code, which is what lets it run on the machine the
control plane runs on rather than only inside the image.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest
from training_sdk.contract import BudgetConfig, EnvironmentConfig

from entries.rtrrl_aaai import SPACE, iterations, settings

ENVIRONMENT = EnvironmentConfig(
    id="brax::hopper",
    backend="spring",
    num_envs=1,
    observed=(0, 1, 2, 3, 4),
)
BUDGET = BudgetConfig(total_steps=2_000_000, epoch_steps=100_000, eval_steps=100)


class Recording(dict):
    """A parameter mapping that remembers which names were asked for."""

    def __init__(self, values: Mapping[str, Any]) -> None:
        super().__init__(values)
        self.read: set[str] = set()

    def __getitem__(self, key: str) -> Any:
        self.read.add(key)
        return super().__getitem__(key)


def smallest(spec: Any) -> Any:
    return spec[0] if isinstance(spec, list) else spec["low"]


def defaults(**overrides: Any) -> Recording:
    values = {name: smallest(spec) for name, spec in SPACE.items()}
    return Recording({**values, **overrides})


def test_every_declared_parameter_is_one_the_entry_reads() -> None:
    """A declared name nothing reads is a knob that does nothing when turned."""

    params = defaults()
    settings(params, ENVIRONMENT, BUDGET)

    assert params.read == set(SPACE)


def test_the_budget_becomes_their_iteration_count() -> None:
    chosen = settings(
        defaults(scan_steps=1000),
        ENVIRONMENT,
        BUDGET,
    )

    assert chosen["episodes"] == 2000
    assert chosen["steps"] == 1000
    assert chosen["eval_every"] == 100


def test_a_budget_that_does_not_divide_is_refused() -> None:
    """Rounding it either way misreports what the run cost or what it reached."""

    with pytest.raises(ValueError, match="whole iterations"):
        iterations(total_steps=2_000_001, scan_steps=1000, num_envs=1)


def test_the_environments_share_the_budget_rather_than_multiplying_it() -> None:
    """Their loop counts transitions per environment, not per iteration."""

    assert iterations(total_steps=2_000_000, scan_steps=1000, num_envs=2) == 1000


def test_the_environment_is_named_and_configured_the_way_their_factory_reads_it() -> (
    None
):
    chosen = settings(defaults(), ENVIRONMENT, BUDGET)["environment"]

    # `make_env` splits the name on its first hyphen to find the Brax task.
    assert chosen["env_name"] == "brax-hopper"
    assert chosen["env_kwargs"] == {"backend": "spring"}
    assert chosen["obs_mask"] == (0, 1, 2, 3, 4)
    assert chosen["render"] is False
    assert chosen["max_ep_length"] == 1000


def test_the_mask_is_hashable() -> None:
    """`RTRRLParams` is declared with `unsafe_hash=True` and holds this.

    A list here makes the whole dataclass unhashable, which is the kind of
    failure that waits for whichever of their code paths hashes it first.
    """

    hash(settings(defaults(), ENVIRONMENT, BUDGET)["environment"]["obs_mask"])


def test_the_learning_rates_reach_the_two_optimisers_separately() -> None:
    chosen = settings(
        defaults(td_lr=1e-4, rnn_lr=2e-4, rnn_grad_clip=1.0),
        ENVIRONMENT,
        BUDGET,
    )

    assert chosen["td"] == {"opt_name": "adam", "learning_rate": 1e-4}
    # No clip on the TD side: their own comment says clipping that update makes
    # the eligibility traces explode.
    assert "gradient_clip" not in chosen["td"]
    assert chosen["rnn"]["learning_rate"] == 2e-4
    assert chosen["rnn"]["gradient_clip"] == 1.0


def test_rflo_is_not_offered() -> None:
    """Their default, which the LRU path raises on rather than implements."""

    assert "rflo" not in SPACE["gradient_mode"]


def test_a_seed_of_zero_cannot_be_sampled() -> None:
    """`args.seed or np.random.randint(1e6)` reads zero as `pick one for me`."""

    assert SPACE["seed"]["low"] == 1


RESERVED = frozenset(
    {
        "environment",
        "env_mode",
        "env_backend",
        "observed",
        "num_envs",
        "total_steps",
        "epoch_steps",
        "eval_steps",
    }
)


def test_the_entry_declares_neither_the_environment_nor_the_budget():
    from entries import rtrrl_aaai

    assert not RESERVED & set(rtrrl_aaai.SPACE)


def _defaults() -> dict:
    """One value per declared parameter, taken as the first of each domain."""

    from entries import rtrrl_aaai

    chosen = {}
    for name, spec in rtrrl_aaai.SPACE.items():
        chosen[name] = spec[0] if isinstance(spec, list) else spec["low"]
    return chosen


def test_the_kept_indices_reach_their_obs_mask():
    from training_sdk.contract import BudgetConfig, EnvironmentConfig

    from entries import rtrrl_aaai

    chosen = rtrrl_aaai.settings(
        _defaults(),
        EnvironmentConfig(
            id="brax::hopper", backend="spring", num_envs=1, observed=(0, 1, 2, 3, 4)
        ),
        BudgetConfig(total_steps=2000, epoch_steps=1000, eval_steps=100),
    )

    assert chosen["environment"]["obs_mask"] == (0, 1, 2, 3, 4)
    assert chosen["environment"]["batch_size"] == 1


def test_a_fully_observed_task_asks_for_no_mask():
    from training_sdk.contract import BudgetConfig, EnvironmentConfig

    from entries import rtrrl_aaai

    chosen = rtrrl_aaai.settings(
        _defaults(),
        EnvironmentConfig(id="brax::hopper", backend="spring", num_envs=1),
        BudgetConfig(total_steps=2000, epoch_steps=1000, eval_steps=100),
    )

    assert chosen["environment"]["obs_mask"] is None
