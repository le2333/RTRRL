"""Instrument the pinned AAAI25 training body without restating its equations."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import importlib.util
from pathlib import Path
import sys
import tempfile
from typing import Any, NamedTuple

import numpy as np

SOURCE_SHA256 = "082914b2dbe95481e30c738945b58f7948d4065eb917ef1554f8127227ad0edf"
NONE_SENTINEL = np.asarray("<none>")


class OracleEnvironmentState(NamedTuple):
    obs: Any
    reward: Any
    done: Any
    phase: Any
    last_action: Any


class _Progress:
    def __init__(self, count: int):
        self._values = range(count)

    def __iter__(self):
        return iter(self._values)

    def set_description(self, *_args, **_kwargs):
        return None

    def write(self, *_args, **_kwargs):
        return None


class _Environment:
    action_size = 2

    def __init__(self, jnp: Any, batch_size: int):
        self.jnp = jnp
        self.batch_size = batch_size

    def reset(self, key):
        del key
        if self.batch_size == 1:
            observation = [[0.25]]
            reward = [-0.5]
        else:
            observation = [[0.25], [-0.35]]
            reward = [-0.5, 0.25]
        return OracleEnvironmentState(
            obs=self.jnp.asarray(observation, dtype=self.jnp.float32),
            reward=self.jnp.asarray(reward, dtype=self.jnp.float32),
            done=self.jnp.zeros((self.batch_size,), dtype=self.jnp.bool_),
            phase=self.jnp.array(0, dtype=self.jnp.int32),
            last_action=self.jnp.zeros((self.batch_size, 2), dtype=self.jnp.float32),
        )

    def step(self, state, action):
        phase = state.phase
        if self.batch_size == 1:
            observation = self.jnp.asarray(
                [[0.1 + 0.05 * phase]], dtype=self.jnp.float32
            )
            reward = self.jnp.asarray([0.625 - 0.125 * phase], dtype=self.jnp.float32)
            done = self.jnp.asarray([phase == 1])
        else:
            observation = self.jnp.stack(
                (
                    self.jnp.asarray([0.1 + 0.05 * phase]),
                    self.jnp.asarray([-0.2 + 0.025 * phase]),
                )
            ).astype(self.jnp.float32)
            reward = self.jnp.asarray(
                [0.625 - 0.125 * phase, -0.25 + 0.05 * phase],
                dtype=self.jnp.float32,
            )
            done = self.jnp.asarray([False, phase == 0])
        return OracleEnvironmentState(
            obs=observation,
            reward=reward,
            done=done,
            phase=phase + 1,
            last_action=action,
        )


def _instrument_source(source_path: Path) -> str:
    source = source_path.read_text()
    digest = hashlib.sha256(source.encode()).hexdigest()
    if digest != SOURCE_SHA256:
        raise RuntimeError(
            f"pinned rtrrl.py source mismatch: {digest} != {SOURCE_SHA256}"
        )
    replacements = (
        (
            "    @jax.jit\n    def step_fn(_carry, _):",
            "    def step_fn(_carry, _):",
        ),
        (
            "        seed, action_key, dropout_key = jrandom.split(seed, 3)\n",
            "        seed, action_key, dropout_key = jrandom.split(seed, 3)\n"
            '        _oracle_capture_phase("start", locals())\n',
        ),
        (
            "        updates, opt_state = optimizer.update(\n",
            '        _oracle_capture_phase("directions", locals())\n'
            "        updates, opt_state = optimizer.update(\n",
        ),
        (
            "        z = jax.tree.map(lambda a, b: jax.vmap(jnp.where)(env_state.done, a, b), z0, z)\n",
            '        _oracle_capture_phase("incoming", locals())\n'
            "        z = jax.tree.map(lambda a, b: jax.vmap(jnp.where)(env_state.done, a, b), z0, z)\n",
        ),
        (
            "        _carry = (\n",
            '        _oracle_capture_phase("final", locals())\n' "        _carry = (\n",
        ),
        (
            "    carry = (\n",
            "    _oracle_capture_init(locals())\n" "    carry = (\n",
        ),
    )
    for old, new in replacements:
        if source.count(old) != 1:
            raise RuntimeError(
                f"instrumentation anchor count is {source.count(old)}: {old!r}"
            )
        source = source.replace(old, new)
    return source


def _copy_tree(tree: Any, jax: Any, jnp: Any) -> Any:
    return jax.tree.map(
        lambda value: None if value is None else jnp.asarray(value),
        tree,
        is_leaf=lambda value: value is None,
    )


def _environment_tree(state: OracleEnvironmentState) -> dict[str, Any]:
    return {
        "obs": state.obs,
        "reward": state.reward,
        "done": state.done,
        "phase": state.phase,
        "last_action": state.last_action,
    }


def _state_tree(
    carry: tuple[Any, ...],
    *,
    model_input: Any,
    initial_recurrent_state: Any,
    step_count: int,
) -> dict[str, Any]:
    (
        parameters,
        slow_parameters,
        environment_state,
        optimizer_state,
        action,
        recurrent_state,
        traces,
        value,
        average_reward,
        emphasis,
        observation_statistics,
        reward_statistics,
        _key,
    ) = carry
    return {
        "parameters": parameters,
        "slow_parameters": slow_parameters,
        "optimizer_state": optimizer_state,
        "environment_state": _environment_tree(environment_state),
        "action": action,
        "recurrent_state": recurrent_state,
        "traces": traces,
        "value": value,
        "average_reward": average_reward,
        "emphasis": emphasis,
        "observation_statistics": (
            NONE_SENTINEL if observation_statistics is None else observation_statistics
        ),
        "reward_statistics": (
            NONE_SENTINEL if reward_statistics is None else reward_statistics
        ),
        "model_input": model_input,
        "initial_recurrent_state": initial_recurrent_state,
        "step_count": np.asarray(step_count, dtype=np.int32),
    }


def _store_tree(arrays, prefix, tree, jax):
    for path, leaf in jax.tree_util.tree_leaves_with_path(tree):
        parts = []
        for component in path:
            for attribute in ("key", "idx", "name"):
                if hasattr(component, attribute):
                    parts.append(str(getattr(component, attribute)))
                    break
            else:
                parts.append(str(component))
        arrays[f"{prefix}/{'/'.join(parts)}"] = np.asarray(leaf)


def _load_module(source_path: Path):
    source = _instrument_source(source_path)
    temporary = tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", prefix="aaai25_instrumented_", delete=False
    )
    try:
        temporary.write(source)
        temporary.close()
        spec = importlib.util.spec_from_file_location(
            "_aaai25_instrumented_rtrrl", temporary.name
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("could not load instrumented oracle module")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module, Path(temporary.name)
    except BaseException:
        Path(temporary.name).unlink(missing_ok=True)
        raise


def _run_complete_path(
    *,
    source_path: Path,
    seed: int,
    batch_size: int,
    steps: int,
    optimizer_td: Any,
    optimizer_rnn: Any,
    update_period: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    module, temporary_path = _load_module(source_path)
    jax, jnp = module.jax, module.jnp
    environment = _Environment(jnp, batch_size)
    init_snapshot: dict[str, Any] = {}
    phases: list[dict[str, Any]] = []

    def capture_init(values):
        names = (
            "params",
            "slow_params",
            "env_state",
            "opt_state",
            "initial_action",
            "rnn_state",
            "z0",
            "v_prev",
            "r_bar",
            "initial_I",
            "obs_rms",
            "reward_rms",
            "key_step",
            "key_model",
            "key_init",
            "key_env",
            "_key",
            "initial_input",
            "h0",
        )
        init_snapshot.update(
            _copy_tree({name: values[name] for name in names}, jax, jnp)
        )

    def capture_phase(name, values):
        if name == "start":
            phases.append({})
        names = {
            "start": ("_carry", "action"),
            "directions": ("updates",),
            "incoming": ("z", "v_targ", "d", "updates"),
            "final": (
                "params",
                "slow_params",
                "env_state",
                "opt_state",
                "action",
                "rnn_state",
                "z",
                "v_hat",
                "r_bar",
                "_I",
                "_obs_rms",
                "_reward_rms",
                "seed",
                "f_input",
                "loss_info",
                "grads_next",
                "non_td_grads",
            ),
        }[name]
        phases[-1][name] = _copy_tree(
            {field: values[field] for field in names}, jax, jnp
        )

    def eager_scan(function, initial, xs):
        carry = initial
        outputs = []
        for item in np.asarray(xs):
            carry, output = function(carry, jnp.asarray(item))
            outputs.append(output)
        return carry, jax.tree.map(lambda *items: jnp.stack(items), *outputs)

    setattr(module, "_oracle_capture_init", capture_init)
    setattr(module, "_oracle_capture_phase", capture_phase)
    setattr(
        module,
        "make_env",
        lambda *_args, **_kwargs: (
            environment,
            {
                "observation_size": 1,
                "discrete": False,
                "action_size": 2,
                "obs_mask": None,
                "act_clip": None,
            },
            environment,
        ),
    )
    setattr(module, "print_env_info", lambda *_args, **_kwargs: None)
    setattr(module, "pprint", lambda *_args, **_kwargs: None)
    setattr(
        module,
        "trange",
        lambda count, **_kwargs: _Progress(count),
    )
    original_scan = jax.lax.scan
    jax.lax.scan = eager_scan
    try:
        args = module.RTRRLParams(
            seed=seed,
            episodes=1,
            steps=steps,
            patience=0,
            eval_every=0,
            eval_batch_size=batch_size,
            env_params=replace(
                module.RTRRLParams().env_params,
                batch_size=batch_size,
                render=False,
            ),
            optimizer_params_td=optimizer_td,
            optimizer_params_rnn=optimizer_rnn,
            hidden_size=2,
            gradient_mode="rtrl",
            update_period=update_period,
        )
        module.train_rtrrl(args, logger=module.DummyLogger())
    finally:
        jax.lax.scan = original_scan
        temporary_path.unlink(missing_ok=True)

    initial_carry = (
        init_snapshot["params"],
        init_snapshot["slow_params"],
        init_snapshot["env_state"],
        init_snapshot["opt_state"],
        init_snapshot["initial_action"],
        init_snapshot["rnn_state"],
        init_snapshot["z0"],
        init_snapshot["v_prev"],
        init_snapshot["r_bar"],
        init_snapshot["initial_I"],
        init_snapshot["obs_rms"],
        init_snapshot["reward_rms"],
        init_snapshot["key_step"],
    )
    arrays: dict[str, np.ndarray] = {}
    arrays["init/keys/root"] = np.asarray(jax.random.PRNGKey(seed))
    _store_tree(
        arrays,
        "init/state",
        _state_tree(
            initial_carry,
            model_input=init_snapshot["initial_input"],
            initial_recurrent_state=init_snapshot["h0"],
            step_count=0,
        ),
        jax,
    )
    for key_name, local_name in (
        ("model", "key_model"),
        ("step", "key_step"),
        ("carry", "key_init"),
        ("environment", "key_env"),
        ("outer", "_key"),
    ):
        arrays[f"init/keys/{key_name}"] = np.asarray(init_snapshot[local_name])

    for index, phase in enumerate(phases, start=1):
        start = phase["start"]
        directions = phase["directions"]
        incoming = phase["incoming"]
        final = phase["final"]
        final_carry = (
            final["params"],
            final["slow_params"],
            final["env_state"],
            final["opt_state"],
            final["action"].reshape((batch_size, -1)),
            final["rnn_state"],
            final["z"],
            final["v_hat"],
            final["r_bar"],
            final["_I"],
            final["_obs_rms"],
            final["_reward_rms"],
            final["seed"],
        )
        prefix = f"step_{index}"
        arrays[f"{prefix}/key_in"] = np.asarray(start["_carry"][-1])
        arrays[f"{prefix}/key_out"] = np.asarray(final["seed"])
        arrays[f"{prefix}/environment_action"] = np.asarray(start["action"])
        for name, value in (
            ("model_input", final["f_input"]),
            ("sampled_next_action", final_carry[4]),
            ("value_target", incoming["v_targ"]),
            ("td_error", incoming["d"]),
            ("value", final["v_hat"]),
            ("actor_loss", final["loss_info"]["actor_loss"]),
            ("entropy", final["loss_info"]["entropy"]),
        ):
            arrays[f"{prefix}/{name}"] = np.asarray(value)
        for name, value in (
            ("gradients", final["grads_next"]),
            ("direct_gradients", final["non_td_grads"]),
            ("incoming_traces", incoming["z"]),
            ("carried_traces", final["z"]),
            ("mean_directions", directions["updates"]),
            ("optimizer_updates", incoming["updates"]),
        ):
            _store_tree(arrays, f"{prefix}/{name}", value, jax)
        _store_tree(
            arrays,
            f"{prefix}/state",
            _state_tree(
                final_carry,
                model_input=final["f_input"],
                initial_recurrent_state=init_snapshot["h0"],
                step_count=index,
            ),
            jax,
        )
    return arrays, {
        "batch_size": batch_size,
        "steps": steps,
        "source_sha256": SOURCE_SHA256,
    }


def capture_complete_state_machine(
    *,
    oracle_module: Any,
    optimizers: Any,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    source_path = Path(oracle_module.__file__).resolve()
    default_td = optimizers.OptimizerConfig(
        opt_name="adam", learning_rate=1e-3, gradient_clip=None
    )
    default_rnn = optimizers.OptimizerConfig(
        opt_name="adam", learning_rate=1e-3, gradient_clip=None
    )
    one, one_metadata = _run_complete_path(
        source_path=source_path,
        seed=seed,
        batch_size=1,
        steps=3,
        optimizer_td=default_td,
        optimizer_rnn=default_rnn,
        update_period=1.0,
    )
    two, two_metadata = _run_complete_path(
        source_path=source_path,
        seed=seed,
        batch_size=2,
        steps=1,
        optimizer_td=default_td,
        optimizer_rnn=default_rnn,
        update_period=1.0,
    )
    arrays = {
        **{f"state_machine/{path}": value for path, value in one.items()},
        **{f"state_machine/two_env/{path}": value for path, value in two.items()},
    }
    jax, jnp = oracle_module.jax, oracle_module.jnp
    optimizer_config = optimizers.OptimizerConfig(
        opt_name="adam",
        learning_rate=3e-3,
        kwargs={"b1": 0.73, "b2": 0.84, "eps": 2e-5},
        decay_type="exponential",
        lr_kwargs={
            "transition_steps": 2,
            "decay_rate": 0.5,
            "staircase": True,
        },
        weight_decay=0.03,
        gradient_clip=0.4,
        multi_step=2,
    )
    optimizer = optimizers.make_optimizer(config=optimizer_config, direction="max")
    parameters = {
        "bias": jnp.asarray([0.5], dtype=jnp.float32),
        "weight": jnp.asarray([1.0, -2.0], dtype=jnp.float32),
    }
    arrays["optimizer_characterization/slow_target/previous"] = np.asarray(
        [1.0, -2.0], dtype=np.float32
    )
    arrays["optimizer_characterization/slow_target/fast"] = np.asarray(
        [3.0, 2.0], dtype=np.float32
    )
    arrays["optimizer_characterization/slow_target/period"] = np.asarray(
        0.25, dtype=np.float32
    )
    arrays["optimizer_characterization/slow_target/result"] = np.asarray(
        oracle_module.optax.incremental_update(
            jnp.asarray([3.0, 2.0], dtype=jnp.float32),
            jnp.asarray([1.0, -2.0], dtype=jnp.float32),
            0.25,
        )
    )
    optimizer_state = optimizer.init(parameters)
    _store_tree(
        arrays,
        "optimizer_characterization/init_state",
        optimizer_state,
        jax,
    )
    gradients = (
        (jnp.asarray([0.2]), jnp.asarray([3.0, -4.0])),
        (jnp.asarray([-0.1]), jnp.asarray([0.5, 0.25])),
        (jnp.asarray([0.3]), jnp.asarray([-2.0, 1.0])),
        (jnp.asarray([0.4]), jnp.asarray([0.1, -0.2])),
        (jnp.asarray([-0.2]), jnp.asarray([1.5, 2.0])),
    )
    for index, (bias_gradient, weight_gradient) in enumerate(gradients, start=1):
        updates, optimizer_state = optimizer.update(
            {"bias": bias_gradient, "weight": weight_gradient},
            optimizer_state,
            parameters,
        )
        parameters = oracle_module.optax.apply_updates(parameters, updates)
        _store_tree(
            arrays,
            f"optimizer_characterization/step_{index}/updates",
            updates,
            jax,
        )
        _store_tree(
            arrays,
            f"optimizer_characterization/step_{index}/parameters",
            parameters,
            jax,
        )
        _store_tree(
            arrays,
            f"optimizer_characterization/step_{index}/state",
            optimizer_state,
            jax,
        )
    return arrays, {"one_env": one_metadata, "two_env": two_metadata}
