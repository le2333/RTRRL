"""Blocking S0 exactness gate for the S1 minimal DiagSSM experiment.

This is intentionally independent of Brax: it checks the learner algebra on the
same JAX backend that will execute training.  A non-zero exit status means that
S1 training must not start.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", False)


def unpack_params(params, modes, input_size):
    offset = 0
    nu = params[offset : offset + modes]
    offset += modes
    theta = params[offset : offset + modes]
    offset += modes
    bre = params[offset : offset + modes * input_size].reshape(modes, input_size)
    offset += modes * input_size
    bim = params[offset : offset + modes * input_size].reshape(modes, input_size)
    return nu, theta, bre, bim


def coefficients(params, modes, input_size):
    nu, theta, bre, bim = unpack_params(params, modes, input_size)
    exp_nu = jnp.exp(nu)
    radius = jnp.exp(-exp_nu)
    omega = jnp.exp(theta)
    cosine, sine = jnp.cos(omega), jnp.sin(omega)
    gamma = jnp.sqrt(jnp.maximum(1.0 - radius * radius, 1e-12))
    d_radius = -exp_nu * radius
    d_gamma = (-radius / gamma) * d_radius
    return bre, bim, radius, omega, cosine, sine, gamma, d_radius, d_gamma


def state_step(state, params, inputs, reset, modes, input_size):
    bre, bim, radius, _, cosine, sine, gamma, _, _ = coefficients(
        params, modes, input_size
    )
    state = jnp.where(reset, jnp.zeros_like(state), state)
    real, imag = state[:, 0], state[:, 1]
    rotated_real = cosine * real - sine * imag
    rotated_imag = sine * real + cosine * imag
    return jnp.stack(
        [
            radius * rotated_real + gamma * (bre @ inputs),
            radius * rotated_imag + gamma * (bim @ inputs),
        ],
        axis=1,
    )


def rtrl_step(carry, item, modes, input_size):
    state, sensitivity = carry
    params, inputs, reset = item
    bre, bim, radius, omega, cosine, sine, gamma, d_radius, d_gamma = (
        coefficients(params, modes, input_size)
    )
    state = jnp.where(reset, jnp.zeros_like(state), state)
    sensitivity = jnp.where(reset, jnp.zeros_like(sensitivity), sensitivity)
    real, imag = state[:, 0], state[:, 1]
    input_real, input_imag = bre @ inputs, bim @ inputs
    rotated_real = cosine * real - sine * imag
    rotated_imag = sine * real + cosine * imag
    next_state = jnp.stack(
        [
            radius * rotated_real + gamma * input_real,
            radius * rotated_imag + gamma * input_imag,
        ],
        axis=1,
    )

    propagated_real = radius[:, None] * (
        cosine[:, None] * sensitivity[:, 0] - sine[:, None] * sensitivity[:, 1]
    )
    propagated_imag = radius[:, None] * (
        sine[:, None] * sensitivity[:, 0] + cosine[:, None] * sensitivity[:, 1]
    )
    next_sensitivity = jnp.stack([propagated_real, propagated_imag], axis=1)
    local = jnp.zeros_like(next_sensitivity)
    local = local.at[:, 0, 0].set(d_radius * rotated_real + d_gamma * input_real)
    local = local.at[:, 1, 0].set(d_radius * rotated_imag + d_gamma * input_imag)
    local = local.at[:, 0, 1].set(
        radius * (-sine * real - cosine * imag) * omega
    )
    local = local.at[:, 1, 1].set(
        radius * (cosine * real - sine * imag) * omega
    )
    local = local.at[:, 0, 2 : 2 + input_size].set(gamma[:, None] * inputs)
    local = local.at[:, 1, 2 + input_size :].set(gamma[:, None] * inputs)
    next_sensitivity += local
    return (next_state, next_sensitivity), (next_state, next_sensitivity)


def local_to_full(local, modes, input_size):
    parameter_count = 2 * modes + 2 * modes * input_size
    full = jnp.zeros((2 * modes, parameter_count), dtype=local.dtype)
    for mode in range(modes):
        rows = slice(2 * mode, 2 * mode + 2)
        full = full.at[rows, mode].set(local[mode, :, 0])
        full = full.at[rows, modes + mode].set(local[mode, :, 1])
        start = 2 * modes + mode * input_size
        full = full.at[rows, start : start + input_size].set(
            local[mode, :, 2 : 2 + input_size]
        )
        start = 2 * modes + modes * input_size + mode * input_size
        full = full.at[rows, start : start + input_size].set(
            local[mode, :, 2 + input_size :]
        )
    return full


def unroll(params, inputs, resets, modes, input_size):
    state = jnp.zeros((modes, 2), dtype=jnp.float32)
    states = []
    for step in range(inputs.shape[0]):
        state = state_step(state, params, inputs[step], resets[step], modes, input_size)
        states.append(state.reshape(-1))
    return jnp.stack(states)


def errors(actual, expected):
    difference = actual - expected
    return {
        "max_abs": float(jnp.max(jnp.abs(difference))),
        "relative": float(
            jnp.linalg.norm(difference) / (jnp.linalg.norm(expected) + 1e-12)
        ),
    }


def actor_log_prob(hidden, actor_weights, action):
    output = hidden @ actor_weights
    location, raw_scale = jnp.split(output, 2)
    bounded_log_scale = -2.0 + 4.0 * jax.nn.sigmoid(raw_scale)
    scale = jax.nn.softplus(bounded_log_scale)
    action = jax.lax.stop_gradient(action)
    return jnp.sum(
        -0.5 * ((action - location) / scale) ** 2
        - jnp.log(scale)
        - 0.5 * jnp.log(2.0 * jnp.pi)
    )


def globalize(local_gradient, modes, input_size):
    return jnp.concatenate(
        [
            local_gradient[:, 0],
            local_gradient[:, 1],
            local_gradient[:, 2 : 2 + input_size].reshape(-1),
            local_gradient[:, 2 + input_size :].reshape(-1),
        ]
    )


def run_gate(relative_tolerance, absolute_tolerance):
    modes, input_size, action_size, steps = 4, 9, 6, 25
    keys = jax.random.split(jax.random.PRNGKey(20260812), 12)
    nu = jnp.full((modes,), -2.0) + 0.1 * jax.random.normal(keys[0], (modes,))
    theta = jnp.log(jnp.linspace(0.2, 2.0, modes))
    bre = 0.1 * jax.random.normal(keys[1], (modes, input_size))
    bim = 0.1 * jax.random.normal(keys[2], (modes, input_size))
    params = jnp.concatenate([nu, theta, bre.ravel(), bim.ravel()])
    inputs = jax.random.normal(keys[3], (steps, input_size))
    reset_positions = [0, 7, 16, 24]
    resets = jnp.zeros((steps,), dtype=bool).at[jnp.array(reset_positions)].set(True)

    initial_state = jnp.zeros((modes, 2), dtype=jnp.float32)
    local_size = 2 + 2 * input_size
    initial_sensitivity = jnp.zeros((modes, 2, local_size), dtype=jnp.float32)
    scan = jax.lax.scan(
        lambda carry, item: rtrl_step(carry, item, modes, input_size),
        (initial_state, initial_sensitivity),
        (jnp.broadcast_to(params, (steps, params.size)), inputs, resets),
    )
    _, (states, local_sensitivities) = scan
    rtrl_jacobians = jnp.stack(
        [local_to_full(value, modes, input_size) for value in local_sensitivities]
    )
    ad_jacobians = jax.jacfwd(
        lambda value: unroll(value, inputs, resets, modes, input_size)
    )(params)

    per_step_j = [errors(rtrl_jacobians[t], ad_jacobians[t]) for t in range(steps)]
    vector = jax.random.normal(keys[4], params.shape)
    covector = jax.random.normal(keys[5], (steps, 2 * modes))
    rtrl_jvp = jnp.einsum("thp,p->th", rtrl_jacobians, vector)
    ad_jvp = jax.jvp(
        lambda value: unroll(value, inputs, resets, modes, input_size),
        (params,),
        (vector,),
    )[1]
    ad_vjp = jax.vjp(
        lambda value: unroll(value, inputs, resets, modes, input_size), params
    )[1](covector)[0]
    rtrl_vjp = jnp.einsum("thp,th->p", rtrl_jacobians, covector)

    actor_weights = 0.1 * jax.random.normal(keys[6], (2 * modes, 2 * action_size))
    critic_weights = 0.1 * jax.random.normal(keys[7], (2 * modes,))
    epsilon = jax.random.normal(keys[8], (action_size,))
    hidden = states[-1]
    output = hidden @ actor_weights
    location, raw_scale = jnp.split(output, 2)
    scale = jax.nn.softplus(-2.0 + 4.0 * jax.nn.sigmoid(raw_scale))
    executed_action = jnp.clip(location + scale * epsilon, -1.0, 1.0)
    actor_source_ad = jax.grad(actor_log_prob)(
        hidden, actor_weights, jax.lax.stop_gradient(executed_action)
    )
    # This is the explicit current-source algebra used by the online learner.
    d_location = (executed_action - location) / (scale**2)
    d_scale = (executed_action - location) ** 2 / (scale**3) - 1.0 / scale
    bounded_log_scale = -2.0 + 4.0 * jax.nn.sigmoid(raw_scale)
    d_bounded_log_scale = d_scale * jax.nn.sigmoid(bounded_log_scale)
    sigmoid_raw = jax.nn.sigmoid(raw_scale)
    d_raw_scale = d_bounded_log_scale * 4.0 * sigmoid_raw * (1.0 - sigmoid_raw)
    actor_source = actor_weights @ jnp.concatenate([d_location, d_raw_scale])
    critic_source = critic_weights
    final_j = rtrl_jacobians[-1]
    actor_rtrl = actor_source @ final_j
    critic_rtrl = critic_source @ final_j
    actor_ad = jax.grad(
        lambda value: actor_log_prob(
            unroll(value, inputs, resets, modes, input_size)[-1],
            actor_weights,
            jax.lax.stop_gradient(executed_action),
        )
    )(params)
    critic_ad = jax.grad(
        lambda value: critic_weights
        @ unroll(value, inputs, resets, modes, input_size)[-1]
    )(params)
    combined_ad = jax.grad(
        lambda value: actor_log_prob(
            unroll(value, inputs, resets, modes, input_size)[-1],
            actor_weights,
            jax.lax.stop_gradient(executed_action),
        )
        + critic_weights @ unroll(value, inputs, resets, modes, input_size)[-1]
    )(params)

    # Exercise eligibility with the real recurrent-gradient shape produced by J.
    actor_sources = jax.random.normal(keys[9], (steps, 2 * modes))
    critic_sources = jax.random.normal(keys[10], (steps, 2 * modes))
    local_gradients = jnp.einsum(
        "th,tmhq->tmq", actor_sources + critic_sources, local_sensitivities
    )
    recurrent_gradients = jax.vmap(
        lambda value: globalize(value, modes, input_size)
    )(local_gradients)
    importance = jnp.linspace(0.3, 1.2, steps, dtype=jnp.float32)
    decay = jnp.float32(0.99 * 0.9)
    eligibility = jnp.zeros_like(params)
    recurrence_values = []
    for step in range(steps):
        eligibility = jnp.where(resets[step], jnp.zeros_like(eligibility), eligibility)
        eligibility = decay * eligibility + importance[step] * recurrent_gradients[step]
        recurrence_values.append(eligibility)
    recurrence_values = jnp.stack(recurrence_values)
    explicit_values = []
    for end in range(steps):
        start = max(position for position in reset_positions if position <= end)
        explicit = jnp.zeros_like(params)
        for source_step in range(start, end + 1):
            power = end - source_step
            explicit += (
                decay**power
                * importance[source_step]
                * recurrent_gradients[source_step]
            )
        explicit_values.append(explicit)
    explicit_values = jnp.stack(explicit_values)

    metrics = {
        "J_all_steps": errors(rtrl_jacobians, ad_jacobians),
        "J_per_step": per_step_j,
        "J_reset_steps": {
            str(step): per_step_j[step] for step in reset_positions
        },
        "JVP": errors(rtrl_jvp, ad_jvp),
        "VJP": errors(rtrl_vjp, ad_vjp),
        "actor_head_source": errors(actor_source, actor_source_ad),
        "actor_current_source": errors(actor_rtrl, actor_ad),
        "critic_current_source": errors(critic_rtrl, critic_ad),
        "source_additivity": errors(
            actor_rtrl + critic_rtrl, combined_ad
        ),
        "eligibility_explicit_sum": errors(recurrence_values, explicit_values),
    }
    arrays = [
        states,
        local_sensitivities,
        rtrl_jacobians,
        ad_jacobians,
        rtrl_jvp,
        ad_jvp,
        rtrl_vjp,
        ad_vjp,
        actor_rtrl,
        actor_ad,
        actor_source,
        actor_source_ad,
        critic_rtrl,
        critic_ad,
        recurrence_values,
        explicit_values,
    ]
    finite = bool(all(bool(jnp.all(jnp.isfinite(value))) for value in arrays))

    checks = {}
    for name, value in metrics.items():
        if name in {"J_per_step", "J_reset_steps"}:
            continue
        checks[name] = bool(
            value["relative"] <= relative_tolerance
            or value["max_abs"] <= absolute_tolerance
        )
    checks["J_per_step"] = all(
        value["relative"] <= relative_tolerance
        or value["max_abs"] <= absolute_tolerance
        for value in per_step_j
    )
    checks["reset_boundary_semantics"] = all(
        per_step_j[step]["relative"] <= relative_tolerance
        or per_step_j[step]["max_abs"] <= absolute_tolerance
        for step in reset_positions
    )
    checks["finite"] = finite
    passed = all(checks.values())
    return {
        "schema_version": 1,
        "gate": "S1 server-backend S0",
        "backend": jax.default_backend(),
        "jax_version": jax.__version__,
        "dtype": "float32",
        "shape": {
            "modes": modes,
            "real_states": 2 * modes,
            "input_size": input_size,
            "recurrent_parameters": int(params.size),
            "local_sensitivity": list(local_sensitivities.shape[1:]),
            "global_sensitivity": list(rtrl_jacobians.shape[1:]),
            "eligibility": list(recurrence_values.shape[1:]),
        },
        "reset_positions": reset_positions,
        "tolerance": {
            "relative": relative_tolerance,
            "absolute": absolute_tolerance,
            "rule": "relative <= rtol OR max_abs <= atol",
        },
        "metrics": metrics,
        "checks": checks,
        "finite": finite,
        "pass": passed,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--rtol", type=float, default=3e-5)
    parser.add_argument("--atol", type=float, default=3e-6)
    return parser.parse_args()


def main():
    args = parse_args()
    result = run_gate(args.rtol, args.atol)
    document = json.dumps(result, indent=2, sort_keys=True)
    print(document)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(document + "\n", encoding="utf-8")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
