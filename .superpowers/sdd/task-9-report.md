# Task 9 Report: Closed Program and Logging Compatibility

## Status

Implemented the strict closed RTRRL program, fixed-shape scan/JIT boundary,
historical epoch aggregation, host-side logging translation, and import-compatible
legacy façade.

- `build_rtrrl_program(config, components, env) -> AgentProgram` resolves and
  closes over components exactly once.
- Production epochs return only the final fixed pytree state and scalar
  `RTRRLEpochSummary`; full parameter, gradient, trace, and debug histories are
  not scan outputs.
- Environment dataclass translation is performed once at the program boundary.
- Evaluation events use `ActionDecision` and `EvalSummary`.
- `RTRRL.init/warmup/train/evaluate` delegate to one `LegacyProgram` constructed
  in `RTRRL.__post_init__`; the former second update implementation was removed.
- `RTRRL`, `RTRRLConfig`, `RTRRLState`, `_find_leaf`, and `_tree_norm` remain
  importable.

## Strict TDD Evidence

Both requested test modules were created before production implementation.

Initial closure/JIT RED:

```text
ImportError: cannot import name 'program' from 'memorax.algorithms.rtrrl'
```

Initial logging RED, after correcting test import paths:

```text
ImportError: cannot import name '_historical_rtrrl_metrics' from 'experiment'
```

The first Batch GREEN attempt exposed that the Task 8 mock environment state
was an ordinary dataclass and therefore not a valid JAX carry. The existing
fixed-pytree test failed three program/JIT cases. Registration was moved to the
environment/program boundary; focused Batch job
`268a0d90-bee1-4526-a18c-933c306289de` then passed all four program-contract and
three logging-contract tests.

## Closed Program, JIT, and Retrace Evidence

`test_program_contract.py` verifies:

- the init and production-step factories are each selected once during build;
- the declared schemas are `RTRRLState` and `RTRRLEpochSummary`;
- init and final state have the same pytree structure;
- every epoch-summary leaf is scalar;
- recursive JAXPR inspection contains none of `debug_callback`, `io_callback`,
  `pure_callback`, or `outside_call`;
- two calls through the same fixed-shape `jax.jit` wrapper increment the Python
  trace counter exactly once;
- evaluation events carry an `ActionDecision`.

No host callback is used. `num_steps` is a fixed scan length at each compiled
shape, and repeated calls with identical state/summary structures reuse the
trace.

## Output Tree Size and Schema

Authorized schema job `de8b504b-f55b-4614-9981-43b3d20b57e1` compiled and ran a
three-step production epoch and reported:

```text
state_leaves=80
state_bytes=1289
summary_leaves=46
summary_bytes=184
```

All 46 summary leaves are scalar. The count includes 33 scalar historical
`norms/*` leaves, not stacked parameter arrays. Summary fields are:

```text
steps, mean_reward, num_episodes, mean_delta, mean_r_bar, mean_v,
total_td_loss, actor_loss, critic_loss, entropy, v_targ, magnitude_loss,
learning_rate_td, learning_rate_rnn, norms
```

`magnitude_loss` and learning-rate keys remain optional at host translation;
their absent values are not pytree leaves.

## Historical Logging Evidence

Synthetic fixed step summaries assert the exact historical reductions:

- `steps = num_steps * num_envs`, dtype `int32`;
- `num_episodes = sum(done counts)`, dtype `int32`;
- reward and delta are summed over environments and divided by
  `max(num_episodes, 1)`;
- `mean_r_bar` uses final average reward divided by the same divisor;
- values and loss observables use historical means;
- all non-counter metrics are `float32`;
- `total_td_loss = actor_loss + critic_loss`;
- `norms/*` are scalar leaf L2 norms over `z`, `params`, and `slow_params`.

The asserted historical keys are:

```text
steps, mean_reward, num_episodes, mean_delta, mean_r_bar, mean_v,
total_td_loss, actor_loss, critic_loss, entropy, v_targ,
optional magnitude_loss, optional lr/td, optional lr/rnn, norms/*
```

The logger test proves that epoch index 4 with three scan steps logs at logger
step 15 while the batched `steps` metric is 6. At a non-logging epoch only
evaluation keys are emitted. It also verifies `eval/rewards`,
`eval/best_eval_reward`, and `env/video` naming and best-reward behavior.

## Legacy Delegation

The legacy class now contains no optimizer, gradient, trace, TD, or environment
update implementation. Construction creates one composed `LegacyProgram`;
lifecycle methods and the retained `_update_step` compatibility shim delegate
to it. `build_rtrrl_agent` returns that same already-constructed façade rather
than constructing another program.

Normalization ownership is explicit at construction:

- direct historical `RTRRL(...)` keeps environment wrappers;
- the experiment builder selects program-owned normalization and strips only
  outer normalization wrappers.

The authorized final job passed six targeted existing RTRRL builder tests,
including normalized train/eval parity and one-step delegation.

## Batch Provenance and Verification

No JIT/full parity or complete RL environment ran locally. Local work was
limited to RED collection and static checks.

Final authorized Batch job:

- job: `dbdf3052-2154-4a40-a750-621715f7fbec`;
- queue/definition: `rtrrl-cpu2-queue` / `rtrrl-cpu-job:14`;
- resources: 4 vCPU, 8192 MiB;
- runtime: Python 3.12, JAX/JAXLIB 0.10.0, Flax 0.12.7, CPU;
- full `tests/rtrrl_parity`: 97 passed, 5 pre-existing opt-in directional
  finite-difference skips;
- targeted existing RTRRL builders: 6 passed;
- ruff: all checks passed;
- pyright: 0 errors, 0 warnings, 0 informations;
- compileall: exit 0.

Schema-size Batch job:

- job: `de8b504b-f55b-4614-9981-43b3d20b57e1`;
- resources: 4 vCPU, 8192 MiB;
- result: succeeded with the output sizes recorded above.

## Concerns

- A diagnostic whole-file builder run
  (`ebef1404-a1c2-4b21-b98e-9f9d9a780691`) found two pre-existing
  StreamAC-RTRL-only dtype mismatches: the composed state uses explicit
  `int32` counters while the old direct `StreamACRtrlState` stores Python
  integers observed as `int64`. No RTRRL builder failed. This unrelated
  StreamAC issue was not changed in Task 9.
- Production summary size scales with the number of parameter leaves only
  through scalar `norms/*` values. It never scales with epoch length.
- Experimental RTU, prediction, independent-topology component restoration is
  intentionally deferred to Task 10. The external legacy script remains Task 11.
