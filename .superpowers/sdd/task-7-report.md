# Task 7 Report: Pure RTRRL Update Rules

## Status

Implemented only the pure strict-RTRRL numerical rules requested by Task 7:

- `td_error`
- `update_traces`
- `combine_update_directions`
- `update_emphasis_or_average_reward`
- `update_slow_target`

No initialization, sampling, environment interaction, optimizer construction,
hidden randomness, state machine, or training-program orchestration was added.

## RED

The table-driven scalar/tree suite was written before `rules.py`. The first
local run failed during collection for the missing production module:

```text
ModuleNotFoundError: No module named 'memorax.algorithms.rtrrl.rules'
```

The host had only 246 MiB available RAM and was already using swap, so all
subsequent numerical runs used the authorized 8 GiB Batch worker.

Review of the initial GREEN implementation found that Dutch trace accumulation
alone did not cover AAAI25 `compute_updates`. A dedicated true-online critic
direction test was added before changing production code. Batch job
`d0d1f036-cee7-43fe-a51c-8f5cde0cd8da` failed as expected:

```text
TypeError: combine_update_directions() got an unexpected keyword argument
'trace_mode'
```

This proved the true-online correction was absent before it was implemented.

## Equations and Edge Cases

The rules encode the source operation order and keep the environment axis until
the final update reduction.

### TD error

```text
delta_i = reward_i
          + gamma * next_value_i * (1 - terminated_i)
          - average_reward
          - value_i
```

Terminal and nonterminal cases are table-driven. A heterogeneous two-environment
case rejects computing means before the per-environment delta.

### Eligibility traces

Accumulated actor/critic/recurrent trace:

```text
reset_old_i = (1 - terminated_i) * old_i
fresh_i = gamma * lambda * reset_old_i + emphasis_i * gradient_i
```

The Dutch critic trace is leaf-local, matching `RTRRL-AAAI25/traces.py`:

```text
fresh_i = gamma_lambda * reset_old_i
          + (1 - alpha * gamma_lambda * <reset_old_i, gradient_i>) * gradient_i
```

As in the pinned AAAI25 branch, Dutch critic increments do not use episodic
emphasis. The strict `lambda_rnn == 0` branch carries the unweighted immediate
recurrent gradient rather than `emphasis * gradient`. Both `incoming` and
`fresh` update timing are explicit while the fresh trace is always carried.

### Update directions

Accumulated domains use:

```text
mean_i(delta_i * trace_i + direct_i)
```

Only the recurrent traced term receives `eta_f`/`recurrent_scale`; actor and
critic traced terms do not. Entropy and other direct gradients are neither
delta-weighted nor recurrent-scale-weighted.

Dutch critic updates include the AAAI25 true-online correction:

```text
mean_i(
    delta_i * trace_i
    + alpha * (next_value_i - value_i) * (trace_i - immediate_gradient_i)
    + direct_i
)
```

Tests explicitly reject:

- mean-before-delta reduction;
- selecting the wrong incoming/fresh trace;
- applying entropy scaling twice or multiplying direct entropy by delta;
- applying recurrent parameter-domain scaling to actor/critic or direct terms;
- omitting the Dutch true-online correction.

### Emphasis, average reward, and slow target

The mutually exclusive post-transition branches are:

```text
episodic:   emphasis_next = gamma * emphasis * (1 - terminated) + terminated
continuing: average_reward_next = average_reward + eta * mean(delta)
```

Polyak targeting consumes post-update fast parameters:

```text
period == 1: slow_next = fast
otherwise:   slow_next = period * fast + (1 - period) * slow
```

Periods `1.0` and `0.1` are covered.

## Existing `online_ac` Helper Decisions

Each candidate was compared to independent equations before deciding whether
strict rules could delegate to it:

- `make_td0` passes its supported scalar TD equation, but requires the caller
  to prebuild `bootstrap_discount` and has no average-reward operand. The strict
  function remains explicit rather than adding an adapter that changes the
  operation boundary.
- `make_rtrrl_trace` matches the accumulated, nonzero-lambda subset exactly.
  It does not implement Dutch critic traces or the pinned unweighted
  `lambda_rnn == 0` branch, so it is not reused for strict trace semantics.
- `make_slow_subtree_target` matches the independent period-`0.1` numerical
  equation, but also owns forward views, destination routing, sensitivity, and
  torso-specific state. The pure tree rule is kept separate.
- `make_grouped_adam` and `make_whole_tree_obgd` are optimizer kernels, not the
  requested RTRRL direction-combination equation, so neither is reused.

Only the already-pure `TraceDirections` data carrier is shared. No `online_ac`
production helper needed modification, and no adapter reorders numerical
operations.

## GREEN and Regression Evidence

Focused rule GREEN ran on authorized Batch job
`e39ecd7a-824d-469f-afe6-9f7f08056e3e`:

```text
18 passed in 1.04s
```

The same job also produced:

```text
ruff: All checks passed!
pyright: 0 errors, 0 warnings, 0 informations
compileall: exit 0
```

Final full RTRRL parity ran on authorized Batch job
`22c3b28a-d5fe-4ba4-8759-3ac7b6a7add4` with 4 vCPU and 8192 MiB:

```text
69 passed, 5 skipped
```

The five skips are the existing directional finite-difference tests gated by
`RTRRL_RUN_ACCELERATED_NUMERICS`; all prior oracle parity tests executed and
passed. Local `git diff --check` also exited zero.

## Concerns

- Pure tree rules assume a leading environment axis on every trace/direct
  gradient leaf. Task 8 must preserve that contract when wiring the state
  machine.
- Dutch inner products intentionally remain leaf-local because that is the
  pinned `jax.tree.map` source behavior; they are not a whole-parameter-tree
  dot product.
- Focused equation tests used the pinned JAX/JAXLIB 0.4.38 runtime. The full
  repository lock currently resolves JAX/JAXLIB 0.10.0 in Batch; both passed.
- The controller remains too memory-constrained for safe local JAX regression
  runs, so numerical evidence is from the authorized Batch workers.
