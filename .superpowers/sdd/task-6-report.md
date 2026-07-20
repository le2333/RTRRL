# Task 6 Report: Strict LRU Online Credit

## Status

Implemented the compact AAAI25 LRU credit state, its two-step sensitivity
recurrence, recurrent parameter gradients, and a custom VJP wrapper around the
already verified `AAAI25LRU.forward`. The strict Task 5 forward implementation
was not changed, and no full training state machine was wired.

## RED

The two-step oracle test was added before production code. Its local RED run
failed at the missing requested type:

```text
$ uv run pytest \
    tests/rtrrl_parity/test_lru_credit_parity.py::test_two_step_credit_states_and_recurrent_gradients_match_oracle -q
F
E AttributeError: module 'memorax.algorithms.rtrrl.lru' has no attribute
E 'LRUCreditState'
```

The five directional finite-difference parameter groups were then run before
implementation on the authorized 8 GiB AWS Batch worker:

- Job: `b2b581ab-ac72-4437-a11d-c44906116ea9`
- Queue / definition: `rtrrl-cpu2-queue` / `rtrrl-cpu-job:14`
- Resources: 4 vCPU, 8192 MiB on `c7a.2xlarge`
- Runtime: Python 3.12, JAX/JAXLIB 0.4.38, Flax 0.10.2, CPU
- Result: five expected failures because `LRUCreditState` was absent

Thus both the fixture contract and every finite-difference group were observed
RED because the feature was missing, not because of fixture or environment
setup.

## Oracle Fixture and Provenance

The final oracle was generated in a separate clean AAAI25 runtime:

- AWS Batch job: `22f2803e-a0d3-4d19-a6ab-3bd13c80d09a`
- Job name: `rtrrl-oracle-task6-final-20260720`
- Status: `SUCCEEDED`
- Queue / definition: `rtrrl-cpu2-queue` / `rtrrl-cpu-job:14`
- Resources: 4 vCPU, 8192 MiB on `c7a.2xlarge`
- Source: clean `RTRRL-AAAI25` checkout at
  `4301943c349171d828d0fcf3e40944c286451415`
- Runtime: Python 3.12.13, JAX/JAXLIB 0.4.38, Flax 0.10.2, CPU
- Seed: 7
- NPZ size: 23208 bytes
- Manifest size: 12276 bytes
- NPZ SHA-256:
  `5043f2b0adf7f429e409310068f374bd9803e7d9affdf2620b9cb23cc51e767a`

For each of two explicit transitions, the fixture stores actual pinned:

- lambda sensitivity, shape `(2,)`, complex64;
- gamma sensitivity, shape `(2,)`, complex64;
- B sensitivity, shape `(2, 4)`, complex64;
- output cotangent, shape `(2,)`, float32;
- pinned hidden cotangent scalar, float32; and
- gradients for `nu_log`, `theta_log`, `gamma_log`, `B_real`, and `B_img`.

The source's `force_trace_compute` returns a flat four-tuple despite its
initializer returning `(hidden, (lambda, gamma, B))`. The capture invokes the
actual pinned update, then repacks only that returned state into the
initializer layout before step two. No expected sensitivity or gradient value
was constructed in Memorax or locally.

## Implementation

`LRUCreditState` keeps only the three AAAI25 sensitivity leaves:

```text
lambda_sensitivity
gamma_sensitivity
B_sensitivity
```

`AAAI25LRU.credit` updates those leaves and returns the five recurrent
gradients. It preserves the pinned `B_img = -imag(complex contraction)` rule.
`forward_with_credit` returns the exact Task 5 forward values and uses a custom
VJP to replace only recurrent parameter cotangents with online-credit values;
ordinary cotangents remain in place for readout parameters, carry, and input.

No environment, optimizer, trace owner, or training update was added.

## Numerical Evidence

Focused GREEN ran on the authorized 8 GiB Batch environment:

- Job: `4731744f-7a38-4e9f-ad32-8b5a0f053ae0`
- Status: `SUCCEEDED`
- Runtime: Python 3.12, JAX/JAXLIB 0.4.38, Flax 0.10.2, CPU
- Result: `10 passed`

Maximum absolute differences against the pinned oracle:

```text
step  sensitivity state  recurrent gradients
1     0                  0
2     0                  0
```

Eager versus JIT maximum absolute difference was
`2.98023224e-08`.

Central finite differences used float32 `epsilon=1e-3` over basis directions:

```text
group       cosine       relative error
nu_log      1.00000012   1.19965051e-04
theta_log   0.99999994   7.30149404e-05
gamma_log   1.00000000   2.06126366e-04
B_real      1.00000000   2.86421237e-05
B_img       1.00000000   3.88202425e-05
```

All analytical and numerical values were finite. Every group exceeds cosine
`0.999` and remains below relative error `1e-2`.

The final full RTRRL parity suite also ran on the authorized Batch environment:

- Job: `af671ebb-78b1-491f-9be7-2e7ff9ab88ab`
- Status: `SUCCEEDED`
- Resources: 4 vCPU, 8192 MiB on `c7a.2xlarge`
- Result: `53 passed`

## Static Checks

```text
$ uv run ruff check <four changed Python files>
All checks passed!

$ uv run pyright <four changed Python files>
0 errors, 0 warnings, 0 informations

$ uv run python -m compileall -q <four changed Python files>
# exit 0

$ git diff --check
# exit 0
```

## Concerns

- The pinned custom VJP indexes the unbatched hidden cotangent with `[0]`,
  broadcasting that first scalar across all hidden units. The oracle gradients
  prove this behavior, and the strict implementation preserves it.
- Calling the pinned custom VJP directly with a batched carry fails because its
  `(batch, hidden)` lambda cotangent is passed to a VJP expecting `(hidden,)`.
  The actual AAAI25 online-credit path is per-environment/unbatched; this task
  preserves batched forward support but does not invent batched credit
  semantics.
- The pinned forced-credit output changes carry nesting, as described above.
  `LRUCreditState` provides a stable compact layout without exposing that
  upstream structural defect.
- The custom VJP emits JAX's complex-to-real cotangent warning in the pinned
  0.4.38 runtime. Values and gradients remain finite and oracle-exact.
