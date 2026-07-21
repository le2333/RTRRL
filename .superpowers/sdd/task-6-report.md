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

## Review Follow-Up: Explicit Credit Boundary and Full Custom VJP

### RED

Shape-contract and compiled-backward tests were added before production
changes. The local shape-only RED produced two failures:

```text
test_batched_credit_is_rejected_before_calculation
  expected "credit requires unbatched"
  got a later incompatible-broadcast failure

test_batched_custom_vjp_is_rejected_before_calculation
  failed: DID NOT RAISE ValueError
```

The complete pre-fix review suite then ran on AWS Batch:

- Job: `fb5cdb84-3346-45be-9211-b64859067a4a`
- Resources: 4 vCPU, 8192 MiB on `c7a.2xlarge`
- Runtime: Python 3.12, JAX/JAXLIB 0.4.38, Flax 0.10.2, CPU
- Result: three expected failures

The two boundary failures matched the local RED. The new compiled custom-VJP
test also demonstrated that exact eager-versus-JIT output equality was too
strict due a small float32 compiled-rounding difference (`2.38418579e-07`);
its final contract uses the same narrow measured numerical policy as other
compiled paths.

### Explicit Unbatched Contract

Both `credit` and `forward_with_credit` now validate, before recurrence,
forward, or VJP work:

- lambda and gamma sensitivity shapes `(hidden_dim,)`;
- B sensitivity shape `(hidden_dim, input_dim)`;
- carry shape `(hidden_dim,)`;
- input shape `(input_dim,)`; and
- credit cotangent shape `(output_dim,)`.

Any batched or otherwise mismatched leaf raises `ValueError` beginning with
`online LRU credit requires unbatched pinned shapes` and reports every
mismatched leaf. The verified `forward` method is unchanged and remains
batched/unbatched.

### Oracle Extension

The fixture was regenerated in the clean pinned runtime to capture, after each
step, the ordinary custom-VJP outputs that AAAI25 defines:

- `C_real`, `C_img`, and `D` gradients;
- input gradient; and
- hidden, lambda-sensitivity, gamma-sensitivity, and B-sensitivity carry
  gradients.

Final review capture:

- AWS Batch job: `9488ff92-5fab-4f23-8e69-046fe94578f9`
- Status: `SUCCEEDED`
- Source commit: `4301943c349171d828d0fcf3e40944c286451415`
- Runtime: Python 3.12.13, JAX/JAXLIB 0.4.38, Flax 0.10.2, CPU
- Resources: 4 vCPU, 8192 MiB on `c7a.2xlarge`
- NPZ size: 28372 bytes
- Manifest size: 15079 bytes
- NPZ SHA-256:
  `dc4a21d037ac42f966a90cffcec342df09c477d7a694a0665d7025d11096ed29`

The compiled wrapper is compared both to these actual pinned leaves and to an
independent `jax.vjp(AAAI25LRU.forward)` pullback. The latter is the correct
merge-boundary invariant because the pinned backward first obtains the
ordinary VJP, then overwrites only the five recurrent parameter leaves. Thus
`C_real`, `C_img`, `D`, input, and carry must remain the ordinary forward VJP.

### Compiled Backward GREEN

Focused review GREEN:

- AWS Batch job: `594fc8a8-2179-449d-974f-8177dc61fd38`
- Status: `SUCCEEDED`
- Resources: 4 vCPU, 8192 MiB on `c7a.2xlarge`
- Runtime: Python 3.12, JAX/JAXLIB 0.4.38, Flax 0.10.2, CPU
- Result: `13 passed`

The test runs `jax.vjp(forward_with_credit)` inside `jax.jit`, supplies the
nontrivial second-step cotangent, and validates the full parameter dictionary,
`LRUCreditState`, `LRUCarry`, input gradient, and float0 reset gradient.

```text
ordinary params vs pinned oracle max abs   2.98023224e-08
ordinary params vs standard VJP max abs    2.98023224e-08
carry vs standard VJP max abs               0
input vs pinned/standard VJP max abs         2.38418579e-07
recurrent gradients vs pinned oracle max abs 0
```

Full RTRRL parity GREEN:

- AWS Batch job: `4827734a-6b9b-4409-8082-ff664ede4ffa`
- Status: `SUCCEEDED`
- Resources: 4 vCPU, 8192 MiB on `c7a.2xlarge`
- Result: `56 passed`

Fresh static checks:

```text
ruff: all checks passed
pyright: 0 errors, 0 warnings, 0 informations
compileall: exit 0
git diff --check: exit 0
```
