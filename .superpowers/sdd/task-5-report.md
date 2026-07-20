# Task 5 Report: Strict AAAI25 LRU Forward Component

## Status

Implemented only the strict AAAI25 LRU initialization, one-step forward/readout,
carry type, and recurrent-component protocol. Online credit/custom VJP and
training wiring remain out of scope and were not added.

## RED: Generic Memorax Differences

Before adding the wished-for component, the generic Memorax LRU was initialized
and characterized:

```text
$ uv run pytest \
    tests/rtrrl_parity/test_lru_forward_parity.py::test_generic_memorax_lru_documents_strict_aaai25_differences -q
.                                                                        [100%]
```

The characterization records the incompatibilities that prevent substituting
the generic implementation:

- generic input parameters use `params/B_imag`, while AAAI25 uses `B_img`;
- generic carry is `(state, decay)` with an inserted singleton time dimension,
  while strict AAAI25 carry is the batched complex hidden state;
- generic `read` retains a singleton time dimension;
- generic configuration defaults to an unspecified computation dtype; and
- generic initialization requests `theta_log` before `nu_log`, unlike the
  AAAI25 parameter request order.

The final wished-for tests were then installed before production code. Their
first run failed during collection:

```text
$ uv run pytest tests/rtrrl_parity/test_lru_forward_parity.py -q
E   ModuleNotFoundError: No module named 'memorax.algorithms.rtrrl.components'
```

The oracle contract test independently failed because the committed fixture
lacked the required LRU parameters and intermediate values:

```text
$ uv run pytest \
    tests/rtrrl_parity/test_public_api.py::test_oracle_fixture_has_required_sections -q
F
E   AssertionError: required LRU leaves are absent
```

After the first minimal implementation, two meaningful GREEN-stage failures
were retained and resolved from oracle evidence:

- current JAX initialization differed in every random parameter because JAX
  0.10 uses partitionable Threefry by default, unlike pinned JAX 0.4.38;
- the second transition proved that the pinned AAAI25 one-step path does not
  prepend the incoming carry to its one-element associative scan.

The initializer now scopes `jax.threefry_partitionable(False)`. The strict
forward preserves the observed AAAI25 carry behavior rather than silently
substituting the generic Memorax recurrence.

## Implementation

`RecurrentComponent` exposes exactly:

```text
initialize(key, input_shape)
forward(params, carry, inputs, reset)
credit(params, credit_state, carry, inputs, cotangent)
```

All dynamic values remain arguments. `AAAI25LRU` construction stores only
`input_dim`, `hidden_dim`, `output_dim`, and the static activation mode.
`credit` is an explicit `NotImplementedError` boundary for Task 6; no credit
equations or custom VJP were introduced.

`AAAI25LRU` reproduces:

- AAAI25 parameter names, shapes, dtypes, request order, and initializers;
- `nu_log`, `theta_log`, and derived `gamma_log`;
- complex `lambda = exp(-exp(nu_log) + i exp(theta_log))`;
- normalized `B = (B_real + i B_img) * exp(gamma_log)`;
- the exact pinned one-element scan hidden-state behavior;
- `Re((C_real + i C_img) h)` projection;
- `x D^T` skip;
- preactivation addition and SiLU output; and
- pinned reset behavior.

No public legacy facade or training builder was changed.

## Oracle Fixture Extension and Provenance

The capture now stores actual pinned AAAI25 leaves, not fitted parameters:

- initialization key;
- all eight LRU parameter leaves;
- complex lambda and normalized B;
- first and second transition inputs, carries, and outputs;
- C projection, D skip, preactivation, and SiLU output; and
- reset input, carry, and output.

The capture uses the real `OnlineLRULayer` parameter tree and helper functions
from the clean pinned repository. It derives intermediate values from those
captured parameters inside the separate oracle runtime.

Final regeneration:

- AWS Batch job: `15ee0d6a-2964-4ae2-8fde-29dfda0edcae`
- Job name: `rtrrl-oracle-task5-final-20260720`
- Status: `SUCCEEDED`
- Queue/job definition: `rtrrl-cpu-queue` / `rtrrl-cpu-job:14`
- Resources: 4 vCPU, 7168 MiB on the authorized `c7a.xlarge` environment
- Source: clean `RTRRL-AAAI25` checkout at
  `4301943c349171d828d0fcf3e40944c286451415`
- Runtime: Python 3.12.13, JAX/JAXLIB 0.4.38, Flax 0.10.2, CPU
- Seed: 7
- NPZ size: 15054 bytes
- Manifest size: 7910 bytes
- NPZ SHA-256:
  `926d87f232ba0da0de1ed9cde9d5f33e9e9d9433c54060ea40647e5dffa852cd`

The oracle environment was separate from Memorax. No full RL environment ran
locally; local verification only loaded the committed NPZ/JSON fixture.

## Numerical Evidence

Paths, shapes, and dtypes are checked before values. Exact comparison succeeds
for initialization and every forward leaf. Measured maxima against the pinned
oracle are:

```text
leaf             eager abs  eager rel  eager ULP  JIT abs  JIT rel  JIT ULP
lambda           0          0          0          0        0        0
normalized_B     0          0          0          0        0        0
next_hidden      0          0          0          0        0        0
projection       0          0          0          0        0        0
skip             0          0          0          0        0        0
preactivation    0          0          0          0        0        0
output           0          0          0          0        0        0
```

JIT versus eager also measured maximum absolute, relative, and ULP differences
of zero for every listed leaf. Therefore no tolerance was whitelisted; final
tests assert exact values.

## GREEN and Static Checks

Focused GREEN:

```text
$ uv run pytest \
    tests/rtrrl_parity/test_lru_forward_parity.py \
    tests/rtrrl_parity/test_public_api.py::test_oracle_fixture_has_required_sections -q
........                                                                 [100%]
```

Fresh full RTRRL parity suite:

```text
$ uv run pytest tests/rtrrl_parity -q
..........................................                               [100%]
42 passed
```

Static checks:

```text
$ uv run ruff check <five changed Python files>
All checks passed!

$ uv run pyright <five changed Python files>
0 errors, 0 warnings, 0 informations

$ uv run python -m compileall -q <five changed Python files>
# exit 0
```

The sole test warning is the pre-existing TensorFlow Probability use of
deprecated `jax.core.pytype_aval_mappings`.

## Concerns

- The pinned AAAI25 `LRUCell` performs `associative_scan` over the current input
  without prepending its incoming carry. For one-step calls, both normal and
  reset transitions therefore produce `B_norm @ input`; the incoming hidden
  state and reset flag do not alter the result. The second-transition and reset
  leaves prove this from the pinned implementation. This strict component
  preserves that behavior, but it is likely an upstream algorithmic defect and
  must not be “fixed” without changing the parity target.
- Exact initializer replay requires the JAX 0.4.38 Threefry partitioning mode.
  The compatibility scope is local to initialization.
- `credit` intentionally raises `NotImplementedError`; Task 6 must implement
  online credit and any custom VJP without changing this verified forward path.

## Files

- Added `memo/memorax/algorithms/rtrrl/components.py`
- Added `memo/memorax/algorithms/rtrrl/lru.py`
- Added `memo/tests/rtrrl_parity/test_lru_forward_parity.py`
- Extended `memo/tests/rtrrl_parity/oracle_capture.py`
- Extended `memo/tests/rtrrl_parity/test_public_api.py`
- Regenerated `memo/tests/rtrrl_parity/golden/aaai25_lru.npz`
- Regenerated `memo/tests/rtrrl_parity/golden/manifest.json`
- Added `.superpowers/sdd/task-5-report.md`
