# Task 4 Report: Parity-Verified Heads

## Status

Implemented only the isolated strict AAAI25 dense, TD, and policy-head behavior
in the new `memorax.algorithms.rtrrl.heads` module. The legacy facade and
training builders remain unchanged; no LRU or state-machine work was started.

## TDD RED Evidence

Before production implementation, the generic Memorax Gaussian was exercised
against the strict contract:

```text
$ cd memo
$ uv run pytest tests/rtrrl_parity/test_heads_parity.py -q
FFF

test_strict_actor_has_no_bias:
  generic Dense contained bias shape (4,)
test_strict_kernel_uses_aaai25_initializer:
  8/8 elements differed; max absolute difference 0.25715518
test_strict_distribution_is_per_dimension_and_reduces_by_mean:
  MultivariateNormalDiag was not distrax.Normal
```

These failures demonstrated all required intended differences before the
strict implementation: actor bias, initializer, and distribution/reduction
semantics.

The final wished-for module tests were then installed before production code:

```text
$ uv run pytest tests/rtrrl_parity/test_heads_parity.py -q
ERROR collecting tests/rtrrl_parity/test_heads_parity.py
ModuleNotFoundError: No module named 'memorax.algorithms.rtrrl.heads'
```

The first implementation run additionally exposed two relevant compatibility
issues rather than hiding them:

- JAX 0.10 defaults to partitionable Threefry, while the fixture records JAX
  0.4.38 sampling. The test now scopes `jax.threefry_partitionable(False)` to
  reconstruct the pinned oracle key split and sample.
- Current Flax/JAX custom-VJP validation requires the module cotangent to retain
  the exact outer mapping and inner `FrozenDict` structure. The backward rule
  now preserves that structure while keeping the AAAI25 equations unchanged.

## Implementation

- `FADense`
  - defaults to AAAI25
    `glorot_normal(in_axis=-1, out_axis=-2)`;
  - preserves ordinary dense forward values;
  - stores feedback matrices at `falign/B`;
  - routes input VJPs through `B`, parameter VJPs through the forward input;
  - preserves actor/critic bias behavior.
- `RTRRLTDHead`
  - strict linear actor at `params/actor/kernel`, without bias;
  - critic at `params/critic/{kernel,bias}`;
  - continuous actor width is exactly `2 * action_dim`.
- `make_action_distribution`
  - discrete output is `distrax.Categorical`;
  - continuous output splits loc/log-scale, applies AAAI25 sigmoid bounds and
    softplus, and remains a per-dimension `distrax.Normal`;
  - consequently log-probability and entropy retain action dimensions and the
    strict objective convention is `.mean()` across those dimensions.

No new heads are wired into training.

## Fixture Leaves and Explicit Parameters

Used committed leaves:

- `heads/input`
- `heads/actor_loc`
- `heads/actor_scale`
- `heads/value`
- `init/action`

The forward test supplies an explicit float32 variable tree rather than
depending on random initialization:

- `params/actor/kernel`: `(2, 4)`, no actor bias;
- `params/critic/kernel`: `(2, 1)`;
- `params/critic/bias`: `(1,)`.

The explicit replay values are chosen to reproduce the committed fixture at
`heads/input`. Initializer identity is tested separately and exactly by
evaluating the strict default and AAAI25 initializer with the same key, shape,
and dtype. Parameter paths, shapes, and dtypes are checked before float values.

The feedback-alignment VJP test uses the committed `heads/input` leaf and
explicitly distinct forward and feedback matrices. Its forward output is exact,
and the input VJP is exactly `[[11.0, 15.0]]`, proving that `B`, not the forward
kernel, controls the input cotangent.

## Tolerances and Differences

Forward/distribution policy: `rtol=2e-6`, `atol=2e-7`, after exact path, shape,
and dtype checks.

Measured maximum absolute differences:

```text
loc:      0
scale:    1.192092896e-07
value:    0
sample:   1.788139343e-07
log_prob: 0
entropy:  2.980232239e-07
```

The initializer and feedback-alignment forward/VJP assertions are exact.
Per-dimension fixture-derived log probabilities are
`[[-0.9792221, -0.40963465]]`; their strict mean is `-0.6944284`.
Per-dimension entropies are `[[0.64159447, 0.452748]]`; their strict mean is
`0.54717124`.

## GREEN and Compatibility Evidence

Focused GREEN:

```text
$ uv run pytest tests/rtrrl_parity/test_heads_parity.py -q
....                                                                     [100%]
4 passed
```

Final fresh related suite:

```text
$ uv run pytest \
    tests/rtrrl_parity/test_heads_parity.py \
    tests/rtrrl_parity/test_public_api.py \
    tests/rtrrl_parity/test_config_compat.py -q
...................................                                      [100%]
35 passed
```

The only warning is the pre-existing TensorFlow Probability use of deprecated
`jax.core.pytype_aval_mappings`.

Static checks:

```text
$ uv run ruff check \
    memorax/algorithms/rtrrl/heads.py \
    tests/rtrrl_parity/test_heads_parity.py
All checks passed!

$ uv run pyright \
    memorax/algorithms/rtrrl/heads.py \
    tests/rtrrl_parity/test_heads_parity.py
0 errors, 0 warnings, 0 informations

$ uv run python -m compileall -q \
    memorax/algorithms/rtrrl/heads.py \
    tests/rtrrl_parity/test_heads_parity.py
# exit 0

$ git diff --check
# exit 0
```

## Batch and Oracle Provenance

No new Batch job was needed for Task 4: the committed fixture and tiny isolated
head/VJP checks fit local resources, and no full RL environment ran.

The numerical oracle remains the committed fixture generated from
`RTRRL-AAAI25` commit
`4301943c349171d828d0fcf3e40944c286451415`, Python 3.12.13, JAX/JAXLIB
0.4.38, CPU, seed 7. Its prior successful regeneration provenance is AWS Batch
job `210bc314-30ec-4639-9ffb-48d25e9181d1` on the authorized
`rtrrl-cpu-queue` / `c7a.xlarge` compute environment (4 vCPU instance, 8 GiB;
4 vCPU and 7168 MiB assigned). The oracle environment was separate from the
Memorax runtime.

## Files

- Added `memo/memorax/algorithms/rtrrl/heads.py`
- Added `memo/tests/rtrrl_parity/test_heads_parity.py`
- Added `.superpowers/sdd/task-4-report.md`

## Concerns

- The committed fixture contains head inputs/outputs and sampled action, but not
  the oracle actor/critic parameter tree or VJP leaves. Therefore the test uses
  explicit replay parameters for the committed input and an independently
  explicit feedback-alignment VJP characterization; it does not claim that
  those replay parameter values are the original oracle initialization.
- Exact sampled-action replay depends on the fixture's JAX 0.4.38 Threefry
  partitioning behavior. The test scopes the old setting and documents it;
  without that setting JAX 0.10 produces a different deterministic sample for
  the same key.
- The existing TensorFlow Probability deprecation warning remains unrelated to
  these heads.

## Review Blocker Follow-Up: Oracle-Backed Parameters and VJPs

The original Task 4 tests did not establish parameter or VJP parity: they
constructed a kernel to fit one committed output and hand-wrote distribution
and feedback-gradient expectations. This follow-up replaces every such value
with leaves generated by the pinned AAAI25 implementation.

### RED

The required fixture paths and tests were added before changing the capture
contract or fixture:

```text
$ uv run pytest \
    tests/rtrrl_parity/test_public_api.py::test_oracle_fixture_has_required_sections \
    tests/rtrrl_parity/test_heads_parity.py -q
FFF.F
4 failed, 1 passed
```

The failures were the intended old-fixture failures:

- the manifest/archive exposed only the original four head output leaves;
- `heads/params/actor/kernel` and all other real parameter leaves were absent;
- sampled-action, log-probability, entropy, reduction, and sample-key leaves
  were absent; and
- no cotangent, input VJP, parameter VJP, or zero feedback-matrix cotangent
  existed.

This directly demonstrated that the artificial replay approach could not
satisfy the reviewed contract.

### Extended Oracle Contract

The standalone capture now initializes a separate strict linear AAAI25 head
with feedback alignment enabled, without changing the original end-to-end LRU
transition leaves. It records exactly 24 `heads/` leaves:

- input and raw actor output;
- actor loc/scale, critic value, sample key and sampled action;
- per-dimension log-probability and entropy plus their mean reductions;
- actual actor kernel, critic kernel/bias, actor feedback matrix and critic
  feedback matrix;
- the documented nontrivial output cotangent
  (`actor=[[0.25,-0.5,0.75,-1.25]]`, `value=[[0.625]]`);
- input VJP;
- actor/critic kernel VJPs, critic bias VJP; and
- exact zero cotangents for both feedback matrices.

The manifest's `head_vjp` section documents the differentiated function,
cotangent output order, cotangent leaves, input-VJP leaf and differentiated
variable collections. Required-section tests assert that the `heads/` path set
is exact, not merely a subset. Existing metadata tests continue to validate
every one of the fixture's 33 total paths, shapes, dtypes and finite values.

Task 4 tests now construct the Flax variable tree solely from committed oracle
leaves. Forward, sampling, per-dimension distribution metrics, mean reductions,
complete variable VJP tree and input VJP all compare path/shape/dtype before
values. No constructed-to-fit or handwritten numerical expectation remains.

### Batch Regeneration Provenance

The fixture was regenerated from a clean clone of
`https://github.com/FranzKnut/RTRRL-AAAI25.git` at exact commit
`4301943c349171d828d0fcf3e40944c286451415` in a runtime separate from Memorax.

- AWS Batch job: `6d69891a-ce03-4e69-95ac-a6fb45f258ec`
- Job name: `rtrrl-oracle-task4-review-fix-20260720`
- Queue/job definition: `rtrrl-cpu-queue` / `rtrrl-cpu-job:14`
- Status: `SUCCEEDED`
- Resources: 4 vCPU, 7168 MiB container allocation on the authorized
  `c7a.xlarge` compute environment (8 GiB instance)
- Runtime: Python 3.12.13, JAX 0.4.38, JAXLIB 0.4.38, Flax 0.10.2,
  Distrax 0.1.5, CPU
- Seed: 7
- `aaai25_lru.npz`: 9444 bytes
- NPZ SHA-256:
  `1ad7dc9eebd0b181d84aee5e0552333e953e5de4e536e4b8bc4e95182d0a6071`
- `manifest.json`: 5120 bytes

No full RL environment was run locally. Local work loaded only the committed
NPZ/JSON fixture; AAAI25 generation and its dependency environment remained in
Batch.

### Numerical Differences and GREEN

Under exact comparisons, all measured maximum absolute differences between the
current isolated head and committed AAAI25 leaves are zero:

```text
forward:
  actor_output 0
  actor_loc 0
  actor_scale 0
  value 0
  sampled_action 0
  log_prob 0
  entropy 0
  log_prob_mean 0
  entropy_mean 0
vjp:
  input 0
  params/actor/kernel 0
  params/critic/kernel 0
  params/critic/bias 0
  falign/actor/B 0
  falign/critic/B 0
```

The previous `(rtol=2e-6, atol=2e-7)` policy is no longer needed for these
leaves; forward and complete VJP parity are asserted exactly after exact tree,
shape and dtype checks.

Focused and related GREEN:

```text
$ uv run pytest \
    tests/rtrrl_parity/test_public_api.py \
    tests/rtrrl_parity/test_heads_parity.py \
    tests/rtrrl_parity/test_config_compat.py -q
...................................                                      [100%]
35 passed
```

Static checks:

```text
$ uv run ruff check \
    tests/rtrrl_parity/oracle_capture.py \
    tests/rtrrl_parity/test_public_api.py \
    tests/rtrrl_parity/test_heads_parity.py \
    memorax/algorithms/rtrrl/heads.py
All checks passed!

$ uv run pyright \
    tests/rtrrl_parity/oracle_capture.py \
    tests/rtrrl_parity/test_public_api.py \
    tests/rtrrl_parity/test_heads_parity.py \
    memorax/algorithms/rtrrl/heads.py
0 errors, 0 warnings, 0 informations

$ uv run python -m compileall -q <same four Python files>
# exit 0

$ git diff --check
# exit 0
```

The only test warning remains the pre-existing TensorFlow Probability
`jax.core.pytype_aval_mappings` deprecation. Exact sample replay intentionally
scopes the JAX 0.4.38 Threefry partitioning mode recorded by the oracle.
