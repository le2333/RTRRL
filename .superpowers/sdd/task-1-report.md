# Task 1 Report: Establish the Oracle Fixture Contract

## Status

Implemented the oracle fixture loader, stable pytree assertions, standalone
AAAI25 capture CLI, and the committed LRU fixture. No production RTRRL module
was changed.

Commit: `2f4accd` (`memo(rtrrl): establish AAAI25 oracle fixture contract`)

## RED

The first local invocation established that the development dependency group
was required:

```text
$ cd memo
$ uv run python -m pytest tests/rtrrl_parity/test_public_api.py -q
/memo/.venv/bin/python: No module named pytest
```

The required RED was then observed with the development group:

```text
$ uv run --group development python -m pytest \
    tests/rtrrl_parity/test_public_api.py -q
E   ModuleNotFoundError: No module named 'rtrrl_parity.oracle_capture'
ERROR tests/rtrrl_parity/test_public_api.py
```

After adding the assertion contract tests, their focused RED was:

```text
$ uv run --group development python -m pytest \
    tests/rtrrl_parity/test_public_api.py -q
E   ModuleNotFoundError: No module named 'rtrrl_parity.assertions'
ERROR tests/rtrrl_parity/test_public_api.py
```

## GREEN

Fixture-contract GREEN:

```text
$ uv run --group development python -m pytest \
    tests/rtrrl_parity/test_public_api.py -q
......                                                                   [100%]
```

Final contract and compatibility verification:

```text
$ uv run --group development python -m pytest \
    tests/rtrrl_parity/test_public_api.py \
    tests/online_ac/test_legacy_characterization.py::test_flatten_paths_and_tree_assertion_are_leaf_exact \
    -q
.......                                                                  [100%]
```

The compatibility test emitted one pre-existing TensorFlow Probability
deprecation warning about `jax.core.pytype_aval_mappings`; it did not fail.

Static checks:

```text
$ uv run --group development ruff check \
    tests/rtrrl_parity tests/online_ac/golden.py
All checks passed!

$ uv run --group development pyright tests/rtrrl_parity
0 errors, 0 warnings, 0 informations
```

Fixture integrity:

```text
$ uv run --group development python -c '<manifest/path/finite validation>'
13 leaves finite
aaai25_lru.npz: 3682 bytes
manifest.json: 2031 bytes
combined fixture files: 5713 bytes
```

## Files Changed

- `memo/tests/rtrrl_parity/__init__.py`
- `memo/tests/rtrrl_parity/assertions.py`
- `memo/tests/rtrrl_parity/oracle_capture.py`
- `memo/tests/rtrrl_parity/test_public_api.py`
- `memo/tests/rtrrl_parity/golden/manifest.json`
- `memo/tests/rtrrl_parity/golden/aaai25_lru.npz`
- `memo/tests/online_ac/golden.py`
  - delegates reusable flattening/comparison to the new assertion module while
    preserving its existing `(leaves, treedef)` public result
- `.superpowers/sdd/task-1-report.md`

## Fixture Provenance

- Source: `https://github.com/FranzKnut/RTRRL-AAAI25.git`
- Source commit: `4301943c349171d828d0fcf3e40944c286451415`
- Capture implementation: `memo/tests/rtrrl_parity/oracle_capture.py`
- Algorithm: `lru`
- Seed: `7`
- Dimensions: hidden `2`, input `4`, action `2`, batch `1`
- Inputs/transitions: deterministic explicit arrays; Brax was imported only as
  an AAAI25 module dependency and no reinforcement-learning environment ran
- Runtime: Python `3.12.13`, JAX `0.4.38`, JAXLIB `0.4.38`, CPU backend
- Dtype policy: `float32-complex64`
- AWS Batch successful job:
  `f262e6c8-ce4e-4e5a-81eb-4cfc6b543347`
- Batch compute: `rtrrl-cpu-queue`, `c7a.xlarge` compute environment
  (4 vCPUs, 8 GiB instance memory), with 4 vCPUs and 7168 MiB assigned to the
  container
- The AAAI25 dependencies were installed into `/tmp/aaai25-env`, separate from
  the Memorax environment.

The generated manifest records all 13 leaf paths, leaf shapes and dtypes, exact
source commit, runtime versions, backend, dimensions, seed, and transition
protocol. All float and complex leaves were independently checked as finite.

## EC2 and Resource Actions

- Confirmed the local controller is a `t3.micro`; no local memory or CPU
  expansion was attempted.
- `ec2:RunInstances` was denied for the controller role, so no standalone EC2
  instance was created.
- Reused the authorized AWS Batch `c7a.xlarge` compute environment, satisfying
  the minimum 4-vCPU/8-GiB instance requirement.
- Failed Batch attempts were retained in AWS history and diagnosed:
  missing `git`, a source-only `pytinyrenderer` dependency, missing `mujoco`,
  missing `mujoco-mjx`, missing logging dependencies, an incompatible JAX
  version, and the correct Distrax `Normal.scale` API.
- No complete RL environment or parity compilation was run on the local host.

## Self-Review

- Loader reads only JSON/NPZ and does not import AAAI25.
- Capture prepends only the explicit `--oracle-root`, verifies the exact Git
  commit, imports `RNNActorCritic`, `OnlineLRULayer`, trace helpers, and the
  optimizer, and removes its path entry afterward.
- Existing fixtures are rejected before oracle import or generation unless the
  CLI receives `--overwrite`.
- Stable paths cover mapping keys and tuple/list indices.
- Comparisons check paths, shape, and dtype before values; integers and booleans
  are exact; floating/complex non-finite values fail; exact, `(rtol, atol)`,
  and named per-leaf ULP policies are supported; signed zero is normalized for
  ULP ordering.
- Existing online-AC helper behavior remains compatible through its wrapper.
- Diff review found no credentials, environment secrets, production-module
  changes, or unrelated refactoring.

## Concerns

- The oracle repository does not pin JAX/JAXLIB. The working remote runtime
  (`0.4.38`) is therefore recorded in the manifest and must be treated as part
  of fixture provenance.
- Direct EC2 creation remains unavailable to the controller role; the
  authorized Batch path completed successfully.

## Review Fixes

The review fixes were developed test-first. The focused RED command was:

```text
$ cd memo
$ uv run --group development python -m pytest \
    tests/rtrrl_parity/test_public_api.py -q
...FFFF....                                                              [100%]
4 failed, 7 passed
```

The four expected failures demonstrated:

1. invalid manifest leaf metadata was accepted;
2. a dirty oracle worktree was accepted;
3. `main(..., overwrite=True)` was not implemented and the CLI pre-deleted
   existing files; and
4. importing `tests/online_ac/golden.py` permanently inserted `memo/tests`
   into `sys.path`.

The implementation now:

- rejects tracked, staged, or untracked AAAI25 worktree changes using
  `git status --porcelain --untracked-files=all` before reading `HEAD`;
- validates every archive path, manifest metadata entry, shape, dtype, and
  float/complex finiteness in `load_oracle`;
- stages both generated files in a temporary directory and uses
  `os.replace` only after capture, validation, archive creation, and manifest
  creation have all succeeded; and
- removes the temporary test import path in a `finally` block.

The focused GREEN command was:

```text
$ uv run --group development python -m pytest \
    tests/rtrrl_parity/test_public_api.py -q
...........                                                              [100%]
```

The final compatibility and static verification was:

```text
$ uv run --group development python -m pytest \
    tests/rtrrl_parity/test_public_api.py \
    tests/online_ac/test_legacy_characterization.py::test_flatten_paths_and_tree_assertion_are_leaf_exact \
    -q
............                                                             [100%]

$ uv run --group development ruff check \
    tests/rtrrl_parity tests/online_ac/golden.py
All checks passed!

$ uv run --group development pyright tests/rtrrl_parity
0 errors, 0 warnings, 0 informations
```

The exact fixture was regenerated from a clean clone in AWS Batch job
`210bc314-30ec-4639-9ffb-48d25e9181d1` on the existing `c7a.xlarge`
compute environment. The downloaded artifact matched the previous committed
artifact byte-for-byte:

```text
aaai25_lru.npz byte size: 3682
aaai25_lru.npz SHA-256:
e9e5537bfe399bd0f53d6967fafb67596b08f1e1e6019763d54e527c9ea3dc03
```

The regenerated manifest is 2031 bytes. The local loader tests inspect all 13
manifest leaves and verify exact path coverage, recorded shape, recorded dtype,
and finiteness for every float or complex leaf.

Review-fix commit: `c7cc355` (`fix(rtrrl): harden oracle fixture provenance`).

## Task 4 Oracle Contract Extension

The byte sizes, checksum and 13-leaf count above describe the Task 1 artifact
at the time of its original review. Task 4 review required actual strict-head
parameters and complete VJP evidence, so the same fixture contract was extended
and regenerated from the same clean pinned source.

Current committed artifact provenance:

- AWS Batch job: `6d69891a-ce03-4e69-95ac-a6fb45f258ec`
- Source commit: `4301943c349171d828d0fcf3e40944c286451415`
- Runtime: Python 3.12.13, JAX/JAXLIB 0.4.38, CPU
- Total leaves: 33, including exactly 24 `heads/` leaves
- `aaai25_lru.npz`: 9444 bytes
- NPZ SHA-256:
  `1ad7dc9eebd0b181d84aee5e0552333e953e5de4e536e4b8bc4e95182d0a6071`
- `manifest.json`: 5120 bytes

The original LRU, credit, initialization and TD-error leaves remain generated
by the same capture path. The extension adds a separately initialized strict
feedback-aligned head, its actual parameter collections, distribution evidence,
documented nontrivial cotangent and complete relevant input/variable VJPs.
