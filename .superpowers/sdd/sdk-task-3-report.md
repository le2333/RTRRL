# SDK Task 3 Report

## Result

Implemented the complete-episode Rerun adapter without changing training or JIT
logic. Selected episodes are written as one atomic `.rrd` artifact each under
collision-free encoded experiment/run path components. The public
`training_sdk` package remains backend-neutral.

## RED / GREEN

- RED: `PYTHONPATH=. uv run --with pytest pytest tests/training_sdk/test_rerun_adapter.py -v`
  failed during collection because `training_sdk.rerun_adapter` did not exist.
- GREEN (fake factory): the adapter tests passed after implementing selection,
  metadata, complete array logging, timeline alignment, safe paths, cleanup,
  and no-overwrite behavior.
- RED: the `TrainingRun.log_episode` delegation test returned `None` instead of
  the sink's `Path`.
- GREEN: `RerunSink`, `NullRerun`, and `TrainingRun.log_episode` now consistently
  return `Path | None`.
- RED: new `Episode.number` tests accepted zero, negative, non-integer, boolean,
  and seven-digit values.
- GREEN: `Episode` now requires a true integer in the inclusive range
  `1..999999`.
- RED: the real Rerun test promoted warnings to errors and exposed that Rerun
  tensors do not accept NumPy boolean dtype.
- GREEN: the Rerun boundary serializes boolean indicators losslessly as `uint8`;
  the generated non-empty RRD passes `rerun rrd verify`.

## Tests and Checks

- SDK Task 1-3: `92 passed` with
  `PYTHONPATH=. uv run --with pytest pytest tests/training_sdk -v`.
- Rerun real-file check: generated RRD was non-empty and
  `rerun rrd verify` returned success.
- Ruff: `uvx ruff check training_sdk tests/training_sdk` passed.
- Lock: `uv lock --check` passed.
- Patch hygiene: `git diff --check` passed.
- IDE diagnostics: no errors in changed source or tests.

## Dependencies

Ran `uv add rerun-sdk` as required. It selected `rerun-sdk>=0.34.1` and updated
`uv.lock`; `pyarrow==25.0.0` is a transitive dependency. Existing JAX/JAXlib
pins remain unchanged.

## Commit

This report is included in the Task 3 commit. The final commit hash is reported
in the task response because a Git commit cannot contain its own hash.

## Self-review

- Scope is limited to the Task 3 adapter, dependency manifests, Episode and
  TrainingRun contracts, and their tests.
- Aim experiment naming is untouched; only artifact path components are
  percent-encoded at the UTF-8 byte level.
- Failed writes remove temporary files. Existing episode targets deterministically
  raise `FileExistsError` and are never intentionally overwritten.
- N+1 observations/environment states and N transition arrays retain distinct
  episode-step timelines. Environment states with N values are also explicitly
  supported.
- Ragged, object, and non-numeric arrays fail at the adapter boundary with the
  offending field name.
- No unresolved concerns found.

## Review Follow-up: Atomic No-replace Publication

The Task 3 review identified a TOCTOU race between the final `exists()` check
and `Path.replace()`: another writer could create the target after the check,
and `replace()` would silently overwrite it.

### RED / GREEN

- RED: `test_publish_race_never_overwrites_competing_artifact` simulated a
  competing artifact appearing at the publication syscall. The old
  implementation did not call `os.link`, did not raise `FileExistsError`, and
  published its own bytes.
- RED: `test_successful_publish_fsyncs_target_directory` observed no parent
  directory fsync after publication.
- GREEN: publication now uses same-filesystem `os.link(temp, target)`, whose
  create-if-absent semantics atomically reject an existing target. On success,
  the temporary link is removed and the target parent directory is fsynced.
  There is no overwrite-capable rename/replace fallback.
- GREEN: the race test confirms the competing artifact bytes remain unchanged,
  `FileExistsError` propagates, and only the competing target remains in the
  directory.

### Follow-up Verification

- SDK Task 1-3: `94 passed` with
  `PYTHONPATH=. uv run --with pytest pytest tests/training_sdk -v`.
- Ruff: `uvx ruff check training_sdk tests/training_sdk` passed.
- Lock: `uv lock --check` passed.
- Patch hygiene: `git diff --check` passed.
- IDE diagnostics: no errors in the changed source or test.

### Follow-up Commit

This follow-up is included in a separate review-fix commit; its hash is reported
in the final task response.

### Follow-up Self-review

- The early `exists()` check remains only as a fast failure; correctness comes
  from atomic hard-link creation.
- A failed link never removes the target path, so a competitor's artifact is
  preserved. The adapter's temporary file is still cleaned in `finally`.
- Directory fsync occurs only after the temporary hard link is removed, making
  the final target directory state durable.
- No unresolved concerns found.
