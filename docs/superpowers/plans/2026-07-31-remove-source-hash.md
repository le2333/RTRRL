# Remove source_hash Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Delete the `source_hash` field, its three computations, its one comparison, and every test and fixture that names it, so that "which code ran" is answered by the image digest alone.

**Architecture:** `source_hash` is a digest of an image's Python sources, computed at catalog-build time, carried on `EntryDescriptor` and `RunConfig`, and compared in preflight against the repository's current sources. The image digest already answers the same question and is what the control plane pins. The source hash additionally fails whenever a source file changes for a reason that does not change behaviour — a comment removal invalidated it during phase one — which makes it a source of false rejections rather than a guard. Removing it is deletion only; nothing replaces it.

**Tech Stack:** Python, pydantic v2 (`extra="forbid"` on every contract model), pytest, uv, GitHub Actions.

## Global Constraints

- NEVER run pytest, python, or any test on this machine. It is a micro instance with roughly 250 MiB free and running a suite has killed the editor session. All verification happens in CI. This is absolute and has no exceptions.
- NEVER run docker in any form.
- Work on the current branch, `feature/rtrrl-lru-paper-parity`. NEVER commit to main.
- Do NOT write rationale into code or configuration files. No comments explaining a change or recording a past decision.
- Stage files with explicit paths. Do NOT use `git add -A` or `git add .`.
- `workflow_dispatch` runs against the REMOTE state of a ref, so commit AND `git push` before triggering.
- `memo/tests/test_stream_ac_golden.py` has five failures that predate this plan and are sanctioned by the human. Memo CI is "green" for this plan's purposes when those five, and nothing else, fail.
- `training-sdk/src/training_sdk/images.py` around line 172 uses `hashlib.sha256` to verify that a config blob downloaded from ECR matches the digest its manifest declares. That is content-addressed integrity checking, unrelated to `source_hash`, and MUST NOT be touched.
- Historical launch records under any `archive/` directory keep their `source_hash` field. Do not edit them.
- `memo/docs/rtrrl-task12-evidence.json` contains a `source_hashes` key in a recorded evidence artifact. Leave it.

---

### Task 1: Remove the field and everything that produces, carries, or checks it

The contract models set `extra="forbid"`, so the moment `source_hash` leaves `EntryDescriptor` every catalog builder still emitting it raises `ValidationError`, and the moment it leaves `RunConfig` every fixture still passing it does the same. There is no ordering of smaller commits that keeps all five projects green in between, so this is one task. Commit it in the steps below — several commits, one review gate, one CI sweep at the end.

**Files:**

Contract and its consumers:
- Modify: `training-sdk/src/training_sdk/contract.py:69` (drop `source_hash` from `EntryDescriptor`), `:162` (drop it from `RunConfig`), `:7` (bump `CONTRACT_VERSION` from 3 to 4)
- Modify: `training-sdk/src/training_sdk/sinks/aim.py:26` (drop the Aim run attribute)
- Modify: `training-sdk/tests/test_contract.py:29,84,96,125,150,166`
- Modify: `training-sdk/tests/test_aim_sink.py:49`
- Modify: `training-sdk/tests/test_reporter.py:27`
- Modify: `training-sdk/tests/test_worker.py:39,112`

Catalog builders:
- Modify: `memo/runner/catalog.py:38-56` (delete the `source_hash` function), `:82,:89` (its call and the emitted key)
- Modify: `rtrrl/scripts/build_catalog.py:64` (delete the function), `:95,:102`
- Modify: `rtrrl/infra/mock-trainer/scripts/build_catalog.py:22` (delete the function), `:44`

Control plane:
- Modify: `rtrrl/infra/control-plane/src/trainer_infra/preflight.py:134-139` (delete the comparison block)
- Modify: `rtrrl/infra/control-plane/src/trainer_infra/launch.py:59` (launch.json), `:112` (`RunConfig`)
- Modify: `rtrrl/infra/control-plane/src/trainer_infra/loop.py:40` (study user attribute)

Tests and fixtures:
- Modify: `rtrrl/infra/control-plane/tests/conftest.py:161`, `helpers.py:23`, `test_launch.py:49,66`, `test_local_backend.py:30`, `test_preflight_offline.py:33`, `test_space.py:18`, `test_study.py:147`
- Delete from `rtrrl/infra/control-plane/tests/test_preflight_aws.py:386-398`: `test_image_source_hash_drift_is_rejected`
- Delete from `memo/tests/test_loop.py`: the `source_hash` import at `:19` and `test_the_source_hash_ignores_bytecode` at `:197-205`
- Delete from `rtrrl/tests/test_catalog.py`: the `source_hash` import at `:18`, the assertion at `:43`, and the three tests at `:46-90`
- Delete from `rtrrl/infra/mock-trainer/tests/test_catalog.py`: the three tests at `:17-51`
- Modify: `rtrrl/infra/mock-trainer/tests/test_runtime_cpu.py:47`, `test_train.py:118`

Checked-in catalog artifacts, regenerated not hand-edited:
- Modify: `rtrrl/catalog.json`, `rtrrl/infra/mock-trainer/catalog.json`

**Interfaces:**
- Consumes: nothing from earlier tasks; this is the first task.
- Produces: `EntryDescriptor(command, metrics, space)` and `RunConfig(contract, run_id, experiment, name, launch_id, trial, entry, digest, environment, budget, params, logging, score)` — both without `source_hash`. `CONTRACT_VERSION == 4`.

- [ ] **Step 1: Write the failing tests**

Two tests, one per model, asserting the field is gone rather than merely unused. `extra="forbid"` makes a rejected key the observable behaviour.

In `training-sdk/tests/test_contract.py`:

```python
def test_an_entry_descriptor_has_no_source_hash() -> None:
    """The image digest answers which code ran; a source digest also failed on
    edits that changed no behaviour."""

    with pytest.raises(ValidationError):
        EntryDescriptor(
            command=("train",),
            metrics=("eval/episode_return",),
            space={},
            source_hash="sha256:0",
        )


def test_a_run_config_has_no_source_hash() -> None:
    with pytest.raises(ValidationError):
        RunConfig(**(run_config_kwargs() | {"source_hash": "sha256:0"}))
```

`run_config_kwargs()` does not exist yet. Look at how the existing tests in that file build a `RunConfig` (around `:150`) and either reuse the helper already there or add one that returns the full valid kwargs dict. Do not invent field values — copy them from the existing construction.

- [ ] **Step 2: Run the tests and watch them fail for the right reason**

```bash
git add training-sdk/tests/test_contract.py
git commit -m "test(contract): require the source digest to be gone"
git push origin HEAD
gh workflow run tests.yml --ref "$(git branch --show-current)"
```

Read the run with `gh run list --workflow=tests.yml --branch "$(git branch --show-current)" --limit 1` then `gh run view <id> --log-failed`.

Expected: both new tests FAIL, because `source_hash` is still an accepted field so no `ValidationError` is raised. Confirm the failure is `DID NOT RAISE`, not an import error or a `NameError` from `run_config_kwargs`. If it is either of those, fix the test and repeat this step — a red run for the wrong reason is not evidence.

- [ ] **Step 3: Remove the field from the contract and bump the version**

In `training-sdk/src/training_sdk/contract.py`, delete `source_hash: str` from `EntryDescriptor` (`:69`) and from `RunConfig` (`:162`), and change `CONTRACT_VERSION = 3` to `CONTRACT_VERSION = 4` (`:7`). The bump is required: the catalog's JSON shape changes, and an old image's catalog carrying the key is now rejected by this model, which is exactly what the version number exists to announce.

In `training-sdk/src/training_sdk/sinks/aim.py`, delete line 26 (`self._run["source_hash"] = config.source_hash`).

Update `training-sdk`'s own tests at the lines listed under **Files** by deleting the `source_hash` key from each fixture dict and the two assertions that read it (`test_aim_sink.py:49`, `test_contract.py:166`).

- [ ] **Step 4: Remove it from the three catalog builders**

In each of `memo/runner/catalog.py`, `rtrrl/scripts/build_catalog.py`, and `rtrrl/infra/mock-trainer/scripts/build_catalog.py`: delete the `source_hash` function definition entirely, delete the local that calls it, and delete the `"source_hash": ...` key from the emitted entry dict. Also delete any now-unused imports (`hashlib`, and `SOURCE_ROOTS`/`SOURCE_ROOT`/`PACKAGE_ROOT` if nothing else uses them) — the linters run in CI and will catch a stale import, but catch it yourself first.

- [ ] **Step 5: Remove it from the control plane**

In `preflight.py`, delete the whole `if image_entry.source_hash != offline_entry.source_hash:` block at `:134-139`. The surrounding function compares command, metrics and space and keeps doing so.

In `launch.py`, delete `"source_hash": plan.entry.source_hash,` at `:59` and `source_hash=launch.plan.entry.source_hash,` at `:112`. In `loop.py`, delete `"source_hash": launch.plan.entry.source_hash,` at `:40`.

- [ ] **Step 6: Remove it from every remaining test and fixture**

Delete the key from each fixture dict listed under **Files**, and delete outright these tests, whose entire subject is the hash: `test_image_source_hash_drift_is_rejected` (`test_preflight_aws.py`), `test_the_source_hash_ignores_bytecode` (`memo/tests/test_loop.py`), the three `source_hash` tests in `rtrrl/tests/test_catalog.py:46-90`, and the three in `rtrrl/infra/mock-trainer/tests/test_catalog.py:17-51`. Remove the corresponding imports.

In `rtrrl/tests/test_catalog.py`, note that `test_the_source_hash_leaves_out_the_projects_this_image_does_not_carry` (`:73-90`) is about which source roots the hash covered. It goes with the rest.

- [ ] **Step 7: Regenerate the two checked-in catalogs**

Do NOT hand-edit these. They are build outputs.

```bash
uv run --project rtrrl python rtrrl/scripts/build_catalog.py
uv run --project rtrrl/infra/mock-trainer python rtrrl/infra/mock-trainer/scripts/build_catalog.py
```

Check the exact invocation each script expects first — one of them takes `--print-label` in CI, which is a different mode. If a script writes to a path other than the checked-in one, follow what the workflow does (`.github/workflows/build-aaai-image.yml` and `build-infra-acceptance-image.yml` both build a catalog and will show you the real command).

If neither can run here without importing jax, say so in your report and instead edit the two JSON files by deleting only the `source_hash` key and changing `"contract": 3` to `"contract": 4`, and flag it as hand-edited so the reviewer knows to confirm against a CI-built catalog.

Afterwards confirm: `rg -n "source_hash" rtrrl/catalog.json rtrrl/infra/mock-trainer/catalog.json` returns nothing, and both files say `"contract": 4`.

- [ ] **Step 8: Commit the removal**

```bash
git add training-sdk/src training-sdk/tests memo/runner/catalog.py memo/tests/test_loop.py \
        rtrrl/scripts/build_catalog.py rtrrl/tests/test_catalog.py \
        rtrrl/catalog.json rtrrl/infra/mock-trainer/catalog.json \
        rtrrl/infra/mock-trainer/scripts rtrrl/infra/mock-trainer/tests \
        rtrrl/infra/control-plane/src rtrrl/infra/control-plane/tests
git commit -m "refactor: drop the source digest in favour of the image digest"
git push origin HEAD
```

- [ ] **Step 9: Verify every affected workflow**

Three workflows cover the five projects. Run all three and read all three.

```bash
gh workflow run tests.yml --ref "$(git branch --show-current)"
gh workflow run build-aaai-image.yml --ref "$(git branch --show-current)"
gh run list --branch "$(git branch --show-current)" --limit 6
```

- `Tests` (`tests.yml`) covers training-sdk, control-plane and mock-trainer. Expected: SUCCESS, with the two new contract tests passing.
- `Memo CI` (`memo-ci.yml`) triggers automatically on push and covers memo. Expected: only the five sanctioned `test_stream_ac_golden.py` failures. Compare the failure set against run `30594733778`, which is the current baseline; anything beyond those five is yours.
- `Build AAAI training image` (`build-aaai-image.yml`) covers `rtrrl/`. Leave its `push` input at the default false. Expected: SUCCESS, including the in-container probe step, which asserts on the entry's parameters and must still print `configured: brax-hopper 5 of 11 observed`.

`build-infra-acceptance-image.yml` and `build-memo-image.yml` build images and are triggered by path filters. You do not need to run them, but note in your report that both build a catalog and will exercise the changed builders on their next run.

- [ ] **Step 10: Confirm nothing was missed**

```bash
rg -n "source_hash" --glob '!**/archive/**' --glob '!*.jsonl' --glob '!*.md' --glob '!*.diff' .
```

Expected: only `memo/docs/rtrrl-task12-evidence.json` (a recorded artifact, deliberately left). Anything else is a miss. In particular confirm `training-sdk/src/training_sdk/images.py` still has its `hashlib.sha256` blob check — that one stays.

---

## Self-Review

**Spec coverage.** This plan implements the first paragraph of §7 of `docs/superpowers/specs/2026-07-30-configuration-surface-design.md`: the two contract fields, the three builder computations, the preflight comparison, the launch.json and study-attribute and Aim-attribute references, the control-plane fixtures, and the `test_preflight_aws.py` drift case. §7's `CONTRACT_VERSION` bump is Step 3. §7's exclusion of `images.py` is a global constraint and Step 10. §7's instruction to leave `archive/` alone is a global constraint. The rest of §7 (migrating experiment files) was completed in phase one, and the rest of the spec (§1–§6) is out of scope for this plan and belongs to the two that follow.

**Placeholders.** None: every step names exact files and lines, and the two new tests are written out. Step 1 deliberately does not invent `run_config_kwargs`'s contents and says to copy them from the existing construction, because guessing thirteen field values would be worse than reading them.

**Type consistency.** The only signatures this plan changes are the two model definitions, and the **Interfaces** block states their post-change field lists in full.
