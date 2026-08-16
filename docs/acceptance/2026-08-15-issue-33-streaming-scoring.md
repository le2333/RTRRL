# Issue 33 — Streaming Scoring and Trial Settlement: Remote Acceptance Request

Requested 2026-08-15. This is a request for work on the two machines a
development checkout cannot reach: the micro control-plane instance, and AWS
Batch. It is not a report; nothing below has been run remotely yet.

## What is being accepted

Branch `fix/issue-33-hpo-scoring-oom`, commit `6a60f97`.

Scoring used to read the whole of `metrics.jsonl` into memory — fetched as
bytes, copied to a temporary file, then read back with
`read_text().splitlines()`. A finished 20M-step Hopper trial leaves 3.6 GB, so
the control plane held several gigabyte-sized copies at once and was killed
with exit 137 while both workers had already finished and uploaded
`success: true`. The change folds the reduction row by row, decodes the S3 body
a megabyte at a time, writes no temporary copy, and adds a way to settle trials
whose work finished but was never read back into the study.

## What the local suite already proves, and what it cannot

Passing locally, on a development checkout (Windows, Python 3.11):

- `uv run ruff check .` — clean.
- `uv run pytest -q` — 85 passed, 2 deselected.
- `uv run pytest -m stress` — 1 passed in 87 s. A two-round Batch HPO over a
  real 1 GiB metrics file: both rounds submitted, both trials `COMPLETE`, peak
  traced allocation under 64 MiB.

Three things that evidence does not cover, which is the whole reason for this
request:

1. **The real body.** The tests read through a fake S3 that hands back a
   `BytesIO`. Production reads a botocore `StreamingBody`, and the new reader
   depends on `body.read(n)` returning at most `n` bytes and `b""` at the end.
   Only a real object proves that.
2. **Real memory, not traced memory.** `tracemalloc` counts Python
   allocations. It does not count urllib3 and socket buffers, which is exactly
   where a streaming download could hide a copy. What matters is the process's
   maximum resident set size on the box that has ~250 MiB free.
3. **The size that actually failed.** 3.6 GB, not 1 GiB, and on Linux with the
   deployed interpreter.

## Inputs the operator must supply

These are not known here and every step below needs them:

| Name | Where it comes from |
| --- | --- |
| `EXPERIMENT` | The Hopper HPO YAML the killed launch was started from |
| `LAUNCH_ID` | The launch id of that run; it is what names the artifacts |
| `DATABASE` | The Optuna SQLite holding trials 0 and 1 as `RUNNING` |
| `CATALOG` | The catalog JSON for image `sha256:af108ad5…0a708` |

Known from the issue: the two finished Batch jobs are
`5ccf7db0-8d56-4817-9ddc-18122436975e` and
`76e1506d-9a19-40b5-af4d-5343912ec181`; region `eu-north-1`; the final 50M eval
return of the fixed reproduction was 13.37–13.56, so a settled score far
outside that range is a signal something is wrong.

`EXPERIMENT` must still declare the same `name`, `experiment`, `storage` and
pinned `image` as the killed launch. Those four fields plus `LAUNCH_ID` are
what rebuild each trial's artifact root; change any of them and settlement will
look in a prefix that was never written.

## Precondition

The branch must be pushed before anything remote runs. A `workflow_dispatch`
against an unpushed commit tests the previous state and its green means
nothing.

```bash
git push -u origin fix/issue-33-hpo-scoring-oom
```

## Step 1 — The stress test on Linux, at the real size (free, no AWS)

On a checkout with real memory, not the micro instance:

```bash
cd infra
export UV_PROJECT_ENVIRONMENT=~/.venvs/trainer-infra UV_CACHE_DIR=~/.cache/uv
uv run pytest -m stress -q                              # 1 GiB, ~90 s
TRAINER_STRESS_METRICS_BYTES=4294967296 uv run pytest -m stress -q   # 4 GiB
```

The 4 GiB run needs ~4 GiB of free disk for the generated file and takes
several minutes. **Pass:** both exit 0. The test asserts internally that the
second round was asked for and that peak allocation stayed under 64 MiB.

If it is wanted in CI rather than by hand, `tests.yml` currently runs a fixed
`pytest` command and would need a `workflow_dispatch` input to select `-m
stress`. That change is not in this branch.

## Step 2 — Score one real 3.6 GB object on the micro instance (read-only)

This is the measurement that matters: the same machine, the same object, the
same code path, watched with `/usr/bin/time`. It submits nothing, writes
nothing, and mutates no AWS state. Substitute the artifact prefix of one of the
two finished jobs above.

```bash
cd infra
/usr/bin/time -v uv run python - <<'PY'
import boto3

from trainer_infra.batch import _lines
from trainer_infra.scoring import ScoreSpec, score_lines

BUCKET = "rtrrl-artifacts-007122174918"
KEY = "trainer/<experiment>/<LAUNCH_ID>/<run-id>/metrics.jsonl"

spec = ScoreSpec(
    metric="eval/episode/return_per_step",   # must match the experiment's score block
    window_steps=(0, 20_000_000),
    reduce="last",
    direction="maximize",
    non_finite="worst",
)
body = boto3.client("s3", region_name="eu-north-1").get_object(Bucket=BUCKET, Key=KEY)["Body"]
try:
    print("score:", score_lines(_lines(body), spec))
finally:
    body.close()
PY
```

**Pass:**

- it completes without being killed;
- `Maximum resident set size` is under 250 MiB — report the number whatever it
  is, since Python plus boto3 plus optuna is already ~100 MiB of it;
- the printed score is a plausible return for that trial.

**Fail, and stop:** an exit 137, or an RSS that scales with the object.

## Step 3 — Settle trials 0 and 1 (read-only against AWS)

The recovery path, run on the control plane. It reads `result.json` and
`metrics.jsonl` for each trial the study is still waiting on and tells the
study their scores. It submits no job and starts no worker; the only thing it
writes is the local SQLite.

First record what the study currently believes:

```bash
sqlite3 "$DATABASE" 'select number, state from trials order by number;'
# expected before: 0|RUNNING  1|RUNNING
```

Then:

```bash
cd infra
/usr/bin/time -v uv run trainerctl settle "$EXPERIMENT" \
  --backend batch \
  --catalog "$CATALOG" \
  --database "$DATABASE" \
  --launch-id "$LAUNCH_ID"
```

**Pass:**

```json
{
  "launch_id": "...",
  "settled": [{"trial": 0, "value": 13.4}, {"trial": 1, "value": 13.5}],
  "still_running": []
}
```

with both trials `COMPLETE` in SQLite afterwards, and RSS again under 250 MiB.

**Fail, and stop before Step 4:**

- a trial appears under `still_running` with a `NoSuchKey` or `FileNotFound`
  reason although Batch reports its job `SUCCEEDED`. That means the artifact
  root was rebuilt from a different `LAUNCH_ID`, `name`, `experiment` or
  `storage` than the one that ran. Report the reason string; do not re-run the
  worker to work around it.
- `artifact result does not complete trial N` — the result exists but does not
  claim success, or names another trial. Report it; that is a different
  failure from this issue.

## Step 4 — Resume the remaining rounds (spends money; needs the owner's word)

Only after Steps 2 and 3 pass, and only with explicit authorization: the
original study was 3 rounds × 2 trials × 20M steps, of which round 1 is now
settled, so this is 4 remaining trials of about 45 minutes each.

Two things about resuming that are easy to get wrong:

- **`hpo.rounds` is how many more rounds to run, not the study's total.** The
  runner asks `rounds` fresh rounds every time it starts. Set `rounds: 2` in
  the YAML used for the resumed launch, or it will run three more rounds and
  the study will end up with eight trials.
- **Use a new `--launch-id`.** The already-settled trials keep their artifacts
  and their scores wherever they are; a new launch id only affects the trials
  sampled from here on, and it keeps the resumed run from overwriting the
  `round-000` control objects the killed launch left behind.

```bash
cd infra
uv run trainerctl run "$EXPERIMENT_ROUNDS_2" \
  --backend batch \
  --catalog "$CATALOG" \
  --database "$DATABASE" \
  --launch-id "$(date -u +%Y%m%d-%H%M%S)" \
  --queues run \
  > report.json
```

**Pass:** exit 0; six trials in the study, all `COMPLETE`; the controller
survives every scoring phase; RSS stays flat across rounds.

## What to report back

1. Step 1: the two exit codes and durations.
2. Step 2: the full `/usr/bin/time -v` block, or at minimum
   `Maximum resident set size`, plus the printed score and the object size from
   `aws s3api head-object`.
3. Step 3: the settle JSON verbatim, the SQLite states before and after, and
   the RSS.
4. Step 4, if authorized: `report.json`, the best trial, and the peak RSS
   observed during the run.
5. Any traceback in full. A `BatchExecutionError` naming a trial, and a
   `still_running` entry, mean different things and the distinction is the
   useful part.

## Authorization boundary

Steps 1 to 3 read AWS and spend nothing beyond S3 GET requests; they start no
container and terminate no job. Step 4 pays for four GPU-hours-equivalent of
Batch and is not authorized by this document. Nothing here retries, resubmits,
or cancels the two jobs from the failed launch — their artifacts are the
evidence Step 3 reads and must not be touched.
