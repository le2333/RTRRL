# Running the GPU image and the parallel channel

Everything a machine with AWS credentials needs to run what is currently
untested, and nothing it has to work out for itself. Three jobs, in order,
each answering a question the one before it does not.

## What is under test

| | |
| --- | --- |
| Branch | `fix/cuda-toolkit-cohort` |
| Commit | the tip of that branch; `git log -1` after checkout |
| GPU image | `007122174918.dkr.ecr.eu-north-1.amazonaws.com/rtrrl@sha256:73ff0adce076ab3cd549e1c6472bfe25967e9718ac52a4e27e8162c7a1273ac4` |
| CPU image | `…/rtrrl@sha256:2dc96e279e07629a72bc30d22c803a3c090421a5f34a1efce19fe0742952f9ff` |
| Contract | 13 |

The three experiment files below already name the GPU digest. There is nothing
to fill in.

**Earlier digests are not interchangeable.** `sha256:bbc84fa8…` is contract 12
and the control plane here speaks 13. `sha256:c61793b6…` is contract 13 but
predates the fix that lets a group span trials, which is what blocked the first
swept launch -- runs 1 and 2 are unaffected by it, run 3 needs this image.

## Prerequisites

A checkout of the branch, `uv`, and credentials that reach Batch, S3 and
CloudWatch. The digest is what actually pins what runs -- it is immutable, and
the experiment files name it -- so the checkout only has to be new enough to
carry the same catalog. The CI role cannot do this: `rtrrl-github-actions-role` exists to
push images and has no `batch:SubmitJob`.

```bash
git fetch origin && git checkout fix/cuda-toolkit-cohort && git pull
```

The control plane reads the image's catalog from a local file, so generate it
from the same commit the image was built from — which is this one, so the two
agree by construction:

```bash
cd memo && uv run --frozen python -m deployment.catalog && cd ..
```

`memo/catalog.json` is generated and gitignored. Every command below passes it
with `--catalog`.

## The runs

All three go to `dev-gpu-queue` (`--queues dev`) on `g6.xlarge`. Run them from
`infra/`, since that is where `trainerctl` lives.

### 1. The baseline: one member, the serial channel

```bash
cd infra
uv run --frozen trainerctl run "../experiments/drqn gpu smoke.yaml" \
  --backend batch --queues dev \
  --catalog ../memo/catalog.json \
  --database ./runs/drqn-gpu-smoke.db
```

`entry: drqn`, one seed. This is the channel every launch has used, and the
number it produces is what the next run is divided against.

**Read:** exit 0; two evaluation checkpoints of ten episodes each; the wall
clock of the *second* chunk. The first chunk pays for compilation and the second
does not, which is why the file runs two.

### 2. DRQN's seeds as one graph

```bash
uv run --frozen trainerctl run "../experiments/drqn gpu ensemble.yaml" \
  --backend batch --queues dev \
  --catalog ../memo/catalog.json \
  --database ./runs/drqn-gpu-ensemble.db
```

The same file as run 1 with `entry: drqn_ensemble` and `seeds: [0, 1, 2]`.
Nothing else differs, which is what makes the pair comparable.

**Read:** three results, three metrics streams, three Aim runs. The members must
differ from each other — three identical runs would mean one computed thrice.
Then this job's wall clock ÷ 3, against run 1's. **That ratio is the first real
number about whether filling the device is worth anything.** Everything said
about it so far has been inference.

### 3. RTRRL through the parallel channel, sweeping a value

```bash
uv run --frozen trainerctl run "../experiments/rtrrl gpu ensemble.yaml" \
  --backend batch --queues dev \
  --catalog ../memo/catalog.json \
  --database ./runs/rtrrl-gpu-ensemble.db
```

Two gammas × three seeds = six members of one graph.

**Read:** six results, six Aim runs, all six differing. This is the run with the
most that has never been tried: RTRRL has never gone through the ensemble entry,
no ensemble has run on a GPU, and no swept value has reached a device.

## What has been run

Runs 1 and 2 have been, on `sha256:c61793b6…`, and the fix in this image does not
touch what they exercise. They are here for a repeat, or for the second-chunk
numbers below.

| | wall clock | members | per member |
| --- | --- | --- | --- |
| Run 1, one member, serial | 40.805 s | 1 | 40.805 s |
| Run 2, three seeds, parallel | 40.525 s | 3 | 13.508 s |

Three times the work in the same wall clock, so the device was not close to
saturated at one member and is not at three. **3× is a floor, not a ceiling**,
and the question worth asking next is where it stops being linear rather than
whether three is enough.

One thing those totals cannot separate: how much of the gain is the device being
filled and how much is compilation being paid once instead of three times. Only
the first keeps scaling with members. Each file runs two chunks so that the
second one answers this — **the second chunk's wall clock from each job is the
number still wanted.**

Run 3 has never succeeded. Its first attempt was refused before reaching the
device, by an entry that required a group to differ only in its seed while the
control plane had correctly grouped two trials. That is what this image fixes.

Also known, and not needing a repeat: the GPU image compiles RTRRL's graph
(verified on an L4 via `scripts/run-gpu-abort-probe.sh` — the July double free
does not occur, though which of the three simultaneous changes is responsible is
unknown), and RTRRL's swept values work at the algorithm layer on CPU, 99 of 105
float leaves diverging across a member axis. Never through the worker, never on
a device. That gap is run 3.

## Reading a failure

| What you see | What it means |
| --- | --- |
| `probe landed on 'cpu', not a GPU` | The job did not get a device. The image or the job definition, not the algorithm. |
| `the group varies <leaf>, which is static` | Two members disagree about a width or a `kind`. The control plane should have split them; if it did not, its static set and the image's disagree. |
| `a group may differ only in its seed` | An image older than this one. Two trials in a group is correct and this message predates that being allowed. |
| `is not pinned to a digest` | An experiment file still says `TBD`. |
| `image catalog declares no entry` | `--catalog` is stale. Regenerate it (see Prerequisites). |
| Exit code -6, no Python traceback | A native abort — the July failure's signature. Keep the CloudWatch stream and the job's `statusReason`: a SIGABRT leaves its trace in the reason, not in the log. |

## If a probe is wanted instead of a launch

`scripts/run-gpu-abort-probe.sh` submits one job that compiles a graph and
nothing else — no S3, no Aim, no manifest. It is the cheapest way to ask whether
something compiles.

```bash
PROBE_ENTRY=rtrrl scripts/run-gpu-abort-probe.sh <gpu-digest>
PROBE_ENTRY=rtrrl PROBE_DIFFERENTIATION=tbptt scripts/run-gpu-abort-probe.sh <gpu-digest>
```

The second replaces the jacobian-in-scan with truncated backpropagation, so a
crash on the first and not the second would name the defect. It refuses to pass
on the CPU: a job that lands without a device says so rather than compiling
happily and reporting that nothing went wrong.

## One thing to keep in mind when comparing numbers

An ensemble member is **not** bit-identical to the same seed under the serial
channel, and cannot be. `jax.vmap` rewrites the computation into batched
operations and XLA reduces them in a different order; a one-member ensemble
already diverges from the driver. What does hold, and is tested, is that a
member is a function of its seed and of nothing else about the round — not its
size, not its position in it, not which other members it travelled with.

So run 2 and run 1 are compared on wall clock, not on scores. A score that
differs between them is expected.
