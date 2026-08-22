# Running the GPU image and the parallel channel

Everything a machine with AWS credentials needs to run what is currently
untested, and nothing it has to work out for itself. Three jobs, in order,
each answering a question the one before it does not.

## What is under test

| | |
| --- | --- |
| Branch | `fix/cuda-toolkit-cohort` |
| Commit | `bf785f1` |
| GPU image | `007122174918.dkr.ecr.eu-north-1.amazonaws.com/rtrrl@sha256:c61793b630ce46aa11217ea0a82f5da71374cb901755bc287d8c6b938d8d7ffe` |
| CPU image | `…/rtrrl@sha256:e0ee3d672a23e28487679b702354a05ffc4b18d686949e17ba657179cc129bf1` |
| Contract | 13 |

The three experiment files below already name the GPU digest. There is nothing
to fill in.

**Do not use the earlier GPU digest `sha256:bbc84fa8…`.** It is contract 12 and
the control plane at this commit speaks 13, so a launch against it stops at
validation.

## Prerequisites

A checkout at `bf785f1`, `uv`, and credentials that reach Batch, S3 and
CloudWatch. The CI role cannot do this: `rtrrl-github-actions-role` exists to
push images and has no `batch:SubmitJob`.

```bash
git fetch origin && git checkout bf785f1
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

## What is already known, so it need not be re-run

- **The GPU image compiles RTRRL's graph.** Verified on an L4 via
  `scripts/run-gpu-abort-probe.sh`. The double free that parked the GPU image in
  July does not occur on the rebuilt image with the rebuilt algorithm. Which of
  the three simultaneous changes is responsible is unknown.
- **DRQN's seeds and swept values work end to end** through the real worker and
  entry — but on CPU, in the test suite, not on a device.
- **RTRRL's swept values work at the algorithm layer** on CPU: gamma, eta_pi and
  a learning rate across a member axis, 99 of 105 float leaves diverging. Never
  through the worker, never on a device. That gap is run 3.

## Reading a failure

| What you see | What it means |
| --- | --- |
| `probe landed on 'cpu', not a GPU` | The job did not get a device. The image or the job definition, not the algorithm. |
| `the group varies <leaf>, which is static` | Two members disagree about a width or a `kind`. The control plane should have split them; if it did not, its static set and the image's disagree. |
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
