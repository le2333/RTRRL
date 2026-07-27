# Working agreements

## What this machine is for

It is a permanently-on micro instance — the only size that stays free — and its
job is to *host* things, not to compute:

- the `trainerctl` control plane and its Optuna study,
- the Aim server,
- Rerun.

All of those are small and IO-bound. Everything else — training, container
builds, test suites — happens elsewhere. Treat any local workload heavier than
polling and bookkeeping as a mistake in the making.

## Never run tests on this machine

This box is a micro instance: 911 MiB of total memory, ~250 MiB free, 2 cores.
Running the test suite here has already exhausted memory and killed the editor
session more than once.

Do not run `pytest` here — not the full suite, not one module, not one test.
The same goes for anything else that spawns real servers or subprocesses:
`moto`, an Aim server, a worker subprocess, a training framework.

Do not run `docker` in any form. A single GPU image does not fit on the disk.

Tests are **written** here and **executed** remotely:

- Python suites run in GitHub Actions.
- Anything needing a container, a GPU, or a real training framework runs on the
  `dev-*` AWS Batch queues.

Static checks that read files without executing them are fine: `ruff check`,
`git`, reading and editing sources.

A change is not verified because it was reasoned about carefully. It is verified
when a remote run reports it green.

## When two implementations have to agree

Do not check them by hand. Write the numerical test for every module that is
supposed to agree — the initialisation, each piece of the update, the quantities
one transition passes through — commit them together, and push. The run names
the leaves that disagree and how far apart they are; those, and only those, are
worth reading.

Reading a diff line by line to decide whether two expressions compute the same
thing, or guessing what a type checker will say about a file nothing has checked
yet, spends the attention the real disagreement is going to need. CI is the
instrument. Point it at everything at once, then read what it says.

## AWS

- `dev-*` Batch queues are for infrastructure development. Delivered
  `trainerctl run` workflows use `run-*` queues, and selecting `dev-*` requires
  an explicit `--queues dev`.
- Anything that spends money or mutates AWS needs the owner's go-ahead first.

## Scope

- `memo/` is what this repo develops, and it is not off limits. The rule that
  said otherwise belonged to the infra branch, which had to leave it alone while
  the control plane was being built; on `main` that no longer holds.
- `../memorax-upstream` is a read-only clone of upstream Memorax, kept as the
  reference that restored files answer to. Never edit it, and never let it join
  the import path — a restored file lives in `memo/` and is checked there.
