# Working agreements

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

## AWS

- `dev-*` Batch queues are for infrastructure development. Delivered
  `trainerctl run` workflows use `run-*` queues, and selecting `dev-*` requires
  an explicit `--queues dev`.
- Anything that spends money or mutates AWS needs the owner's go-ahead first.

## Scope

- Do not modify anything under `memo/`. Another team is refactoring it, and this
  branch must leave it untouched.
