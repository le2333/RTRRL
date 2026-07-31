# Working agreements

## Two machines, two sets of limits

This repository is worked on from two places, and what may be run locally
depends on which one you are sitting in. Establish which before running
anything heavier than `git`.

**The permanently-on micro instance** — the only size that stays free. Its job
is to *host*, not to compute:

- the `trainerctl` control plane and its Optuna study,
- the Aim server,
- Rerun.

It has 911 MiB of total memory, ~250 MiB free, and 2 cores. Running the test
suite there has already exhausted memory and killed the editor session more than
once. On that box do not run `pytest` — not the full suite, not one module, not
one test — and nothing else that spawns real servers or subprocesses: `moto`, an
Aim server, a worker subprocess, a training framework. Do not run `docker` in
any form; a single GPU image does not fit on the disk. Static checks that read
files without executing them are fine: `ruff check`, `git`, reading and editing
sources.

**A development checkout** with real memory. Run `ruff check` and `pytest`
locally here; that is the fast loop and it is worth using. Keep virtualenvs and
caches outside the repository (`UV_PROJECT_ENVIRONMENT`, `UV_CACHE_DIR`) so a
local run never shows up in `git status`.

## What is remote regardless of the machine

AWS Batch is not reachable from either checkout. Anything that needs a
container, a GPU, or a real training framework runs on the `dev-*` queues, and
getting there means commit, push, and dispatch against the remote ref —
`workflow_dispatch` tests the remote state, so a dispatch on an unpushed commit
silently tests the previous one and its green means nothing.

If the Batch round trip becomes the bottleneck, a local mock Batch backend is
worth building rather than working around.

A change is not verified because it was reasoned about carefully. A local pass
is evidence for the code it ran; the merge gate is still a green remote run.

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
