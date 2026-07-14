# Project Workflow

This repository follows the Issue-driven workflow defined by the workspace
Cursor rules. The Issue is the contract, state, and acceptance entry for
repository changes.

## Repository Boundary

Use this `streaming-rtrrl` directory as the working repository for this project.
Do not assume the parent directory is a git repository.

The project should only use the maintainer fork as its normal remote. Do not
restore or push to an upstream remote unless a later Issue explicitly authorizes
that change.

## Confirmation Gates

Issue-driven work has two separate gates before execution:

1. **Gate 1: Requirement Contract**
   - Confirms goal, scope, non-goals, acceptance criteria, risks, and
     assumptions.
   - Must be reviewed internally before user confirmation.
   - Must be archived as an Issue comment after user confirmation.
   - Does not authorize execution.

2. **Gate 2: Execution Plan And Authorization**
   - Confirms implementation steps, verification, review handoff, and explicit
     authorization boundaries.
   - Must be reviewed internally before user confirmation.
   - Must be archived as an Issue comment after user confirmation.
   - Only after Gate 2 is confirmed and archived may the Issue be considered
     `ready`.

If archival fails, do not advance state. Report the blocker in chat.

## Lifecycle Labels

Use Issue labels as the execution state source. Do not maintain a separate local
state in files.

- `ready`: The user has confirmed Gate 1 and Gate 2, and the plan is authorized
  for scheduling.
- `wip`: The agent is executing the authorized local work.
- `blocked`: Execution cannot continue because required user/admin input,
  permission, environment, or verification is missing.
- `review`: The agent believes the Issue-scoped staged delivery is complete and
  is waiting for user acceptance.

At most one Issue should occupy `wip`, `blocked`, or `review`.

## Task Snapshots

Task snapshots live under `docs/tasks/`.

`docs/tasks/<issue>.md` is an execution aid and durable delivery record. It is
not the source of truth for state; Issue labels are the execution state source,
and Issue comments remain the durable record for confirmed contracts,
authorization, blockers, acceptance, and close-out.

Create or update the task snapshot after Gate 2 is confirmed and archived, and
before moving the Issue label from `ready` to `wip`.

A task snapshot should include:

- Issue number and title.
- Current Issue lifecycle label at the time the snapshot was written, if useful
  for audit context.
- Links to archived Gate 1 and Gate 2 comments.
- Confirmed scope and non-goals.
- Authorized execution plan.
- Verification checklist.
- Review and acceptance handoff placeholders.

## Review And Acceptance

Before asking for user acceptance:

1. Complete the authorized work.
2. Stage only the Issue-scoped delivery set.
3. Run the required internal review.
4. Run code review tools only if executable behavior or source code changed.
5. Present the chat acceptance packet to the user.

The user accepts the staged Issue-scoped delivery state, not unstaged or later
changes. Raw diffs, Issue comments, and reviewer output are audit records; they
do not replace the chat acceptance handoff.

Only close an Issue after user acceptance is confirmed and archived. Commit,
push, PR creation, and Issue close require explicit authorization.

## Issue Templates And Labels

GitHub templates, labels, and project automation are optional. If they are not
available, document the state and gates in Issue comments and task snapshots.

This project intentionally keeps workflow infrastructure minimal. Add templates
or automation only when a later Issue authorizes them.

## Experiment Management Boundary

This workflow document does not define experiment tracking, metrics schemas,
plotting standards, MLflow, or artifact layouts. Those concerns belong to
separate experiment-management Issues.
