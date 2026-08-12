# Semantic Refactor Behaviour Baseline

This document fixes the behaviours that the semantic framework refactor must
preserve. It records promises, not preferred implementation structure. A known
broken or transitional API is listed separately and is not made permanent merely
because it exists before the refactor.

## Shipped execution surface

- The image catalog contract is version 7.
- The only discovered executable entry is `stream_ac`.
- Its command is `python -m entries.stream_ac`.
- `entries.stream_ac.PARAMETERS` is the parameter declaration copied into the
  image catalog; `entries.stream_ac.METRICS` is the metric declaration copied
  into it.
- The serialized version-7 catalog, manifest, run configuration, and metric
  examples under `tests/contracts/v7/` are the wire baseline. Memo and Infra
  must read those files rather than import one another when the wire contract is
  migrated.

The current entry owning the StreamAC parameter tree and graph assembly is a
boundary defect, not a behaviour to preserve. The declarations and the graph it
builds are preserved while their ownership moves into the algorithm.

## Algorithm execution promise

Both refactored algorithms must present one closed program with three host-side
operations:

1. initialize state from a random key;
2. train the existing state for a requested number of environment steps;
3. evaluate the existing state for a requested number of environment steps.

Training returns updated state and readings. Evaluation returns readings and
does not update training state. Environment-step budgets, reset order,
normalization, transition fields, and declared readings are observable behaviour
and must not change during structural moves.

## StreamAC numerical references

- `memo/reference/stream_ac.py` is the local flat-kernel migration oracle for
  the layered `memorax.algorithms.stream_ac` implementation.
- `memo/tests/test_layered_parity.py` compares complete state and reading trees
  across the configurations listed in that test. This is local parity and must
  run without another checkout.
- `memo/tests/test_paper_parity.py` compares individual equations and updates to
  `mohmdelsayed/streaming-drl` commit
  `407dca7a8b584c1c20bc649053557f66e270b1e6`.
- The external comparison requires `STREAMING_DRL` and the `paper` dependency
  group. It is an explicit external-parity gate, not part of the fast suite.

## RTRRL numerical references

- `memo/memorax/algorithms/rtrrl_aaai.py` is the new implementation under
  refactor. `memorax.algorithms.rtrrl` remains a different older public path at
  this baseline.
- `memo/tests/test_rtrrl.py` specifies the local coupling, routing, trace, target
  following, reset, and reading behaviour of the new implementation.
- `memo/tests/test_rtrrl_parity.py` compares its trace algebra, complete wiring,
  and recurrent cell to the published RTRRL-AAAI25 implementation.
- The external reference is RTRRL-AAAI25 commit
  `4301943c349171d828d0fcf3e40944c286451415` and is selected through
  `RTRRL_AAAI25`.
- The external RTRRL checkout is not currently prepared by `memo-ci`; a skipped
  test is therefore not evidence of parity. The future external-parity gate must
  fetch the pinned source before claiming coverage.

## Known defects and temporary paths

- `memorax.algorithms.__init__` advertises `EvalSummary`, but the contract module
  no longer defines it. This is a broken export to remove, not preserve.
- `IndependentRTRRL` still depends on the older algorithm contract.
- The new RTRRL implementation has no executable Entry or catalog item.
- `program_of`, `drive`, and Runtime's evaluation return-shape compatibility are
  migration aids, not target public contracts.
- The version-7 run configuration carries HPO score policy and sink-specific S3
  destinations. The fixture preserves the old wire for migration tests; version
  8 deliberately removes them.

## Test classes

- `parity`: numerical agreement with another implementation.
- `external`: requires a separately obtained reference checkout or framework.
- `service`: starts or embeds a real service implementation such as Moto or an
  Aim repository.
- `integration`: crosses two or more production ownership boundaries.
- `container`: requires a built container and is never collected as an ordinary
  local pytest suite.

These classes describe evidence. Tests are not made unit tests merely because
they happen to execute quickly.
