# Observability SDK Task 5 Report

Status: `BLOCKED`

## Blocking evidence

The pinned dependency is Brax 0.10.5 (`rtrrl/uv.lock`, package entry at
lines 240-242). Its PPO and SAC public training APIs do not expose completed
training-episode returns or lengths to the scripts.

- PPO calls `progress_fn` only with the result of
  `Evaluator.run_evaluation(..., training_metrics)`:
  `.venv/lib/python3.12/site-packages/brax/training/agents/ppo/train.py:431-469`.
- SAC has the same callback boundary:
  `.venv/lib/python3.12/site-packages/brax/training/agents/sac/train.py:453-503`.
- `Evaluator.run_evaluation` obtains episode reward and episode length from an
  explicit evaluation rollout, prefixes those values with `eval/`, and merges
  only optimizer/timing `training_metrics`:
  `.venv/lib/python3.12/site-packages/brax/training/acting.py:117-148`.
- The current script callbacks confirm that their episode source is
  `eval/episode_reward`: `rtrrl/ppo_baseline.py:235-255` and
  `rtrrl/sac_baseline.py:201-217`.

Consequently, neither script has a truthful source for the mandatory
`train/episode_return` and `train/episode_length` summary. A post-training
rollout with the returned policy can produce an evaluation trajectory and
evaluation summary, but relabeling it as training data is expressly forbidden.
No source-string or AST-only test can make that runtime contract true.

The original RTRRL script also currently reduces training output to rewards and
done counts (`rtrrl/rtrrl.py:867-876`) without retaining completed-episode
lengths. `rtrrl_lru.py` does expose genuine aggregated training returns and
lengths through `RecordEpisodeStatistics`, but all four registered scripts must
satisfy the contract.

## Minimal unblock options

Choose one of these API changes before Task 5 is implemented:

1. **Recommended:** vendor a narrowly patched Brax 0.10.5 PPO/SAC trainer that
   carries training `env_state.info` completed-episode return/length aggregates
   to a separate host-side training-progress callback. Keep policy updates,
   PRNG splits, and return values byte-for-byte equivalent, and cover the fork
   with upstream parity tests.
2. Extend the upstream/facility Brax trainer API with a backend-neutral
   `training_episode_fn(env_steps, returns, lengths)` callback populated from
   the wrapped training environment, then pin that patched dependency.
3. Explicitly relax the mandatory-summary requirement for PPO/SAC. This changes
   the approved SDK contract and is not recommended.

Host callbacks from inside the JIT/pmap training path and relabeled evaluation
statistics are not acceptable alternatives.

## Implementation and tests

No Task 5 production code or contract tests were added because a green
four-script contract would require fabricated data or a prohibited
implementation. Child-process bootstrap and the otherwise feasible
RTRRL/RTRRL-LRU integration were intentionally not landed as a partial Task 5.

Verification results are recorded below after running the unchanged SDK test
suite and repository hygiene checks.

### Verification

- `uv run --with pytest python -m pytest tests/training_sdk -q`
  - Passed: 121 tests.
- `uv run --with pytest python -m pytest tests -k "parity or jit or smoke" -q`
  - No matching tests: 121 deselected (pytest exit 5).
- Four-script Task 5 contract smoke
  - Not added or run because the mandatory PPO/SAC training-summary source is
    unavailable; a passing smoke would encode fabricated behavior.
- `uvx ruff check logging_util.py training_sdk tests/training_sdk rtrrl.py
  rtrrl_lru.py ppo_baseline.py sac_baseline.py`
  - Failed on the pre-existing unused `functools.partial` import at
    `rtrrl/rtrrl_lru.py:17`; Task 5 changed no Python files.
- `git diff --check`
  - Passed.

## Commit

This report is the only task artifact; the commit hash is returned in the final
task response.

## Concerns

- Task 5 cannot meet its mandatory four-script contract without expanding the
  allowed change scope to include a Brax training API/fork or changing the
  contract.
- The requested broad Ruff invocation is not clean on the baseline because of
  the unrelated existing `rtrrl_lru.py:17` violation.
