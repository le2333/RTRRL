# Experiment Parameter Evolution

This file summarizes the main parameter evolution paths currently recorded in
Aim. It is intended as a compact presentation aid rather than a full run log.

## Hopper PPO Exploration

```mermaid
flowchart LR
    H001["PPO-HOP-001\nDefault-ish Hopper PPO\nbest 487"]
    H002["002\nspring + gamma=0.99\nbest 392"]
    H003["003\nnum_evals=10\nbest 503"]
    H004["004\nbatch_size=8\nbest 528"]
    H005["005\nbatch_size=1 GPU\nbest 360"]
    H006["006\n+ normalize_obs GPU\nbest 583"]
    H007["007\n+ reward_scaling=10 CPU\nbest 846"]
    H008["008\nparallel64 normobs GPU\nbest 440"]
    H014["014\nparallel64 normobs CPU\nbest 539"]

    H009["009\nreward_scaling=5\nbest 486"]
    H010["010\nreward_scaling=20\nbest 669"]
    H011["011\nunroll=5 minibatch=8 updates=4\nbest 380"]
    H012["012\ndefault batch + normobs + reward10\nbest 445"]
    H013["013\nupdates_per_batch=4\nbest 538"]
    H015["015\n10M extension\nbest 752; final 237"]

    H016["016\nbatch32 5M\nbest 430"]
    H017["017\nbatch8 + entropy=0.001 5M\nbest 529"]
    H018["018\nbatch8 + num_envs=4 5M\nbest 678"]
    H019["019\nbatch32 + entropy=0.001 5M\nbest 735"]
    H020["020\nbatch16 + env4 + entropy 5M\nbest 450"]
    H021["021\nbatch16 + entropy 5M\nbest 488"]
    H022["022\nbatch32 + entropy + unroll20\nbest 736"]
    H023["023\nbatch32 + entropy + unroll30\nbest 648"]
    H024["024\nbatch8 + reward10 + unroll20 5M\nbest 815; no collapse"]
    H025["025\n024 + gae_lambda=0.97\nbest 783"]
    H026["026\n025 + reward_scaling=12.5\nbest 804"]

    H001 --> H002 --> H003 --> H004 --> H006 --> H007
    H003 --> H005
    H003 --> H008 --> H014
    H007 --> H009
    H007 --> H010
    H007 --> H011
    H007 --> H012
    H007 --> H013
    H007 --> H015
    H007 --> H016
    H007 --> H017
    H007 --> H018
    H016 --> H019
    H019 --> H020
    H019 --> H021
    H019 --> H022
    H019 --> H023
    H007 --> H024 --> H025 --> H026
```

Key reading:

- `PPO-HOP-007` remains the strongest 2M Hopper PPO run.
- `PPO-HOP-024` is the most useful longer PPO branch: it is no-mask, improves
  steadily to about `815`, and does not show late collapse within 5M.
- `PPO-HOP-015` is the clearest late-collapse example: it peaks near `752` and
  falls to about `237` by 10M.
- None of the Hopper PPO exploration runs reaches the official SAC level.

## Brax Baseline Anchors

```mermaid
flowchart LR
    SAC0["BRAX-HOP-SAC-OFFICIAL\nHopper SAC no mask\nbest 2175; final 1804"]
    SACM["BRAX-HOP-SAC-OFFICIAL-MASK-EVEN\nHopper SAC masked\nbest 1033; final 703"]
    PPOH1["BRAX-HOP-PPO-CGPAX-S1\nHopper PPO cgpax seed1\nbest 457"]
    PPOH2["BRAX-HOP-PPO-CGPAX-S2\nHopper PPO cgpax seed2\nbest 538"]
    HCPPO["BRAX-HC-PPO-OFFICIAL\nHalfCheetah PPO no mask\nbest 4523; final 4418"]
    HCPPO_M["BRAX-HC-PPO-OFFICIAL-MASK-EVEN\nHalfCheetah PPO masked\nbest 2122"]
    HCCGPAX["BRAX-HC-PPO-CGPAX-S1\nHalfCheetah PPO cgpax\nbest 4175; final 4093"]

    SAC0 -->|"obs_mask=even"| SACM
    HCPPO -->|"obs_mask=even"| HCPPO_M
    PPOH1 --> PPOH2
    HCPPO --> HCCGPAX
```

Key reading:

- Hopper's strongest baseline is official SAC, not the explored PPO settings.
- Masked Hopper SAC still reaches about `1033`, but is much noisier than no-mask
  SAC.
- HalfCheetah PPO validates that the baseline pipeline works; masking lowers the
  reward substantially but does not prevent learning.

## RTRRL Hopper And HalfCheetah

```mermaid
flowchart LR
    R001["RTRRL-HOP-001\nparal64 lr1e-3 no-mask earlystop\nbest 100"]
    R002["RTRRL-HOP-002\nparal64 lr1e-3 no-mask 10M\nbest 295; final 4"]
    R003["RTRRL-HOP-003\nparal1 lr1e-3 no-mask 10M\nbest 94"]
    R004["RTRRL-HOP-004\nparal1 lr1e-4 no-mask\nbest 231"]
    R005["RTRRL-HOP-005\nparal64 lr1e-4 no-mask\nbest 40"]
    R006["RTRRL-HOP-006\nparal1 lr1e-4 mask lambda0.9\nbest 133"]
    R007["RTRRL-HOP-007\nparal1 lr1e-4 mask lambda0.99\nbest 31"]
    R008["RTRRL-HOP-008\nauthor-standard mask\nbest 515 at first eval"]
    R009["RTRRL-HOP-009\nfixed trace/action no-mask\nbest 66"]
    R010["RTRRL-HOP-010\neta_f=0 author-standard mask\nbest 188"]
    R011["Current active direction\nparal64 lr1e-3 normobs no-mask\nnot yet finalized in notes"]

    HC1["RTRRL-HC-001\nHalfCheetah paral64\ncollapse"]
    HC2["RTRRL-HC-002\nHalfCheetah paral1\ncollapse"]

    R001 --> R002
    R002 --> R003
    R003 --> R004
    R002 --> R005
    R004 --> R006 --> R007
    R006 --> R008 --> R010
    R004 --> R009
    R002 --> R011
    HC1 --> HC2
```

Key reading:

- Current RTRRL results are not yet strong enough to claim Brax reproduction.
- `paral64` helps Hopper relative to `paral1`, but this may mean parallel envs
  are acting like batch learning.
- Masked RTRRL remains weak or ambiguous; `RTRRL-HOP-008` has high best reward,
  but the best occurs at the first eval, so it should not be treated as a
  learned curve.
- HalfCheetah RTRRL currently collapses.
