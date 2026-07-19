# Independent Dual-Path RTRRL and HPO Design

## Goal

Run three controlled experiment groups on Hopper partial observability (`mode: P`):

1. Re-evaluate the best StreamAC-RTU run `d9fe098699004327b330f35f` with explicit seeds 1 and 2.
2. Run 20 HPO trials for StreamAC with an LRU-RTRL backbone.
3. Implement an RTRRL variant with no shared learned representation between actor and critic, then run 20 HPO trials for it.

“20 HPO trials” means five rounds of four AWS Batch jobs, not twenty rounds.

## Isolation and Reproducibility

- RTU seed replications reuse the original generated YAML and job-definition revision 12.
- StreamAC-LRU gets a new base config, spec, study ID, and result directory. It must not import RTU trials.
- Independent RTRRL is implemented in the `feature/composable-online-ac` worktree and gets a new preset, config, study ID, image tag, and AWS job definition.
- The current shared job definition and existing studies are not overwritten.
- Every submitted config records an explicit nonzero seed. Seed 0 is forbidden because the experiment layer replaces it with a random seed.

## StreamAC-LRU HPO

The experiment uses the existing masked MuJoCo StreamAC entry point and sets `agent_type: lru_rtrl`. Actor and critic remain independent, as in the existing StreamAC-RTRL implementation.

The base budget matches the RTU comparison:

- environment: Hopper
- observation mode: P
- 2,000,000 environment steps
- 20 evaluation epochs
- 16 parallel environments
- Aim logging

The initial search space mirrors the existing StreamAC-RTU study where parameter meanings are unchanged:

- actor and critic learning rates
- actor and critic kappa
- trace lambda
- entropy coefficient
- hidden dimension

The study objective is the existing `streamac_stable` score: mean `eval/rewards` over the 1M–2M step interval after minimum-completion filtering. The study uses a distinct `streamac-hopP-lru-v1` identity and starts with no imported RTU history.

## Independent Dual-Path RTRRL

### Representation Boundary

Actor and critic share no learned representation or recurrent state. Each path owns:

- feature-extractor parameters
- recurrent torso parameters
- hidden carry
- exact RTRL sensitivity
- eligibility traces
- slow recurrent target
- optimizer moments

They share only the raw transition, TD error, episodic emphasis, and bound hyperparameter values.

### Preserved RTRRL Semantics

The new program preserves the existing RTRRL algorithm:

- TD(0) error construction
- policy, value, and recurrent trace decays
- episodic emphasis
- incoming/fresh trace timing option
- grouped Adam ascent
- direct entropy gradient
- slow-target update
- terminal reset behavior

It is not implemented by switching to `StandardProgram`, because that program encodes StreamAC objectives, fresh traces, and whole-tree OBGD rather than RTRRL semantics.

### State and Parameters

The program state contains separate actor and critic path states. Each path contains current parameters, slow torso parameters, carry, sensitivity, recurrent trace, and optimizer state. Policy-head traces belong only to the actor path; value-head traces belong only to the critic path.

The first version binds actor and critic recurrent hyperparameters:

- equal hidden dimension per path
- equal recurrent learning rate
- equal recurrent lambda
- equal slow-target rate
- equal update period

“Hidden dimension” means width per branch, so total recurrent capacity is approximately twice that of the shared model at the same value.

### Step Data Flow

For each environment transition:

1. Run independent actor and critic acting forwards.
2. Sample the action and step the environment.
3. Run critic-only bootstrap without committing bootstrap carry.
4. Compute the shared TD error and episodic emphasis.
5. Run independent actor and critic differentiation forwards.
6. Update actor traces from policy and actor-recurrent objectives.
7. Update critic traces from value and critic-recurrent objectives.
8. Apply grouped Adam updates to each owned parameter group.
9. Update actor and critic slow torso targets independently.
10. Reset both recurrent states, sensitivities, and traces on terminal transitions.

Entropy gradients may enter actor parameters only. Value gradients may enter critic parameters only. No cross-path recurrent Jacobian is permitted.

Prediction-head support is excluded from the first implementation because ownership is ambiguous and is not needed for these Hopper trials.

## Independent RTRRL HPO

The new topology uses an independent study and does not import shared-RTRRL history. Its base budget matches the existing Hopper-P RTRRL comparison:

- 2,000,000 environment steps
- 20 evaluation epochs
- one environment
- Aim logging

The first 20-trial search keeps the existing eight-variable RTRRL space while binding both paths:

- hidden dimension per branch
- TD/head learning rate
- recurrent learning rate
- gamma
- policy lambda
- value lambda
- entropy rate
- update period

Recurrent lambda and slow-target rate remain fixed at the validated base-config
values. Expanding those dimensions is deferred because 20 trials are not enough
to search the larger topology reliably.

The objective is the existing stability-penalized score, `mean(eval/rewards) - std(eval/rewards)` over the 1M–2M interval. The study runs five rounds of four jobs.

## Tests and Acceptance Criteria

Before building an experiment image:

- actor and critic parameter trees are independently initialized
- actor and critic carry and sensitivity advance independently
- actor loss has no critic recurrent gradient path
- critic loss has no actor recurrent gradient path
- entropy cannot update critic parameters
- bootstrap does not commit recurrent carry
- slow targets update independently
- both paths reset correctly on terminal transitions
- JIT state tree structure remains stable
- existing shared RTRRL and Standard StreamAC parity tests still pass
- a short local smoke run emits finite evaluation metrics

The image is eligible for HPO only after all targeted tests and the smoke run pass.

## AWS Execution

- Submit at most four jobs concurrently.
- Keep each study’s generated configs, plan, report, Aim runs, JSONL results, and Optuna database separate.
- Use immutable image tags and a dedicated job-definition revision for the independent RTRRL implementation.
- Verify no active job with the same run name before each submission.
- `suggest` must be preceded by `sync-aim`; submission is previewed before using `--yes`.
- Stop launching new rounds if infrastructure errors, non-finite metrics, or systematic trial failures appear.

The already-approved RTU replications use seeds 1 and 2. Their results are comparisons only and are not imported into either new HPO study.
