Task training-evaluation-1: complete (commits 82962d4..dfade53, review clean, WSL ruff pass, WSL pytest 41 passed)
Task training-evaluation-2: complete (commits dfade53..36a0136, review clean after fixes, WSL ruff pass, WSL non-Task5 pytest 106 passed)
Task training-evaluation-3: complete (commits 36a0136..98a050d, review clean after minor fix, WSL ruff pass, targeted mock-trainer pytest pass)
Task training-evaluation-4: complete with one gap handed to Task 5 (commits 98a050d..2df1f3b; WSL ruff pass on all four projects; training-sdk 79 passed, mock-trainer 100 passed, control-plane 160 passed + 27 Task-5 failures, memo 276 passed + 5 sanctioned golden + 31 Task-5 failures; gap: test_hopper_reproduction.py:75 still calls the old two-argument build())
Task training-evaluation-5: complete (commits 75be204..98db37a; 27 YAMLs migrated, four seed sweeps split into 20 files, manual rewritten; memo back to exactly its 5 sanctioned golden failures, control-plane 203 passed, rtrrl 26 passed, training-sdk 79 passed; ruff clean on all four projects)
Plan 2026-07-31-training-evaluation-sections.md: all five tasks done. Next: 2026-07-31-parameter-catalog-and-conditional-sampling.md (spec phase 2), then phases 3 and 4, which have no plans yet.

--- spec phase 2: parameter catalog and conditional sampling ---

Seed sweeps deleted (ecdc4b7): the four streamac-hopper-seeds-* experiments existed only to vary the seed, which is an environment input and never a search dimension. experiments/ is 23 files plus two templates.

Spec decisions taken this round, all recorded in 2026-07-30-configuration-surface-design.md and authoritative over any plan:
- Every parameter declares a search; a parameter that should not move gets a single-point search rather than omitting the field. placeholder narrows to what an inactive branch collapses to.
- Structures are not searched. A study fixes its structure by experiment override or placeholder; only parameters are drawn. Comparing structures is two studies. Grid needs no special case.
- Branch parameters are keyed by the component, e.g. ob.kappa, rtu.hidden_dim, adam.b1. flatten raises when two branches offer the same component name.
- An omitted parameter means the entry's declared search. The memo guard that required every parameter be named is deleted (7aad33c).
- Components own their declarations beside their implementation.
- Structure fields belong to the algorithm, not to a shared set: stream_ac has an actor and a critic optimizer, rtrrl has one. (Decided; not yet implemented.)
- Phase 5 grows: network and normaliser become components, and a component may not name its call sites. FeatureExtractor is deleted outright.

Task 2-1: complete (fd9c649..4136c79, then 91ec5d5 and d19afa8) - contract v6 with ParameterSpec/StructureSpec/EntryDescriptor.parameters, one-sided valid bounds, and training_sdk/parameters.py giving param()/structure()/describe_parameters(). training-sdk 93 passed, ruff clean.
Task 2-2: complete (0baa4b5..65a106e, d5cbfe4) - resolve_parameters/sample_parameters/grid_distributions; study asks empty trials; preflight, launch, loop and every fixture migrated. control-plane 194 passed, ruff clean. Found en route: tests/data/experiment.yaml still named env and backend under space, missed in phase 1. enable_rerun added to LoggingSpec and LoggingConfig.
Task 2-3: complete (c429fb7) - the three catalog builders take PARAMETERS. mock-trainer 100 passed; its catalog regenerated at contract 6 and its declarations stay hand-written, being a harness rather than an algorithm.
Task 2-4: partial (aa17d16) - stream_ac declares PARAMETERS against the final surface; the bound and base components live in memorax/rl/updates.py and the backbones in memorax/networks/torso.py. The normaliser became one component carrying cold_start, variance and reward_trace_reset_on_done, and normalization_upstream.py is deleted. Left open by intent: stream_ac.build still reads the retired names, the other three entries still declare SPACE, memo is red.
Task 2-4 closed at one entry. The plan was trimmed to match (Task 4 is stream_ac only; the old Task 5 removed): this batch delivers the standard template and nothing else. upstream_stream_ac, rtrrl and rtrrl_aaai keep their SPACE, memo's catalog does not build, and the 27 experiment YAMLs plus the two templates still use the retired names. All of that is the next batch, written against stream_ac.

Plans still missing: spec phases 3 and 5. Phase 4 has 2026-08-01-metric-surface.md, which reports that reward and done are only recorded under record_trajectory=True, so the new reward metrics would report nothing without that being fixed first.

Open, to do next round - the kernel should hold component instances rather than flat fields:
- Bound: StreamAC already builds two rules (stream_ac.py:206,213), but StreamACConfig shares bounded_rule/beta2/eps across the roles while kappa and lr are already per-role. That shared triple is the only reason stream_ac's _optimizer raises when the two roles ask for different bounds; it is not a kernel limit. Give StreamACConfig two bound component instances and that raise goes. Does not need the phase 3 decomposition. Touches the golden tests and test_hopper_reproduction, so not an isolated edit.
- Normaliser: one Normalizer handles both streams inside step, so two instances need the observation/reward split first. That is phase 5, and stream_ac's _normalization raise stays until then.
- What genuinely waits for phase 3: optimizer_base=adam and optimizer_bound=none, which this kernel cannot express at all.

--- spec phase 4: training loop and logging ---

Executed 2026-08-02 against 2026-08-01-training-loop-and-logging.md, all four tasks, TDD.
Phase 4 is independent of phase 3 and was taken first because phase 3's last task wants a
real experiment file to run from.

What the surface is now: a completed episode is the only reporting occasion.
`training_sdk.episode` owns both `statistics(episode)` and `metric_names(phase, series)`,
built from one list so an entry's METRICS cannot drift from what it reports. Names are
`<phase>/episode/<quantity>`; mean and variance are `<name>` and `<name>_variance`; a
family inside a quantity uses a dot (`train/episode/actor_grad_norm.torso`) so the three
slashes stay the three axes.

`Reporter.log_episode` is the loop's only call. It reduces to statistics, reports them
through `report`, then hands the episode on for its series. Aim and metrics.jsonl take the
first, rerun the second. `named_scalars`, `report_evaluation`, the per-epoch report and
`logging.every_steps` are gone. `enable_rerun` became the rerun switch; it had been in the
contract with no consumer while a destination turned the sink on by accident.

`record_trajectory` became `record: Iterable[str]`, and each entry passes
`EPISODE_FIELDS | TRAINING_METRICS`, so a declared metric is recorded by construction.
Both kernels now record the reward the environment paid rather than the normalised one,
which retired `drive`'s `eval_reward` callback in favour of a `reward` path both phases
share. `UpstreamStreamACStepMetrics` gained reward and done; upstream's arithmetic is
untouched.

Evaluation episodes are dated at the epoch boundary: `complete_episodes` takes a `stride`
and evaluation's is zero. Spreading them along the axis dated a measurement of the current
policy to training that had not happened.

Migrated: 29 YAMLs off `every_steps` and onto `eval/episode/return`, the manual, both
control-plane specs, the AAAI arm's RENAMED table and rtrrl/catalog.json.

Verified: training-sdk 122 passed, control-plane 194 passed, mock-trainer 100 passed;
memo 219 passed, 43 skipped, 9 xfailed, 9 failed and 3 errored. ruff clean on all four
projects, black and isort clean on memo's CI scope (three files were already unformatted
from phase 2 and are now fixed).

Every memo failure traced and none is phase 4's: five golden StreamAC leaves are the
sanctioned set; five in test_hopper_reproduction and one in test_paper_parity are phase 2
casualties (`stream_ac.SPACE` and `NormalizationConfig.statistics` are both gone);
test_loop's catalog test and the two modules that fail to collect are phase 3's, because
`discover()` refuses rtrrl and upstream_stream_ac for still declaring SPACE.

Not fixed, and pre-existing: `rtrrl/entries/rtrrl_aaai.py` and `rtrrl/tests/test_entry.py`
fail the AAAI image's black and isort gates over a line phase 1 wrapped. Left alone rather
than folded into this commit; phase 3 does not touch them either.

Left open: the AAAI arm's training scalars keep two-part names, being per scanned
iteration rather than reduced over an episode; `evaluation.num_envs` still has no
consumer.
