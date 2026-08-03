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

--- 记录面统一到 SDK ---

问过是否该让"记录什么"从配置文件传入。答案是不:preflight 在花钱之前用 `entry.metrics`
校验 `score.metric`,实验文件若能新增指标名这条校验就空了;而且指标名要绑到内核对象上的
一条路径(`actor_grad_norm.torso`),让实验文件写这条路径就等于让它知道实现的拼写。声明留
在代码里,实验只做窄化 —— 与 `PARAMETERS` 同一个论证。算法内部量本来就由 `TRAINING_METRICS`
承担,加一条只需往元组里写一个名字。

真正搬动的两样:

- `training_sdk/rollout.py` —— `complete_episodes` 与 `read` 从 `memo/runner/episodes.py`
  搬进 SDK。它是纯 numpy over `(T, num_envs)` 加一个点号读取器,没有一处是 memo 专有的;
  搬完之后第二个 trainer 继承的是同一条边界规则,而不是在旁边另发明一条。切分的测试跟着
  搬到 `training-sdk/tests/test_rollout.py`。
- `training_sdk/episode.py` 增 `WINDOWS` 与 `check_names()` —— 三段、中段必须是一个真有东西
  在其上归约的窗口。词汇归 SDK,因为 sink 才是名字最终被读的地方。

两道闸**故意没接**,因为它们会打死这次不重构的部分:

1. `EntryDescriptor` 用 `check_names` 校验 `metrics`。会让 mock-trainer 的 catalog
   (`episode_return`/`episode_length`)构建失败。
2. `RunConfig` 增 `metrics`,`Reporter.report` 拒绝未声明的名字。这是唯一还敞着的贵洞——
   "声明 X 上报 Y",preflight 抓不到,要等打分时才炸。接上会打死 mock-trainer 的 100 个
   用例和 control-plane 里那个内联 fake trainer(约 17 个),而它们验证的正是控制面本身。
   需要 `CONTRACT_VERSION` → 7。

两道闸各自等对应 trainer 迁移时接,那时每个只值一次改动。memo 侧现在由构造保证:`METRICS`
从 `TRAINING_METRICS` 推出,`test_loop.py` 直接对 `stream_ac.METRICS` 跑 `check_names`。

验证:training-sdk 141 passed,control-plane 194 passed,memo test_loop/test_upstream 17
passed(catalog 那条仍是阶段 3 的红);ruff、black、isort 全绿。

--- rerun 改为按环境步采样 ---

原来是 `episode.number % rerun_every_episodes`。序号是先按流、再按时间数的
(`complete_episodes` 外层循环是流),所以取模采到的是"(流, 时间)"这个枚举顺序上的均匀,
不是训练进度上的均匀。

新规则一句话:每 `rerun_every_steps` 个环境步取一个采样点,该步属于哪条流、落在那条流的
哪个回合里,就记那个回合。环境步计数把每条流的每一步都数了进去,所以第 S 步是第
`S // num_envs` 行、第 `S % num_envs` 条流 —— 一个采样点恰好指定一条流和一个回合。

两件事是推论而不是另外的规则:
- **不记评估。** 评估回合的跨度为零(标在它所测量的那个边界上,自己不花训练步),没有采样点
  落在 `[start, end)` 里。
- **流会轮转。** 步长不是 `num_envs` 倍数时采样点在流之间移动;是倍数时固定在一条流上。
  这是取值的后果,不是藏在代码里的政策。

`Episode` 增 `stream`,否则 sink 分不出手上这个回合是谁的。`LoggingConfig` 与 `LoggingSpec`
的 `rerun_every_episodes` 改名为 `rerun_every_steps`,21 个实验文件的取值按 `total_steps/100`
重设(2M 的给 20000,约一百份记录)。

采样点决定流的算术里有一条值得记住:**一个回合跨越 `num_envs × 回合长度` 个环境步**。16 条流、
hopper 约 500 步的回合,就是 8000 环境步。步长要按 `num_envs` 放大才是稀疏采样。

mock-trainer 不受影响:它用自己的 `RecordingRerun` 假 sink,从不走真的 `RerunSink`。

验证:training-sdk 144 passed,control-plane 194 passed,memo test_loop/test_upstream 17
passed(catalog 那条仍是阶段 3 的红);ruff、black、isort 全绿。

--- 阶段 3 task 2:终止与截断分开(以及跟着掉出来的两件) ---

提交 3499943、57629b7、46893be、8a156a1。

**终止与截断。** brax 的 `EpisodeWrapper` 把"跑满 episode_length"也写进 `done`,而旧的
`gamma * (1 - done)` 因此在截断那一步把 bootstrap 抹成 0 —— 等于教 critic"跑满和摔倒一样
糟"。wrapper 现在带出 `truncation`,`td0` 收 `terminal` 与 `gamma` 而不是乘好的 discount。
不区分两种结束的环境,`terminal_of` 读成"每次结束都是失败",所以仓库里其他环境一个都不用改。
`episode_length` 进 `environment` 段:同一个策略在 500 与 1000 下的回报不是同一个数。

**重置挪出环境。** 先是把 brax 的 auto-reset 搬进我们的 wrapper(为了留住回合结束时那个被
覆盖的观测),然后按"rst 是给 act 的、旧 state 是给 learn 的"这条分解,重置移到 `_step` 顶上,
和 carry、迹读同一个 flag —— wrapper 因此比动手之前还小。行为变化:每个回合从新采的初始状态
开始,brax 那套把 hopper 的初始分布塌成每条流一个固定点。

**指标容器按出处重切。** `terminal` 是训练容器和评估容器不得不各存一份的第二个字段,而旧形状里
"读不到"分不出"没有更新发生"和"这个容器没这个字段"。现在是 `InteractionMetrics`(在契约里,
每个算法一样,切分器读它)加各算法自己的 `ForwardMetrics` / `UpdateMetrics`,由 `StepMetrics`
装起来。指标名带上出处:`update.td_error`、`forward.value`。`EvalSummary` 与三个
`*StepMetrics` 删除。

验证:training-sdk 146、control-plane 171 passed;memo 七条红,与开工前同一批 —— 5 条
golden(现在是数值上的,因为重置与 terminal 都改了行为)、1 条 paper_parity 归一化(阶段 2 的)、
1 条 catalog(阶段 3 的)。ruff / black / isort 全绿。

`test_hopper_reproduction.py` 读的实验文件已被删除,按指示不管。
