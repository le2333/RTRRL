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

--- 阶段 3 task 1 与 task 3 ---

提交 761d88f(优化器两轴)、a7c185d(归一化器一个估计器两个实例),中间 8a156a1 是指标容器改名。

**Task 1.** `make_bounded_rule(bound=, base=)` 收组件而不是一个把两件事捆在一起的字符串。
`StreamACConfig` 每个角色一个界一个基;原先共享的 `bounded_rule`/`beta2`/`eps` 三元组是入口
必须拒绝"两个角色要不同界"的唯一原因。`adaptive_ob` 与 `adaptive_ob_fixed` 原本共用一个组件
类,分支交回来的东西说不出自己是哪个 —— 拆出 `AdaptiveObBoundFixed`。朴素界不再累它从来没读过
的第二矩(那让规则需要一个什么都不改变的衰减率),`test_blocks` 的矩比对因此只对两个自适应界成立。
`optimizer_bound=none` 走通,界套在 adam 上被拒绝而不是猜。

**Task 3.** 一个估计器两个实例:`center` 决定减不减均值,`discount` 决定它看到的是单步值还是
折扣累加。`reset`/`step` 变 `begin`/`observe`。奖励估计器从不 begin —— 它看的是累加,回合结束
时丢掉累加是 `reset_on_done` 的事,而旧的 `reset` 只更新观测那一路,悄悄跳过了奖励那路。内核各
持两个实例两份统计量。入口的调和整个消失,那条"这个内核只带一套统计量"的 raise 是这个面上最后
一条。`raw_episode_return` 与每步归一化指标删除:没有入口声明,却每步都算。

契约测试 `test_normalization.py` 直接断言组件不认识自己的调用场景:`Normalizer` 的方法名和三个
配置类的字段名里不许出现 observation 或 reward(改之前命中七个)。

验证:每一步之后 memo 都是同一批七条红 —— 5 golden、1 paper_parity 归一化(阶段 2 遗留,它的
helper 是已删除的 upstream statistics 臂的死代码)、1 catalog(阶段 3 的)。ruff/black/isort 全绿。

**一次自造的麻烦:** 跑 black 时给了整个 `memorax` 而 memo-ci 只检查其中一部分,42 个无关文件被
重排;按 CHECKED 逐一还原了。格式化只按 CI 的范围跑。

Task 4 的设计已写进计划文件(序列的 carry 是逐组件的列表、信用包住那一个循环组件、meta_rl 的
拼接移出网络、以及 blast radius),留待下一段执行。

--- 阶段 3 task 4:网络是一个序列 ---

`Sequence(components=(...))`,一步是 `(carries, x) -> (carries, y)`,别的什么都不过这条线。
组件超出 `x` 需要什么由自己用 `reads` 声明,序列只递声明过的东西;被替掉的三个固定槽是把观测、
结束、上一步动作、上一步奖励一起推给每一级,所以每个槽为了用一个而得接四个。

carry 是逐组件一条。无状态组件把拿到的那条原样交回,循环组件交回新的 —— "无状态组件不理会
carry"因此是可检查的而不是约定。序列里只允许一个循环组件:跨两个的精确敏感度要一个稠密的层间
雅可比,规格把它推迟了,所以在被要求的地方拒绝,而不是把第二个做的事记到第一个头上。

新增 `memorax/networks/{sequence,components,backbones}.py`;`components.py` 里一层只做一件事
(`FFN` 只是一个仿射、`LayerNorm`、四个激活各自一个、`Readout` 把先于这套协议存在的头包一下)。
`torso.py` 与 `blocks/stack.py` 删除 —— 后者是同一个想法但没有 carry 契约、没有命名、也不拒绝。
`blocks/ffn.py` 里那个 Dense→激活→Dense 改名 `TransformerFFN`:它是另一个东西,不该跟序列的
前馈层同名。

`meta_rl` 的拼接移进内核:上一步动作与奖励并到观测旁边是输入组合,发生在这些值已经在的地方,
序列看到一个向量。

**三个槽全删,只有它们能驱动的东西一起没了。** 我先把 `Network`/`FeatureExtractor` 搬到
`memorax/algorithms/slots.py` 想保住 `test_blocks` 的对照,被否了 —— 那些内核反正要重构。
现在 `Network`、`FeatureExtractor`、`torso.py`、`blocks/stack.py` 全删。代价写在这里:

- `entries/upstream_stream_ac.py` 与 `entries/rtrrl.py` 不再能 import。它们本来就因为还声明
  `SPACE` 被 `discover()` 拒,catalog 那条红不变,只是失败点从"拒绝"变成"导入"。
- `test_upstream_stream_ac.py` 进排除集:它端到端驱动 upstream 的内核,没有网络可驱动了。
- `test_blocks.py` 保住了每一块算术的比对 —— upstream 用 `None` 网络构造,因为那些块比的是
  已经算出来的量,没有一块碰网络。丢掉两条:**截断梯度对 upstream 的比对**(这是其余所有块
  留下的那道缝 —— 别的比算术,只有它比"参数的信用是怎么拿到的"),和"一个种子给两个内核同一个
  起点"。两条都要 upstream 的前向。`test_exact_credit_is_not_the_truncated_one` 改用我们
  自己的 `init` 造状态,活下来了。
- `test_algorithms.py` 丢掉 RTRRL 的两个 program 和两条门控消融。RTRRL 按槽名路由它的三域
  梯度(`RECURRENT_DOMAINS` 就是 `("feature_extractor", "torso")`),不重写就收不了序列。
  重写时把它们加回来。

**一个种子不再给两个内核同一个起点。** flax 按持有参数的模块路径抽参数,序列里的位置和具名
的槽拼法不同。组合是同一个 —— 截断梯度经测试里的一次改名之后仍然逐叶等于 upstream —— 但抽出来
的值不同,所以 `stream_ac` 与 `upstream_stream_ac` 不能再在单个种子上比。Task 5 把每条 backbone
按各自来源摆回去,本来也会终结这个比较。那条测试改成断言还成立的部分,并在名字里说清剩下的不成立了。

**逐部件的梯度范数改成逐位置。** `PARTS` 原来是三个槽名;序列的部件数随 backbone 变,而
`METRICS` 是 catalog 读的模块常量。`Sequence.split` 把树分成 `before` / `recurrence` / `after`
—— 这本来就是 `subtree_norms` 当初要拆的那个区别 —— 声明出去的指标名因此不随 backbone 变。

验证:`test_sequence.py` 十四条,建立前在收集期就是红的。之后 memo 仍是同一批七条红。
其中 5 条 golden 现在是**名字**上的红而不只是数值上的:快照记的是
`actor_params/params/feature_extractor/...`,这些叶子路径已经不存在了。重录金快照时要按序列的
拼法录。`test_blocks` 里的翻译函数 `as_sequence` 就是这次改名的对照表。

memo-ci 的 CHECKED 里 `memorax/networks/torso.py` 换成三个新文件加 `slots.py`。

**一处归属写错了,顺手改掉。** `entries/stream_ac.py` 和 `memorax/rl/credit.py` 都写着 `tbptt`
"就是已发表的 StreamAC"。不是。已发表的 StreamAC 是前馈的 —— `test_paper_parity` 驱动的是论文
自己的 `stream_ac_continuous.py`,构造参数只有 `n_obs / n_actions / hidden_size`,没有 carry
—— 所以里面没有东西可截断,`mlp` backbone 下 `rtrl` 与 `tbptt` 是同一个计算。`tbptt` 复现的是
"StreamAC 的更新 + 循环 backbone + 不带敏感度",也就是这个仓库继承来的那条循环基线
(`entries/upstream_stream_ac.py` 的注释本来就写对了:upstream 是 by construction 截断的)。

**这影响消融怎么读**:照原来那句话,`rtrl` vs `tbptt` 像是"我们的方法 vs 已发表基线";实际上
`tbptt` 那一臂已经是循环扩展了。对已发表版本的基线是 `mlp` backbone,是另一个比较。

--- credit 从参数改成结构 ---

`credit` 原本是 `param(search=list(CREDITS))`,两个值都在搜索域里,而同文件其它计算图选择
(backbone、两个归一化、四条优化器轴)都是 `structure`。规格 §4 把梯度门控列在结构里,并且
结构不参与搜索、一个 study 内固定。

两条代码上的证据说它是结构:`rtrl` 与 `tbptt` 的 `actor_sensitivity` pytree 结构不同
(`TruncatedBPTT.initialize` 返回 None,`ExactRTRL.initialize` 返回敏感度树);两者初始化
递推单元走的方法也不同(`ExactRTRL.initialization()` 装 `_delegate_rtu_init_forward`,按
`local_jacobian` 建参数)。

改成 `structure(placeholder="tbptt", branches={"rtrl": (), "tbptt": ()})`,读取用
`read_branch`。效果:`space` 里给两个分支现在会被 preflight 拒
(`control-plane/space.py:133`),给一个分支照旧。`experiments/streamac template.yaml:47`
本来就是 `credit: [tbptt]`,不受影响。manifest 仍带裸键 `credit`。

契约测试 `memo/tests/test_credit.py` 三条,声明那条改之前是红的。

**顺带纠正一处我读错的**:那些 `stop_gradient(carry)`(stream_ac 4 处、rtrrl 5 处、
independent_rtrrl 7 处)不起作用 —— 三个内核都是 `jax.grad`/`jax.jacobian` 只对参数树求导,
carry 是另一个位置参数、在那次求导里是常数。实测把它们摘掉梯度不变。真正的"一步 BPTT"来自
逐步更新本身,不来自这些调用。

--- 阶段 3 task 5:backbone 对齐来源 ---

`backbone()` 现在交出每条分支自己的完整前段,入口不再自己加编码器:

- `rtu`:`FFN(feature_dim) → LayerNorm → Tanh → RNN(RTUCell)`
- `mlp`:`FFN(hidden) → LayerNorm → LeakyReLU` 两遍

`RTUCell` 去掉 `activation_fn` 字段,直接用 `tanh`。`LRUCell` 本来就没有这个字段,计划里说
"两个都有"只对了一半。`Memoryless`(六件事焊在一起、层数写死为二)没有使用者了,删除。

**没能按计划"驱动着对照来源"**,两条都不行:RTU 那篇论文的代码不在仓库里;streaming-drl 的文件
要 `STREAMING_DRL` 指向一个 checkout,这里没有,`test_paper_parity` 里需要它的每一条都 skip。
所以 `test_backbones.py` 比的是组合对不对得上规格 §4 写下的配方,文件开头写明了这一点。

**去 GitHub 读了原版,结论跟 spec §4 那张表对不上,按原版改回来。**

`mlp` —— `mohmdelsayed/streaming-drl` 的 `stream_ac_continuous.py`。actor 和 critic 都是
`Linear(obs,128) → layer_norm → leaky_relu → Linear(128,128) → layer_norm → leaky_relu`
再接头。落地的就是这个。改之前我们在前面还多一层 `Dense → relu`,他们没有。

另外两处不一致,不在 backbone 里所以不属于本 task:他们的 `initialize_weights` 对每个
`nn.Linear`(含头)做 `sparse_init(sparsity=0.9)`、bias 全零,我们是 `lecun_normal`;他们的
actor 头是两个独立 `Linear` 加 `softplus`,我们的 `heads.Gaussian` 是一个 Dense 加一个全局可学
的 `log_std`。

`rtu` —— 没有东西可核对。它的设置出自一篇不公开代码的论文;而 memorax(这个仓库跟踪的上游)
只定义了 `RTUCell`,没有任何围绕它的组合 —— 它自己的 StreamAC 例子是
`FeatureExtractor(Dense(120)→relu→Dense(84)→relu)` 加 `Stack(Residual(RNN(GRUCell)))`。
所以 `rtu` 保持仓库原有的 `FFN → ReLU → RNN`,`RTUConfig` 也把 `activation_fn` 放回去 ——
memorax 有这个字段、默认 `jnp.tanh`,删掉它是计划的主张而不是任何来源的。

**教训**:spec §4 那张表不是对来源的转写,不能当转写读。我按它改了 `rtu` 的非线性和归一化位置,
是照转述改数值。凡是"对齐来源",先去读来源。


--- 初始化改成结构 ---

`memorax/networks/initialization.py`:`INITIALIZATION_BRANCHES = {"lecun": (), "sparse": Sparse}`,
`Sparse` 带一个 `sparsity` 参数(`search=[0.9]`,单点)。入口声明
`initialization: str = structure(placeholder="lecun", branches=...)`,读出来的初始化器传给
backbone 里每个 `FFN` 和两个头。

placeholder 取 `lecun` 而不是 `sparse`:不让默认行为悄悄变。实验要复现 streaming-drl 就在
`space` 里钉 `initialization: [sparse]`。

偏置两边都是零,所以只有 kernel 有分支。

`Sparse` 加进 `test_component_contract.py` 的 COMPONENTS。契约测试 `test_initialization.py` 六条,
其中"是不是 StructureSpec"和"稀疏的份额对不对"在实现前是红的;稀疏那条按 streaming-drl 的
`sparse_init` 语义断言:每个输出单元的 fan_in 里恰好 `ceil(sparsity * fan_in)` 个零。

**另一处顺带查清的**:memorax 的 `Gaussian` 头是一个 Dense 加全局可学 `log_std` 再 `exp`,跟我们
的 `bound=False` 一致 —— 之前说"他们的 actor 头是两个独立 Linear"只指 streaming-drl,不包括
memorax。这条差异是我们对 streaming-drl 的,不是对 memorax 的。

--- actor 头改成结构 ---

`heads.Gaussian` 原来用一个 `bound: bool` 装两种参数化,按"组件内不混合"拆成三个类:

- `Gaussian` —— 一个 Dense 给均值 + 全局可学 `log_std`,`std = exp(log_std)`。**恢复成 memorax 的原样**
  (`bound`/`loc_bounds`/`log_std_bounds` 三个字段是我们加的,现在挪走了)。
- `StateStdGaussian` —— 两个 Dense,第二个过 `softplus`。**这是 streaming-drl 的**,之前仓库里没有。
- `BoundedGaussian` —— 原来的 `bound=True` 那条路,loc 与 log_scale 各自 sigmoid 到区间再 softplus。

`memorax/networks/policy.py` 里 `ACTOR_HEAD_BRANCHES = {"global_std": (), "state_std": (), "bounded": ()}`,
入口声明 `actor_head: structure(placeholder="global_std", ...)`。placeholder 取 `global_std`
(= memorax 的 = 现状),默认行为不变。

`entries/rtrrl.py` 的 `bound_actor` 布尔改成在两个类之间选(那个入口本来就是红的)。

测试 `test_policy_head.py` 九条,按"尺度从哪来"区分三条分支:`global_std` 换一个观测尺度不变,
另外两条会变;`bounded` 的 loc 落在区间内;还有一条断言 `Gaussian` 上没有 `bound` 字段。
