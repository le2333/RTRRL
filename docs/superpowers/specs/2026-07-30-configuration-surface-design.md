# 配置面设计

范围:实验配置文件的分段、观测的指定方式、算法侧的参数声明、结构的表达与采样、OBGD 的分解。

不在本范围:数值偏差判据(见 `2026-07-29-numerical-testbench-design.md`)、金快照重录、具体实验的取值选择、多 seed 聚合语义、按环境拆解奖励分量(见 §6 末)。

取代 `2026-07-26-algorithm-config-contract-design.md`。

## 1. 配置文件的分段

实验 YAML 的配置面按四段组织:

```yaml
environment:
  id: brax::hopper
  backend: spring
  observed: [0, 1, 2, 3, 4]
  seed: 0

training:
  num_envs: 1
  total_steps: 2000000
  epoch_steps: 100000
  chunk_steps: 1000
  early_stop_patience: null

evaluation:
  steps: 1000
  num_envs: 1

space: {}
```

`environment` 只描述任务本身:环境 id、后端、观测列与随机种子。`seed` 定义这一条随机环境流与初始化流,由控制面写进 manifest 后注入给 entry,不作为算法参数声明,也不进入 HPO。

`training` 描述训练流与训练预算:`num_envs` 是并行训练流数,`total_steps` 是总环境步预算,`epoch_steps` 是报告和评估间隔。`evaluation` 描述测试流:`steps` 是每次评估 rollout 长度,`num_envs` 是评估并行环境数。

`chunk_steps` 是训练循环的内部切块长度,对应 AAAI entry 旧 `scan_steps`:作者代码外层循环每次进入一个编译后的 scan,scan 内跑 `chunk_steps` 个 transition。它影响运行组织和编译形状,不改变环境、预算或算法定义。memo 系 entry 的循环已由 `memo.runner.loop.drive()` 管理,不需要声明该字段;AAAI adapter 从 `training.chunk_steps` 把 `training.total_steps` 与 `training.epoch_steps` 翻译成作者代码的 `episodes` 与 `eval_every`。

`early_stop_patience` 是训练停止策略,对应 AAAI entry 旧 `patience`。公平比较的实验应设为 `null` 或足够大的值使其不触发;它仍属于训练执行策略,不属于算法搜索空间。

`environment`、`training` 与 `evaluation` 的取值定义任务、开销与评估方式,不同取值之间的 trial 不可比,因此不进入搜索。控制面直接读取它们,不经过采样器。`score.window_steps` 与 `training.total_steps` 直接比较,`minimum_total_steps`(`space.py:50`)不再需要。

`space` 只放两类东西:要固定的结构选择,以及要覆盖默认搜索范围的参数。常态是空的。

`training.total_steps % training.epoch_steps == 0`、`training.epoch_steps % training.num_envs == 0` 是 `TrainingConfig` 自身的合法性。若声明了 `chunk_steps`,则 `training.total_steps` 与 `training.epoch_steps` 也必须整除 `chunk_steps * training.num_envs`。这不是 HPO 的跨参数约束机制,而是同一训练配置对象内的运行形状校验;失败时 preflight 拒绝,不提交作业。

## 2. 观测的指定与实现

`observed` 是**保留**的观测维度下标。命名为 `observed` 而非 `mask`,因为"掩码数组"未指明列出的是留下的还是去掉的。省略该键表示全观测。与 AAAI 实现的 `obs_mask` 同义(`RTRRL-AAAI25/envs/environments.py:34,54-67`)。

实现为删维度而非置零:包装器在 reset 与 step 中做 `o[..., observed]`,并覆写 `observation_space` 报告缩减后的形状。首层 Dense 的扇入随之从完整观测维数降到保留维数。

改动面三处:`memo/memorax/environments/wrappers/mask_observation.py`(现状在 `:24,37` 做 `o * m`)、`memo/memorax/environments/brax.py:82` 的唯一调用点、`memo/tests/test_masking.py`。`brax.py:9-30` 的 F/P/V 表删除。`mask_rate` 属性在仓库内无消费者,删除。`test_masking.py` 的三个测试都是在论证置零与删维度等价,失去对象,删除。

## 3. 参数声明

算法侧用 dataclass 声明,每个参数三个字段:

```python
lambda_v: float = param(valid=(0.0, 1.0),    search=(0.5, 0.99),  placeholder=0.9)
meta_rl:  bool  = param(valid=[False, True], search=[False, True], placeholder=True)
eta_pi:   float = param(valid=(0.0, None), search=[0.0],           placeholder=0.0)
```

- **`valid`** 是硬边界,只用于校验。二元组为数值边界,`None` 表示该侧无界;列表为允许取值的集合。实验请求的单值或范围越界时 preflight 拒绝,并指出参数名与越界的一侧。
- **`search`** 是默认搜索空间,**每个参数都必须声明,没有例外**。二元组为连续区间,列表为离散候选集;不想让它变动就给单点列表 `[值]`。
- **`placeholder`** 是单值,未激活分支下该参数塌缩到的取值。

这里有两个都可以叫"默认"的东西,必须分开:`search` 是**搜索范围的默认**,即今天各入口 `SPACE` 里那些区间与候选列表,实验不覆盖时 HPO 就按它搜;`placeholder` 是**未激活分支下的取值**。命名为 `placeholder` 而非 `default`,因为后者读起来像"推荐值",会诱使实现者去考据它该是多少。

不允许"省略 `search` 表示不搜索"这条例外。一个参数不该变动时写单点 `search=[值]`,采样器照样走它,只是只有一个候选。这样每个参数都有一个采样器走的域,manifest 里的每个值都来自一次抽样而不是某条兜底规则,读 catalog 也不必分两种情况。

`placeholder` 的取值不重要:任何实验真正在意的参数都会显式给出范围或单值,不会落到 `placeholder`。有现成出处的照抄(rtrrl 系取 `RTRRL-AAAI25/config/brax.yml`,streamac 系取 reproduce 实验里已钉死的值),没有的取任意安全值即可。唯一的硬要求是它落在 `valid` 内,且不是会引爆下游的退化值——例如作除数的参数不取 0,因为未激活分支的参数照样会写进 manifest 并被算法读到。

`search` 与 `placeholder` 都必须落在 `valid` 内,导出 catalog 时检查。`log=True` 时 `search` 的下界必须严格大于零。

catalog 从这些声明导出,不手写字面量,覆盖算法的全部配置面。`EntryDescriptor` 中该字段命名为 `parameters`,因为它是入口声明的完整参数面,不等同于一次实验要搜索的空间。实验 YAML 顶层仍叫 `space`,表示本次实验固定结构选择或覆盖默认搜索范围。实验 YAML 未提及的参数按其 `search` 搜索。

manifest 携带全部参数,算法一律 `params["x"]` 取值。禁止 `params.get("x", v)` 这类 Python 端兜底,缺键即报错。

`seed` 不在这里声明。entry 需要随机种子时从 manifest 的 `environment.seed` 读取并传给算法构造或训练循环;算法内部仍可把它当作输入参数使用,但 catalog 与 HPO 不把它视作算法搜索维度。

运行、预算、观测与评估字段同样不在这里声明。`training.num_envs`、`training.total_steps`、`training.epoch_steps`、`training.chunk_steps`、`training.early_stop_patience`、`evaluation.steps`、`evaluation.num_envs` 都由对应配置段持有,不在 catalog 的参数树中出现。

`eps` 不作为跨算法统一参数名。AAAI 版 RTRRL 不暴露 optimizer eps,entry 不声明它,继续使用作者代码/Optax 的默认值。StreamAC 与 upstream StreamAC 统一拆成两个名字:

- `optimizer_eps`:优化器/Adaptive OBGD 的 eps,只在自适应有界规则下有意义,保留默认搜索范围。
- `normalization_eps`:观测或奖励归一化的 eps,只影响 normalization wrapper/config,`search` 取单点,实验需要改时覆盖。

两个值可以有相同 placeholder,但不共享一个参数名,也不由一个采样维度同时驱动。

### 归一化:一个开关拆成三个,其中一个只有一个候选

`normalization_statistics: ours | upstream`(`memo/memorax/rl/normalization.py:20`)按**代码出自谁**命名,而不是按它是什么。配置面上的取值不该指向仓库归属。它同时也太粗:`UpstreamNormalizer` 覆写三个方法,合并成一个开关之后,跟 streaming-drl 的曲线对不上时说不出是哪一件造成的,而这条臂存在的全部目的就是把这三件事从"我们的框架"里分离出来。拆成三个:

- `normalization_cold_start`:`seeded`(播一个均值 0、M2 1、count 1 的伪观测)| `first_sample`(第一个样本直接成为均值、M2 置零,第一个方差恰好 1)。
- `normalization_variance`:`population`(`M2/count`)| `sample`(`M2/(count-1)`,count<2 时钉在 1)。
- `reward_trace_reset_on_done`:折扣回报迹在终止步存回时是否再掩一次码。

前两个是两种都站得住的估计量约定,几千个样本之后差别消失,是真正的选项,带 `search`。

第三个不是选项。`reset_on_done=false` 时终止步之后 `G` 等于终止奖励,下一个回合带着它(衰减一次)开始,整段跑里每个回合边界都漏一次;这是缺陷,gymnasium 的 `NormalizeReward` 也带了多年。它的 `search` 是单点 `[true]`,因此采样器只有一个候选可选,只能被实验显式覆盖成 `false` 才会打开——否则 TPE 会把"跨回合漏奖励"当超参数去调,而在 Hopper 上漏进去的存活奖励很可能让它分数更高,于是被选中并当作结论报出来。

`STATISTICS` 表与 `normalization_statistics` 一并删除。

### 评估期的归一化统计量

`reset_on_start` 与 `update_during_eval`(`normalization.py:29-30`)今天是 `NormalizationConfig` 的字段,不在任何 `SPACE` 里,配置文件够不着,于是每次运行都是 `True`/`True`。两者都声明为参数,placeholder 取 `false`/`true`。现有约束"`reset_on_start=True` 要求 `update_during_eval=True`"(`normalization.py:252`)保留,在该 placeholder 组合下自然满足。

## 4. 结构与条件空间

结构是计算图的可替换部分:backbone、归一化的开关、梯度门控、优化器的界与底。声明为一个选择字段加若干分支,每个分支可携带自己的子参数:

```python
optimizer_bound: Structure = structure(
    placeholder="ob",
    branches={
        "none": (),
        "ob": ObBound,
        "adaptive_ob": AdaptiveObBound,
        "adaptive_ob_fixed": AdaptiveObBound,
    },
)
```

catalog 导出这棵树。

**结构不参与搜索,只有参数参与。** 一个 study 里结构是固定的:实验在 `space` 里把它钉成某个分支,或者不提、取 `placeholder`。所以结构声明没有 `search` 字段,实验给它两个以上分支时 preflight 拒绝。HPO 每一轮抽的只是被选中分支下的那些参数。

这样一个 study 内所有 trial 的搜索维度是同一组,trial 之间可比;结构本身要比较时,是两个 study 而不是一个 study 内的两个 trial。GridSampler 也因此不需要特判——网格里从来没有结构这一维。

未激活分支下的参数**取 `placeholder`**,不是从参数面上消失。它仍然存在,仍然带着该值写进 manifest,只是不构成搜索维度。

### 参数键是它在树里的位置

分支下的参数拍平成 `<结构字段>.<分支>.<参数>`,顶层参数保持裸名:

```
gamma                            顶层
backbone.rtu.hidden_dim          backbone 结构的 rtu 分支
actor_optimizer.sgd.lr           actor 那一路的优化器
critic_optimizer.sgd.lr          critic 那一路的
```

三段都不能省。**结构字段名**是必需的,因为同一个组件会被实例化多次——StreamAC 的 actor 与 critic 各有一个优化器,两边都选 `sgd` 时只有它能把 `sgd.lr` 分开。**分支名**也是必需的,因为未激活分支的参数照样进 manifest,而 `adam` 的 `b1` 在 `sgd` 分支根本不存在,省掉分支名之后同一个键在不同分支下含义不同。

不为"只有一个实例"的结构开省略前缀的特例。省了之后,加第二个实例会把已有参数改名,写着旧名字的实验文件全部失效。

结构字段属于算法,不是一套所有算法共用的字段:StreamAC 声明 `actor_optimizer` 与 `critic_optimizer` 两条,RTRRL 只声明一条。

没有分支特有参数的结构(两个归一化开关、两个梯度门控)在机制上与普通离散参数一致,不特判。

### 网络由组件组合,组件内不混合

前馈层、层归一化、各个激活函数、LRU、RTU 都是**独立组件**。算法把它们组合成一个 sequence。单个组件内部不得混合两件事。

这条约束不是风格偏好。三条通路在编码、非线性位置、归一化上两两都不同,任何一段公共接线都会同时破坏其中两条;而只有当网络是一串可排列的组件时,"后续用配置文件描述网络结构"才可能——固定槽位的接线没法被数据描述。

现状有四处违反:

- `Memoryless`(`memo/memorax/networks/sequence_models/memoryless.py:48-53`)把 `Dense → LayerNorm → 激活 → Dense → LayerNorm → 激活` 六件事焊成一个 torso,层数还写死为二。
- 三个 entry 的 `encoder()`(`memo/entries/stream_ac.py:120`)是 `nn.Sequential((Dense, relu))`,两件事。
- `Network`(`memo/memorax/networks/network.py:8-11`)是固定三槽 `feature_extractor / torso / head`,不是序列。
- `RTUCell` 与 `LRUCell` 各自带 `activation_fn` 字段,让"哪个非线性"看起来像递推组件的一个参数。

### 组件不得认识自己的调用场景

上一节说的是一个组件内部不混两件事。还有一种混法:组件把**它被用在哪**写进了自己的字段名和方法名。识别方法是看命名——组件里出现了调用方才该知道的词,它就已经把两个场景焊进来了。

四处:

**归一化器**同时认识"观测"和"奖励":`_update_observation`/`_normalize_observation` 与 `_update_reward`/`_scale_reward` 四个方法,Welford 那段算术两边一字不差。真差异只有两条:统计量喂什么(观测本身,还是折扣回报迹 `G = r + γ·G_prev·(1-done)`),以及变换要不要减均值(观测减,奖励不减)。这两条都是**配置**,不是代码路径。拆成 `RunningNormalizer(cold_start, variance, center, eps)` 与 `DiscountedTrace(gamma, reset_on_done)` 两个组件,观测是 `center=True` 直接喂,奖励是 `center=False` 喂迹的输出,四个方法塌回一对,`step` 里那两段只有变量名不同的 `if ... is not None: if update:` 整块消失。`reward_trace_reset_on_done` 随之归到迹上,名字也就自解释了。

**`FeatureExtractor`** 有四个具名槽 `observation_extractor` / `action_extractor` / `reward_extractor` / `done_extractor`,再拼出 `*_embedding` 三个键——而全仓没有任何下游读那三个键。它认识的正是四种输入,而它做的事只是"对若干输入各过一个编码器再拼接"。

它**整个删除**,不做拆分。需要编码的分支(`rtu` 的来源规定了 encoder)就在 sequence 里放一个 FFN 组件,和其它层没有区别;不需要的分支(`lru` 按 AAAI 版是观测直连)就没有这一层。`meta_rl` 把上一步动作与奖励并到观测旁边,是**输入组合**,按前一节归到网络之外,不需要网络里有一个知道"动作"和"奖励"的槽位来做它。

**`Network`** 的三个固定槽与推给每一级的 `action=`/`reward=`/`done=`,同上一节。

**`StreamACConfig`** 把 `actor_lr`/`critic_lr`、`actor_kappa`/`critic_kappa` 并列成四个字段。优化器被实例化了两次,参数各一份;组件化之后那是同一个优化器组件的两个实例,前缀由图给出,配置里不该有 `actor_`/`critic_` 这种前缀成对出现。

判据统一为:**组件的字段名里不得出现只有调用方知道的名词。** 出现了就说明那是两个实例或两个配置,不是一个组件的两条分支。

**递推组件的非线性是例外,不算混合。** RTU 的定义就是把非线性引入递推(`arXiv 2409.01449`,Elelimy 等,NeurIPS 2024):它作用在 carry 上,外提就改变了下一步的递推,那是另一个算法而不是同一组件后面加一层。LRU 是线性的,名字里的 Linear 就是这个意思。两者是并列的两个原子组件,不是"LRU 加激活等于 RTU";去掉 RTU 的非线性它就退回成 LRU,而 LRU 正是 RTU 论文用来对照的基线。因此 `RTU` 与 `LRU` 各自完整,但都不再暴露"换一个激活函数"的字段。

### 三条 backbone 分支各自对齐自己的原版

每条分支有各自的已发表来源,通路互不相同:

| 分支 | 来源 | sequence |
|------|------|----------|
| `rtu` | `arXiv 2605.24709`,Farr 等,Masked MuJoCo 正是本项目的任务 | `FFN(64) → LayerNorm → tanh → RTU(hidden 192) → FFN(head)` |
| `lru` | RTRRL AAAI | `LRU → SiLU → FFN(head)` |
| `mlp` | streaming-drl | `FFN(128) → LayerNorm → LeakyReLU → FFN(128) → LayerNorm → LeakyReLU → FFN(head)` |

`rtu` 的来源明写了三个参数组:"an encoder φ producing features, a recurrent RTU layer producing the state, and a feedforward head",并给出 Masked MuJoCo 的取值:全连接宽度 64、该基准上用 tanh(离散动作基准才用 LeakyReLU)、每个激活之前 LayerNorm、单层 RTU 隐藏维 192、policy 与 value 各自独立的网络、初始化 90% 稀疏。所以编码器是这条分支的原版规定的,不是多出来的一层;`feature_dim` 保留,它就是编码器宽度,placeholder 取 64。memo 现有的 `initializers/sparse.py` 已实现稀疏初始化,只是 entry 没有用它。

`lru` 按 AAAI 版的通路写。他们的 `OnlineLRULayer` 在 `C` 读出与 `D` 直连之后作用一个 SiLU,memo 的 `LRUCell` 没有这一步——要补上,但作为 LRU 之后的独立组件,不是塞进 `LRUCell`。抄的是通路,不是他们把激活焊进层里的封装方式。

`mlp` 是 streaming-drl 的两个前馈块,宽度 128,不是一层裸 `Dense+relu`。

`stream_ac` 与 `upstream_stream_ac` 的 backbone 值域去掉 `lru`,留 `rtu` 与 `mlp`;LRU 归 rtrrl 那条线。

`test_paper_parity.py` 今天覆盖 OBGD、归一化、TD 误差与 actor/critic 方向,没有一条断言网络结构,所以这些通路从未对照任何参考被检查过。拆成组件之后,每条分支都应当能被驱动着与它的来源逐层比对。

### 采样的改动

现状 `ask_round` 调用 `study.ask(dict(distributions))`,一次给出固定分布集(`study.py:79`),`distributions` 由整个 space 一次性构造(`space.py:34`)。改为 `study.ask()` 取得 trial 后按结构树逐层调用 `trial.suggest_*`。条件性在控制面解析完毕,作业提交前参数已全部确定,worker 不受影响。

结构既然不搜,网格就总是给得出:`grid_distributions` 只枚举被选中分支下的参数。`check_sampler`(`study.py:14`)保持原样,仍然只拒绝非法采样器名与 grid 下的连续区间。

本轮不引入通用跨参数约束语言。算法参数之间若存在"某选择下该参数才有意义",用结构树表达;训练循环的整除性归 `TrainingConfig` 校验;除此之外,参数声明保持独立。preflight 可以拒绝显然不成立的固定结构组合,但不把这类检查扩展成一套任意布尔/算术约束系统。

## 5. OBGD 的分解

现状 `make_obgd_rule` 一个函数同时做四件事:算步长界、按界与 TD 误差加权迹、按二阶矩归一化、对环境轴取均值(`memo/memorax/rl/updates.py:109-222`)。Adam 路径则是 `make_optax_rule` 包一个 optax 变换(`updates.py:82-103`)。两条路径不对称,且"OB 界 + Adam 底"无法表达。

拆成两条独立的结构轴:

```
optimizer_bound: none | ob | adaptive_ob | adaptive_ob_fixed
optimizer_base:  sgd  | adam
```

两个名字共享 `optimizer_` 前缀。manifest 把参数拍平成 `params["x"]`,前缀是拍平之后唯一还能说明这两个字段是同一次分解的东西。

值里去掉 `gd`:`obgd` 指的是 Overshoot-Bounded Gradient Descent,名字里已经含了底,拿它当界这条轴的取值,`optimizer_bound=obgd, optimizer_base=adam` 就成了"用 Adam 的梯度下降界的梯度下降"。`ob` 只说界,正是这条轴表达的东西。`none` 也因此能进同一条轴——今天想说"不加界"要去改另一个字段(`update_rule=adam`),一个概念散在两处。

更新链固定为三段:**bound → 环境轴均值 → base**。

- `bound=none` 时第一段是 `delta * trace`。
- `bound=ob` 时第一段在此基础上乘以步长界 `lr / max(1, |δ̄| · Σ|z̃| · lr · κ)`。
- 自适应变体在第一段内部除以二阶矩分母。该分母同时进入界的计算与最终上升方向,拆成两个变换会迫使二者共享状态,因此留在同一段内。两个自适应分支的区别只在 eps 相对平方根的位置。

接口用 `optax.GradientTransformationExtraArgs`(optax 0.2.8 已提供,`memo/uv.lock`):迹作为 `updates`,TD 误差作为额外参数传入。界的公式需要 δ 与迹分别可见,还需要一次跨整棵树的 L1 求和,把 δ 预先乘进迹之后界无法还原,所以单树接口不够用。

学习率由链中唯一消费它的那一段应用,不重复应用:`bound` 不为 `none` 时它已在界的公式内部,base 段不再乘;`bound=none` 时由 base 段应用,即现状 `optax.scale(config.td_lr)` 所处的位置(`memorax/algorithms/rtrrl.py:244-249`)。

分解后的数值结果不要求与现状逐位一致。`bound=ob, base=sgd` 是已发表的 OBGD,`bound=none, base=adam` 是现状的 Adam 路径,`bound=ob, base=adam` 是新增组合,没有对照实现。

`freeze_gamma` 在有界时报错的现状(`memorax/algorithms/rtrrl.py:200-205`)保留:界按组整体缩放,无法单独按住一个叶子。

## 6. 指标面

### 上报由回合结束驱动

一个回合完成时发出一个事件,携带这个回合的全部统计量。没有别的上报时机。

回合是唯一有训练含义的窗口:它的边界由环境给出(摔倒,或到达截断长度),不由我们的实现给出。chunk 是 JAX 的调度单位——调小它是为了显存,不该让任何曲线的粒度跟着变,所以**测量面看不见 chunk**。epoch 也不是窗口,它只剩"每隔多久评估一次"。

按环境步的固定间隔采样没有语义:间隔与任何自然边界都不对齐,那个点既不是某一步的值,也不是某个完整窗口的均值。现状把 epoch 内的窗口均值标在一个累计环境步的坐标上,读起来正是这种不存在的采样;`logging.every_steps` 这个名字加固了它,而它做的是丢弃已经算完的写入。该字段删除。

**per-env 是白送的。** 每个回合只属于一条流,所以按回合统计天然分得开各条流。今天丢掉这个区分的是 `named_scalars` 里的 `jnp.nanmean`,它把时间轴和 env 轴一起压平——是聚合方式丢的,不是测不到。

### 一个回合发出什么

`length`、`return`、`return_per_step`,以及**方差**。

只报一个均值是不够的:`return_per_step` 说不出这个回合是稳定拿分还是几次尖峰撑起来的。方差和均值一起给,才是对分布的最小交代。

算法自己的步级诊断量(`td_error`、`value`、`entropy`、各种范数)同样按回合归约——它们没有天然窗口,借回合这个窗口是唯一有意义的选择,同样是均值与方差一起。

回合内的细节——`td_error` 是不是在开头尖峰——不进标量库,进 rerun 的完整轨迹。两者分工:**rerun 存轨迹,Aim 存统计量。**

### 名字

`<阶段>/<粒度>/<量>`:

```
train/episode/length          eval/episode/length
train/episode/return          eval/episode/return
train/episode/return_per_step eval/episode/return_per_step
```

中段是**归约所用的窗口**,当前只有 `episode` 一种。保留三段是为了以后加别的窗口时不必改名;`step` 这一段不存在,因为没有步级的上报。

粒度由名字承担,不用 Aim 的 context。`score.metric` 保持是一个字符串,preflight 拿它对入口声明的指标校验的做法不用改。

横轴仍是累计环境步:回合在它结束的那个步数上发出事件。

### 一处要接受的代价

回合不结束就没有点。训练早期回合短、点密;策略变好后回合变长、点变稀。这是诚实的——它反映的正是"这个策略活多久"。被 `total_steps` 截断的最后一个未完成回合不发事件。

### 按环境拆解,后续

Hopper 的奖励是存活、前进、控制代价三项之和,Brax 在 `info` 中分别给出。分别记录比只看总和更有意义,并且对不同环境该拆法不同。这需要指标声明能随环境变化,与本设计的参数声明是同一类问题但不是同一件事,留待后续。

## 7. 迁移

`BudgetConfig` 不再作为实验与 manifest 的独立字段。契约改为 `EnvironmentConfig(seed, id, backend, observed)`、`TrainingConfig(num_envs, total_steps, epoch_steps, chunk_steps?, early_stop_patience?)`、`EvaluationConfig(steps, num_envs)`。memo 系 entry 从 `training` 与 `evaluation` 读循环参数;AAAI entry 从 `training.chunk_steps` 计算作者代码的 `episodes` 与 `eval_every`,从 `evaluation.num_envs` 设置作者代码的 `eval_batch_size`。

`EntryDescriptor.space` 改名为 `EntryDescriptor.parameters`,内容从平铺 `FloatSpec | IntSpec | ChoiceSpec` 扩展为参数树与结构树。实验 YAML 顶层 `space` 保留该名字,但它只是实验覆盖,不是 catalog 字段。控制面的 reserved 名增加 `seed`、`num_envs`、`total_steps`、`epoch_steps`、`chunk_steps`、`early_stop_patience`、`eval_steps`、`eval_envs`,旧实验若把这些名字放在 `space` 下必须迁出。

入口参数声明删除 `seed`、AAAI 的 `scan_steps`、`eval_envs`、`patience`。AAAI 不新增 `eps`。StreamAC 与 upstream StreamAC 同时把旧 `eps` 拆成 `optimizer_eps` 与 `normalization_eps`,旧实验 YAML 中的 `eps` 必须按语义改名;若只是复现实验里钉死的 `1e-8`,两个字段都写 `1e-8`。

`EntryDescriptor` 与 `RunConfig` 去掉 `source_hash` 字段,连同以下引用一并删除:三个 catalog 构建脚本里的计算(`memo/runner/catalog.py:38-56`、`rtrrl/scripts/build_catalog.py`、`rtrrl/infra/mock-trainer/scripts/build_catalog.py`)、`preflight.py:135` 的比对、`launch.py` 与 `loop.py` 写进 launch.json 与 study 属性的引用、`sinks/aim.py:26` 写进 Aim 运行属性的引用,以及控制面测试里的相关夹具与 `test_preflight_aws.py` 中那条漂移检测用例。镜像 digest 已经回答"跑的是哪个镜像"。

`images.py:172` 的 `hashlib.sha256` 不在此列。它校验从 ECR 下载的 config blob 字节是否等于清单声明的摘要,是内容寻址的完整性检查,决定读到的 catalog 标签是否可信。

归档目录 `rtrrl/infra/control-plane/archive/` 下历史 launch.json 中的该字段不动。

`CONTRACT_VERSION` 递增。catalog 结构改变,旧镜像的 catalog 不再被接受,memo 与 rtrrl 两个镜像都要重建。

`experiments/` 下现存的实验一并迁移到新格式,旧格式不保留,控制面不同时接受两种语义。`experiments/streamac template.yaml` 与 `experiments/rtrrl template.yaml` 是目标格式的两份样板,实验按它们写。

`seed` 只作为 `environment.seed` 出现,一个值,不是列表,任何时候都不进搜索。因此不存在"多 seed 实验"这种文件:仅靠改变 seed 来重复的实验就是重复启动同一个文件。原先四个 `streamac-hopper-seeds-*` 实验的 `space` 是十八个钉死的值加 `seed: [0,1,2,3,4]`,拿 grid 采样器的 trial 数当重复循环,seed 注入之后无可搜索,已删除。多 seed 的聚合语义不在本设计范围内,也不用复制文件的方式补回来。

`LoggingSpec` 拆成两个组件,Aim 与 Rerun 各自完整,各自可关:今天 `rerun_every_episodes` 是一个可选整数,靠"是否为 None"兼任开关,而 Aim 没有开关。新增 `enable_rerun`,两个 sink 的启用与参数分别声明。

`normalization_statistics`、`STATISTICS` 表、`feature_dim` 作为公共编码宽度的用法一并处理:前两者删除并换成三个参数,`feature_dim` 保留但归入 `rtu` 分支。`RTUCell` 与 `LRUCell` 去掉 `activation_fn` 字段。

实施分四个阶段,每阶段自身可验证:

1. **分段** —— 环境、训练、评估分段,`seed`/预算/循环/评估参数出 `space`,`observed` 与删维度。
2. **配置加载** —— 只做 infra 一侧:契约的参数树、声明与读回的助手、控制面的解析与条件采样、catalog 构建器。它规定的是**加载的行为**,不是任何算法的参数长什么样。
3. **算法与组件** —— 网络与归一化拆成组件,组件内不混合、不认识自己的调用场景,三条 backbone 分支各自对齐来源,OBGD 拆成 `optimizer_bound` 与 `optimizer_base` 两轴,内核收组件实例而非扁平字段。
4. **指标面** —— 三段命名、训练侧两条、`eval/episode/return_per_step`。

### 两个阶段各自的产出

**阶段 2 交两样东西:**

1. **一套配置格式契约** —— 组件怎么声明自己才算合法:冻结的 dataclass,每个字段用 `param()` 或 `structure()` 声明,`param` 三件套齐全且 `search` 与 `placeholder` 都落在 `valid` 内,`log` 的 `search` 下界大于零,结构的 `placeholder` 是它的一个分支,组件只有声明没有方法。
2. **一个配置加载组件** —— 对任何满足该契约的算法,把它全部组件的配置展开成参数树,再用实验配置里的超控参数覆盖,产出一份扁平 manifest。它不认识任何具体算法。

**阶段 3 交两样东西:** 全部组件都符合该契约的库,以及一个能用目标实验配置跑起来的 `stream_ac` 入口。

### 阶段 2 的测试测契约,不测内容

阶段 2 的测试**可以用真实组件**,但断言只能是契约层的性质,不能落到某个字段上。判据是:**把 `rtu` 或 `adam` 的字段增删改一遍,展开的测试仍应绿**——它断言的是"每个声明的字段都成为一个键"、"每个键的 placeholder 落在它的 valid 内"、"`read_branch` 读回的与声明的一致",而不是"`rtu` 有 `hidden_dim` 且 placeholder 是 192"。

反过来,**违反契约的组件必须红**:字段没有用 `param()`/`structure()` 声明、`search` 越出 `valid`、`log` 的下界为零、结构的 `placeholder` 不是分支、组件上挂了方法。这几条各有一个测试,用测试内构造的不合规组件。

一个断言里出现 `gamma` 或 `kappa` 的取值就说明它测错了层——那是算法的事。这样加载器的行为与算法的取值互不牵连,改一个参数的范围不会让 infra 的测试变红。


### 阶段 3 的测试复用阶段 2 的加载器

阶段 3 是 TDD。`stream_ac` 的测试**必须经由阶段 2 的加载器取得参数**——声明一棵树、解析、采样,拿到扁平 manifest 再喂给入口——而不是手写参数字典。手写字典会绕开加载器,于是声明和读取是否真的对得上就没有被测到。

本阶段只动 `stream_ac` 与它用到的组件。`upstream_stream_ac`、`rtrrl`、`rtrrl_aaai` 不动,它们的测试是红的,原因明确。

阶段 2 依赖阶段 1 腾空 space。阶段 3 依赖阶段 2 的结构树来表达分支下的组件参数,并且会改变已记录分数的可比性——现有网络的参数量与非线性位置都变,旧分数不再是对照。阶段 4 与其余三个无依赖,插在任何位置都可以。

## 8. 未定

- `evaluation.num_envs` 目前没有消费者。`drive()` 拿训练的流数做评估(`memo/runner/loop.py:133`),memo entry 只读 `evaluation.steps`。规格 §7 要求 memo entry 从 `evaluation` 读循环参数,所以 `evaluate()` 应当接收自己的流数;`evaluate()` 内部的 reset key、动作零张量、timestep、carry、sensitivity 全部按宽度当场新建,从训练带过来的只有网络参数与 normalizer 状态,两者都不依赖 batch 宽度,所以这个宽度本就是独立的。
- `mlp` 分支的层数、`rtu` 分支编码器的层数,以及各分支里 LayerNorm 的开关,是分支下的参数还是固定接线,未定;placeholder 一律取各自来源的取值。
- 90% 稀疏初始化(`rtu` 来源规定)是分支参数还是固定接线,未定。
