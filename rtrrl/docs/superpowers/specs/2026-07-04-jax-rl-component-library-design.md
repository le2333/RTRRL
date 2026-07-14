# jax_rl 组件库:从 rtrrl.py 抽离可复用算法组件

- 日期:2026-07-04
- 状态:设计待复核
- 范围:`streaming-rtrrl` 中将可替换的算法模块独立成库,并将算法与训练进程管理分离
- 不含:训练脚本重写、config schema 变更、AWS 基础设施改动(均留后续,见 §8)

## 1. 目标

把 `rtrrl.py` 中可复用的 RL 组件抽离成中性的、不绑定 RTRRL 的组件库 `jax_rl/`,使:

1. **组件级可复用**:TD、优化器、模型各自独立,可单独 `import`、单独替换。
2. **算法可替换**:RTRRL 只是 `algorithms/` 下一个具体装配;换算法 = 新增一个子包,其余组件复用。
3. **算法与训练分离**:算法核心(`algo_step`)与训练编排(主循环/eval/日志/early stopping)解耦。

### 不做

- 不改 `RTRRLParams` 字段名(AGENTS.md:config 是不可变审计记录,改 schema 破坏所有历史 config 与 AWS Batch 任务)。
- 不动 `envs/`、`logging_util.py`(环境与日志是训练侧设施,不属于算法组件库)。
- 不改 JAX 版本(AGENTS.md:`jax==0.5.0` 锁定)。

## 2. 同类库调研与借鉴

| 库 | 定位 | 借鉴点 | 反例 |
|---|---|---|---|
| **RLax** (DeepMind) | 纯原语库:TD-λ、资格迹、policy gradient loss 等数学运算,无完整算法、无训练循环 | `jax_rl/traces/` 对标 RLax 的原语风格——暴露 `trace_update`/`compute_updates` 这类原子函数 | — |
| **coax** | 模块化 RL,明确**不隐藏训练循环**:"你决定如何以及何时更新函数逼近器";tracer/updater 作独立组件 | 印证训练调度不入库(§5) | — |
| **Ajax** (YannBerthelot) | 模块化 agent 框架:`base.py` ActorCritic 基类 + 各算法子目录 + `modules/` + `networks/` + `logging/` | `algorithms/<name>/` + `models/` 的子包结构 | — |
| **Stoix** | 单文件风格,允许代码重复换可读性,**明确不是可 import 的库** | — | 我们的库要可 import,不走 Stoix 路线 |
| **skrl** | 多框架(PyTorch/JAX/Warp)重量级 | — | 偏离轻量定位,不参考 |

**结论**:`jax_rl` 定位介于 RLax(纯原语)与 Ajax(完整 agent 框架)之间——暴露可复用组件 + 可替换算法装配,但**不含训练调度**(同 coax/RLax)。

### 2.1 外部库/算法引入评估

| 来源 | 处理 | 归属 |
|---|---|---|
| **RLax** (DeepMind, JAX) | **不引入为依赖**:`jax>=0.7.0` 与项目锁定 `jax==0.5.0` 冲突;且 RLax 的 `td_lambda` 是 λ-return 误差计算(batch 视角),非 RTRRL 用的参数级 eligibility trace 维护,抽象层级不对 | 仅作 `jax_rl/traces/` 的**正确性测试基准**(dev/test extras,λ=0/1 退化数值比对) |
| **coax** (JAX) | 不引入:完整框架,与轻量定位冲突;其 tracer 是 transition tracer,非 eligibility trace | 参考"不隐藏训练循环"哲学(已写入 §5) |
| **streaming-drl** (PyTorch, arXiv 2410.14606) | **部分引入并移植到 JAX**:ObGD 必须引入;TD 误差与现有差别不大(复用 `td/`);归一化不引入(用现有 `running_statistics`);`sparse_init` 不适用 | `optimizers/streaming/`(ObGD 移植)+ `algorithms/streaming/`(新算法装配) |

streaming-drl 关键点:ObGD 主体是**限幅**(`step_size = lr / max(|delta|_bar · z_sum · lr · kappa, 1)`),trace 维护(`e ← γλ·e + grad`)与 `delta·e` 组合都是标准原语,可外部维护复用 `traces/`。唯一接口问题:optax 标准签名缺 `delta`/`z_sum` 标量,ObGD 做成带 extra args 的 optax 变换解决。详见 §4 插入契约。

## 3. 分层架构

当前 `rtrrl.py` 把三层混在一个 `train_rtrrl` 函数里。重构后分三层:

| 层 | 职责 | 当前位置 | 目标位置 |
|---|---|---|---|
| **L1 工具层** | 纯函数,无状态:meta-输入拼接、done 重置、归一化、tree ops、param labeling | 散落在 `train_rtrrl` 闭包(4 处 meta 输入重复、3 处 done 重置重复) | `jax_rl/utils/` |
| **L2 算法层** | `algo_step(state, obs, key) -> (state, outputs)` + `AlgorithmState` pytree;含 TD-λ/资格迹/RFLO 梯度/trace 更新/optimizer.step | `step_fn` 内嵌套闭包 `grads_step`/`td_loss`/`non_TD_loss`/`trace_updates` | `jax_rl/traces/` + `jax_rl/td/` + `jax_rl/algorithms/rtrrl/` |
| **L3 训练管理层** | 主循环、`eval_model`、日志节奏、early stopping、`with_logger`、profiling | `train_rtrrl` 外层 + `__main__` | **留 `rtrrl.py` 顶层,不入库**(见 §5) |

### L1 缺口(用户首轮指出)

当前 L1 完全不存在,被当作"就地写两行 lambda"散在各处。需抽出的工具:

- `build_meta_input(obs, action, reward, obs_rms, reward_rms, discrete, act_size)` — 消灭 4 处重复(`rtrrl.py` 365/429/458/554)
- `reset_on_done(done, zero_state, state)` — 消灭 3 处重复(534/712/791),含变体(544 done 时置零 action/reward)
- `tree_add` / `tree_mean` — 消灭 620/655/745/749 的 `jax.tree.map(lambda x,y: x+y, ...)` 重复
- `make_normalizer` / `normalize` / `update_rms` — `running_statistics` 薄封装(349-360/528-530)
- `label_params(params, rnn_keys)` — 取代 387-400 的 `"rnn" in k` 字符串匹配

## 4. `jax_rl/` 包结构

```
streaming-rtrrl/
├── rtrrl.py / rtrrl_lru.py / ppo_baseline.py / sac_baseline.py   # 训练脚本,留顶层
├── envs/                # 留顶层(被所有脚本共用)
├── logging_util.py      # 留顶层(训练侧设施)
├── config/ docs/ infra/ # 不动
└── jax_rl/              # 通用 RL 组件库(基于外部 optax/distrax/brax,见 §4.3)
    ├── __init__.py
    ├── backbones/      # RNN 单元(独立于 Meta-RL 架构,通用 backbone)
    │   ├── __init__.py
    │   ├── ctrnn.py    #   CT-RNN(Murray 2019)← 现 models/ctrnn.py
    │   ├── online_lru.py # LRU(Zucchet 2023)← 现 models/online_lru.py
    │   ├── wirings.py  #   CT-RNN wiring ← 现 models/wirings.py
    │   └── seq_util.py #   LRU parallel scan ← 现 models/seq_util.py
    ├── models/         # 神经网络 heads(linear policy/value + feedback alignment)
    │   ├── __init__.py
    │   ├── neural_networks.py # MLP/FADense ← 现 models/neural_networks.py
    │   └── jax_util.py #   sigmoid_between 等 ← 现 models/jax_util.py
    ├── traces/         # 资格迹原语(独立组件,不绑定 TD;通用信用分配)
    │   ├── __init__.py #   暴露 trace_update / compute_updates / init_trace
    │   ├── traces.py   #   ← 现 traces.py 迁入
    │   └── config.py   #   TraceConfig(trace_mode/gamma/lambda_*)
    ├── td/             # TD 误差 + 账本(TD 特有,不复用流式外的算法)
    │   ├── __init__.py
    │   ├── td_error.py #   ← 片 C 从 step_fn 抽出(td_error/update_I/update_r_bar)
    │   └── config.py   #   TDConfig(eta/gamma)
    ├── optimizers/      # 优化器组件:基于外部 optax 的薄包装 + 自定义变换
    │   ├── __init__.py  #   暴露 make_optimizer / OptimizerConfig / get_current_lrs
    │   ├── factory.py   #   保留现有 make_optimizer 工厂(按 opt_name 调度到 adam/ 或 streaming/)
    │   ├── schedules.py #   cosine / cosine_warmup / exponential decay(共享,主要 adam 用)
    │   ├── config.py    #   OptimizerConfig(共享)
    │   ├── util.py      #   get_current_lrs(共享)
    │   ├── adam/        #   直接用 optax.adam(chain(scale_by_adam, scale(-lr)))
    │   └── streaming/   #   ← 后续开发:ObGD 自定义 optax 变换(限幅+scale 内联;带 extra args: delta, z_sum)
    ├── utils/           # L1 工具层
    │   ├── __init__.py
    │   ├── tree_utils.py    # tree_add / tree_mean / reset_on_done / tree_abs_sum
    │   ├── meta_input.py    # build_meta_input
    │   └── normalization.py # brax.running_statistics 薄封装
    ├── algorithms/      # 可替换算法锚点(用上述组件装配)
    │   ├── __init__.py
    │   ├── rtrrl/       # Meta-RL 装配:backbones/ + models/ + traces/ + td/ + optimizers/
    │   │   ├── __init__.py
    │   │   ├── algorithm.py  # ← 片 C:algo_step + AlgorithmState(含 RFLO/RTRL Jacobian 维护)
    │   │   └── config.py     # RTRRLAlgoConfig(从 RTRRLParams 拆出的算法字段)
    │   └── streaming/   # ← 新增:streaming-drl 装配:backbones/ + models/ + traces/ + ObGD(无 td/r_bar)
    │       ├── __init__.py
    │       ├── algorithm.py  # streaming_algo_step(片 F,后续)
    │       └── config.py     # StreamingAlgoConfig
    └── tests/           # 数值一致性测试(见 §7.1)
```

### 组件契约

**ModelProtocol**(L2 算法依赖模型的最小接口,片 B 定义):

```python
class ModelProtocol(Protocol):
    def initialize_carry(self, rng, input_shape) -> Carry: ...
    def rnn_step(self, carry, obs, training=True) -> (hidden, carry): ...
    def value(self, hidden, x=None) -> Array: ...
    def action_dist(self, hidden, x=None) -> Distribution: ...
```

算法侧只通过这 4 个方法访问模型(当前 `rtrrl.py` 569/580/583/637 的 `method=model.*` 调用正好就是这 4 个),接口面已很窄。

**AlgorithmState**(L2 算法状态,片 C 定义,取代现 13 元裸 tuple):

```python
@dataclass  # pytree-registered
class AlgorithmState:
    params: Params
    slow_params: Params
    opt_state: OptState
    z: Trace               # 资格迹
    rnn_state: Carry
    v_prev: Array
    r_bar: Array
    I: Array
    obs_rms: RMS | None
    reward_rms: RMS | None
```

**algo_step**(L2 算法一步,片 C 定义):

```python
def algo_step(state: AlgorithmState, transition, key) -> (AlgorithmState, AlgoOutputs):
    # 内含:grads_step / td_loss / non_TD_loss / trace_updates / compute_updates / optimizer.step / Polyak / tau hook
```

### 优化器拆分说明(用户:工厂保留 + adam/streaming 目录独立,无 gd/ 共享层)

现 `optimizers.py` 167 行单文件拆成**工厂 + 共享 + 两个独立实现**的结构。`optax.sgd`/`optax.scale` 已是纯 SGD,adam 和 ObGD 真正共享的代码几乎没有(动态 lr 注入、z_sum、sign 门控都只 ObGD 用),故不设 `gd/` 共享层:

- `factory.py`:保留现有 `make_optimizer` 工厂(`optax.chain` 装配 + 按 `config.opt_name` 调度到 `adam/` 或 `streaming/`)。
- 共享层:`schedules.py`(3 种 decay schedule)、`config.py`(`OptimizerConfig`)、`util.py`(`get_current_lrs`)。
- `adam/`:直接用 `optax.adam`(`chain(scale_by_adam, scale(-lr))`),非开发重点,仅归类。
- `streaming/`:ObGD 自定义 optax 变换,开发重点。trace **外部维护**(复用 `traces/`),ObGD 内部 `updates = step_size * (delta * z)`(限幅 + scale 内联,一行 SGD scale 不必抽 gd/)。ObGD 需要的 `tree_abs_sum`/`sign` 等工具放 `utils/tree_utils.py`。

**两个实现目录独立**的理由:adam 是现成 optax 组合;ObGD 是开发重点需独立迭代。两者通过 `factory.py` 统一调度,互不耦合。不强求共享执行器代码(optax 已提供纯 SGD)。

**实现插入契约**:

核心问题:optax 标准 `update(grads, state, params)` 签名缺 `delta`(标量 TD 误差)与 `z_sum`(trace 范数)两个标量,而 ObGD 限幅需要它们。解法——trace 外部维护,ObGD 做成**带 extra args 的 optax 变换**:

```python
# optimizers/streaming/obgd.py
def obgd_init(params) -> ObGDState: ...
def obgd_update(updates, state, params=None, *, delta, z_sum):
    # updates = delta * z (外部 td/ 已组合)
    step_size = lr / jnp.maximum(jnp.maximum(jnp.abs(delta), 1.0) * z_sum * lr * kappa, 1.0)
    updates = jax.tree.map(lambda u: step_size * u, updates)   # 限幅 + SGD scale 内联
    return updates, state
ObGD = optax.GradientTransformation(obgd_init, obgd_update)
```

- `algo_step(streaming)` 路径:`z = trace_update(grad, z, γλ)`(td/)→ `z_sum = tree_abs_sum(z)`(utils)→ `updates = delta * z`(`compute_updates`)→ `obgd_update(updates, state, delta=d, z_sum=z_sum)`(限幅 + scale)→ `apply_updates`
- `AdaptiveObGD` 多一个 RMSProp 式二阶矩 `v`(有状态),仍是标准 optax 变换,trace 仍外部
- `config.opt_name` 指向 `"adam"` 或 `"streaming"` 即切换,无需改 `factory.py`
- optax 0.2+ 有 `GradientTransformation` 带 `extra_args` 的正式机制;项目锁定的 optax 版本需确认是否支持,不支持则用普通函数签名扩展(等价)

`multi_step` 保留(虽当前 config 未用,语义已实现且未来 HPO 可能启用)。

## 4.2 组件 ↔ 论文算法对应(每个组件可追溯到论文的一个算法部分)

论文([arXiv:2311.04830](https://arxiv.org/abs/2311.04830))将 RTRRL 分为 3 个 building blocks(line 47):Meta-RL RNN 架构 + TD(λ) backward view + RFLO/RTRL online autodiff。Algorithm 2(附录)的 4 步伪代码 + Figure 7/8 给出完整流程。`jax_rl/` 每个组件对应关系:

| jax_rl 组件 | 论文对应 | 论文位置 |
|---|---|---|
| `backbones/ctrnn.py` (`OnlineCTRNNCell`) | **CT-RNN backbone** + ODE forward-Euler solver(Murray 2019,通用 RNN,不绑定 Meta-RL) | §"CT-RNNs"、Algorithm 2 line 13、Appendix 公式 10-12(RFLO Jacobian for W/τ) |
| `backbones/online_lru.py` (`OnlineLRULayer`) | **LRU backbone** trained with RTRL(diagonal connectivity) | §"LRUs"、Table 3 RTRRL-LRU |
| `models/neural_networks.py` (`FADense`/`MLP`) | **Linear policy** π(a\|h,θA) + **Linear value** v̂(h,θC) + **feedback matrices** B_A/B_C(FA) | Figure 1、Algorithm 2 line 10/14、line 22-23(`gC=BC·1, gA=BA·∂h lnπ`) |
| `backbones/wirings.py` | CT-RNN wiring(NCP / fully_connected) | §"CT-RNNs" wiring |
| `backbones/seq_util.py` | LRU parallel scan(RTRL 高效计算) | §"LRUs" diagonal connectivity |
| `models/jax_util.py` (`sigmoid_between`) | 连续动作 action bounding | 工程设施(非论文核心) |
| `traces/traces.py` (`trace_update`/`compute_updates`/`init_trace`) | **eligibility traces**(独立原语):e_A/e_C/e_R 更新 `γλe+∇f` + `δ·e` 组合 | Algorithm 2 line 16-17(eC/eA)、line 27-28(δe)、§"Eligibility Traces" |
| `traces/config.py` (`TraceConfig`) | trace_mode / γ / λ_[A,C,R] | Table 1、§"hyperparameters" |
| `td/td_error.py` (`td_error`/`update_I`/`update_r_bar`) | **TD-error** δ = R + γv' - v(+ 平均奖励 r_bar 扩展,TD 特有) | Algorithm 2 line 15、§"TD(λ)" |
| `td/config.py` (`TDConfig`) | η(平均奖励)/ γ | Table 1、§"hyperparameters" |
| `optimizers/adam/` | **parameter update** θ ← θ + α·(δe + entropy) via adam | Algorithm 2 line 27-29、Table 1(optimizer=adam) |
| `optimizers/streaming/` (ObGD) | 同一 update step 的**替代实现**(streaming-drl,非论文) | —(非 RTRRL 论文,见 §2.1) |
| `optimizers/{factory,schedules,config,util}.py` | update 支撑设施(clip / lr decay / weight decay) | Table 1(gradient_clip=1.0、lr_decay) |
| `algorithms/rtrrl/algorithm.py` (`algo_step`) | **Algorithm 2 整体装配** + **RFLO/RTRL approximate Jacobian Ĵ 维护**(online autodiff) | Algorithm 2、Figure 7 4步、Figure 8、§"Putting it together"(line 192)、Appendix RFLO derivation |
| `algorithms/rtrrl/config.py` | Algorithm 2 超参 | Table 1 |
| `algorithms/streaming/` | streaming-drl StreamAC 装配(非论文) | — |
| `utils/meta_input.py` (`build_meta_input`) | **Meta-RL input** [o, a, r] 拼接 | Figure 1、line 192("observation, previous action, reward are concatenated")、Algorithm 2 line 11/13 |
| `utils/normalization.py` | obs/reward 归一化 | 工程设施(Table 1 normalize_obs=False) |
| `utils/tree_utils.py` (`reset_on_done`/`tree_abs_sum`) | done 时重置 trace(Algorithm 2 line 5 `e←0`)+ z_sum(ObGD) | Algorithm 2 line 5、ObGD `z_sum=Σ|e|` |

**论文 3 个 building blocks ↔ 组件映射**(backbone 与 Meta-RL 架构分层,traces 与 td 分离):

1. **Meta-RL RNN 架构**(basal ganglia)→ `backbones/`(RNN 单元,通用)+ `models/`(linear heads + FA)+ `utils/meta_input`([o,a,r] 拼接)+ `algorithms/rtrrl/`(装配出 Meta-RL 架构)
   - backbone 是通用 RNN(CT-RNN/LRU),独立于 Meta-RL;Meta-RL 是"用 backbone + heads + [o,a,r] 组装"的上层架构
2. **TD(λ) backward view** → `traces/`(eligibility traces,通用信用分配原语,独立)+ `td/`(TD 误差 + 账本,TD 特有)
   - traces 不绑定 TD(`γλe+∇f` 不依赖 δ);只有 `δ·e` 组合时才与 td/ 协作
3. **RFLO/RTRL online autodiff**(approximate Jacobian Ĵ)→ `backbones/` 内的 Jacobian 维护(CTRNN/LRU 的 RTRL/RFLO trace)+ `models/FADense`(feedback alignment B_A/B_C)+ `algorithms/rtrrl/algorithm.py` 的 `jax.vjp(rnn_step, slow_params)`

**Figure 7 的 4 步 ↔ `algo_step` 的 4 阶段**:

| Figure 7 步 | algo_step 阶段 | 论文 line |
|---|---|---|
| 1. Step RNN | `backbones/` rnn_step(含 Ĵ 维护) | line 13 |
| 2. Act | `models/` action_dist + sample | line 10-11 |
| 3. Compute TD-error | `td/td_error` | line 15 |
| 4. Update using TD-error + Ĵ | `traces/` trace_update + compute_updates + `optimizers/` step | line 16-29 |

streaming-drl(ObGD)非论文算法,作为 `algorithms/streaming/` 单列,不混入 RTRRL 对应。

## 4.3 外部依赖声明

`jax_rl` 不从零实现基础 ML 设施,而是在外部库之上做薄包装/扩展。明确依赖边界,避免重复造轮子:

| 外部库 | 用途 | jax_rl 中的体现 |
|---|---|---|
| **jax** / **jaxlib** | 自动微分、jit、数组计算 | 全库基础(锁定 0.5.0,见 §6) |
| **flax** (`flax.linen`) | 神经网络模块定义 | `backbones/`、`models/` 的 `nn.Module` |
| **optax** | 优化器原语(`scale`/`adam`/`chain`/`schedule`) | `optimizers/adam/` 直接用 `optax.adam`;`optimizers/streaming/` 用 `optax.GradientTransformation` 协议自定义 ObGD;`schedules.py` 用 optax schedule |
| **distrax** | action distribution(`Normal`/`Categorical`) | `models/` 的 actor 输出(RTRRL `action_dist` 用 distrax) |
| **brax.training.acme.running_statistics** | obs/reward 归一化 | `utils/normalization.py` 薄封装(不重新实现 Welford) |
| **chex** | 类型/形状断言 | 各组件按需 |

`jax_rl` 自己实现的算法部分:`traces/`(eligibility trace 原语)、`td/`(TD 误差+账本)、`optimizers/streaming/`(ObGD 限幅)、`backbones/` 内的 RFLO/RTRL Jacobian 维护、`algorithms/` 的算法装配。其余是外部库的薄包装。

## 5. 训练调度的归属:独立,不入 jax_rl

**结论**:训练调度(主循环/`eval_model`/日志节奏/early stopping/checkpoint/profiling)留 `rtrrl.py` 顶层,**不进 `jax_rl/`**。

### 依据

1. **业界共识**:RLax 完全不涉及训练循环;coax 明确"你决定如何以及何时更新函数逼近器";Ajax 也只把 `logging/` 放库内,训练循环用户写。
2. **边界清洁**:`jax_rl` 应是"纯组件",可被任意训练框架驱动。训练调度与实验执行基础设施(Aim/W&B/AWS Batch,见 `infra/`)强绑定,塞进库会污染库边界。
3. **耦合未解**:当前训练循环与 RTRRL 的 `lax.scan` + 13 元 carry 结构耦合;要抽象成通用 `Trainer`,需先有 `algo_step` 契约(片 C),而契约尚未定义。

### 后续选项(留 issue,本轮不做)

- 若片 C 落地后多个算法(PPO/SAC/RTRRL)想共享训练循环,可**另建独立的训练框架层**(`jax_rl_training/` 或顶层 `training.py`),驱动 `jax_rl` 组件。它和 `jax_rl` 是"使用方/被使用方"关系,不嵌套。
- 与 `docs/superpowers/specs/2026-07-04-execution-infra-s3-resume-design.md` 片7"训练组件从脚本重构为框架"对齐,作为该片的先决技术工作。

## 6. 配置兼容性约束(AGENTS.md binding)

- `RTRRLParams` 顶层字段名**不变**,否则破坏 `config/*.yml` 审计记录与在跑 AWS Batch 任务。
- 库内子 config(`td/config.py`、`optimizers/config.py`、`algorithms/rtrrl/config.py`)各自定义自己的 dataclass;`rtrrl.py` 仍保留 `RTRRLParams` 作聚合,内部字段映射到子 config。
- 字段拆分留到接口稳定后(片 E),本轮只搬代码、不改 schema。

## 7. 迁移路径(按风险递增)

**总原则(用户约束)**:
1. **迁移时不动源代码逻辑**——片 A1 只做"文件搬运 + 改 import",不改任何函数体;A2 起的重构(提取/重写)单独成片,与纯迁移分开。
2. **迁移后写测试确认算法结果一致**——每个片都配数值一致性测试,见 §7.1。

| 片 | 内容 | 风险 | 验证 |
|---|---|---|---|
| **A1** | 建 `jax_rl/`,纯搬运(不改函数体):`traces.py`→`traces/traces.py`(独立组件)、`optimizers.py`→`optimizers/{factory,schedules,config,util,adam}.py`(`adam/` 现有路径;`streaming/` 留空占位)、`models/`拆分→`backbones/`(ctrnn/online_lru/wirings/seq_util,RNN 单元)+ `models/`(neural_networks/jax_util,heads);改库内互引(`traces.py` 的 `from models.neural_networks import FADense`→`from jax_rl.models...`);各 `__init__.py` 重新导出 | 零(纯搬运) | §7.1 测试 T1:迁移前后同输入同 seed,输出 bitwise 一致 |
| **A2** | 建 `utils/`,抽 `tree_utils`/`meta_input`/`normalization`;`rtrrl.py` 内 4 处 meta 输入、3 处 done 重置改调工具函数 | 低(纯函数提取,不碰 jit) | §7.1 测试 T2:工具函数 vs 原 inline 逻辑数值一致 + 同 seed `eval reward` 不变 |
| **B** | `rtrrl.py`/`rtrrl_lru.py` 改 `import`(从根目录 `from traces ...`→`from jax_rl.td ...` 等);定义 `ModelProtocol`,算法侧通过协议访问模型 | 低 | 同 A2 验证 |
| **C** | 定义 `AlgorithmState` pytree + `algo_step`;把 `grads_step`/`td_loss`/`non_TD_loss`/`trace_updates`/`compute_updates`/optimizer.step/Polyak/tau-hook 整体迁入 `algorithms/rtrrl/algorithm.py`;`vjp(slow_params)` 链路**整体内迁不拆** | **中高** | §7.1 测试 T3:algo_step vs 原 step_fn 同 seed eval reward 一致;重点查 RFLO 双梯度路径 |
| **D** | `train_rtrrl` 只保留 L3:主循环调 `algo_step`,`eval_model`/日志/early stopping 留脚本 | 中 | 同 C 验证 |
| **E** | (可选,后做)拆 `RTRRLParams` 为子 config 聚合,字段名保持兼容 | 高(碰 schema) | 全部 config yml 跑通 |
| **F** | **ObGD 移植**:`optimizers/streaming/obgd.py` 把 `streaming-drl/optim.py` 的 ObGD/AdaptiveObGD 从 PyTorch 移植到 JAX,做成带 extra args(`delta`, `z_sum`)的 optax 变换(trace 外部维护,复用 `traces/`;限幅+scale 内联,无 gd/ 层);`algorithms/streaming/` 新增 `streaming_algo_step` | 中(限幅逻辑移植 + Adaptive 的 `v` 状态;trace 复用现有) | §7.1 测试 T4:ObGD 与外部 trace+adam 在 λ=0 退化时数值一致;streaming-drl 原始论文环境复现 |

### 7.1 数值一致性测试策略

测试放 `jax_rl/tests/`(随库一起,可独立 `pytest` 运行)。每个片配一组测试,原则:**同 seed + 同输入,迁移/重构前后输出 bitwise 一致**(JAX 在固定 seed 下确定性)。

**T1(片 A1 后,纯搬运)**——确认搬运无行为变化:
- `test_traces`:对 `trace_update`/`compute_updates`/`init_trace`,用固定 `grads`/`z`/`gamma_lambda`/`d`,比对迁移前后输出(因代码未改,应 bitwise 一致)
- `test_optimizers`:`make_optimizer` 对同 `OptimizerConfig` 产出等价变换;对同 `grads`/`opt_state`/`params`,`update` 输出一致
- `test_models`:`OnlineCTRNNCell`/`OnlineLRULayer`/`MLP`/`FADense` 对同输入同 `params` 输出一致
- 比对基准:迁移前的旧路径(`from traces import ...`)与迁移后新路径(`from jax_rl.td import ...`)并存跑一次,逐函数 assert

**T2(片 A2 后,工具提取)**:
- `test_utils`:`build_meta_input`/`reset_on_done`/`tree_add` 对模拟输入,与 `rtrrl.py` 原 inline lambda 逻辑数值一致
- 端到端:同 seed 跑 `--episodes 小数 --steps 小数`,`eval reward` 与重构前一致

**T3(片 C 后,algo_step 抽离)**:
- 单元:`algo_step` 单步对固定 `AlgorithmState` + transition,与原 `step_fn` 单步输出(`params`/`z`/`v_hat`/`d` 等)bitwise 一致
- 端到端:同 seed 跑完整训练,`eval/best_eval_reward` 与原 `rtrrl.py` 一致(允许浮点误差 < 1e-6)
- 重点:`vjp(slow_params)` 双梯度路径(RFLO)数值比对

**T4(片 F 后,ObGD 移植)**:
- λ=0 退化:ObGD 在 `lamda=0` 时 trace 退化为单步 grad,与"外部 trace + adam 单步"数值一致(验证移植正确)
- `AdaptiveObGD` 的 `v` 状态:与 PyTorch 原版 `optim.py` 同 seed 同序列数值比对(跨框架,允许 1e-5 误差)
- streaming-drl 论文环境复现:在 CartPole-v1 / Ant-v4 上跑 `streaming_algo_step`,reward 曲线与原 `stream_ac_continuous.py` 趋势一致

### JIT 边界注意(片 C 关键)

现 `step_fn` 是单个 `@jax.jit`,算法与训练在同一个编译单元。片 C 拆分时**保持"算法一步仍是一个完整 jit"**:`algo_step` 自身被 jit 包裹,训练侧在外层 `lax.scan` 调用它。不要把 `algo_step` 内部拆成两个 jit 单元——会改变 XLA 融合与内存计划,尤其 `vjp(slow_params)` 那段。

## 8. 风险

1. **RFLO 双梯度路径搬迁**(片 C 最大风险):`jax.vjp(rnn_step, slow_params, has_aux=True)` + `jax.grad(slow_params)` 的"目标网络前向、梯度回灌 params"链路(573-620 行)微妙,搬动易静默出错。对策:整体内迁不拆,同 seed 数值比对。
2. **资格迹参数树过滤**:现按 `"rnn" in k` 字符串匹配(388/398-400 行)。抽库时改显式 `label_params(params, rnn_keys)` 注入,语义不靠名字猜。
3. **`tau` 裁剪 HACK**(768-775 行):ctrnn 特定的后处理,夹在 optimizer.step 后。做成可插拔 `post_update_hook`,不写死在算法主路径。
4. **JIT 性能**:片 C 拆分若改 jit 边界,可能影响编译/融合。对策:对比重构前后单步编译时间与 `cost_analysis()`(现 `debug==2` 已有 profiling 入口)。
5. **配置兼容**:片 A-D 不碰 `RTRRLParams` 字段;片 E 才考虑拆分,且必须保持字段名。
6. **`jax_rl` 命名**:PyPI 上可能已有人占。当前仅本地包,不影响;若未来发布需查重。

## 9. 后续切片(留 issue,本轮不做)

- 片 E:config schema 拆分(等接口稳定)。
- 训练框架层:独立于 `jax_rl` 的 `Trainer` 抽象(等片 C 契约落地,多算法共享需求出现后再做)。
- 与 S3 任务发布片(`2026-07-04-execution-infra-s3-resume-design.md` 片7)协同:训练组件框架化后,快照/续训接口再定。
