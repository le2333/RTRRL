# JAX-first 在线循环 Actor-Critic 架构设计

## 1. 决策摘要

`memo` 当前有两套在线循环 Actor-Critic：

- RTRRL：共享 recurrent torso 的 meta-AC，三组资格迹、episodic emphasis、
  slow torso、Adam。
- StreamACRtrl：actor/critic 独立网络的 standard AC，RTRL、两组资格迹、OBGD。

新架构不建立一个运行时可动态切换组件的“万能算法类”。JAX 适配性优先，灵活性只
存在于 JIT 前的构建阶段。

核心决策：

1. standard 与 meta 使用两个具体 program kernel。
2. TD、trace、credit、target、update、normalization 以 JAX-pure 函数工厂复用。
3. 每个 resolved 配置生成一个 state 结构固定、scan 长度固定的 AgentProgram。
4. 配置解析、环境构建、Aim、evaluation 编排和 I/O 位于 JIT 外。
5. 不使用运行时 registry、字符串分派、反射实例化或 universal optional state。
6. 接受 standard/meta kernel 中少量重复代码，换取可预测的 JAXPR、HLO 和编译行为。
7. 当前 RTRRL/StreamACRtrl 保持数值语义；RTRRL-OBGD 在通用入口稳定后组合。
8. “保持数值语义”包括 forward/RNG/trace/reset/bootstrap/update 的调用顺序；新语义
   必须由显式配置启用，不能在重构中静默替换 legacy 行为。

## 2. 范围

### 当前范围

- 两种具体 program：`MetaProgram`、`StandardProgram`。
- LRU、RTU recurrent core。
- RFLO/RTRL-compatible credit、TBPTT(1)。
- RTRRL/StreamAC objective。
- 两组 trace、三组 trace + emphasis。
- TD(0)、无 target、slow subtree。
- Grouped Adam、domain OBGD。
- observation/reward/TD normalization。
- 环境、evaluation protocol 和 Aim recorder 的统一配置。
- 旧 API、YAML、Aim 字段和 HPO 配置的兼容翻译。

### 非目标

- 不支持 JIT 内动态切换算法。
- 不为尚不存在的第三方插件设计注册系统。
- 不统一 MetaState 与 StandardState。
- 不在首次迁移中更正历史 trace timing、熵路径或 target 语义。
- 当前不实现 trajectory、Rerun 和结构化 FailureEvent。

## 3. 总体架构

```text
ConfigLoader
    ↓
ResolvedExperimentSpec
    ↓
Explicit Factories / Validation
    ↓
BuiltExperiment
├── JAXEnvAdapter
├── AgentProgram
│   ├── init_fn
│   ├── train_epoch_fn
│   ├── evaluate_fn
│   └── concrete State pytree
├── EvaluationRuntime
└── Recorder
```

训练热路径：

```text
MetaProgram 或 StandardProgram
    ↓
具体 step_fn closure
    ↓
jax.lax.scan
    ↓
单个复用的 XLA executable
```

## 4. JAX-first 约束

### 4.1 静态层

下列内容在 `jax.jit` 前确定：

- topology：meta / standard；
- core：LRU / RTU；
- credit：RFLO / TBPTT(1)；
- objective、trace、target、update、normalization；
- num_envs；
- input/hidden/output/action dimensions；
- 参数域布局；
- RNG 路径；
- train epoch vector-step 数；
- evaluation 最大 vector-step 数和结果 buffer 容量；
- evaluation normalization 的重置/更新策略；
- evaluation protocol 与 legacy compatibility semantics；
- metric tree 的 key、shape 和 dtype。

静态配置对象使用 frozen/hashable dataclass、Flax Module 或闭包，不包含可变运行时
array，也不在 trace/train 时修改 Python 属性。环境不可变模型 array 优先作为显式
`env_params` 传入；受控 closed-over constant 是允许的例外。

Optimizer/OBGD transform、参数 path label 和 update closure 必须在 program 构建期
产生。不得沿用当前 RTRRL 在 jitted `init()` 中写入 `self.optimizer` 的做法。若
transform 依赖参数树结构，builder 先用 `jax.eval_shape(init_params_fn, ...)` 获得
抽象树，再构造 transform；抽象树和 transform 均不进入 State。

### 4.2 动态层

下列内容进入固定结构 JAX pytree：

- params；
- carry；
- credit/sensitivity；
- traces/emphasis；
- target/slow params；
- optimizer/OBGD state；
- normalization statistics；
- timestep/env state；
- counters；
- 固定结构 metric arrays。

### 4.3 禁止项

JIT/scan 内禁止：

- 查 registry；
- 解析 config/type 字符串；
- Python `if optimizer == ...`；
- 修改捕获的 agent/self；
- 动态 append/list/dict key；
- Aim、文件、S3 和网络 I/O；
- 根据数据创建新模块或参数；
- 返回长度或 pytree 结构动态变化的数据。

若抽象会引入动态 shape、host callback、每步 Python 分派、额外重编译或阻碍 fusion，
保留具体 kernel 中的重复实现。

## 5. 配置模型

### 5.1 顶层

```yaml
experiment:
  seed: 3
  total_env_steps: 2000000
  steps_per_epoch: 100000

algorithm: ...
environment: ...
evaluation: ...
recorder: ...
```

配置通过带 `type` 判别字段的严格联合类型解析。统一入口不等于巨型 dataclass；每个
分支只接受自己的字段，未知字段报错。

### 5.2 算法示例

```yaml
algorithm:
  topology:
    type: meta
    feedback: [action, reward]

  recurrent:
    type: lru
    hidden_dim: 128
    output_dim: 128
    activation: silu

  credit:
    type: rflo

  objective:
    type: rtrrl
    gamma: 0.95
    eta_pi: 0.38
    eta_f: 0.5
    entropy_rate: 3.0e-5
    logprob_reduction: mean
    prediction:
      enabled: false
      coefficient: 1.0

  trace:
    type: three_trace_emphasis
    lambda_actor: 0.97
    lambda_critic: 0.90
    lambda_recurrent: 0.945
    timing: incoming

  target:
    type: slow_subtree
    subtree: torso
    tau: 0.1

  update:
    type: grouped_adam
    actor_lr: 3.0e-5
    critic_lr: 3.0e-5
    recurrent_lr: 2.0e-6
    recurrent_clip: 1.0
    freeze_paths: []
    b1: 0.9
    b2: 0.999
    eps: 1.0e-8
```

### 5.3 环境与 recorder

```yaml
environment:
  type: brax
  name: hopper
  backend: spring
  mode: P
  num_envs: 1
  wrappers:
    episode_statistics: true
  action_transform:
    type: clip
    low: -1.0
    high: 1.0
```

```yaml
recorder:
  type: aim
  repo: aim://172.31.62.192:53800
  experiment_name: masked-hopper
  run_name: rtrrl-lru-h128
  description: LRU capacity under partial observation
  tags: [rtrrl, lru, masked, h128]
  properties:
    family: online-recurrent-ac
  step_window_env_steps: 10000
  log_system_params: true
  append_hash_to_name: true
```

```yaml
evaluation:
  protocol: fixed_vector_steps
  num_steps: 1000
  deterministic: true
  normalization:
    reset_on_start: true
    update_during_eval: true
```

`normalization.reset_on_start` 与 `update_during_eval` 是两个独立、构建期固定的布尔值：

- `true, true`：测试归一化统计冷启动并在线更新；这是当前 RTRRL/StreamAC legacy 行为；
- `false, false`：复制训练统计并冻结；
- `false, true`：复制训练统计并继续在线更新；
- `true, false`：冻结空统计，构建时拒绝。

normalization 未启用时 resolved spec 将二者归一为 `disabled`。Aim 必须记录最终 resolved
组合；不同组合属于不同 evaluation protocol，不能作为同一指标口径静默混合。
字段缺省时使用当前领域惯例 `reset_on_start=true, update_during_eval=true`；
evaluation protocol 缺省为 `fixed_vector_steps`。其他行为必须显式配置。

### 5.4 解析

Hydra/OmegaConf 只负责：

- preset 展开；
- YAML 合并；
- CLI/HPO dotted override；
- 基础类型解析。

随后转换为 frozen dataclass。Hydra/OmegaConf 对象不进入 AgentProgram 或 State。

构建使用显式、穷尽 factory：

```text
build_environment()
build_meta_program() / build_standard_program()
make_credit_forward()
make_td()
make_trace()
make_target()
make_update()
make_normalization()
build_recorder()
```

当前不使用 plugin registry。未来如出现外部插件需求，可在 factory 前增加受控映射，
但不改变 runtime。

## 6. 构建上下文

环境先于 program 构建，以产生静态 shape 信息：

```text
BuildContext
├── observation_spec
├── action_spec
├── num_envs
├── has_natural_episodes
├── global_steps_per_vector_step
└── termination/truncation capabilities
```

`input_dim: auto` 在此解析。MetaProgram 根据 feedback 配置计算输入宽度；
StandardProgram 使用同一个 core recipe 分别实例化 actor/critic core。相同 recipe
不表示参数共享，共享关系只由 program 类型决定。

语义验证在 program 构建前完成：

- core 支持所选 credit method；
- objective 支持 action distribution；
- trace/update 域匹配 program 参数布局；
- target subtree 存在；
- normalization 不重复启用；
- evaluation normalization 组合有效；
- total/epoch env steps 可转换为整数 vector-step 数；
- evaluation buffer shape 可静态确定。

## 7. AgentProgram

```python
@dataclass(frozen=True)
class AgentProgram:
    init_fn: Callable
    train_epoch_fn: Callable
    evaluate_fn: Callable
    state_schema: StateSchema
    metric_schema: MetricSchema
```

`AgentProgram` 只在 JIT 外保存。实际调用：

```python
program = build_program(resolved_spec, env_adapter)

init = jax.jit(program.init_fn)
train_epoch = jax.jit(program.train_epoch_fn)
evaluate = jax.jit(program.evaluate_fn)
```

`steps_per_epoch` 已闭包固定；不在每个 epoch 重新传 static argument。一个实验的每个
epoch 复用同一 executable。

`AgentProgram` 本身只保存在 host 侧，不作为 `jax.jit` static argument，不放入任何
State。callable field、Module、optimizer transform、schema 和环境构建对象都不得成为
scan carry。

## 8. 两个具体 Program

### 8.1 MetaProgram

参数布局：

```text
feature
torso
actor_head
critic_head
optional auxiliary heads
```

状态：

```text
MetaState
├── params
├── carry
├── credit
├── traces
├── target
├── update
├── normalization
├── timestep
├── env_state
└── counters
```

特点：

- 单 feature/torso/carry/credit；
- actor/critic heads 共享 recurrent representation；
- actor、critic、recurrent、auxiliary、frozen 参数域由静态 path rule 定义；
- RTRRL 与 RTRRL-OBGD 都由该 program 生成。

### 8.2 StandardProgram

状态：

```text
StandardState
├── actor: NetworkState
│   ├── params
│   ├── carry
│   ├── credit
│   ├── traces
│   └── update
├── critic: NetworkState
│   ├── params
│   ├── carry
│   ├── credit
│   ├── traces
│   └── update
├── normalization
├── timestep
├── env_state
└── counters
```

特点：

- actor/critic 完全独立；
- core recipe 调用两次得到不同参数；
- actor/critic update domain 独立；
- StreamACRtrl 由该 program 生成。

两个 program 分别实现具体 `step_fn`。只抽取确定不会改变数据流的 pure kernel；
不通过一个万能 step 模板消除所有重复。

## 9. 统一 transition 语义

禁止用一个模糊 `done` 同时表达 reset、termination 和 bootstrap：

```text
reset_before_t
terminated_after_t
truncated_after_t
bootstrap_discount_t
reset_before_next
termination_reason_code
```

`termination_reason_code` 是固定 dtype 整数 enum；字符串仅在 JIT 外解码。

环境 step 返回完整：

```text
next_observation
reward
terminated_after_t
truncated_after_t
bootstrap_discount_t
reset_before_next
termination_reason_code
env_info
```

Transition 必须保存 next observation。

## 10. 动作语义

```text
ActionDecision
├── sampled_action
├── logprob_action
├── env_action
├── bootstrap_feedback_action
└── persisted_feedback_action
```

Program 产生分布和 sampled action；JAXEnvAdapter 的静态 ActionTransform 产生其余
表示。log-prob 始终针对 `logprob_action`。Program 还显式产生
`bootstrap_feedback_reward` 和 `persisted_feedback_reward`。

Legacy RTRRL：

- `logprob_action = sampled_action`；
- `env_action` 可 clip；
- bootstrap 使用真实 `env_action` 与当前 `next_reward`；
- 持久化 feedback 使用 `env_action/next_reward`，在 `next_done` 时归零。

Legacy StreamAC：

- `logprob_action = sampled_action`；
- 算法内 `env_action = sampled_action`；若 wrapper clip，不反写 feedback；
- bootstrap 使用 sampled action 与当前 `next_reward`；
- 持久化 feedback 在 `next_done` 时归零。

这些规则由 topology preset 固定，不能由一个含糊的“feedback action”默认值推断。

## 11. Pure Kernel 工厂

Factory 在 Python 阶段返回固定签名函数；返回函数进入 program closure。

### 11.1 Credit

```text
credit_init(core, batch_shape) -> credit_state
credit_forward(params, inputs, carry, credit_state, reset_mask, rngs)
    -> output, next_carry, next_credit
```

RFLO 与 TBPTT(1) 返回不同 closure。RNG 数量和 mutable collection 在 closure 中固定。
`credit_forward` 必须调用 core 自己的 `local_jacobian` 与
`initialize_sensitivity`；LRU/RTU 的 exact local-Jacobian 递推不得被通用近似替换。

每个 program 还构造参数化的 `differentiation_forward(params)` closure。它必须在
`jax.jacobian` 内部重新执行 credit forward，使 phantom 对 `params` 的依赖保留。
不能把 acting forward 的普通输出直接传给 objective 后再对 params 求 Jacobian。

每个 transition 有三个不同视图：

1. acting forward：从 carried carry/credit 出发，结果写回；
2. bootstrap forward：从 acting 后、stop-gradient 的 carry/credit 出发，结果丢弃；
3. differentiation forward：从 acting 前、stop-gradient 的 carry/credit 重跑，
   只为 objective Jacobian，结果丢弃。

RTU legacy 的单步顺序固定为：
`phantom(S_old) → inject(stop_gradient(carry)) → reset carry → reset S →
local_jacobian → S_new`。LRU 使用其现有 Memoroid `local_jacobian` 顺序。reset 时点属于
credit contract，不允许 program 自行重排。

### 11.2 TD

```text
target = reward + bootstrap_discount * next_value
delta = target - value
```

TD kernel 不访问网络、参数、trace 或 optimizer。

### 11.3 Objective

返回固定参数域 tree：

```text
traced_directions_by_domain
direct_directions_by_domain
objective_metrics
```

RTRRL：

- actor traced：`eta_pi * logprob_scale * log_prob`；
- critic traced：`value`；
- feature/torso traced：同一 shared objective 的 Jacobian，更新时乘 `eta_f`；
- entropy direct：actor + feature + torso，不乘 `eta_f`；
- prediction direct：prediction head + feature + torso，不乘 `eta_f`；
- prediction head 使用 TD/heads Adam domain；
- logprob/entropy 的 sum/mean 由静态 `logprob_reduction` 固定。

StreamAC：

- actor traced：
  `log_prob + entropy_coefficient * sign(stop_gradient(delta)) * entropy`；
- critic traced：value；
- 保持后续乘 TD 后的历史熵缩放语义。

### 11.4 Trace

```text
trace_update(trace_state, gradients, boundary)
    -> carried_traces, update_traces, next_emphasis
```

closure 固定：

- domain lambdas；
- accumulate/dutch；
- emphasis；
- reset 边界；
- incoming/fresh timing。

Legacy 公式与 preset 必须显式保留：

```text
RTRRL:
z_new = gamma * lambda_domain * (1 - terminated_after_t) * z_old
        + emphasis_t * gradient_t
carried_trace = z_new
update_trace = z_new if update_trace_before_td else z_old
I_next = gamma * I * (1 - terminated_after_t) + terminated_after_t

StreamAC:
z_new = gamma * trace_lambda * (1 - reset_before_t) * z_old + gradient_t
carried_trace = update_trace = z_new
```

因此 RTRRL 使用 transition 后边界，StreamAC 使用当前 timestep 的 reset 边界；不能用
一个统一 `done` 替代。legacy 字段映射为
`update_trace_before_td=True → fresh`、`False → incoming`。

### 11.5 Target

Target closure 明确返回：

```text
acting_params
bootstrap_params
differentiation_params
update_destination
gradient_to_destination_map
```

并提供纯函数 `target_update(online_params, target_state)`。

RTRRL `SlowSubtree("torso")` 的 legacy 视图固定为：

- acting/bootstrap/differentiation：fast feature/heads + slow torso；
- update destination：fast params；
- 参数更新后对 fast torso 做 Polyak，供下一 transition 使用；
- 已累积 sensitivity 不因 fast/slow torso 更新而重算。

StreamAC bootstrap 使用更新前 stopped critic params、acting 后 stopped carry/credit；
bootstrap forward 的 next carry/credit 丢弃。

### 11.6 Update

```text
update_init(params) -> update_state
update_apply(context, params, update_state)
    -> ascent_increments, next_update_state, update_metrics
```

Grouped Adam 与 Domain OBGD 返回不同 closure/state。

OBGD：

- bound 按 topology preset 的 optimizer domain 独立；
- 每个 env 先计算有效步长，再聚合；
- Standard StreamAC legacy 仅有 `actor_whole_tree`、`critic_whole_tree` 两个 domain；
  feature/core/head 不再拆分；
- Meta RTRRL-OBGD 才单独决定 actor/critic/recurrent domain，不能继承 Standard 规则。

Legacy adaptive OBGD 的二阶矩、bias correction、`delta_bar=max(|delta|,1)`、整树 bound
求和、逐 env 有效步长和 env-axis mean 的运算顺序必须原样保留。`adaptive=false` 时仍
更新二阶矩 state，以保持现有 state 语义。

RTRRL-OBGD 的 direct direction 处理公式在加入该组合时单独确定；不把 StreamAC
trace 内熵语义隐式带入。

### 11.7 Normalization

Normalization 是唯一所有者：

```text
normalize_train(value, state) -> value, next_state
normalize_eval(value, eval_state, update_during_eval)
    -> value, next_eval_state
```

新路径不再同时安装 observation/reward normalization wrapper。evaluation state 由
`reset_on_start` 决定从初始统计或训练统计产生；是否递推由 `update_during_eval`
决定。两者均在 JIT 前固定。

Legacy exact normalization 保留：

- observation Welford 先吸收当前 observation，再标准化当前 observation；
- reward 统计基于 discounted return `G`，normalizer gamma 默认 0.99；
- 统计按 vector env 各自维护；
- 训练跨自然 episode 继承 mean/M2/count，reward `G` 在 done 时清零；
- evaluation 使用 `reset_on_start=true, update_during_eval=true`。

Episode statistics 始终观察原始 reward，不能因 normalization protocol 改变评估回报
口径。

## 12. Meta step 顺序

MetaProgram 的 concrete step：

1. 读取 timestep 和 reset mask。
2. normalization + feedback input。
3. target parameter views。
4. shared acting core/credit forward，写回 carry/credit。
5. actor/critic heads。
6. sample + ActionTransform。
7. environment step。
8. 从 acting 后 stopped carry/credit 做 bootstrap forward，结果状态丢弃。
9. TD。
10. 从 acting 前 stopped carry/credit，在 Jacobian closure 内做 differentiation forward。
11. three-trace update。
12. Adam 或 OBGD update closure。
13. slow target update。
14. normalization/counters/timestep。
15. 固定结构 metrics。

## 13. Standard step 顺序

StandardProgram 的 concrete step：

1. 读取 timestep 和 reset mask。
2. normalization/input。
3. actor core/credit forward。
4. critic core/credit forward。
5. sample + ActionTransform。
6. environment step。
7. 从 acting 后 stopped critic carry/credit 做 bootstrap forward，结果状态丢弃。
8. TD。
9. 从 acting 前 stopped carry/credit，在各自 Jacobian closure 内重跑 forward。
10. actor/critic trace updates。
11. 两个独立 update domain。
12. normalization/counters/timestep。
13. 固定结构 metrics。

## 14. Metric 与 episode 输出

scan 返回固定 shape `StepMetrics`：

```text
[vector_steps, num_envs, ...]
```

至少包含：

- TD/value/entropy/update 等标量；
- episode_done mask；
- episode return/length/index；
- termination reason code；
- global env-step offset。

不返回动态 episode 列表。JIT 外使用 mask 提取 EpisodeSummary。

Aim 接收：

### StepWindowSummary

始终存在，按 global environment transitions 对齐：

- transition/update count；
- mean/median/std/quantiles/max_abs；
- TD/value/entropy/gradient/update/parameter/trace 摘要；
- SPS/system 摘要。

### EpisodeSummary

仅自然 episode 环境：

- env/episode index；
- start/end global env step；
- return/length；
- reward 分布；
- termination/truncation reason；
- episode 内指定诊断摘要。

### EvalSummary

固定 global env step 触发，记录固定 evaluation protocol 的统计。

Recorder 不进入 AgentProgram、State 或 JAXPR。

Legacy recorder compatibility 由 `MetricSchema` 明确列出旧 key、mean/max-abs reduction、
global step 单位和 NaN 处理。新指标可以并存，但不能静默改名或改变聚合口径。

## 15. Evaluation

EvaluationProgram 与训练 program 共用参数/forward kernel，但使用独立固定状态和控制流：

- 参数、target 和 optimizer 永不在 evaluation 中更新；
- recurrent carry 始终是 evaluation-local state，并在 episode 内推进；
- legacy exact 模式仍通过 `local_jacobian` 推进 evaluation-local sensitivity，但不产生
  trace、optimizer 或参数更新；
- normalization statistics 是否重置/更新由两个显式布尔参数决定；
- 固定 reset seeds；
- `protocol=fixed_vector_steps` 保留 legacy 行为；
- `protocol=fixed_completed_episodes` 提供按完整 episode 的新协议；
- 固定最大 vector-step；
- deterministic 或显式 stochastic protocol；
- 预分配 `[num_envs, num_eval_episodes]` 结果 buffer；
- `lax.while_loop` 或固定 scan，carry shape 不变。

EvalSummary 记录请求/实际 episode、实际 transitions、return/length 的
mean/median/std/quantiles。

`lox.spool` 只用于固定 scan 且 schema 已知的路径。evaluation while-loop 内不调用
spool、recorder 或 host callback；evaluation kernel 返回固定 buffer + valid mask，由
JIT 外聚合。legacy fixed-vector-step 使用固定 scan，并保持当前 RNG split 和 reset
顺序。

## 16. Environment 与 Runtime

### JAXEnvAdapter

进入 Program closure：

- reset/step；
- space/static shape；
- vectorization；
- ActionTransform；
- transition/termination 语义；
- `has_natural_episodes`。

环境对象若持有 JAX array（例如 Brax system/params），array 作为显式 `env_params`、
State leaf 或受控 closed-over constant 处理，不作为 Python static argument 参与 hash。
同一 resolved spec 不得在 epoch 间重建 adapter 或 closure。

### ExperimentRuntime

JIT 外：

- 加载 resolved config；
- 构建 env/program/recorder；
- 调用一次 JIT 编译；
- 循环复用 `train_epoch`；
- 触发 evaluation；
- 聚合 metrics；
- 写 Aim；
- 生命周期与外部 I/O。

## 17. 编译与缓存

每个实验构建后：

```python
init = jax.jit(program.init_fn)
train_epoch = jax.jit(program.train_epoch_fn)
evaluate = jax.jit(program.evaluate_fn)
```

epoch scan 长度闭包固定。不得每 epoch 重建 program、closure 或 spooled function。

不同 hidden size、topology、optimizer 或 steps-per-epoch 可产生不同 executable；
这是预期行为。相同 resolved spec 和 shape 应命中持久化 compilation cache。

## 18. Preset

RTRRL：

```text
MetaProgram
+ raw obs/previous action/reward
+ LRU + SiLU
+ RFLO
+ RTRRL objective
+ three traces + emphasis
+ TD0
+ slow torso
+ grouped Adam
+ topology-specific action/feedback semantics
+ legacy exact evaluation/normalization protocol
```

StreamAC-RTU：

```text
StandardProgram
+ independent actor/critic RTU
+ exact RTU local-Jacobian RTRL credit
+ StreamAC objective
+ actor/critic traces
+ TD0
+ no target
+ actor-whole-tree / critic-whole-tree OBGD
+ legacy exact evaluation/normalization protocol
```

RTRRL-OBGD：

```text
RTRRL MetaProgram
- grouped Adam closure/state
+ domain OBGD closure/state
```

## 19. Legacy 兼容

旧 config 先翻译成新 tagged spec：

```text
agent_type=rtu_rtrl  → topology=standard, core=rtu, credit=rflo
agent_type=lru_rtrl  → topology=standard, core=lru, credit=rflo
agent_type=rtu_tbptt → topology=standard, core=rtu, credit=tbptt1
```

旧 `build_rtrrl_agent`、`build_stream_ac_agent` 成为 constructor wrapper，最终调用
相应 program factory。旧字段与显式新字段冲突时失败，不静默覆盖。

resolved config 完整写入 Aim；历史参数名/HPO filter 通过兼容 metadata 保留。

Legacy translation 还必须显式保留：

- algorithm gamma、feature encoder recipe、core output width 和 activation；
- Gaussian 参数化、action clipping、logprob reduction；
- RTRRL `update_trace_before_td`、prediction、freeze gamma、完整 Adam 参数；
- StreamAC exact RTRL、整树 OBGD 和 adaptive state；
- normalization wrapper 顺序、reward-normalizer gamma；
- train floor-division、evaluation step 单位和 RNG split tree；
- `evaluation.normalization.reset_on_start=true`；
- `evaluation.normalization.update_during_eval=true`；
- `evaluation.protocol=fixed_vector_steps`。

严格 env-step 整除、冻结训练统计和 fixed-completed-episodes 只属于显式新配置，不得由
legacy wrapper 默认启用。

## 20. 验收不变量

1. RTRRL/StreamAC 旧 preset 在同一 JAX/device/seed 下逐叶严格容差数值等价。
2. init/train/evaluate 均可 JIT。
3. epoch 间不重新 trace/compile。
4. scan carry pytree、shape 和 dtype 固定。
5. JAXPR/HLO 不含 recorder、配置解析或 host I/O。
6. MetaState/StandardState 不含另一 topology 的 optional 字段。
7. Adam/OBGD 拥有各自具体 update state。
8. OBGD domain bound 相互独立。
9. global env steps 与 vector iterations 单位明确。
10. StepWindowSummary 始终存在；自然 episode 环境额外产生 EpisodeSummary。
11. legacy YAML 可构建并运行。
12. 重构后 SPS 不出现无法解释的显著退化。
13. Golden tests 分别覆盖：
    - init params/carry/sensitivity/trace/update/normalization 全部 leaves；
    - acting action/logprob/value 与 RNG split；
    - bootstrap value、TD 和被丢弃的 bootstrap state；
    - differentiation phantom/Jacobian；
    - RTRRL incoming/fresh trace、emphasis、direct gradient、slow torso；
    - StreamAC fresh trace、整树 OBGD、adaptive on/off；
    - terminal/nonterminal、离散/连续动作和 feedback；
    - 三种有效 evaluation normalization 组合；
    - legacy fixed-vector-step evaluation、metric key 和 reduction。
14. 短轨迹逐步 comparison 不产生超过单步浮点容差累计预期的漂移。

## 21. 展望

### Rerun

Aim 保持跨 run/HPO 系统。未来仅对自然 episode 环境低频保存完整 episode，转换为
`.rrd` 并通过 Aim 深链接打开远程 Rerun Web Viewer。continuing task 默认不构造
伪 episode。

### FailureEvent

当前失败仍由 Batch、CloudWatch、Aim 未完成状态和指标中断表达。失败分类和消费需求
明确后，再定义结构化事件。

### Plugin registry

当前显式 factory 足够。只有出现仓库外第三方组件加载需求时，才在构建层增加受控
registry；训练 runtime 不变。
