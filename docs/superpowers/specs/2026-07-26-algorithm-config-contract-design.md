# 算法配置契约设计

> 已被 `2026-07-30-configuration-surface-design.md` 取代,不再是现行设计。

范围:算法侧如何声明配置面,catalog 如何产生,实验配置如何覆盖。

不在本文范围:算法内部结构、测试锚定策略、旧代码清理。那是后续三个子项目。

## 1. 目标

- 配置只有一个来源,读声明就知道算法有哪些旋钮,不需要读实现。
- 算法容易替换和扩展:加一个算法就是加一个入口,不改契约。
- 搜什么、怎么搜完全由实验决定,契约不表达任何研究意见。

## 2. 算法侧声明

算法用类装饰器声明入口,每个参数一行:

```python
@algorithm_entry(command=["python", "-m", "rtrrl_shared"], metrics=["episode_return"])
@dataclass(frozen=True)
class Config:
    total_steps:     int   = param(default=1_000_000, bound=(1, None))
    learning_rate:   float = param(default=(1e-4, 1e-3), bound=(0, None), log=True)
    trace_lambda:    float = param(default=0.8,          bound=(0.0, 1.0))
    hidden_dim:      int   = param(default=128,          bound=(1, None))
    torso_gradient:  str   = param(default="both",
                                   bound=["both", "actor_only", "critic_only"])
    seed:            int   = param(default=0,            bound=(0, None))
```

`param()` 有两个语义不同的字段:

- **`default`** 决定实验不覆盖时的行为。标量表示固定取该值;二元组表示默认在该区间搜索;列表表示默认在这些取值间搜索。算法作者按参数的常用方式选择——学习率通常搜,所以默认给区间;随机种子通常固定,所以默认给标量。
- **`bound`** 是硬边界,只在实验覆盖时用于校验。数值参数用二元组,`None` 表示该侧无界;分类参数用允许取值的列表。实验请求越界时 preflight 直接拒绝。

`default` 必须落在 `bound` 内,由导出时检查。`log=True` 时 `bound` 的下界必须严格大于零,否则导出即失败——对数刻度无法表达零和负值。

装饰器是函数不是基类。去掉 `@algorithm_entry` 之后剩下的仍然是一个可用的 `dataclass`,算法照常运行,只是不能自动导出 catalog(可临时手写 JSON 兜底)。SDK 因此不会成为算法开发的阻塞点。

## 3. catalog 的产生

`build_catalog.py` 从声明导出,不手写。导出内容:入口名、`command`、`metrics`、每个参数的 `default` 与 `bound`。catalog 覆盖算法的**全部**配置面,不存在只在代码里、不在 catalog 里的参数。

catalog 写入镜像内的 `catalog.json`,同时编码进镜像标签供控制面读取。

`source_hash` 只对**入口自身的模块**求哈希,不含共享原语,不含依赖库,不做调用链分析。它回答的问题只有一个:这个算法本身变了没有。共享原语是稳定基础设施,不是被版本化的对象;依赖库的变化由镜像 digest 承担。

现状是对包内全部 `.py` 求和,导致改一个入口会让同镜像内其他入口的哈希一起变,必须改掉。

## 4. 实验配置的覆盖

实验 YAML 的 `space` 覆盖 catalog 的默认值,语义为逐键替换:

```yaml
space:
  torso_gradient: [actor_only]                                   # 钉死
  learning_rate: {type: float, low: 1.0e-6, high: 1.0e-2, log: true}   # 改范围
  # 未出现的参数使用 catalog 的 default
```

覆盖必须落在该参数的 `bound` 内,否则 preflight 拒绝并指出是哪个参数、越界在哪一侧。实验不能引入 catalog 未声明的参数,这一条现有实现已具备。

搜什么、钉什么、用 `tpe` 还是 `grid`,全部是实验设计,契约不干预也不给建议。

## 5. 运行时

控制面对解析后的空间调用 Optuna,每个参数都会得到取值(钉死的参数取值唯一),完整参数集写入 manifest。

算法一律从 manifest 取值:`params["learning_rate"]`。**禁止 Python 端默认兜底**,不允许 `params.get("x", 3)` 这类写法——manifest 必然携带全部参数,兜底只会制造第二个来源。缺键即报错。

## 6. 入口划分

一个入口对应一个参数面。判据是:**该变体是否改变参数面或 state 形状**。

- 改变的,独立入口。例:shared torso 与 independent actor/critic 的 state 布局不同;不同 recurrent backbone 的参数不同。
- 不改变的,是同一入口下的一个参数。例:`torso_gradient` 的三种取值,参数面与 state 形状完全相同。

这条判据同时消除了条件依赖:只有当某个配置选择改变别的参数是否有意义时才需要条件依赖,而这类选择按判据一律提升为独立入口,于是任何单个入口的空间内都不存在无效组合,契约无需支持条件参数。

## 7. 与现有实现的差距

| 项 | 现状 | 需要改动 |
| --- | --- | --- |
| catalog 来源 | `build_entry()` 手写字面量 | 从声明导出 |
| 参数边界 | 无,覆盖即完全替换 | 增加 `bound` 与 preflight 校验 |
| `source_hash` | 对包内全部 `.py` 求和 | 改为只对入口自身模块求和 |
| 声明原语 | 无 | SDK 新增 `algorithm_entry` 与 `param` |
| 算法侧兜底 | mock trainer 用 `params.get(k, v)` | 改为 `params[k]`,并加测试禁止该写法 |

`resolve_space` 的逐键覆盖语义、`total_steps` 保留参数、grid 采样器对离散取值的校验均已实现,不需改动。

新增 `bound` 改变了 catalog 结构,`CONTRACT_VERSION` 需要递增。旧镜像的 catalog 将不再被接受;现有镜像只有验收用的 mock trainer,重建成本可忽略。

镜像仍然按 profile 共用,不按入口拆分。digest 只表示"跑的是哪个镜像",算法是否变化由 `source_hash` 回答,两者职责不重叠。
