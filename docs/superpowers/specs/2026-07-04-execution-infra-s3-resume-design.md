# 片1:S3 任务发布 + 顺序执行

- 日期:2026-07-04
- 状态:设计待复核
- 范围:`streaming-rtrrl` 执行基础设施**第一片**;按模块横切,敏捷推进
- 不含:续练、parallel 模式、快照/状态记录、训练脚本改动(均留后续,见 §7)

## 1. 目标

把"配置注入"从 base64 换成 S3,并引入"任务"概念(一个任务 = 一份清单 + 一组 config,一个 Batch 作业顺序跑完)。三个可独立替换的组件,靠接口契约对接;替换任一组件,系统仍正常运行(老 `submit.sh` 作 fallback)。

## 2. 接口契约(三模块对接的核心,本轮定型)

### S3 key 布局

```
s3://<bucket>/configs/<experiment>/<NNN>.yaml   # 配置真源(S3),结构与 Aim 实验一致
s3://<bucket>/tasks/<task_id>/manifest.json      # 任务清单
```

- 配置路径**镜像 Aim 的 experiment→run 层级**:`<experiment>` = Aim 实验名(algorithm-environment,如 `ppo-hc`),`<NNN>` = 该实验内 run 编号(如 `013`)。
- 例:`configs/ppo-hc/013.yaml` ↔ Aim `ppo-hc` 实验的第 013 号 run。
- **配置真源在 S3 `configs/`,不在 git**。git `config/` 退居手工基准/模板。
- config 写入后**不可变**(AGENTS.md 规则2:不删、不覆盖);衍生 = 起新编号写新对象,不在原对象上改。
- 一个任务 = 一个实验(algo+env)的调参,故 task 内所有 run 共享同一 `<experiment>`。

### manifest.json

```json
{
  "task_id": "ppo-hc_20260704_001",
  "experiment": "ppo-hc",
  "entry": "ppo_baseline.py",
  "logging": "aim+wandb",
  "runs": ["013", "014"]
}
```

- `experiment`:Aim 实验名(本任务所有 run 共享)。`runs`:该实验内 run 编号有序列表 = 执行顺序。S3 key = `configs/<experiment>/<NNN>.yaml`。
- 无 override、无 run_name(脚本自动按 env 生成,Aim hash 作唯一编号)。
- **命名待对齐**:`<experiment>` 与 `<NNN>` 的具体规则依赖 issue #10/#12(Aim 架构重组 + 脚本自动生成实验 ID/编号);本片先采用 `<experiment>/<NNN>` 结构,具体名称待 #10/#12 落地对齐。HPO 是否产出 config 文件也 open。

### 容器环境变量

- `TASK_ID`:S3 任务 key(容器据此拉 manifest)。
- `RUN_NAMES`:该作业负责跑的 run 编号子集(packed=全部),对应 `configs/<experiment>/<NNN>.yaml`。`experiment` 从 manifest 读。
- 现有 `AIM_SERVER` / W&B 注入机制不变。

## 3. 模块 M1:S3 配置中继

- 职责:把 config 文件上传到 `configs/<experiment>/<NNN>.yaml`、manifest 上传到 `tasks/<task_id>/manifest.json`;容器侧按 `experiment`+`NNN` 从 `configs/` 拉。
- 替代现有 `CONFIG_B64` base64 注入。
- config 写入后不可变(规则2);衍生 = 起新编号写新对象。
- 独立可替换:换成别的存储(本地/其他对象存储)只需改本模块,契约不变。

## 4. 模块 M2:容器侧任务调度组件

- 新共享模块 `runner.py`(复用 `run_many.py` 顺序骨架)。
- 流程:读 `TASK_ID`/`RUN_NAMES` → 从 S3 拉 manifest(取 `experiment`+`entry`+`logging`)→ 对 `RUN_NAMES` 每个 `NNN` 拉 `configs/<experiment>/<NNN>.yaml` → 调 `python <entry> --config_path <下载的config> --logging <mode> [--log_repo <aim>]` → 全跑完退出。
- **不写状态/快照**(本轮);**不改训练脚本**。
- 入口 `entrypoint.sh` 改为:传 `TASK_ID`/`RUN_NAMES`,调 `runner.py`。

## 5. 模块 M3:跳板侧任务调度组件

- 新 Python CLI `rtrrl-publish`(boto3 直连,复用 `infra/env.sh` 资源名)。
- `publish --experiment <exp> --config <本地路径>... [--entry ...] [--logging ...]`:为本批 config 分配 run 编号(扫 S3 `configs/<exp>/` 取最大值+1,或显式指定)→ 上传到 `configs/<exp>/<NNN>.yaml` → 写 manifest → 提交一个 Batch 作业(packed)。
- `list`:列出 S3 上的 task。
- 编号分配细节可后续与 #12(脚本自动生成编号)协调;本片先由 publish 承担。
- 本轮**无 status/resume/clean**(留后续)。

## 6. 替换兼容性

- 老 `submit.sh` / `submit_many.sh` / `run_many.py` **不删**,作 fallback;`rtrrl-publish` 是上层新增。
- 任一模块替换实现,只要遵守 §2 契约,系统仍正常运行。
- `AGENTS.md` 实验规则需相应修订(本轮配合本片落地):
  - **规则1 改**:配置真源从 git `config/` 挪到 S3 `configs/`;"run only from config/" → "run only from S3 configs"。
  - **规则6 改**:把"on-disk configs"改成"in-S3 configs";精神保留——衍生仍读真实 config 文件(从 S3),不从记忆/Aim hparams 重构。
  - **规则2 保留**:不删配置;S3 `configs/` 不可变,衍生起新名。
  - 不变:publish 前需显式确认、不自动重提交。

## 7. 后续切片(留 issue,本轮不做)

- 片2:parallel 模式(每 run 一个作业)+ `status`/`clean` + etag。
- 片3:run 级续练(controller 用 Batch API 驱动,不依赖容器快照)。
- 片4:快照记录 + 状态管理(先实现接口,脚本里调用)。
- 片5:训练心跳(step)→ hang 检测;容器内调度器重试;`watch` 守护。
- 片6:epoch 级 checkpoint 续训;experiment_group 实验组建模。
- 片7(更后续):训练组件从脚本重构为框架,届时再定快照等接入。
- **依赖**:config 路径结构 `configs/<experiment>/<NNN>.yaml` 本片已定(镜像 Aim experiment→run);`<experiment>` 具体名称与 `<NNN>` 分配规则待 issue #10/#12 落地后对齐。HPO 是否用 config 文件亦待 #12 定。
