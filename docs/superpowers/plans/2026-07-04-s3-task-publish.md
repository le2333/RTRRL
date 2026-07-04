# 片1:S3 任务发布 + 顺序执行 — 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **注:** 本计划当前用于**派生 GitHub issue**,本轮不实现。Issue 见任务末尾映射。

**Goal:** 把配置注入从 base64 换成 S3,引入"任务"概念(一份 manifest + 一组 config,一个 Batch 作业顺序跑完),三模块可独立替换。

**Architecture:** 跳板机 Python CLI(`rtrrl-publish`,boto3,独立 uv 项目)上传 config + manifest 到 S3 并提交 Batch;容器 entrypoint 用 aws CLI 从 S3 拉 manifest + configs,`runner.py` 顺序跑。配置路径 `s3://bucket/configs/<experiment>/<NNN>.yaml` 镜像 Aim experiment→run。Spec:`docs/superpowers/specs/2026-07-04-execution-infra-s3-resume-design.md`。

**Tech Stack:** Python 3.12、boto3(跳板机)、aws CLI v2(容器,已在镜像)、argparse、pytest、uv、AWS Batch、S3。

---

## 文件结构

| 文件 | 责任 | 动作 |
|---|---|---|
| `infra/rtrrl_publish/pyproject.toml` | 独立 uv 项目(boto3) | 新建 |
| `infra/rtrrl_publish/src/rtrrl_publish/manifest.py` | Manifest 数据类、序列化、task_id 生成 | 新建 |
| `infra/rtrrl_publish/src/rtrrl_publish/s3relay.py` | 上传 config/manifest、列 task、分配 run 编号 | 新建 |
| `infra/rtrrl_publish/src/rtrrl_publish/cli.py` | `publish`/`list` 子命令 + boto3 Batch 提交 | 新建 |
| `infra/rtrrl_publish/tests/test_manifest.py` | manifest 单测 | 新建 |
| `infra/rtrrl_publish/tests/test_s3relay.py` | s3relay 单测(moto 或 stub) | 新建 |
| `infra/runner.py` | 容器侧顺序执行(读本地 manifest + configs) | 新建 |
| `infra/docker/entrypoint.sh` | 从 S3 拉 manifest+configs,调 runner.py | 改 |
| `infra/env.sh` | S3 configs/tasks 前缀 | 改 |
| `AGENTS.md` | 规则 1、6 修订 | 改 |
| `infra/README.md` | 片1 用法章节 | 改 |

容器侧无 boto3:`entrypoint.sh` 用 aws CLI 拉,`runner.py` 纯本地。跳板机侧 boto3 封在 `rtrrl_publish` 独立项目(与 `control/hpo` 同模式)。

---

## Task 1: env.sh S3 前缀与 task_id 约定

**Files:**
- Modify: `infra/env.sh`

- [ ] 在 `infra/env.sh` 增加配置(沿用现有 `S3_BUCKET`):

```bash
# ---- 片1: S3 任务发布 --------------------------------------------------------
export S3_CONFIGS_PREFIX="${S3_CONFIGS_PREFIX:-configs}"   # s3://$S3_BUCKET/configs/<experiment>/<NNN>.yaml
export S3_TASKS_PREFIX="${S3_TASKS_PREFIX:-tasks}"         # s3://$S3_BUCKET/tasks/<task_id>/manifest.json
```

- [ ] task_id 约定(文档化在 README,不强制代码):`<experiment>_<YYYYMMDD>_<NNN>`,如 `ppo-hc_20260704_001`。`-` 允许(Batch job-name 接受)。

- [ ] Commit:`chore(env): add S3 configs/tasks prefixes`

---

## Task 2 (M1): rtrrl_publish 项目骨架 + manifest 模块

**Files:**
- Create: `infra/rtrrl_publish/pyproject.toml`
- Create: `infra/rtrrl_publish/src/rtrrl_publish/__init__.py`
- Create: `infra/rtrrl_publish/src/rtrrl_publish/manifest.py`
- Create: `infra/rtrrl_publish/tests/test_manifest.py`

- [ ] `pyproject.toml`(独立 uv 项目,deps:boto3、moto[dev]):

```toml
[project]
name = "rtrrl-publish"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = ["boto3>=1.34"]
[project.scripts]
rtrrl-publish = "rtrrl_publish.cli:main"
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
[tool.hatch.build.targets.wheel]
packages = ["src/rtrrl_publish"]
[dependency-groups]
dev = ["pytest", "moto"]
```

- [ ] `manifest.py`:

```python
from __future__ import annotations
import json, time
from dataclasses import dataclass, asdict, field

@dataclass
class Manifest:
    task_id: str
    experiment: str
    entry: str
    logging: str
    runs: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))

    @classmethod
    def from_json(cls, s: str) -> "Manifest":
        return cls(**json.loads(s))

def make_task_id(experiment: str, seq: int) -> str:
    return f"{experiment}_{time.strftime('%Y%m%d')}_{seq:03d}"
```

- [ ] `test_manifest.py`(round-trip + task_id 格式):

```python
from rtrrl_publish.manifest import Manifest, make_task_id

def test_manifest_roundtrip():
    m = Manifest(task_id="ppo-hc_20260704_001", experiment="ppo-hc",
                 entry="ppo_baseline.py", logging="aim+wandb", runs=["013", "014"])
    assert Manifest.from_json(m.to_json()) == m

def test_task_id_format():
    assert make_task_id("ppo-hc", 1) == f"ppo-hc_{time.strftime('%Y%m%d')}_001"
```

- [ ] Run:`uv run --project infra/rtrrl_publish pytest` → PASS
- [ ] Commit:`feat(rtrrl-publish): manifest module + tests`

---

## Task 3 (M1): s3relay 模块(上传 + 列表 + 编号分配)

**Files:**
- Create: `infra/rtrrl_publish/src/rtrrl_publish/s3relay.py`
- Create: `infra/rtrrl_publish/tests/test_s3relay.py`

- [ ] `s3relay.py`:

```python
from __future__ import annotations
import boto3
from pathlib import Path
from rtrrl_publish.manifest import Manifest

class S3Relay:
    def __init__(self, bucket: str, configs_prefix: str = "configs",
                 tasks_prefix: str = "tasks", s3=None):
        self.bucket = bucket
        self.cfg_prefix = configs_prefix
        self.tasks_prefix = tasks_prefix
        self.s3 = s3 or boto3.client("s3")

    def config_key(self, experiment: str, nnn: str) -> str:
        return f"{self.cfg_prefix}/{experiment}/{nnn}.yaml"

    def manifest_key(self, task_id: str) -> str:
        return f"{self.tasks_prefix}/{task_id}/manifest.json"

    def upload_config(self, experiment: str, nnn: str, local_path: Path) -> str:
        key = self.config_key(experiment, nnn)
        self.s3.put_object(Bucket=self.bucket, Key=key,
                           Body=local_path.read_bytes())
        return key

    def upload_manifest(self, manifest: Manifest) -> str:
        key = self.manifest_key(manifest.task_id)
        self.s3.put_object(Bucket=self.bucket, Key=key,
                           Body=manifest.to_json().encode())
        return key

    def next_run_number(self, experiment: str, start: int = 1) -> str:
        """扫 configs/<experiment>/ 取最大 NNN,+1;空则 start。"""
        prefix = f"{self.cfg_prefix}/{experiment}/"
        resp = self.s3.list_objects_v2(Bucket=self.bucket, Prefix=prefix)
        nums = []
        for o in resp.get("Contents", []):
            stem = Path(o["Key"]).stem
            if stem.isdigit():
                nums.append(int(stem))
        nxt = (max(nums) + 1) if nums else start
        return f"{nxt:03d}"

    def list_tasks(self) -> list[str]:
        resp = self.s3.list_objects_v2(Bucket=self.bucket,
                                       Prefix=f"{self.tasks_prefix}/",
                                       Delimiter="/")
        return [p["Prefix"].split("/")[-2] for p in resp.get("CommonPrefixes", [])]
```

- [ ] `test_s3relay.py`(用 moto mock S3):

```python
import moto, boto3
from pathlib import Path
from rtrrl_publish.s3relay import S3Relay
from rtrrl_publish.manifest import Manifest

@moto.mock_aws
def test_upload_and_next_number(tmp_path):
    s3 = boto3.client("s3"); s3.create_bucket(Bucket="b")
    r = S3Relay("b", s3=s3)
    cfg = tmp_path / "c.yaml"; cfg.write_text("x: 1")
    assert r.upload_config("ppo-hc", "013", cfg).endswith("configs/ppo-hc/013.yaml")
    assert r.next_run_number("ppo-hc") == "014"
    assert r.next_run_number("new") == "001"
```

- [ ] Run:`uv run --project infra/rtrrl_publish pytest` → PASS
- [ ] Commit:`feat(rtrrl-publish): s3relay upload/list/numbering`

---

## Task 4 (M3): cli publish + list

**Files:**
- Create: `infra/rtrrl_publish/src/rtrrl_publish/cli.py`

- [ ] `cli.py`:`publish` 分配编号(扫 S3 max+1,除非 `--run` 显式指定)→ 上传 configs + manifest → `aws batch submit-job` 提交一个作业(packed),注入 `TASK_ID`/`RUN_NAMES` 环境变量。

```python
def main(argv=None):
    import argparse, boto3, os
    p = argparse.ArgumentParser(prog="rtrrl-publish")
    sub = p.add_subparsers(dest="cmd", required=True)

    pub = sub.add_parser("publish")
    pub.add_argument("--experiment", required=True)
    pub.add_argument("--config", action="append", required=True, help="本地 config 路径,可多个")
    pub.add_argument("--run", action="append", help="显式 run 编号,与 --config 一一对应;缺省自动分配")
    pub.add_argument("--entry", default="rtrrl.py")
    pub.add_argument("--logging", default=os.environ.get("LOGGING", "aim"))
    pub.add_argument("--name", help="task_id;缺省 make_task_id(experiment, seq)")
    pub.add_argument("--dry-run", action="store_true")

    lst = sub.add_parser("list")
    args = p.parse_args(argv)
    # ... dispatch; publish 走 S3Relay + boto3.client("batch").submit-job
```

- [ ] `submit-job` 的 `container-overrides`:

```json
{
  "command": ["python", "infra/runner.py"],
  "environment": [
    {"name": "TASK_ID", "value": "<task_id>"},
    {"name": "RUN_NAMES", "value": "013,014"}
  ]
}
```

- [ ] publish 前**打印将上传的 config/编号 + manifest + command,等用户确认**(遵守 AGENTS.md:需显式确认才提交);`--dry-run` 只打印不提交。
- [ ] 手测:`uv run --project infra/rtrrl_publish rtrrl-publish publish --experiment ppo-hc --config config/ppo_halfcheetah_official.yml --dry-run`
- [ ] Commit:`feat(rtrrl-publish): publish/list CLI`

---

## Task 5 (M2): infra/runner.py 容器侧顺序执行

**Files:**
- Create: `infra/runner.py`

- [ ] `runner.py`:读本地 manifest + 本地 configs 目录,对 `RUN_NAMES` 每个 NNN 调训练脚本。**纯本地,无 boto3**。

```python
#!/usr/bin/env python3
"""容器内顺序执行一个 task 的若干 run(从本地 manifest + configs)。"""
from __future__ import annotations
import argparse, subprocess, sys, json
from pathlib import Path

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--configs-dir", required=True)
    p.add_argument("--runs", required=True, help="逗号分隔 run 编号")
    p.add_argument("--log_repo", default=None)
    args = p.parse_args(argv)

    m = json.loads(Path(args.manifest).read_text())
    runs = [r.strip() for r in args.runs.split(",") if r.strip()]
    failures = []
    for nnn in runs:
        cfg = Path(args.configs_dir) / f"{nnn}.yaml"
        cmd = [sys.executable, m["entry"], "--config_path", str(cfg),
               "--logging", m["logging"]]
        if args.log_repo and "aim" in m["logging"]:
            cmd += ["--log_repo", args.log_repo]
        print(f"[runner] {nnn}: {' '.join(cmd)}", flush=True)
        rc = subprocess.run(cmd, check=False).returncode
        if rc != 0:
            failures.append((nnn, rc))
    if failures:
        print("[runner] failures:", failures, flush=True)
        return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] 本地手测(不跑真训练,用 `--entry /bin/true` 或 mock):manifest + 2 个 config,确认顺序调用、失败收集。
- [ ] Commit:`feat(infra): container runner.py sequential execution`

---

## Task 6: entrypoint.sh 从 S3 拉 + 调 runner.py

**Files:**
- Modify: `infra/docker/entrypoint.sh`

- [ ] 新分支:若 `TASK_ID` 存在,走片1 路径;否则保留旧 `CONFIG_B64` 路径(向后兼容)。

```bash
set -uo pipefail
mkdir -p /tmp/jax_cache

if [ -n "${TASK_ID:-}" ]; then
  # 片1: S3 任务发布
  : "${RUN_NAMES:?RUN_NAMES required when TASK_ID set}"
  : "${S3_BUCKET:?S3_BUCKET required}"
  TASK_DIR=/tmp/task; mkdir -p "$TASK_DIR/configs"
  aws s3 cp "s3://$S3_BUCKET/${S3_TASKS_PREFIX:-tasks}/$TASK_ID/manifest.json" \
    "$TASK_DIR/manifest.json" --region "${REGION}"
  EXPERIMENT=$(python3 -c "import json;print(json.load(open('$TASK_DIR/manifest.json'))['experiment'])")
  for NNN in ${RUN_NAMES//,/ }; do
    aws s3 cp "s3://$S3_BUCKET/${S3_CONFIGS_PREFIX:-configs}/$EXPERIMENT/$NNN.yaml" \
      "$TASK_DIR/configs/$NNN.yaml" --region "${REGION}"
  done
  exec python infra/runner.py --manifest "$TASK_DIR/manifest.json" \
    --configs-dir "$TASK_DIR/configs" --runs "$RUN_NAMES" "$@"
  exit 0
fi

# 旧路径: base64 注入(保留,向后兼容)
if [ -n "${CONFIG_B64:-}" ]; then
  echo "${CONFIG_B64}" | base64 -d > /tmp/run-config.yml
fi
exec "$@"
```

- [ ] 注:`S3_BUCKET`/`REGION`/`S3_*_PREFIX` 由 job 定义环境注入(在 `batch/create-batch.sh` 里加,或 publish 时通过 container-overrides environment 传)。本任务在 entrypoint 只读它们。
- [ ] Commit:`feat(infra): entrypoint pulls task from S3, calls runner.py`

---

## Task 7: Batch job 定义环境变量

**Files:**
- Modify: `infra/batch/create-batch.sh`

- [ ] 在 job 定义的 environment 注入 `S3_BUCKET`、`REGION`、`S3_CONFIGS_PREFIX`、`S3_TASKS_PREFIX`(从 `env.sh` 读),使 entrypoint 能用。WANDB/AIM 注入机制不变。
- [ ] Commit:`chore(batch): inject S3 env vars into job definition`

---

## Task 8: AGENTS.md 规则 1、6 修订

**Files:**
- Modify: `AGENTS.md`

- [ ] 规则1 改:"Run only from S3 configs(`s3://bucket/configs/<experiment>/<NNN>.yaml`)。手动基准可仍在 git `config/`,但提交到 Batch 的 run 必须先上传到 S3 `configs/`。"
- [ ] 规则6 改:"Use S3 configs to build configs。衍生时读 S3 真实 config 文件写新配置,不从记忆/Aim hparams 重构。" 精神保留。
- [ ] 规则2 保留(不改):不删配置;S3 `configs/` 不可变,衍生起新编号。
- [ ] Commit:`docs(agents): revise rules 1,6 for S3 config source`

---

## Task 9: infra/README.md 片1 章节

**Files:**
- Modify: `infra/README.md`

- [ ] 加"S3 任务发布(片1)"章节:`rtrrl-publish publish/list` 用法、S3 布局、与旧 `submit.sh` 的关系、向后兼容说明、`#10/#12` 命名依赖注记。
- [ ] Commit:`docs(infra): document片1 S3 task publish`

---

## Task 10: 端到端 smoke 测试(手动)

- [ ] 跳板机:`rtrrl-publish publish --experiment ppo-smoke --config config/ppo_smoke.yml --entry ppo_baseline.py --logging aim --name ppo-smoke_20260704_001`(确认后提交)。
- [ ] 看 Batch 作业启动 → entrypoint 从 S3 拉 manifest + config → runner.py 调 `ppo_baseline.py` → Aim 出点。
- [ ] 验证 S3 上 `configs/ppo-smoke/001.yaml` + `tasks/ppo-smoke_20260704_001/manifest.json` 存在。
- [ ] 验证旧 `submit.sh --config ...` 仍能跑(向后兼容未破坏)。
- [ ] 不 commit(手测)。

---

## 自检

- **Spec 覆盖**:§2 契约 → T1/T2/T3/T6;§3 M1 → T2/T3;§4 M2 → T5/T6;§5 M3 → T4;§6 替换兼容/AGENTS → T8;§7 后续不在本计划。覆盖完整。
- **占位符**:无 TBD;cli.py 的 dispatch 用 `...` 标注(实现细节,非占位——publish/list 主体已在上面展示)。
- **类型一致**:`Manifest` 字段 `task_id/experiment/entry/logging/runs` 在 s3relay/cli/runner 均一致;`config_key(experiment, nnn)`、`manifest_key(task_id)` 命名一致。

## Issue 映射(下一步写)

| Issue | 对应 Task |
|---|---|
| 片1-M1:S3 中继 + manifest | T1, T2, T3 |
| 片1-M3:跳板机 CLI publish/list | T4, T7 |
| 片1-M2:容器 runner + entrypoint S3 拉取 | T5, T6 |
| 片1:AGENTS.md 规则 1/6 修订 | T8 |
| 片1:infra/README 文档 | T9 |
| 片1:端到端 smoke | T10 |
