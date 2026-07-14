"""Jump-host HPO advisor/scheduler.

This is a control-plane tool: it reads Aim history, uses Optuna locally to
suggest the next batch of configs, and optionally submits those configs to AWS
Batch. Training workers remain simple and do not need Optuna installed.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import yaml
from optuna.distributions import (
    BaseDistribution,
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)
from optuna.trial import TrialState, create_trial


# This engine is project-agnostic. It lives in the shared infra repo:
#   trainer/infra/hpo/src/hpo_control/scheduler.py
# parents[3] == trainer/infra, so the shared submit.sh sits next to it.
SHARED_INFRA = Path(__file__).resolve().parents[3]
SUBMIT_SH = SHARED_INFRA / "submit.sh"

# The PROJECT being tuned owns its HPO data under <project>/hpo/{studies,runs,
# specs,snapshots}. These are set at runtime by set_project_root() from
# --project-root / $HPO_PROJECT_ROOT / (inferred from the spec/plan path), so all
# downstream functions can keep reading the module globals.
PROJECT_ROOT: Path = Path.cwd()
HPO_ROOT: Path = PROJECT_ROOT / "hpo"
STUDIES_DIR: Path = HPO_ROOT / "studies"
RUNS_DIR: Path = HPO_ROOT / "runs"


def set_project_root(project_root: Path) -> None:
    """Point the engine at a training project's HPO data dir (<project>/hpo)."""
    global PROJECT_ROOT, HPO_ROOT, STUDIES_DIR, RUNS_DIR
    PROJECT_ROOT = Path(project_root).resolve()
    HPO_ROOT = PROJECT_ROOT / "hpo"
    STUDIES_DIR = HPO_ROOT / "studies"
    RUNS_DIR = HPO_ROOT / "runs"


def rel(path: Path) -> Path:
    """Path relative to PROJECT_ROOT for display; fall back to the path itself."""
    try:
        return Path(path).resolve().relative_to(PROJECT_ROOT)
    except ValueError:
        return Path(path)


# ---- Target-metric registry (pluggable per project) -------------------------
# Objectives are computed from an Aim run by a named extractor. Built-ins cover
# the streaming-rtrrl objectives; a project can add its own without editing this
# engine by shipping <project>/hpo/targets.py that calls register_target(...).
_TARGET_EXTRACTORS: dict[str, Any] = {}


def register_target(name: str, fn: Any) -> None:
    """Register a target extractor fn(run) -> float | None under `name`."""
    _TARGET_EXTRACTORS[name] = fn


def load_project_targets() -> None:
    """Import <project>/hpo/targets.py (if any) so it can register_target()."""
    targets_py = HPO_ROOT / "targets.py"
    if not targets_py.is_file():
        return
    import importlib.util

    spec = importlib.util.spec_from_file_location("hpo_project_targets", targets_py)
    if spec is None or spec.loader is None:
        return
    module = importlib.util.module_from_spec(spec)
    # Expose register_target to the plugin without it importing this module by path.
    module.register_target = register_target  # type: ignore[attr-defined]
    spec.loader.exec_module(module)


@dataclass
class ImportedRun:
    aim_hash: str
    name: str
    target: str
    value: float
    params: dict[str, Any]


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"expected mapping in {path}")
    return data


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True, allow_nan=False, default=json_default)
        f.write("\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            json.dump(row, f, sort_keys=True, allow_nan=False, default=json_default)
            f.write("\n")


def resolve_from_spec(spec_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (spec_path.parent / path).resolve()


def get_nested(data: dict[str, Any], dotted: str) -> Any:
    cur: Any = data
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(dotted)
        cur = cur[part]
    return cur


def set_nested(data: dict[str, Any], dotted: str, value: Any) -> None:
    cur = data
    parts = dotted.split(".")
    for part in parts[:-1]:
        nxt = cur.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[part] = nxt
        cur = nxt
    cur[parts[-1]] = value


def flatten_values(data: dict[str, Any], keys: list[str]) -> dict[str, Any] | None:
    out: dict[str, Any] = {}
    for key in keys:
        try:
            out[key] = get_nested(data, key)
        except KeyError:
            return None
    return out


_MISSING = object()


def flatten_with_defaults(
    data: dict[str, Any], keys_defaults: list[tuple[str, Any]]
) -> dict[str, Any] | None:
    """Like flatten_values but fills ``default`` for missing keys.

    History runs only log the ppo_overrides keys that were explicitly set in
    their config; keys left at the Brax default (e.g. clipping_epsilon, gae_lambda)
    are absent. We fill the spec-declared default so those runs still seed the
    study. A key with no default and no value -> None (run dropped).
    """
    out: dict[str, Any] = {}
    for key, default in keys_defaults:
        try:
            out[key] = get_nested(data, key)
        except KeyError:
            if default is _MISSING:
                return None
            out[key] = default
    return out


def distributions(search_space: dict[str, dict[str, Any]]) -> dict[str, BaseDistribution]:
    dists: dict[str, BaseDistribution] = {}
    for name, cfg in search_space.items():
        typ = cfg["type"]
        if typ == "float":
            dists[name] = FloatDistribution(float(cfg["low"]), float(cfg["high"]))
        elif typ == "log_float":
            dists[name] = FloatDistribution(float(cfg["low"]), float(cfg["high"]), log=True)
        elif typ == "int":
            dists[name] = IntDistribution(int(cfg["low"]), int(cfg["high"]))
        elif typ == "categorical":
            dists[name] = CategoricalDistribution(cfg["choices"])
        else:
            raise ValueError(f"unsupported search_space type for {name}: {typ}")
    return dists


def suggest_param(trial: optuna.Trial, name: str, cfg: dict[str, Any]) -> Any:
    typ = cfg["type"]
    if typ == "float":
        return trial.suggest_float(name, float(cfg["low"]), float(cfg["high"]))
    if typ == "log_float":
        return trial.suggest_float(name, float(cfg["low"]), float(cfg["high"]), log=True)
    if typ == "int":
        return trial.suggest_int(name, int(cfg["low"]), int(cfg["high"]))
    if typ == "categorical":
        return trial.suggest_categorical(name, cfg["choices"])
    raise ValueError(f"unsupported search_space type for {name}: {typ}")


def normalize_params(params: dict[str, Any], search_space: dict[str, dict[str, Any]]) -> list[float]:
    vec: list[float] = []
    for name, cfg in search_space.items():
        value = params[name]
        typ = cfg["type"]
        if typ == "log_float":
            low = math.log10(float(cfg["low"]))
            high = math.log10(float(cfg["high"]))
            val = math.log10(float(value))
            vec.append((val - low) / (high - low))
        elif typ == "float":
            low = float(cfg["low"])
            high = float(cfg["high"])
            vec.append((float(value) - low) / (high - low))
        elif typ == "int":
            low = int(cfg["low"])
            high = int(cfg["high"])
            vec.append((int(value) - low) / max(1, high - low))
        elif typ == "categorical":
            choices = list(cfg["choices"])
            idx = choices.index(value)
            denom = max(1, len(choices) - 1)
            vec.append(idx / denom)
        else:
            raise ValueError(f"unsupported search_space type for {name}: {typ}")
    return vec


def euclidean(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)) / max(1, len(a)))


def params_fit_distributions(params: dict[str, Any], dists: dict[str, BaseDistribution]) -> bool:
    """Whether a historical run's params are all inside the search-space dists.

    Optuna can only import a trial whose values fit the current distributions,
    and normalize_params needs categorical values present in the choices. Runs
    outside the space (e.g. unroll_length=1 streaming runs) are dropped here.
    """
    for name, dist in dists.items():
        if name not in params:
            return False
        value = params[name]
        if isinstance(dist, CategoricalDistribution):
            if value not in list(dist.choices):
                return False
        elif isinstance(dist, FloatDistribution):
            try:
                f = float(value)
            except (TypeError, ValueError):
                return False
            if f < dist.low or f > dist.high:
                return False
        elif isinstance(dist, IntDistribution):
            try:
                i = int(value)
            except (TypeError, ValueError):
                return False
            if i < dist.low or i > dist.high:
                return False
    return True


def _param_value(params: dict[str, Any], dotted: str) -> Any:
    """Read a dotted param from either shape: flat keys (scheduler params use
    "ppo_overrides.batch_size" as a single key) or nested dicts (Aim hparams)."""
    if dotted in params:
        return params[dotted]
    try:
        return get_nested(params, dotted)
    except KeyError:
        return None


def ppo_config_feasible(params: dict[str, Any]) -> bool:
    """Brax PPO hard structural constraints (else the job asserts at startup).

    From brax/training/agents/ppo/train.py:
      assert batch_size * num_minibatches % num_envs == 0
      assert num_envs % device_count == 0
    and the implied ``jax.lax.scan`` length ``batch*mb // num_envs >= 1``.
    Keeping batch/mb/num_envs as powers of two makes divisibility automatic;
    this predicate still validates so any non-power-of-two value is rejected.
    """
    raw_b = _param_value(params, "ppo_overrides.batch_size")
    raw_m = _param_value(params, "ppo_overrides.num_minibatches")
    raw_e = _param_value(params, "ppo_overrides.num_envs")
    # If any of the three isn't in the scanned space we can't evaluate the
    # constraint; assume feasible (the base config already satisfies it).
    if raw_b is None or raw_m is None or raw_e is None:
        return True
    try:
        b = int(raw_b)
        m = int(raw_m)
        e = int(raw_e)
    except (TypeError, ValueError):
        return False
    if b <= 0 or m <= 0 or e <= 0:
        return False
    if (b * m) % e != 0:
        return False
    if b * m < e:
        return False
    return True


def min_distance(vec: list[float], others: list[list[float]]) -> float:
    if not others:
        return float("inf")
    return min(euclidean(vec, other) for other in others)


def finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


def format_distance(value: float | None) -> str:
    return "inf" if value is None else f"{value:.3f}"


def sanitize_job_name(name: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-")
    return (sanitized or "hpo-job")[:128]


def aim_sequence_last(seq: Any) -> Any | None:
    try:
        vals = list(seq.values.values_list()) if hasattr(seq.values, "values_list") else list(seq.values)
    except Exception:
        return None
    return vals[-1] if vals else None


def _metric_curve(run: Any, metric_name: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (steps, values) for a metric, or None.

    Uses the metric's pandas dataframe so steps and values stay aligned (the
    raw ``values_list()`` is NOT in step order).
    """
    try:
        metrics = list(run.metrics())
    except Exception:
        return None
    for seq in metrics:
        if getattr(seq, "name", None) != metric_name:
            continue
        try:
            df = seq.dataframe()
        except Exception:
            continue
        if df is None or len(df) == 0:
            continue
        try:
            df = df.sort_values("step")
            steps = df["step"].to_numpy(dtype=float)
            vals = df["value"].to_numpy(dtype=float)
        except Exception:
            continue
        return steps, vals
    return None


def _eval_curve(run: Any) -> tuple[np.ndarray, np.ndarray] | None:
    """Return HalfCheetah PPO eval reward curve."""
    return _metric_curve(run, "eval/episode_reward")


def _hc_reward_at(run: Any, target_step: float, min_steps: float = 18_000_000.0) -> float | None:
    """Reward at a given env-step on the eval curve, or None if the run didn't
    reach ``min_steps`` (so it can't be scored at that horizon)."""
    curve = _eval_curve(run)
    if curve is None:
        return None
    steps, vals = curve
    if float(steps.max()) < min_steps:
        return None
    return float(vals[int(np.argmin(np.abs(steps - target_step)))])


def hc_score_from_curve(run: Any) -> float | None:
    """hc_target_score = reward@10M + reward@20M (legacy composite objective).

    Kept for backward compatibility; the active HalfCheetah objective is
    ``hc_r20`` (reward@20M only) — see ``hc_r20_from_curve``.
    """
    r10 = _hc_reward_at(run, 10_000_000.0, min_steps=8_000_000.0)
    r20 = _hc_reward_at(run, 20_000_000.0, min_steps=18_000_000.0)
    if r10 is None or r20 is None:
        return None
    return r10 + r20


def hc_r20_from_curve(run: Any) -> float | None:
    """hc_r20 = reward@20M, the active HalfCheetah objective (target = 2500).

    Runs that never reached ~20M return None so they are skipped rather than
    scored at a shorter horizon.
    """
    return _hc_reward_at(run, 20_000_000.0, min_steps=18_000_000.0)


def rtrrl_hop_r1m_from_curve(run: Any) -> float | None:
    """RTRRL Hopper objective: eval/rewards at 1M environment steps.

    RTRRL logs with ``step=(episode + 1) * steps`` while the metrics payload also
    includes env steps as ``step * env_params.batch_size``. Convert the 1M
    env-step target back to the logged step so parallel env counts compare at
    the same training budget.
    """
    curve = _metric_curve(run, "eval/rewards")
    if curve is None:
        return None
    steps, vals = curve
    hparams = run_attr(run, "hparams")
    batch_size = 1.0
    if isinstance(hparams, dict):
        try:
            batch_size = float(get_nested(hparams, "env_params.batch_size") or 1.0)
        except (KeyError, TypeError, ValueError):
            batch_size = 1.0
    target_step = 1_000_000.0 / max(batch_size, 1.0)
    if float(steps.max()) < target_step * 0.9:
        return None
    return float(vals[int(np.argmin(np.abs(steps - target_step)))])


def run_attr(run: Any, key: str) -> Any | None:
    try:
        value = run.get(key, None)
        if value is not None:
            return value
    except Exception:
        pass
    try:
        return run[key]
    except Exception:
        return None


# Built-in objectives (recomputed from the eval curve so historical and new runs
# are scored consistently). Projects add their own via <project>/hpo/targets.py.
register_target("hc_target_score", hc_score_from_curve)
register_target("hc_r20", hc_r20_from_curve)
register_target("rtrrl_hop_r1m", rtrrl_hop_r1m_from_curve)


def target_value(run: Any, target: str) -> float | None:
    # Named extractors (built-in or project-registered) take priority. Runs that
    # did not reach the required horizon return None (skipped).
    extractor = _TARGET_EXTRACTORS.get(target)
    if extractor is not None:
        value = extractor(run)
        if value is not None and math.isfinite(value):
            return value
        return None
    # Fallback: prefer run attrs, then metrics, then final/<metric>.
    for key in (target, f"final/{target}"):
        value = run_attr(run, key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    try:
        for seq in run.metrics():
            if seq.name in (target, f"final/{target}"):
                value = aim_sequence_last(seq)
                if value is not None:
                    return float(value)
    except Exception:
        return None
    return None


def comparable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [comparable(v) for v in value]
    if isinstance(value, list):
        return [comparable(v) for v in value]
    if isinstance(value, dict):
        return {k: comparable(v) for k, v in value.items()}
    return value


def matches_filters(hparams: dict[str, Any], filters: dict[str, Any]) -> bool:
    for key, expected in filters.items():
        try:
            actual = get_nested(hparams, key)
        except KeyError:
            return False
        if comparable(actual) != comparable(expected):
            return False
    return True


def import_aim_runs(spec: dict[str, Any], spec_path: Path) -> list[ImportedRun]:
    from aim import Repo

    aim_cfg = spec["aim"]
    repo_path = resolve_from_spec(spec_path, aim_cfg["repo"])
    repo = Repo(str(repo_path))
    include_pattern = re.compile(aim_cfg.get("include_name_regex", ".*"))
    exclude_pattern = (
        re.compile(aim_cfg["exclude_name_regex"])
        if aim_cfg.get("exclude_name_regex")
        else None
    )
    targets = [aim_cfg["target"], *aim_cfg.get("fallback_targets", [])]
    search_space = spec["search_space"]
    keys_defaults = [
        (k, cfg.get("default", _MISSING)) for k, cfg in search_space.items()
    ]
    filters = spec.get("filters", {})
    skip_archived = bool(aim_cfg.get("skip_archived", True))

    imported: list[ImportedRun] = []
    for run in repo.iter_runs():
        try:
            name = str(getattr(run, "name", "") or "")
            archived = bool(getattr(run, "archived", False))
        except Exception:
            # Aim can leave incomplete runs without props if a job is killed
            # during startup. They cannot be matched or scored, so skip them.
            continue
        if skip_archived and archived:
            continue
        if not include_pattern.search(name):
            continue
        if exclude_pattern and exclude_pattern.search(name):
            continue
        hparams = run_attr(run, "hparams")
        if not isinstance(hparams, dict):
            continue
        if filters and not matches_filters(hparams, filters):
            continue
        # ppo_baseline logs {"ppo": asdict(args), "shared": ...}; rtrrl logs the
        # dataclass dict directly. Try both shapes while keeping generated config
        # keys in the normal top-level form (e.g. ppo_overrides.learning_rate).
        params = flatten_with_defaults(hparams, keys_defaults)
        if params is None and isinstance(hparams.get("ppo"), dict):
            params = flatten_with_defaults(hparams["ppo"], keys_defaults)
        if params is None:
            continue
        if not ppo_config_feasible(params):
            continue
        matched_target = None
        value = None
        for target in targets:
            value = target_value(run, target)
            if value is not None and math.isfinite(value):
                matched_target = target
                break
        if value is None or matched_target is None:
            continue
        imported.append(
            ImportedRun(
                aim_hash=str(getattr(run, "hash", "")),
                name=name,
                target=matched_target,
                value=value,
                params=params,
            )
        )
    return imported


def create_history_study(
    spec: dict[str, Any],
    imported: list[ImportedRun],
    storage_url: str | None = None,
) -> optuna.Study:
    direction = spec["aim"].get("direction", "maximize")
    sampler = optuna.samplers.TPESampler(multivariate=True, seed=spec.get("seed", 0))
    study = optuna.create_study(
        study_name=spec["study"],
        direction=direction,
        sampler=sampler,
        storage=storage_url,
        load_if_exists=True if storage_url else False,
    )

    existing_hashes = {
        t.user_attrs.get("aim_hash")
        for t in study.trials
        if t.user_attrs.get("aim_hash")
    }
    dists = distributions(spec["search_space"])
    for row in imported:
        if row.aim_hash in existing_hashes:
            continue
        trial = create_trial(
            params=row.params,
            distributions=dists,
            value=row.value,
            state=TrialState.COMPLETE,
            user_attrs={"aim_hash": row.aim_hash, "aim_name": row.name, "aim_target": row.target},
        )
        study.add_trial(trial)
    return study


def storage_url_for_study(study_name: str) -> str:
    sqlite_path = STUDIES_DIR / study_name / "optuna.db"
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{sqlite_path}"


def result_cache_path(study_name: str) -> Path:
    return STUDIES_DIR / study_name / "results.jsonl"


def load_result_cache(spec: dict[str, Any]) -> list[ImportedRun]:
    path = result_cache_path(spec["study"])
    if not path.exists():
        return []
    rows: list[ImportedRun] = []
    dists = distributions(spec["search_space"])
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                params = raw["params"]
                value = float(raw["value"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
            if not math.isfinite(value) or not params_fit_distributions(params, dists):
                continue
            rows.append(
                ImportedRun(
                    aim_hash=str(raw.get("aim_hash", "")),
                    name=str(raw.get("name", "")),
                    target=str(raw.get("target", spec["aim"]["target"])),
                    value=value,
                    params=params,
                )
            )
    return rows


def write_result_cache(spec: dict[str, Any], imported: list[ImportedRun]) -> Path:
    by_hash: dict[str, ImportedRun] = {}
    for row in load_result_cache(spec):
        if row.aim_hash:
            by_hash[row.aim_hash] = row
    for row in imported:
        if row.aim_hash:
            by_hash[row.aim_hash] = row
    rows = [
        {
            "aim_hash": row.aim_hash,
            "name": row.name,
            "target": row.target,
            "value": row.value,
            "params": row.params,
        }
        for row in sorted(by_hash.values(), key=lambda r: (r.name, r.aim_hash))
    ]
    path = result_cache_path(spec["study"])
    write_jsonl(path, rows)
    return path


def load_optuna_history(spec: dict[str, Any]) -> list[ImportedRun]:
    storage_url = storage_url_for_study(spec["study"])
    try:
        study = optuna.load_study(study_name=spec["study"], storage=storage_url)
    except Exception:
        return []
    rows: list[ImportedRun] = []
    dists = distributions(spec["search_space"])
    for trial in study.trials:
        if trial.value is None or not math.isfinite(float(trial.value)):
            continue
        params = dict(trial.params)
        if not params_fit_distributions(params, dists):
            continue
        rows.append(
            ImportedRun(
                aim_hash=str(trial.user_attrs.get("aim_hash", "")),
                name=str(trial.user_attrs.get("aim_name", f"trial-{trial.number}")),
                target=str(trial.user_attrs.get("aim_target", spec["aim"]["target"])),
                value=float(trial.value),
                params=params,
            )
        )
    return rows


def tpe_acquisition_values(
    sampler: optuna.samplers.BaseSampler,
    study: optuna.Study,
    dists: dict[str, BaseDistribution],
    candidates: list[dict[str, Any]],
) -> list[float | None]:
    """TPE acquisition log(l(x)/g(x)) for each candidate (= log Expected Improvement).

    Reuses the sampler's private Parzen-estimator builders so the numbers match
    what TPE actually uses to pick points. Returns None per candidate if the
    internal API is unavailable (e.g. Optuna version change) so callers can fall
    back to d_hist only.
    """
    try:
        from optuna.samplers._tpe.sampler import _split_trials  # type: ignore
        from optuna.trial import TrialState
    except Exception:
        return [None] * len(candidates)
    if not isinstance(sampler, optuna.samplers.TPESampler):
        return [None] * len(candidates)
    try:
        trials = study._get_trials(deepcopy=False, states=(TrialState.COMPLETE, TrialState.PRUNED), use_cache=True)
        n = len(trials)
        if n < getattr(sampler, "_n_startup_trials", 10):
            return [None] * len(candidates)
        below, above = _split_trials(study, trials, sampler._gamma(n), False)
        mpe_below = sampler._build_parzen_estimator(study, dists, below, True)
        mpe_above = sampler._build_parzen_estimator(study, dists, above, False)
        samples = {
            name: np.array([dist.to_internal_repr(c[name]) for c in candidates])
            for name, dist in dists.items()
        }
        acq = sampler._compute_acquisition_func(samples, mpe_below, mpe_above)
        return [float(v) if math.isfinite(float(v)) else None for v in acq]
    except Exception:
        return [None] * len(candidates)


def sync_aim_history(spec: dict[str, Any], spec_path: Path) -> tuple[optuna.Study, list[ImportedRun]]:
    imported = import_aim_runs(spec, spec_path)
    dists = distributions(spec["search_space"])
    imported = [row for row in imported if params_fit_distributions(row.params, dists)]
    write_result_cache(spec, imported)
    study = create_history_study(
        spec,
        imported,
        storage_url=storage_url_for_study(spec["study"]),
    )
    return study, imported


def cached_history(spec: dict[str, Any]) -> list[ImportedRun]:
    cached = load_result_cache(spec)
    if cached:
        return cached
    return load_optuna_history(spec)


def round_dir(study: str, round_name: str) -> Path:
    return RUNS_DIR / study / f"round_{round_name}"


def next_round_name(study: str) -> str:
    base = RUNS_DIR / study
    if not base.exists():
        return "001"
    nums = []
    for p in base.glob("round_*"):
        suffix = p.name.removeprefix("round_")
        if suffix.isdigit():
            nums.append(int(suffix))
    return f"{(max(nums) + 1) if nums else 1:03d}"


def build_config(
    base_config: dict[str, Any],
    params: dict[str, Any],
    base_spec: dict[str, Any],
) -> dict[str, Any]:
    cfg = copy.deepcopy(base_config)
    for key, value in params.items():
        set_nested(cfg, key, value)
    if "total_env_steps" in base_spec:
        steps = int(get_nested(cfg, "steps"))
        batch_size = int(get_nested(cfg, "env_params.batch_size") or 1)
        episodes = max(1, math.ceil(int(base_spec["total_env_steps"]) / (steps * batch_size)))
        cfg["episodes"] = episodes
        cfg["patience"] = episodes
        eval_points = int(base_spec.get("eval_points", 20))
        cfg["eval_every"] = max(1, episodes // max(1, eval_points))
    cfg["logging"] = base_spec.get("logging", cfg.get("logging", "aim"))
    if "log_repo" in base_spec:
        cfg["log_repo"] = base_spec["log_repo"]
    return cfg


def rule_matches(config: dict[str, Any], when: dict[str, Any]) -> bool:
    """Evaluate a small declarative resource rule against a generated config."""
    path = when["path"]
    op = when.get("op", "==")
    expected = when["value"]
    try:
        actual = get_nested(config, path)
    except KeyError:
        return False

    if op == "==":
        return actual == expected
    if op == "!=":
        return actual != expected
    if op == ">":
        return float(actual) > float(expected)
    if op == ">=":
        return float(actual) >= float(expected)
    if op == "<":
        return float(actual) < float(expected)
    if op == "<=":
        return float(actual) <= float(expected)
    if op == "in":
        return actual in expected
    raise ValueError(f"unsupported resource rule op: {op}")


def select_profile(spec: dict[str, Any], config: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
    policy = spec.get("resource_policy", {})
    profiles = spec.get("profiles", {})
    profile_name = policy.get("default_profile")
    for rule in policy.get("rules", []):
        if rule_matches(config, rule["when"]):
            profile_name = rule["profile"]
            break
    if profile_name and profile_name not in profiles:
        raise KeyError(f"resource profile not found: {profile_name}")
    return profile_name, profiles.get(profile_name, {})


def sync_aim(args: argparse.Namespace) -> int:
    spec_path = Path(args.spec).resolve()
    spec = load_yaml(spec_path)
    study, imported = sync_aim_history(spec, spec_path)
    completed = [trial for trial in study.trials if trial.state == TrialState.COMPLETE]
    values = [row.value for row in imported]
    best = None
    if values:
        best = min(values) if spec["aim"].get("direction") == "minimize" else max(values)

    print(f"Study      : {spec['study']}")
    print(f"Storage    : {storage_url_for_study(spec['study'])}")
    print(f"Result cache: {result_cache_path(spec['study'])}")
    print(f"Aim runs   : {len(imported)} imported from spec filter")
    print(f"Optuna done: {len(completed)} complete trials in storage")
    print(f"Best value : {best}")
    return 0


def suggest(args: argparse.Namespace) -> int:
    spec_path = Path(args.spec).resolve()
    spec = load_yaml(spec_path)
    study_name = spec["study"]
    round_name = args.round if args.round != "auto" else next_round_name(study_name)
    out_dir = round_dir(study_name, round_name)
    configs_dir = out_dir / "configs"

    imported = cached_history(spec)
    storage_url = storage_url_for_study(study_name)

    # Candidate generation uses an in-memory copy so rejected candidates do not
    # pollute the persistent SQLite study with failed/running trials.
    candidate_study = create_history_study(spec, imported, storage_url=None)
    search_space = spec["search_space"]
    constraints = spec.get("constraints", {})
    n = int(args.n or spec.get("budget", {}).get("max_jobs_per_round", 4))
    pool_multiplier = int(constraints.get("candidate_pool_multiplier", 8))
    max_retries = int(constraints.get("max_duplicate_retries", n * pool_multiplier))
    min_hist = float(constraints.get("min_distance_to_history", 0.0))
    min_batch = float(constraints.get("min_distance_in_batch", 0.0))

    history_vecs = [normalize_params(row.params, search_space) for row in imported]
    selected: list[dict[str, Any]] = []
    selected_vecs: list[list[float]] = []
    rejected = {"near_history": 0, "near_batch": 0, "infeasible": 0}
    attempts = 0

    while len(selected) < n and attempts < max_retries:
        attempts += 1
        trial = candidate_study.ask()
        params = {name: suggest_param(trial, name, cfg) for name, cfg in search_space.items()}
        if not ppo_config_feasible(params):
            rejected["infeasible"] += 1
            continue
        vec = normalize_params(params, search_space)
        d_hist = min_distance(vec, history_vecs)
        if d_hist < min_hist:
            rejected["near_history"] += 1
            continue
        d_batch = min_distance(vec, selected_vecs)
        if d_batch < min_batch:
            rejected["near_batch"] += 1
            continue
        selected.append({"params": params, "distance_to_history": d_hist, "distance_to_batch": d_batch})
        selected_vecs.append(vec)

    base_cfg = load_yaml(resolve_from_spec(spec_path, spec["base"]["config"]))
    dists = distributions(search_space)
    acq_vals = tpe_acquisition_values(
        candidate_study.sampler, candidate_study, dists, [s["params"] for s in selected]
    )
    # Rank by TPE acquisition (log l/g) descending so config_001 is the most
    # promising; candidates with no acq fall to the end.
    ranked = sorted(
        zip(selected, acq_vals),
        key=lambda ia: ia[1] if ia[1] is not None and math.isfinite(ia[1]) else float("-inf"),
        reverse=True,
    )
    plan_items = []
    for idx, (item, acq) in enumerate(ranked, start=1):
        cfg = build_config(base_cfg, item["params"], spec.get("base", {}))
        profile_name, profile = select_profile(spec, cfg)
        cfg_path = configs_dir / f"config_{idx:03d}.yml"
        write_yaml(cfg_path, cfg)
        job_name = sanitize_job_name(f"{study_name}-r{round_name}-{idx:03d}")
        plan_items.append(
            {
                "index": idx,
                "run_name": cfg.get("run_name"),
                "job_name": job_name,
                "config": str(cfg_path.relative_to(PROJECT_ROOT)),
                "entry": spec["base"].get("entry", "ppo_baseline.py"),
                "logging": spec["base"].get("logging", "aim"),
                "profile": profile_name,
                "queue": profile.get("queue"),
                "job_def": profile.get("job_def"),
                "params": item["params"],
                "distance_to_history": finite_or_none(item["distance_to_history"]),
                "distance_to_batch": finite_or_none(item["distance_to_batch"]),
                "acq_log_l_over_g": finite_or_none(acq) if acq is not None else None,
            }
        )

    values = [row.value for row in imported]
    best = max(values) if spec["aim"].get("direction", "maximize") == "maximize" and values else None
    if spec["aim"].get("direction") == "minimize" and values:
        best = min(values)

    plan = {
        "study": study_name,
        "round": round_name,
        "spec": str(rel(spec_path)),
        "storage": storage_url,
        "history": {"imported_runs": len(imported), "best_value": best},
        "rejected": rejected,
        "items": plan_items,
    }
    write_json(out_dir / "plan.json", plan)

    report = [
        f"# HPO Round {study_name} / {round_name}",
        "",
        f"- Imported Aim runs: {len(imported)}",
        f"- Best value: {best}",
        f"- Selected candidates: {len(plan_items)} / requested {n}",
        f"- Rejected near history: {rejected['near_history']}",
        f"- Rejected near batch: {rejected['near_batch']}",
        f"- Rejected infeasible (Brax batch*mb%env): {rejected['infeasible']}",
        "",
        "## Candidates",
        "",
    ]
    for item in plan_items:
        acq = item.get("acq_log_l_over_g")
        acq_str = f" acq={acq:.3f}" if acq is not None else ""
        report.append(
            f"{item['index']}. `{item['config']}` "
            f"profile=`{item['profile']}` d_hist={format_distance(item['distance_to_history'])}{acq_str}"
        )
        report.append(f"   params: `{json.dumps(item['params'], sort_keys=True)}`")
    (out_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    print(f"Wrote {rel(out_dir)}")
    print(f"Plan: {rel(out_dir / 'plan.json')}")
    print(f"Report: {rel(out_dir / 'report.md')}")
    return 0


def submit(args: argparse.Namespace) -> int:
    plan_path = Path(args.plan).resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    for item in plan["items"]:
        cmd = [
            str(SUBMIT_SH),
            "--project",
            str(PROJECT_ROOT),
            "--queue",
            item["queue"],
            "--job-def",
            item["job_def"],
            "--config",
            item["config"],
            "--entry",
            item["entry"],
            "--name",
            item.get("job_name", sanitize_job_name(item["run_name"])),
            "--logging",
            item["logging"],
        ]
        print(" ".join(cmd))
        if args.yes:
            subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
    if not args.yes:
        print("Dry run only. Re-run with --yes to submit.")
    return 0


def aim_server_running() -> bool:
    result = subprocess.run(
        ["pgrep", "-af", r"aim server|aim.ext.transport.run:app"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return bool(result.stdout.strip())


def snapshot_aim(args: argparse.Namespace) -> int:
    # Defaults depend on the resolved project root, so fill them here (after
    # set_project_root) rather than at argparse-construction time.
    source = Path(args.source).resolve() if args.source else (PROJECT_ROOT / "logs" / "aim")
    dest = (
        Path(args.dest).resolve()
        if args.dest
        else HPO_ROOT / "snapshots" / f"aim-{int(time.time())}"
    )
    if aim_server_running() and not args.allow_live:
        print(
            "Aim server appears to be running. Refusing to copy a live Aim repo; "
            "stop experiments/Aim first or pass --allow-live.",
            file=sys.stderr,
        )
        return 2
    if dest.exists():
        raise FileExistsError(dest)
    shutil.copytree(source, dest)
    print(f"Copied {source} -> {dest}")
    return 0


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _run_name_from_hparams(hparams: Any) -> str | None:
    if not isinstance(hparams, dict):
        return None
    raw = hparams.get("run_name")
    return str(raw) if raw else None


def _config_id_from_name(name: str, run_name: str | None) -> str | None:
    for raw in (run_name, name):
        if not raw:
            continue
        match = re.search(r"RTRRL-HOP-(\d+)", raw)
        if match:
            return f"rtrrl_hop_{int(match.group(1)):03d}.yml"
    return None


def aim_manifest_rows(spec: dict[str, Any], spec_path: Path) -> list[dict[str, Any]]:
    from aim import Repo

    repo_path = resolve_from_spec(spec_path, spec["aim"]["repo"])
    repo = Repo(str(repo_path))
    include_pattern = re.compile(spec["aim"].get("include_name_regex", ".*"))
    rows: list[dict[str, Any]] = []
    for run in repo.iter_runs():
        hash_ = ""
        name = ""
        archived = False
        hparams: Any = None
        error: str | None = None
        try:
            hash_ = str(getattr(run, "hash", "") or "")
            name = str(getattr(run, "name", "") or "")
            archived = bool(getattr(run, "archived", False))
        except Exception as exc:
            error = f"props:{type(exc).__name__}"
        if name and not include_pattern.search(name):
            continue
        try:
            hparams = run_attr(run, "hparams")
        except Exception as exc:
            error = error or f"hparams:{type(exc).__name__}"

        run_name = _run_name_from_hparams(hparams)
        metric_points = 0
        finite_points = 0
        nan_points = 0
        r1m = None
        best = None
        last = None
        if error is None:
            curve = _metric_curve(run, "eval/rewards")
            if curve is None:
                error = "missing_eval_rewards"
            else:
                steps, vals = curve
                metric_points = int(len(vals))
                finite_mask = np.isfinite(vals)
                finite_points = int(np.sum(finite_mask))
                nan_points = int(np.sum(~finite_mask))
                if len(vals):
                    last = _safe_float(vals[-1])
                if finite_points:
                    best = _safe_float(np.nanmax(vals))
                if isinstance(hparams, dict):
                    try:
                        batch_size = float(get_nested(hparams, "env_params.batch_size") or 1.0)
                    except (KeyError, TypeError, ValueError):
                        batch_size = 1.0
                else:
                    batch_size = 1.0
                target_step = 1_000_000.0 / max(batch_size, 1.0)
                if len(steps) and float(np.nanmax(steps)) >= target_step * 0.9:
                    r1m = _safe_float(vals[int(np.argmin(np.abs(steps - target_step)))])

        if error:
            cleanup_reason = error
        elif not isinstance(hparams, dict):
            cleanup_reason = "missing_hparams"
        elif metric_points == 0:
            cleanup_reason = "missing_eval_rewards"
        elif finite_points == 0:
            cleanup_reason = "nan_only"
        elif r1m is None and nan_points:
            cleanup_reason = "partial_or_nan_target"
        else:
            cleanup_reason = ""

        rows.append(
            {
                "hash": hash_,
                "name": name,
                "run_name": run_name,
                "config_id": _config_id_from_name(name, run_name),
                "archived": archived,
                "r1m": r1m,
                "best": best,
                "last": last,
                "metric_points": metric_points,
                "finite_points": finite_points,
                "nan_points": nan_points,
                "nan_status": "nan_only" if finite_points == 0 and metric_points else ("has_nan" if nan_points else "finite"),
                "cleanup_reason": cleanup_reason,
            }
        )
    return sorted(rows, key=lambda r: (str(r.get("config_id") or ""), str(r.get("name") or ""), str(r.get("hash") or "")))


def aim_manifest(args: argparse.Namespace) -> int:
    spec_path = Path(args.spec).resolve()
    spec = load_yaml(spec_path)
    rows = aim_manifest_rows(spec, spec_path)
    out = Path(args.output).resolve() if args.output else STUDIES_DIR / spec["study"] / "aim_manifest.jsonl"
    write_jsonl(out, rows)
    flagged = [row for row in rows if row.get("cleanup_reason")]
    print(f"Manifest : {out}")
    print(f"Runs     : {len(rows)}")
    print(f"Flagged  : {len(flagged)}")
    return 0


def cleanup_aim(args: argparse.Namespace) -> int:
    from aim import Repo

    spec_path = Path(args.spec).resolve()
    spec = load_yaml(spec_path)
    manifest = Path(args.manifest).resolve() if args.manifest else STUDIES_DIR / spec["study"] / "aim_manifest.jsonl"
    if not manifest.exists():
        raise FileNotFoundError(manifest)
    if aim_server_running() and not args.allow_live:
        print(
            "Aim server appears to be running. Refusing cleanup of a live Aim repo; "
            "stop Aim first or pass --allow-live.",
            file=sys.stderr,
        )
        return 2
    repo = Repo(str(resolve_from_spec(spec_path, spec["aim"]["repo"])))
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    reasons = set(args.reason or [])
    selected = [
        row
        for row in rows
        if row.get("hash")
        and row.get("cleanup_reason")
        and (not reasons or row.get("cleanup_reason") in reasons)
    ]
    print(f"Selected for cleanup: {len(selected)}")
    for row in selected:
        print(f"{row['hash']} {row.get('run_name') or row.get('name')} reason={row.get('cleanup_reason')}")
    if args.dry_run:
        print("Dry run only. Re-run without --dry-run to delete selected runs.")
        return 0
    for row in selected:
        hash_ = str(row["hash"])
        if repo.run_exists(hash_):
            repo.delete_run(hash_)
    print(f"Deleted: {len(selected)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        default=None,
        help="training project root (owns <project>/hpo/). Default: $HPO_PROJECT_ROOT "
        "or inferred from the --spec/--plan path.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("suggest", help="read Aim history and generate next-round configs")
    p.add_argument("--spec", required=True)
    p.add_argument("--round", default="auto")
    p.add_argument("-n", type=int, default=None)
    p.set_defaults(func=suggest)

    p = sub.add_parser("sync-aim", help="import matching Aim runs into the Optuna SQLite study")
    p.add_argument("--spec", required=True)
    p.set_defaults(func=sync_aim)

    p = sub.add_parser("submit", help="submit a generated plan to AWS Batch")
    p.add_argument("--plan", required=True)
    p.add_argument("--yes", action="store_true", help="actually submit; default is dry run")
    p.set_defaults(func=submit)

    p = sub.add_parser("snapshot-aim", help="copy an Aim repo snapshot into control storage")
    p.add_argument("--source", default=None, help="default: <project>/logs/aim")
    p.add_argument("--dest", default=None, help="default: <project>/hpo/snapshots/aim-<ts>")
    p.add_argument("--allow-live", action="store_true")
    p.set_defaults(func=snapshot_aim)

    p = sub.add_parser("aim-manifest", help="write an Aim run manifest for cleanup/audit")
    p.add_argument("--spec", required=True)
    p.add_argument("--output", default=None)
    p.set_defaults(func=aim_manifest)

    p = sub.add_parser("cleanup-aim", help="delete manifest-flagged bad Aim runs")
    p.add_argument("--spec", required=True)
    p.add_argument("--manifest", default=None)
    p.add_argument("--reason", action="append", help="cleanup_reason to delete; repeatable")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--allow-live", action="store_true")
    p.set_defaults(func=cleanup_aim)
    return parser


def _infer_project_root(args: argparse.Namespace) -> Path | None:
    """Infer <project> from the spec/plan path when --project-root is absent.

    Layout: <project>/hpo/specs/<spec>.yaml  and
            <project>/hpo/runs/<study>/round_<r>/plan.json
    """
    spec = getattr(args, "spec", None)
    if spec:
        parents = Path(spec).resolve().parents
        # .../hpo/specs/<spec>.yaml -> parents[2] == <project>
        if len(parents) >= 3:
            return parents[2]
    plan = getattr(args, "plan", None)
    if plan:
        parents = Path(plan).resolve().parents
        # .../hpo/runs/<study>/round_<r>/plan.json -> parents[4] == <project>
        if len(parents) >= 5:
            return parents[4]
    return None


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    root = args.project_root or os.environ.get("HPO_PROJECT_ROOT") or _infer_project_root(args)
    if root is None:
        parser.error(
            "could not determine the project root; pass --project-root PATH or set "
            "$HPO_PROJECT_ROOT (the project that owns <project>/hpo/)."
        )
    set_project_root(Path(root))
    load_project_targets()

    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
