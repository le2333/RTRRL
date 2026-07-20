"""Utilies for logging."""

import collections
import contextlib
from dataclasses import asdict
import math
import numbers
import os
import traceback
from typing import Callable
from typing_extensions import override

from PIL import Image
from dacite import Config, from_dict

import plotly.express as px
import flax.linen as nn
import jax.tree_util as jtu
from models.jax_util import leaf_norms, tree_norm, tree_stack


class ExceptionPrinter(contextlib.AbstractContextManager):
    """Hacky way to print exceptions in wandb agent."""

    def __enter__(self):  # noqa
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):  # noqa
        if exc_type is not None:
            traceback.print_exception(exc_type, exc_val, exc_tb)
        return False


class DummyLogger(dict, object):
    """Dummy Logger that does nothing besides acting as dictionary."""

    def __repr__(self) -> str:
        """Return name of logger."""
        return "DummyLogger"

    def log(self, metrics: dict, step: int = None):
        """Log a dictionary of metrics (per step).

        Parameters
        ----------
        metrics : dict
            Dictonaries of scalar metrics.
        step : int, optional
            Step number, by default framework will use global step.
        """
        pass

    def log_params(self, params_dict):
        """Log the given hyperparameters.

        Parameters
        ----------
        params_dict : dict
            Dict of hyperparameters.
        """
        pass

    def finalize(self, all_param_norms=None):
        """Log additional plots or media.

        Parameters
        ----------
        all_param_norms : TODO
            _description_
        """
        pass

    def save_model(self, model: nn.Module, filename="model.ckpt"):
        """Save an equinox model to file.

        Parameters
        ----------
        model : equinox.Module
            The model to be saved
        filename : str, optional
            by default "model.ckpt"
        """
        pass

    def log_video(self, name: str, frames, step: int = None, fps=4, **kwargs):
        """Save a video given as array.

        Parameters
        ----------
        name : str
            _description_
        frames : array
            leading dimension for frames, then height, width, channels
        step : int, optional
            Step number, by default framework will use global step.
        fps : int, optional
            _description_, by default 4
        """
        pass

    def log_episode_summary(self, **summary) -> None:
        """Ignore an episode summary."""
        del summary

    def log_episode(self, episode) -> None:
        """Ignore a complete episode."""
        del episode


def update_nested_dict(d, u):
    """Update nested dict d with values from nested dict u.

    Parameters
    ----------
    d : dict
        Base dict
    u : dict
        Updates
    Returns
    -------
    dict
        d with values overwritten by u
    """
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update_nested_dict(d.get(k, {}), v)
        else:
            d[k] = v
    return d


class AimLogger(DummyLogger):
    """Wandb-like interface for aim."""

    def __repr__(self) -> str:
        """Return name of logger."""
        return "AimLogger"

    @override
    def __init__(
        self,
        name,
        repo=None,
        hparams=None,
        run_name="",
        *,
        training_run=None,
    ):
        """Create aim run."""
        self.training_run = training_run
        self._final_metrics = {}
        self._summary = {}
        if training_run is not None:
            return

        global aim
        import aim

        self.run = aim.Run(experiment=name, repo=repo, log_system_params=True)
        if not isinstance(hparams, dict):
            hparams = asdict(hparams)
        self.log_params(hparams)
        self.run.name = run_name + " " + self.run.hash

    @override
    def log(self, metrics, step=None):
        """Loop over scalars and track them with aim."""
        if self.training_run is not None:
            if step is not None and type(step) is not int:
                raise ValueError("explicit step must be an integer")
            canonical_env_steps = metrics.get("train/env_steps")
            if "train/env_steps" in metrics and type(canonical_env_steps) is not int:
                raise ValueError(
                    "canonical train/env_steps metric must be an integer"
                )
            if step is None:
                if "train/env_steps" not in metrics:
                    raise ValueError(
                        "facility logging requires explicit native env_steps "
                        "via step or the train/env_steps metric"
                    )
                env_steps = canonical_env_steps
            else:
                env_steps = step
                if (
                    "train/env_steps" in metrics
                    and canonical_env_steps != env_steps
                ):
                    raise ValueError(
                        "explicit step conflicts with train/env_steps metric"
                    )

            self.training_run.log_metrics(env_steps, metrics)
            self._final_metrics.update(
                {
                    key: value
                    for key, value in metrics.items()
                    if (
                        isinstance(value, numbers.Real)
                        and not isinstance(value, bool)
                        and math.isfinite(float(value))
                    )
                }
            )
            return

        for k, v in metrics.items():
            self.run.track(v, name=k, step=None if step is None else int(step))

    @override
    def log_params(self, params_dict):
        """Log the given hyperparameters.

        Parameters
        ----------
        params_dict : dict
            Dict of hyperparameters.
        """
        if self.training_run is None:
            self.run["hparams"] = params_dict

    def __setitem__(self, key, value):
        """Log scalar for aim."""
        if self.training_run is not None:
            self._summary[key] = value
            return
        if not isinstance(value, dict):
            # Attempt conversion to float if not a dict
            value = float(value)
        self.run[key] = value

    def __getitem__(self, key):
        """Get value from aim run."""
        if self.training_run is not None:
            return self._summary[key]
        return self.run[key]

    @override
    def finalize(self, all_param_norms=None, x_vals=None):
        """Make lineplots for param norms and block until all metrics are logged."""
        if self.training_run is not None:
            self.training_run.finish(self._final_metrics)
            return

        if all_param_norms:
            all_param_norms = tree_stack(all_param_norms)
            self.log(
                {
                    f"Params/{k}": aim.Figure(px.line(x=x_vals, y=list(v.values()), title=k, labels=list(v.keys())))
                    for k, v in all_param_norms.items()
                    if v
                }
            )

        block = os.environ.get("AIM_FINALIZE_BLOCK", "0").lower() in {"1", "true", "yes"}
        self.run.report_successful_finish(block=block)

    @override
    def save_model(self, model, filename="model.ckpt"):
        """Save the model to aim directory using equinox."""
        raise NotImplementedError()
        # run_artifacts_dir = os.path.join('artifacts/aim',  self.run.hash)
        # os.makedirs(run_artifacts_dir, exist_ok=True)
        # model_file = os.path.join(run_artifacts_dir, filename)

    @override
    def log_video(self, name, frames, step=None, fps=30, caption=""):
        """Log a video to wandb."""
        if self.training_run is not None:
            return
        filename = os.path.join("logs/gifs", self.run.hash + ".gif")
        images = [Image.fromarray(frames[i].transpose(1, 2, 0)) for i in range(len(frames))]
        os.makedirs("logs/gifs", exist_ok=True)
        images[0].save(filename, save_all=True, append_images=images[1:], duration=int(1000 / fps), loop=0)
        self.log({name: aim.Image(filename, caption=caption, format="gif")}, step=step)

    def log_episode_summary(self, **summary) -> None:
        """Log an episode summary through the facility SDK when available."""
        if self.training_run is not None:
            self.training_run.log_episode_summary(**summary)

    def log_episode(self, episode):
        """Log a complete episode through the facility SDK when available."""
        if self.training_run is not None:
            return self.training_run.log_episode(episode)
        return None


class WandbLogger(DummyLogger):
    """Wandb-like interface for aim."""

    @override
    def log(self, metrics, step=None):
        """Log metrics to wandb."""
        wandb.log(metrics, step=step)

    def __setitem__(self, key, value):
        """Log scalar for wandb."""
        wandb.run.summary[key] = value

    def __getitem__(self, key):
        """Get value from aim run."""
        return wandb.run.summary[key]

    @override
    def finalize(self, all_param_norms: dict = None, x_vals=None):
        """Make lineplots for all items in all_param_norms."""
        if all_param_norms:
            all_param_norms = tree_stack(all_param_norms)
            wandb.log(
                {
                    f"Params/{k}": wandb.plot.line_series(
                        xs=x_vals,
                        ys=v.values(),
                        title=k,
                        keys=list(v.keys()),
                    )
                    for k, v in all_param_norms.items()
                }
            )

    @override
    def save_model(self, model, filename="model.ckpt"):
        """Save the model to wandb directory using orbax."""
        raise NotImplementedError()

    @override
    def log_video(self, name, frames, step=None, fps=30, caption=""):
        """Log a video to wandb."""
        wandb.log({name: wandb.Video(frames, fps=fps, caption=caption)}, step=step)


class MultiLogger(DummyLogger):
    """Fan out logging calls to several loggers (e.g. Aim + W&B).

    Write-like calls go to every logger; item reads come from the first one.
    """

    def __init__(self, loggers):
        """Store the wrapped loggers (order matters: reads use the first)."""
        self.loggers = list(loggers)

    def __repr__(self) -> str:  # noqa
        return (
            "MultiLogger("
            + ", ".join(repr(logger) for logger in self.loggers)
            + ")"
        )

    @override
    def log(self, metrics, step=None):
        """Forward metrics to every logger."""
        for logger in self.loggers:
            logger.log(metrics, step=step)

    @override
    def log_params(self, params_dict):
        """Forward hyperparameters to every logger."""
        for logger in self.loggers:
            logger.log_params(params_dict)

    @override
    def finalize(self, *args, **kwargs):
        """Forward finalize to every logger."""
        for logger in self.loggers:
            logger.finalize(*args, **kwargs)

    @override
    def save_model(self, *args, **kwargs):
        """Save with whichever logger supports it (skip the ones that don't)."""
        for logger in self.loggers:
            try:
                logger.save_model(*args, **kwargs)
            except NotImplementedError:
                pass

    @override
    def log_video(self, *args, **kwargs):
        """Forward video logging to every logger."""
        for logger in self.loggers:
            logger.log_video(*args, **kwargs)

    def log_episode_summary(self, **summary) -> None:
        """Forward an episode summary to every logger in order."""
        for logger in self.loggers:
            logger.log_episode_summary(**summary)

    def log_episode(self, episode) -> None:
        """Forward a complete episode to every logger in order."""
        for logger in self.loggers:
            logger.log_episode(episode)

    def __setitem__(self, key, value):
        """Set the summary value on every logger."""
        for logger in self.loggers:
            logger[key] = value

    def __getitem__(self, key):
        """Read the summary value from the first logger."""
        return self.loggers[0][key]


def _expand_dotted(flat: dict) -> dict:
    """Expand flat dotted keys into a nested dict.

    `{"ppo_overrides.learning_rate": 1e-3}` -> `{"ppo_overrides": {"learning_rate": 1e-3}}`.
    W&B sweep controllers inject parameters as flat keys; dotted names let a
    sweep target nested dataclass fields (e.g. ppo_overrides.* / env_params.*).
    """
    nested: dict = {}
    for key, value in dict(flat).items():
        parts = str(key).split(".")
        node = nested
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return nested


def _parse_backends(logger_name) -> set:
    """Parse a logger spec like 'aim+wandb' / 'wandb,aim' / 'both' into a set."""
    if not logger_name:
        return set()
    if logger_name == "both":
        return {"aim", "wandb"}
    return {b.strip() for b in logger_name.replace("+", ",").split(",") if b.strip()}


def _resolve_aim_repo(aim_repo, hparams):
    """Pick the effective Aim repo, keeping test (debug) runs out of the main repo.

    Two orthogonal knobs, both env-driven so no config/CLI edits are needed:

    * ``debug`` runs are validation-only and must never pollute or get backed up
      to the shared "main" repo, so they are redirected to an ephemeral local
      scratch repo (``AIM_SCRATCH_REPO``, default ``.aim-scratch``). That
      directory is git-ignored and never synced to S3.
    * For real runs ``AIM_REPO`` (when set) overrides whatever the config/CLI
      passed. This lets local runs anchor to the absolute main repo (or the
      shared ``aim://`` server) without touching the hundreds of configs that
      still carry the historical relative ``log_repo: ".aim"``.
    """
    if getattr(hparams, "debug", 0):
        scratch = os.environ.get("AIM_SCRATCH_REPO") or ".aim-scratch"
        print(f"[aim] debug run -> ephemeral scratch repo {scratch!r} (not backed up)")
        return scratch
    return os.environ.get("AIM_REPO") or aim_repo


def with_logger(
    func: Callable,
    hparams: dict,
    logger_name: str,
    project_name: str,
    aim_repo: str = None,
    run_name="",
    hparams_type=None,
    training_run=None,
):
    """Wrap a training function with one or more loggers.

    `logger_name` may be 'aim', 'wandb', or both ('aim+wandb' / 'both'). When
    W&B is enabled the run is created inside `wandb.init` so a W&B Sweep
    controller can inject swept parameters via `wandb.config`; those are merged
    back into `hparams` (supporting dotted keys for nested fields) before the
    Aim logger and training run see them.
    """
    backends = _parse_backends(logger_name)
    if not backends:
        return func(hparams)

    use_wandb = "wandb" in backends
    use_aim = "aim" in backends
    if use_aim:
        if training_run is None:
            from training_sdk import maybe_current_run

            training_run = maybe_current_run()
        aim_repo = _resolve_aim_repo(aim_repo, hparams)
    base = hparams if isinstance(hparams, dict) else asdict(hparams)

    if use_wandb:
        global wandb
        import wandb

        # debug -> disabled; otherwise let WANDB_MODE env decide (defaults online).
        wandb_mode = "disabled" if getattr(hparams, "debug", 0) else None
        with wandb.init(
            project=project_name,
            config=base,
            mode=wandb_mode,
            dir="logs/",
            name=run_name or None,
        ), ExceptionPrinter():
            # Sweep controller sets wandb.config; merge it back (dotted keys ->
            # nested) so swept params reach nested dataclass fields and Aim.
            merged = update_nested_dict(asdict(hparams), _expand_dotted(wandb.config))
            if hparams_type is not None:
                # Dacite cannot validate some of the existing dataclass annotations
                # after W&B merges config values (notably obs_mask:
                # Optional[Iterable[int] | Literal[...]] with a YAML list). The
                # original CLI/simple_parsing path accepts those values, so keep
                # W&B sweep merging permissive and let the training code validate
                # runtime invariants as usual.
                hparams = from_dict(hparams_type, merged, config=Config(check_types=False))
            if getattr(hparams, "log_code", False):
                wandb.run.log_code()

            loggers = []
            if use_aim:
                loggers.append(
                    AimLogger(
                        project_name,
                        repo=aim_repo,
                        hparams=hparams,
                        run_name=run_name,
                        training_run=training_run,
                    )
                )
            loggers.append(WandbLogger())
            logger = loggers[0] if len(loggers) == 1 else MultiLogger(loggers)
            return func(hparams, logger=logger)

    if use_aim:
        logger = AimLogger(
            project_name,
            repo=aim_repo,
            hparams=hparams,
            run_name=run_name,
            training_run=training_run,
        )
        return func(hparams, logger=logger)

    # Unknown / "none" backend -> run without any logger.
    return func(hparams)


def calc_norms(norm_params: dict = {}, leaf_norm_params: dict = {}):
    """Compute norms and leaf norms of given dict of pytrees."""
    norms = {k: tree_norm(v) for k, v in norm_params.items()}
    param_norms = {k: leaf_norms(v) for k, v in leaf_norm_params.items()}
    return norms, param_norms


def log_norms(pytree):
    """Compute norms and leaf norms of given pytree."""
    flattened, _ = jtu.tree_flatten_with_path(pytree)
    flattened = {jtu.keystr(k): v for k, v in flattened}
    return calc_norms(flattened)[0]
