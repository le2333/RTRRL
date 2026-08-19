"""A run's whole state at one formal boundary, written out and read back.

A checkpoint here is not a copy of the weights. R3.4 branches three update
rules from the same moment of the same run and asks whether they diverge, and
that question has an answer only if the moment is restored completely: the
learner's parameters, the eligibility traces, the rule's own state, the
recurrent carry and its differentiation state, the environment's state, the
normalization statistics, the step counters, and the PRNG the run draws from.
A weight-only checkpoint answers a different question, because the trace and
the carry alone decide the next several hundred updates.

So what is written is the algorithm state tree entire, plus the scheduling
key, and reading one back is checked leaf by leaf against the run that is
about to use it. Nothing is filled in, defaulted, or quietly reshaped: a
checkpoint that does not fit the run reading it is an error naming the leaf
that did not fit.

The one thing a branch may legitimately not share with its parent is the
update rule's own state -- Adam carries moments and the D-RTRRL arms carry
nothing, so a fork onto another arm has nowhere to put the parent's moments.
That is declared per fork, by path, and reported back in what was restored.
It is never inferred from a structure mismatch, which is what would let a
misdirected checkpoint pass as a deliberate branch.

Typed PRNG keys are stored as the integers they are made of, since the key's
implementation is not part of what a run means; they are wrapped again against
the template's own keys on the way back in.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

# The wire version of a checkpoint object, which is not the deployment contract
# version: an image may gain entries or logging blocks without any checkpoint
# it wrote becoming unreadable, and a change to what is stored here must be
# refused even when the deployment contract is unchanged.
CHECKPOINT_FORMAT = "memorax/checkpoint/1"

FILENAME = "step-{steps:012d}.msgpack"


class CheckpointError(ValueError):
    """A checkpoint and the run reading it do not describe the same graph."""


@dataclass(frozen=True)
class Checkpoint:
    """One formal boundary: where a run had got to, and all it was carrying."""

    env_steps: int
    state: Any
    key: Any
    # Which declared paths came from the reading run rather than from the
    # stored document. Empty for an ordinary resume; a fork onto another update
    # rule names the rule's state here and the manifest records it.
    replaced: tuple[str, ...] = ()


def dumps(*, env_steps: int, state: Any, key: Any) -> bytes:
    """Serialize one boundary. The state tree is stored as it stands."""

    return serialization.to_bytes(
        {
            "format": CHECKPOINT_FORMAT,
            "env_steps": int(env_steps),
            "key": jax.random.key_data(key),
            "state": _unwrapped(state),
        }
    )


def loads(
    payload: bytes,
    *,
    state: Any,
    key: Any,
    replacing: Sequence[str] = (),
) -> Checkpoint:
    """Read one boundary into the run that is about to continue from it.

    ``state`` and ``key`` are that run's own freshly built ones. They are the
    template -- what shape everything must be -- and, for the paths named in
    ``replacing``, the value as well.
    """

    stored = serialization.msgpack_restore(payload)
    if not isinstance(stored, dict) or "format" not in stored:
        raise CheckpointError("this is not a checkpoint object")
    if stored["format"] != CHECKPOINT_FORMAT:
        raise CheckpointError(
            f"checkpoint format {stored['format']!r} was written by another "
            f"image; this one reads {CHECKPOINT_FORMAT!r}"
        )

    template = _unwrapped(state)
    wanted = serialization.to_state_dict(template)
    held = stored["state"]
    replaced = tuple(replacing)
    for path in replaced:
        _graft(held, wanted, path)

    problems = list(_differences(held, wanted))
    if problems:
        raise CheckpointError(
            "the checkpoint does not fit this run:\n  " + "\n  ".join(problems)
        )

    restored = _rewrapped(serialization.from_state_dict(template, held), state)
    return Checkpoint(
        env_steps=int(stored["env_steps"]),
        state=restored,
        key=_wrapped_like(stored["key"], key),
        replaced=replaced,
    )


def write(path: Path, *, env_steps: int, state: Any, key: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(dumps(env_steps=env_steps, state=state, key=key))
    return path


def read(
    path: Path,
    *,
    state: Any,
    key: Any,
    replacing: Sequence[str] = (),
) -> Checkpoint:
    return loads(Path(path).read_bytes(), state=state, key=key, replacing=replacing)


class CheckpointDirectory:
    """Where a run files its formal checkpoints, and how many it keeps.

    Retention is stated in checkpoints rather than in bytes because what a fork
    needs is *the boundary before the collapse*, and how far back that is is
    measured in evaluation intervals. ``keep=None`` keeps every one, which is
    what a run whose collapse step is not yet known has to do.
    """

    def __init__(self, directory: Path, *, keep: int | None = None) -> None:
        if keep is not None and keep < 1:
            raise ValueError("a run that keeps no checkpoint cannot be forked")
        self._directory = Path(directory)
        self._keep = keep
        self._written: list[Path] = []

    @property
    def written(self) -> tuple[Path, ...]:
        return tuple(self._written)

    def save(self, *, env_steps: int, state: Any, key: Any) -> Path:
        path = write(
            self._directory / FILENAME.format(steps=env_steps),
            env_steps=env_steps,
            state=state,
            key=key,
        )
        self._written.append(path)
        while self._keep is not None and len(self._written) > self._keep:
            self._written.pop(0).unlink(missing_ok=True)
        return path


def _is_key(leaf: Any) -> bool:
    dtype = getattr(leaf, "dtype", None)
    return dtype is not None and jnp.issubdtype(dtype, jax.dtypes.prng_key)


def _unwrapped(tree: Any) -> Any:
    """Every typed PRNG key replaced by the integers it is made of."""

    return jax.tree.map(
        lambda leaf: jax.random.key_data(leaf) if _is_key(leaf) else leaf, tree
    )


def _wrapped_like(data: Any, template: Any) -> Any:
    """Stored key integers, made a key again under the run's own key type.

    Which PRNG implementation a key is drawn under is part of what the numbers
    will be, so it is taken from the run doing the reading rather than left to
    a default that happens to agree today.
    """

    return jax.random.wrap_key_data(
        jnp.asarray(data), impl=jax.random.key_impl(template)
    )


def _rewrapped(tree: Any, template: Any) -> Any:
    """The inverse, taking which leaves were keys from the run's own state."""

    return jax.tree.map(
        lambda leaf, like: _wrapped_like(leaf, like) if _is_key(like) else leaf,
        tree,
        template,
    )


def _flat(tree: Any, prefix: str = "") -> Iterator[tuple[str, Any]]:
    """Every leaf of a state dict under the dotted path that reaches it."""

    if isinstance(tree, dict):
        if not tree:
            yield prefix.rstrip("."), tree
            return
        for name, value in tree.items():
            yield from _flat(value, f"{prefix}{name}.")
        return
    yield prefix.rstrip("."), tree


def _at(tree: Any, path: str) -> Any:
    found = tree
    for name in path.split("."):
        if not isinstance(found, dict) or name not in found:
            raise CheckpointError(f"neither run has anything at {path!r}")
        found = found[name]
    return found


def _graft(held: Any, wanted: Any, path: str) -> None:
    """Put the reading run's own subtree in place of the stored one.

    Both sides are read before either is written, so a path that names nothing
    is refused rather than silently creating a branch of the state dict that
    the template has no room for.
    """

    replacement = _at(wanted, path)
    _at(held, path)
    parts = path.split(".")
    holder = held
    for name in parts[:-1]:
        holder = holder[name]
    holder[parts[-1]] = replacement


def _differences(held: Any, wanted: Any) -> Iterator[str]:
    """Every way the stored state and the reading run's fail to be the same.

    Structure, shape and dtype, each named by the path that carries it. Shape
    is checked here because nothing else checks it: restoring an array only
    reads the stored one, so a graph whose torso got wider would silently take
    the narrow one and train on it.
    """

    stored = dict(_flat(held))
    template = dict(_flat(wanted))
    for path in sorted(set(stored) | set(template)):
        if path not in stored:
            yield f"{path}: the checkpoint does not hold it"
            continue
        if path not in template:
            yield f"{path}: this run has nowhere to put it"
            continue
        one, other = stored[path], template[path]
        if isinstance(other, dict) or isinstance(one, dict):
            if isinstance(one, dict) != isinstance(other, dict):
                yield f"{path}: one side is a subtree and the other is a value"
            continue
        one, other = np.asarray(one), np.asarray(other)
        if one.shape != other.shape:
            yield f"{path}: stored {one.shape}, this run wants {other.shape}"
        elif one.dtype != other.dtype:
            yield f"{path}: stored {one.dtype}, this run wants {other.dtype}"
