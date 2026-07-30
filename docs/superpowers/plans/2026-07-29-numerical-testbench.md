# Numerical Testbench Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A path-installed `testbench` package that judges whether two theoretically equivalent computations agree, and the conversion of the LRU influence-matrix comparison onto it.

**Architecture:** The package supplies a hardware-testbench vocabulary — stimulus injection checked for totality, probe pairing, a gap metric, a three-class verdict, and a scoreboard that accumulates across a stream and along an axis. The repository supplies a fixture module per implementation pair, holding mounting, stepping, shape conventions and the correspondence model, and case modules that only choose points and demand verdicts.

**Tech Stack:** Python 3.12, numpy, hatchling, uv, pytest. JAX is imported lazily by one function and is not a declared dependency of the package.

**Design:** `docs/superpowers/specs/2026-07-29-numerical-testbench-design.md`

## Global Constraints

- Never run pytest on this machine. `AGENTS.md:16-38` forbids it — 911 MiB, 2 cores. Every "run" step below is executed by pushing and reading the run. A change is verified when a remote run reports it green.
- `testbench` declares `numpy>=2` and nothing else. JAX is imported inside `leaves.flatten` and nowhere else.
- The package contains no reference to LRU, RTRRL, StreamAC, or any algorithm in this repository.
- Verdict thresholds are `bits=8.0` and `growth=2.0`. They are keyword arguments with those defaults and are not per-leaf.
- The rounding verdict is unreachable from a single axis point. A leaf with a non-zero gap measured at one point only is anomalous.
- `memo/tests/conftest.py` keeps working throughout; the eleven other test files are not touched by this plan.
- `memo/memorax/networks/sequence_models/upstream_lru.py` is never edited.
- Commit messages follow the repository's style: imperative, lower case after the type prefix, describing what the change buys.

---

### Task 1: The gap metric stops discarding imaginary parts

`conftest.last_bits` casts both sides with `.astype(np.float64)`. For complex input numpy discards the imaginary part and emits `ComplexWarning`, so a purely imaginary disagreement is measured as zero. `test_lru_parity.py` compares complex quantities throughout — the hidden state and all five sensitivities are `complex64` — so its central comparison has been reading real parts only. This is independent of the rest of the plan and ships on its own.

**Files:**
- Modify: `memo/tests/conftest.py:29-34`
- Test: `memo/tests/test_conftest_gap.py` (create)

**Interfaces:**
- Produces: `conftest.last_bits(wanted, got) -> float`, unchanged signature, now correct for complex input.

- [ ] **Step 1: Write the failing test**

Create `memo/tests/test_conftest_gap.py`:

```python
"""The gap metric, on the one input it used to answer zero for.

``last_bits`` widened both sides to float64 before subtracting. numpy discards
the imaginary part of a complex array on that cast and only warns, so every
disagreement that lived in the imaginary part was measured as no disagreement
at all -- across the hidden state and all five sensitivities, which are the
quantities ``test_lru_parity.py`` exists to compare.
"""

import numpy as np
import pytest
from conftest import deviations, last_bits


def test_a_purely_imaginary_disagreement_is_measured():
    wanted = np.array([1 + 0j], np.complex64)
    got = np.array([1 + 1j], np.complex64)
    assert last_bits(wanted, got) > 1e6


def test_the_real_part_still_measures_what_it_did():
    wanted = np.array([1.0], np.float32)
    got = np.array([np.nextafter(np.float32(1), np.float32(2))], np.float32)
    assert last_bits(wanted, got) == pytest.approx(1.0)


def test_a_complex_leaf_that_differs_only_in_phase_is_reported():
    expected = {"h": np.array([1 + 0j, 2 + 0j], np.complex64)}
    actual = {"h": np.array([1 + 0.5j, 2 + 0j], np.complex64)}
    assert deviations(actual, expected)


def test_two_equal_complex_leaves_are_still_equal():
    same = {"h": np.array([1 + 1j, 2 - 3j], np.complex64)}
    assert not deviations(dict(same), same)
```

- [ ] **Step 2: Fix the cast**

In `memo/tests/conftest.py`, replace the body of `last_bits`:

```python
def _widened(array) -> np.ndarray:
    """The same numbers in the widest format that keeps all of them.

    Complex arrays widen to complex128. Widening them to float64 is what numpy
    does on request and it discards the imaginary part with a warning, which
    turned every imaginary disagreement into no disagreement.
    """

    array = np.asarray(array)
    return array.astype(np.complex128 if np.iscomplexobj(array) else np.float64)


def last_bits(wanted, got) -> float:
    """How many float32 last bits apart two arrays are, at their own scale."""

    wanted, got = _widened(wanted), _widened(got)
    scale = max(float(np.abs(wanted).max()), float(np.abs(got).max()), 1e-6)
    gap = float(np.max(np.abs(got - wanted)))
    return gap / float(np.spacing(np.float32(scale)))
```

- [ ] **Step 3: Turn the warning into an error so it cannot come back**

Add to `memo/pytest.ini` under `[pytest]`, which currently holds only
`testpaths` and `addopts`:

```ini
filterwarnings =
    error::numpy.exceptions.ComplexWarning
```

If pytest reports that it cannot resolve that category — the class moved into
`numpy.exceptions` in numpy 2 and the import path is version-sensitive — match
the text instead, which is stable across both:

```ini
filterwarnings =
    error:.*discards the imaginary part:
```

- [ ] **Step 4: Commit and push**

```bash
git add memo/tests/conftest.py memo/tests/test_conftest_gap.py memo/pytest.ini
git commit -m "fix(tests): measure the imaginary half of every complex comparison"
git push
```

- [ ] **Step 5: Read the run**

```bash
gh run list --workflow=memo-ci.yml --limit 1
gh run view <id> --log-failed
```

Expected: the four new tests pass. `test_lru_parity.py` may now report gaps it never reported before, since the imaginary halves are being compared for the first time. Those are real measurements, not regressions. If a leaf now exceeds its `INFLUENCE` or `READOUT` entry, raise that entry to twice the newly measured worst and say so in the commit — the tables are on their way out anyway, and Task 8 replaces the first of them.

---

### Task 2: Package skeleton and the gap metric

**Files:**
- Create: `testbench/pyproject.toml`
- Create: `testbench/src/testbench/__init__.py`
- Create: `testbench/src/testbench/gap.py`
- Create: `testbench/tests/test_gap.py`
- Modify: `.github/workflows/tests.yml:26-36`

**Interfaces:**
- Produces: `testbench.gap.last_bits(wanted, got) -> float`, `testbench.gap.relative(wanted, got) -> float`. Both take anything `np.asarray` accepts, both widen complex to `complex128` and real to `float64`, both return a single float summarising the whole array.

- [ ] **Step 1: Create the package files**

`testbench/pyproject.toml`:

```toml
[project]
name = "testbench"
version = "0.1.0"
requires-python = ">=3.12,<3.13"
dependencies = [
    "numpy>=2",
]

[dependency-groups]
dev = [
    "pytest",
    "ruff",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
pythonpath = ["."]
```

`testbench/src/testbench/__init__.py`:

```python
"""Judging whether two theoretically equivalent computations agree.

One stimulus, two implementations, probes where they correspond, and a verdict
at each probe. The vocabulary is here; what is being compared is not, and this
package never learns it.
"""

from testbench.gap import last_bits, relative

__all__ = ["last_bits", "relative"]
```

- [ ] **Step 2: Write the failing test**

`testbench/tests/test_gap.py`:

```python
import numpy as np
import pytest

from testbench.gap import last_bits, relative


def test_identical_arrays_are_no_last_bits_apart():
    same = np.array([0.25, -1.5], np.float32)
    assert last_bits(same, same.copy()) == 0.0


def test_one_step_of_the_format_is_one_last_bit():
    wanted = np.float32(1.0)
    got = np.nextafter(wanted, np.float32(2))
    assert last_bits(np.array([wanted]), np.array([got])) == pytest.approx(1.0)


def test_five_steps_are_five_last_bits():
    got = wanted = np.float32(1.0)
    for _ in range(5):
        got = np.nextafter(got, np.float32(2))
    assert last_bits(np.array([wanted]), np.array([got])) == pytest.approx(5.0)


def test_the_scale_is_the_larger_of_the_two_sides():
    wanted = np.array([1.0, 1024.0], np.float32)
    got = np.array([1.0, np.nextafter(np.float32(1024), np.float32(2048))], np.float32)
    assert last_bits(wanted, got) == pytest.approx(1.0)


def test_a_purely_imaginary_difference_is_not_discarded():
    wanted = np.array([1 + 0j], np.complex64)
    got = np.array([1 + 1j], np.complex64)
    assert relative(wanted, got) == pytest.approx(1.0)
    assert last_bits(wanted, got) > 1e6


def test_relative_is_the_gap_over_the_reference_scale():
    wanted = np.array([2.0, -4.0], np.float64)
    got = np.array([2.0, -3.0], np.float64)
    assert relative(wanted, got) == pytest.approx(0.25)


def test_relative_survives_an_all_zero_reference():
    zeros = np.zeros((3,), np.float32)
    assert relative(zeros, zeros) == 0.0
```

- [ ] **Step 3: Implement**

`testbench/src/testbench/gap.py`:

```python
"""How far apart two arrays are, in two units that answer different questions.

``last_bits`` is a statement about one format: how many representable steps of
float32 separate the two, at the scale of the larger of them. ``relative`` is
format-free, which is what a comparison between two formats needs.
"""

from __future__ import annotations

import numpy as np

__all__ = ["last_bits", "relative"]


def widened(array) -> np.ndarray:
    """The same numbers in the widest format that keeps all of them.

    Complex widens to complex128. Widening complex to float64 is a thing numpy
    will do on request, discarding the imaginary part with a warning, and a
    comparison that did it would answer zero to every imaginary disagreement.
    """

    array = np.asarray(array)
    return array.astype(np.complex128 if np.iscomplexobj(array) else np.float64)


def last_bits(wanted, got) -> float:
    """How many float32 last bits apart two arrays are, at their own scale."""

    wanted, got = widened(wanted), widened(got)
    scale = max(float(np.abs(wanted).max()), float(np.abs(got).max()), 1e-6)
    gap = float(np.max(np.abs(got - wanted)))
    return gap / float(np.spacing(np.float32(scale)))


def relative(wanted, got) -> float:
    """The widest gap, scaled by the size of what the reference holds.

    Relative and not in last bits because the two sides of a comparison across
    formats are in different formats, and a last-bit count is a statement about
    one of them.
    """

    wanted, got = widened(wanted), widened(got)
    scale = float(np.max(np.abs(wanted)))
    gap = float(np.max(np.abs(got - wanted)))
    return gap / max(scale, 1e-30)
```

- [ ] **Step 4: Wire the package into CI**

In `.github/workflows/tests.yml`, add to the `matrix.include` list after the
`mock-trainer` entry:

```yaml
          - name: testbench
            project: testbench
            target: ""
```

Then make the workflow run on the branch the work is happening on. It currently
carries `branches: [main]`, which would leave every task in this plan unverified
until merge — and a change is verified when a remote run reports it green, not
before. Replace the `on.push` block so it matches how `memo-ci.yml` already
triggers, by path and not by branch:

```yaml
on:
  push:
    paths:
      - .github/workflows/tests.yml
      - training-sdk/**
      - rtrrl/infra/control-plane/**
      - rtrrl/infra/mock-trainer/**
      - testbench/**
  workflow_dispatch:
```

This also starts running the other three projects on feature branches. That is
the point rather than a side effect, and the minutes are free on a public
repository.

- [ ] **Step 5: Commit and push**

```bash
git add testbench .github/workflows/tests.yml
git commit -m "feat(testbench): measure a gap in last bits and relatively, complex included"
git push
```

- [ ] **Step 6: Read the run**

```bash
gh run list --workflow=tests.yml --limit 1
```

Expected: a `testbench` job appears and its seven tests pass.

---

### Task 3: Probe pairing and pytree flattening

**Files:**
- Create: `testbench/src/testbench/leaves.py`
- Create: `testbench/tests/test_leaves.py`
- Modify: `testbench/src/testbench/__init__.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `testbench.leaves.flatten(tree) -> dict[str, np.ndarray]` and `testbench.leaves.paired(ours: Mapping, theirs: Mapping) -> list[tuple[str, np.ndarray, np.ndarray]]`. `paired` returns triples of `(leaf name, ours, theirs)` sorted by name, and raises `AssertionError` if either side has a leaf the other lacks or if a shared leaf has different shapes.

- [ ] **Step 1: Write the failing test**

`testbench/tests/test_leaves.py`:

```python
import numpy as np
import pytest

from testbench.leaves import paired


def test_every_leaf_comes_back_once_and_sorted():
    ours = {"b": np.zeros((2,)), "a": np.ones((3,))}
    theirs = {"a": np.ones((3,)), "b": np.zeros((2,))}
    assert [name for name, _, _ in paired(ours, theirs)] == ["a", "b"]


def test_a_leaf_the_reference_has_and_we_do_not_is_an_error():
    with pytest.raises(AssertionError, match="answers.*'gone'"):
        paired({"here": np.zeros((1,))}, {"here": np.zeros((1,)), "gone": np.zeros((1,))})


def test_a_leaf_we_have_and_the_reference_does_not_is_an_error():
    with pytest.raises(AssertionError, match="answers.*'extra'"):
        paired({"here": np.zeros((1,)), "extra": np.zeros((1,))}, {"here": np.zeros((1,))})


def test_a_shape_that_does_not_match_is_an_error():
    with pytest.raises(AssertionError, match=r"wide: \(2,\) not \(3,\)"):
        paired({"wide": np.zeros((2,))}, {"wide": np.zeros((3,))})


def test_the_sides_come_back_in_the_order_they_were_asked_for():
    ours = {"x": np.array([1.0])}
    theirs = {"x": np.array([2.0])}
    (_, mine, yours), = paired(ours, theirs)
    assert mine[0] == 1.0 and yours[0] == 2.0
```

- [ ] **Step 2: Implement**

`testbench/src/testbench/leaves.py`:

```python
"""Naming the leaves of two trees, and refusing to compare unpaired ones.

An unpaired probe is the failure that looks like success: the comparison runs,
reports nothing, and establishes nothing. So it is an error here rather than a
leaf quietly left out of the report.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

__all__ = ["flatten", "paired"]


def flatten(tree) -> dict[str, np.ndarray]:
    """A pytree as leaf paths joined by slashes.

    JAX is imported here and nowhere else in this package, so that a caller
    comparing plain dictionaries of arrays never needs it installed.
    """

    import jax

    pairs, _ = jax.tree_util.tree_flatten_with_path(tree)
    return {
        "/".join(str(getattr(key, "key", getattr(key, "idx", key))) for key in path): (
            np.asarray(leaf)
        )
        for path, leaf in pairs
    }


def paired(
    ours: Mapping, theirs: Mapping
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Every leaf of the two, matched by name, with neither side left over."""

    unanswered = sorted(set(theirs) - set(ours))
    assert not unanswered, f"nothing on our side answers {unanswered}"
    unmatched = sorted(set(ours) - set(theirs))
    assert not unmatched, f"nothing on the reference side answers {unmatched}"

    out = []
    for name in sorted(theirs):
        mine, yours = np.asarray(ours[name]), np.asarray(theirs[name])
        assert mine.shape == yours.shape, f"{name}: {mine.shape} not {yours.shape}"
        out.append((name, mine, yours))
    return out
```

- [ ] **Step 3: Export it**

In `testbench/src/testbench/__init__.py`:

```python
from testbench.gap import last_bits, relative
from testbench.leaves import flatten, paired

__all__ = ["flatten", "last_bits", "paired", "relative"]
```

- [ ] **Step 4: Commit and push**

```bash
git add testbench
git commit -m "feat(testbench): pair two trees by leaf, and refuse an unpaired probe"
git push
```

- [ ] **Step 5: Read the run**

Expected: five new tests pass, twelve in the `testbench` job.

---

### Task 4: Stimulus injection, checked for totality

**Files:**
- Create: `testbench/src/testbench/stimulus.py`
- Create: `testbench/tests/test_stimulus.py`
- Modify: `testbench/src/testbench/__init__.py`

**Interfaces:**
- Produces: `testbench.stimulus.injected(tree: Mapping, values: Mapping, spelling: Mapping[str, str] | None = None) -> dict`. Returns a nested dict shaped like `tree` with every leaf replaced by the drawn value whose name matches the leaf's last path element, after `spelling` renames. Raises if any leaf has no drawn value, if any drawn value is unused, or if a shape disagrees.

- [ ] **Step 1: Write the failing test**

`testbench/tests/test_stimulus.py`:

```python
import numpy as np
import pytest

from testbench.stimulus import injected


def test_a_drawn_value_reaches_every_leaf_that_names_it():
    tree = {"outer": {"inner": {"weight": np.zeros((2, 3))}}, "weight2": np.zeros((4,))}
    drawn = {"weight": np.ones((2, 3)), "weight2": np.full((4,), 7.0)}
    out = injected(tree, drawn)
    assert out["outer"]["inner"]["weight"][0, 0] == 1.0
    assert out["weight2"][0] == 7.0


def test_the_nesting_of_the_tree_is_preserved():
    tree = {"a": {"b": {"c": np.zeros((1,))}}}
    out = injected(tree, {"c": np.ones((1,))})
    assert list(out) == ["a"] and list(out["a"]) == ["b"] and list(out["a"]["b"]) == ["c"]


def test_a_leaf_nothing_was_drawn_for_is_an_error():
    with pytest.raises(AssertionError, match="nothing was drawn for outer/missing"):
        injected({"outer": {"missing": np.zeros((1,))}}, {"other": np.zeros((1,))})


def test_a_drawn_value_the_tree_cannot_use_is_an_error():
    with pytest.raises(AssertionError, match=r"drew \['spare'\]"):
        injected({"used": np.zeros((1,))}, {"used": np.zeros((1,)), "spare": np.zeros((1,))})


def test_a_shape_that_does_not_fit_is_an_error():
    with pytest.raises(AssertionError, match=r"w: drew \(3,\), wanted \(2,\)"):
        injected({"w": np.zeros((2,))}, {"w": np.zeros((3,))})


def test_the_two_sides_can_spell_a_parameter_differently():
    tree = {"cell": {"B_img": np.zeros((2,))}}
    out = injected(tree, {"B_imaginary": np.ones((2,))}, {"B_imaginary": "B_img"})
    assert out["cell"]["B_img"][0] == 1.0
```

- [ ] **Step 2: Implement**

`testbench/src/testbench/stimulus.py`:

```python
"""One draw, injected into two trees, with nothing left floating on either.

Two implementations of the same thing nest their parameters differently and
spell some of them differently, so the injection keys on the last element of
each path rather than on the path. What it does not allow is a leaf the draw
does not reach or a draw the tree cannot use: either one is a comparison that
would come out exact because it compared nothing.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

__all__ = ["injected"]


def _flat(tree: Mapping, prefix: tuple[str, ...] = ()) -> dict[tuple[str, ...], object]:
    out: dict[tuple[str, ...], object] = {}
    for key, value in tree.items():
        path = prefix + (str(key),)
        if isinstance(value, Mapping):
            out.update(_flat(value, path))
        else:
            out[path] = value
    return out


def _unflat(flat: Mapping[tuple[str, ...], object]) -> dict:
    out: dict = {}
    for path, value in flat.items():
        here = out
        for key in path[:-1]:
            here = here.setdefault(key, {})
        here[path[-1]] = value
    return out


def injected(
    tree: Mapping, values: Mapping, spelling: Mapping[str, str] | None = None
) -> dict:
    """Put the drawn values into a tree, and account for every one of both."""

    named = {(spelling or {}).get(name, name): value for name, value in values.items()}
    used: set[str] = set()
    out: dict[tuple[str, ...], object] = {}

    for path, leaf in _flat(tree).items():
        name = path[-1]
        assert name in named, f"nothing was drawn for {'/'.join(path)}"
        drawn, wanted = np.shape(named[name]), np.shape(leaf)
        assert drawn == wanted, f"{'/'.join(path)}: drew {drawn}, wanted {wanted}"
        out[path] = named[name]
        used.add(name)

    unused = sorted(set(named) - used)
    assert not unused, f"drew {unused}, which this tree has no parameter for"
    return _unflat(out)
```

- [ ] **Step 3: Export it**

```python
from testbench.gap import last_bits, relative
from testbench.leaves import flatten, paired
from testbench.stimulus import injected

__all__ = ["flatten", "injected", "last_bits", "paired", "relative"]
```

- [ ] **Step 4: Commit and push**

```bash
git add testbench
git commit -m "feat(testbench): inject one draw into two trees, leaving nothing floating"
git push
```

- [ ] **Step 5: Read the run**

Expected: six new tests pass, eighteen in the job.

---

### Task 5: The three verdicts

**Files:**
- Create: `testbench/src/testbench/verdict.py`
- Create: `testbench/tests/test_verdict.py`
- Modify: `testbench/src/testbench/__init__.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `testbench.verdict.Verdict`, an `IntEnum` with `UNBIASED = 0`, `ROUNDING = 1`, `ANOMALOUS = 2`, ordered by severity so `reached > expected` is the failure test.
  - `testbench.verdict.Measurement`, a frozen dataclass with fields `leaf: str`, `point: float`, `at: str`, `bits: float`, `relative: float`.
  - `testbench.verdict.judged(measurements: Iterable[Measurement], *, bits: float = 8.0, growth: float = 2.0) -> dict[str, Verdict]`.

- [ ] **Step 1: Write the failing test**

`testbench/tests/test_verdict.py`:

```python
import pytest

from testbench.verdict import Measurement, Verdict, judged


def taken(leaf, point, bits, rel, at="step=0"):
    return Measurement(leaf=leaf, point=point, at=at, bits=bits, relative=rel)


def test_a_leaf_that_never_differed_is_unbiased():
    reached = judged([taken("h", 5, 0.0, 0.0), taken("h", 40, 0.0, 0.0)])
    assert reached == {"h": Verdict.UNBIASED}


def test_a_small_gap_that_grows_with_the_accumulation_is_rounding():
    reached = judged([taken("h", 5, 1.0, 1e-7), taken("h", 40, 4.0, 4e-7)])
    assert reached == {"h": Verdict.ROUNDING}


def test_a_small_gap_that_does_not_grow_is_anomalous():
    reached = judged([taken("h", 5, 1.0, 1e-7), taken("h", 40, 1.0, 1.1e-7)])
    assert reached == {"h": Verdict.ANOMALOUS}


def test_a_growing_gap_that_is_too_wide_is_anomalous():
    reached = judged([taken("h", 5, 20.0, 1e-6), taken("h", 40, 90.0, 4e-6)])
    assert reached == {"h": Verdict.ANOMALOUS}


def test_rounding_cannot_be_reached_from_one_point():
    reached = judged([taken("h", 5, 1.0, 1e-7), taken("h", 5, 2.0, 2e-7)])
    assert reached == {"h": Verdict.ANOMALOUS}


def test_a_gap_that_appears_only_at_the_longer_run_is_rounding():
    reached = judged([taken("h", 5, 0.0, 0.0), taken("h", 40, 2.0, 2e-7)])
    assert reached == {"h": Verdict.ROUNDING}


def test_the_worst_measurement_at_a_point_is_the_one_that_counts():
    reached = judged(
        [
            taken("h", 5, 1.0, 1e-7),
            taken("h", 5, 0.0, 0.0),
            taken("h", 40, 4.0, 4e-7),
            taken("h", 40, 0.0, 0.0),
        ]
    )
    assert reached == {"h": Verdict.ROUNDING}


def test_each_leaf_is_judged_on_its_own():
    reached = judged(
        [
            taken("good", 5, 1.0, 1e-7),
            taken("good", 40, 4.0, 4e-7),
            taken("bad", 5, 1.0, 1e-7),
            taken("bad", 40, 1.0, 1e-7),
        ]
    )
    assert reached == {"good": Verdict.ROUNDING, "bad": Verdict.ANOMALOUS}


def test_the_thresholds_are_the_callers_to_move():
    stubborn = [taken("h", 5, 1.0, 1e-7), taken("h", 40, 1.0, 1.5e-7)]
    assert judged(stubborn)["h"] is Verdict.ANOMALOUS
    assert judged(stubborn, growth=1.4)["h"] is Verdict.ROUNDING


def test_severity_orders_the_three():
    assert Verdict.UNBIASED < Verdict.ROUNDING < Verdict.ANOMALOUS
```

- [ ] **Step 2: Implement**

`testbench/src/testbench/verdict.py`:

```python
"""What kind of disagreement this is, rather than how much of it is allowed.

Three classes. Bit-identical, of the kind rounding produces, or something else.
The middle one is the only one that has to be established rather than observed,
and it is established by driving the same comparison at more than one point on
an axis that lengthens the accumulation: rounding piles up as the accumulation
grows, and a multiplicative difference in what is computed does not.

That is a weak discriminator -- around one order of magnitude between the two
hypotheses, where recomputing at double precision would give eight -- and it is
here because it costs one extra run rather than a precision-parametric
implementation. It is paired with a magnitude condition because neither growth
nor precision can tell an amplified rounding error from an ordinary one, and an
error that scales correctly while being enormous is a statement about the
conditioning of the computation.

``UNBIASED`` is bit-identity and not a statistical claim. Rounding error is
zero-mean for reassociated sums and products of a sign-symmetric draw, because
IEEE arithmetic is exact under negation, but that symmetry does not survive a
non-odd function or two differently shaped reduction trees -- and testing a mean
needs far more samples than testing a maximum. So bias is worth reporting and is
not worth failing on.
"""

from __future__ import annotations

import enum
from collections.abc import Iterable
from dataclasses import dataclass

__all__ = ["Measurement", "Verdict", "judged"]


class Verdict(enum.IntEnum):
    """The three classes, ordered by severity so they can be compared."""

    UNBIASED = 0
    ROUNDING = 1
    ANOMALOUS = 2


@dataclass(frozen=True)
class Measurement:
    """One leaf, compared once, somewhere on the axis.

    ``point`` is how much accumulation this run carried, in whatever unit the
    caller counts terms in; only its ordering is used. ``at`` says where in the
    run the measurement was taken and is carried for the report alone.
    """

    leaf: str
    point: float
    at: str
    bits: float
    relative: float


def _grew(taken: list[Measurement], growth: float) -> bool:
    """Whether the relative gap is larger where more was accumulated."""

    points = sorted({one.point for one in taken})
    if len(points) < 2:
        return False
    worst = {
        point: max(one.relative for one in taken if one.point == point)
        for point in points
    }
    least, most = worst[points[0]], worst[points[-1]]
    if least == 0.0:
        return most > 0.0
    return most / least >= growth


def judged(
    measurements: Iterable[Measurement], *, bits: float = 8.0, growth: float = 2.0
) -> dict[str, Verdict]:
    """One verdict per leaf, from every measurement taken of it."""

    by_leaf: dict[str, list[Measurement]] = {}
    for one in measurements:
        by_leaf.setdefault(one.leaf, []).append(one)

    reached = {}
    for leaf, taken in by_leaf.items():
        if all(one.bits == 0.0 for one in taken):
            reached[leaf] = Verdict.UNBIASED
        elif max(one.bits for one in taken) > bits:
            reached[leaf] = Verdict.ANOMALOUS
        elif _grew(taken, growth):
            reached[leaf] = Verdict.ROUNDING
        else:
            reached[leaf] = Verdict.ANOMALOUS
    return reached
```

- [ ] **Step 3: Export it**

```python
from testbench.gap import last_bits, relative
from testbench.leaves import flatten, paired
from testbench.stimulus import injected
from testbench.verdict import Measurement, Verdict, judged

__all__ = [
    "Measurement",
    "Verdict",
    "flatten",
    "injected",
    "judged",
    "last_bits",
    "paired",
    "relative",
]
```

- [ ] **Step 4: Commit and push**

```bash
git add testbench
git commit -m "feat(testbench): name the kind of a disagreement, not the size allowed"
git push
```

- [ ] **Step 5: Read the run**

Expected: ten new tests pass, twenty-eight in the job.

---

### Task 6: The scoreboard

**Files:**
- Create: `testbench/src/testbench/scoreboard.py`
- Create: `testbench/tests/test_scoreboard.py`
- Modify: `testbench/src/testbench/__init__.py`

**Interfaces:**
- Consumes: `paired`, `last_bits`, `relative`, `Measurement`, `Verdict`, `judged`.
- Produces: `testbench.scoreboard.Scoreboard(what: str)` with
  - `.watch(*, point: float, at: str, ours: Mapping, theirs: Mapping) -> None`
  - `.close(*, expect: Verdict, bits: float = 8.0, growth: float = 2.0) -> None`, which raises `AssertionError` naming every leaf whose verdict is more severe than `expect`, worst first.

- [ ] **Step 1: Write the failing test**

`testbench/tests/test_scoreboard.py`:

```python
import numpy as np
import pytest

from testbench.scoreboard import Scoreboard
from testbench.verdict import Verdict


def stream(board, point, gaps):
    """Drive one run: a reference of ones, ours off by each gap in turn."""

    for step, gap in enumerate(gaps):
        board.watch(
            point=point,
            at=f"step={step}",
            ours={"h": np.array([1.0 + gap], np.float64)},
            theirs={"h": np.array([1.0], np.float64)},
        )


def test_a_comparison_that_agrees_everywhere_closes_quietly():
    board = Scoreboard("agreeing")
    stream(board, 5, [0.0, 0.0])
    stream(board, 40, [0.0, 0.0])
    board.close(expect=Verdict.UNBIASED)


def test_a_growing_small_gap_closes_when_rounding_is_expected():
    board = Scoreboard("rounding")
    stream(board, 5, [0.0, 1e-7])
    stream(board, 40, [0.0, 5e-7])
    board.close(expect=Verdict.ROUNDING)


def test_the_same_gap_fails_when_bit_identity_is_expected():
    board = Scoreboard("strict")
    stream(board, 5, [1e-7])
    stream(board, 40, [5e-7])
    with pytest.raises(AssertionError, match="strict.*worse than UNBIASED"):
        board.close(expect=Verdict.UNBIASED)


def test_a_gap_that_does_not_grow_fails_and_names_the_leaf():
    board = Scoreboard("stubborn")
    stream(board, 5, [1e-7])
    stream(board, 40, [1e-7])
    with pytest.raises(AssertionError, match="ANOMALOUS  h"):
        board.close(expect=Verdict.ROUNDING)


def test_the_report_says_where_the_worst_was_reached():
    board = Scoreboard("located")
    stream(board, 5, [0.0, 1e-7])
    stream(board, 40, [0.0, 0.0, 9e-3])
    with pytest.raises(AssertionError, match="step=2, point=40"):
        board.close(expect=Verdict.ROUNDING)


def test_the_report_lists_the_gap_at_every_point():
    board = Scoreboard("surveyed")
    stream(board, 5, [1e-7])
    stream(board, 40, [1e-7])
    with pytest.raises(AssertionError, match="point=5 rel=1e-07.*point=40 rel=1e-07"):
        board.close(expect=Verdict.ROUNDING)


def test_closing_with_nothing_watched_is_an_error():
    with pytest.raises(AssertionError, match="empty: nothing was watched"):
        Scoreboard("empty").close(expect=Verdict.UNBIASED)


def test_an_unpaired_probe_is_refused_while_watching():
    board = Scoreboard("unpaired")
    with pytest.raises(AssertionError, match="answers"):
        board.watch(
            point=5,
            at="step=0",
            ours={"h": np.zeros((1,))},
            theirs={"h": np.zeros((1,)), "y": np.zeros((1,))},
        )
```

- [ ] **Step 2: Implement**

`testbench/src/testbench/scoreboard.py`:

```python
"""Watching a whole run, then judging once.

Stopping at the first disagreement measures almost nothing where quantities
accumulate: an influence matrix starts at zero and so does the state it is built
from, so at the first step the two sides agree for a reason that proves nothing.
Every step is watched and the worst each leaf reached is kept, along with where
it reached it.

Closing is an explicit call rather than a context manager. A context manager
would make the closing impossible to forget, and it would move the failure's
line number from the comparison to the end of the block; the line number is
worth more.
"""

from __future__ import annotations

from collections.abc import Mapping

from testbench.gap import last_bits, relative
from testbench.leaves import paired
from testbench.verdict import Measurement, Verdict, judged

__all__ = ["Scoreboard"]


class Scoreboard:
    """Every measurement of one comparison, and one verdict per leaf at the end."""

    def __init__(self, what: str) -> None:
        self.what = what
        self._taken: list[Measurement] = []

    def watch(self, *, point: float, at: str, ours: Mapping, theirs: Mapping) -> None:
        """Measure every paired leaf once, here."""

        for leaf, mine, yours in paired(ours, theirs):
            self._taken.append(
                Measurement(
                    leaf=leaf,
                    point=float(point),
                    at=at,
                    bits=last_bits(yours, mine),
                    relative=relative(yours, mine),
                )
            )

    def close(
        self, *, expect: Verdict, bits: float = 8.0, growth: float = 2.0
    ) -> None:
        """Every leaf is at most as severe as expected, or this raises."""

        assert self._taken, f"{self.what}: nothing was watched"
        reached = judged(self._taken, bits=bits, growth=growth)
        worse = {leaf: got for leaf, got in reached.items() if got > expect}
        if worse:
            raise AssertionError(self._report(worse, expect))

    def _of(self, leaf: str) -> list[Measurement]:
        return [one for one in self._taken if one.leaf == leaf]

    def _report(self, worse: Mapping[str, Verdict], expect: Verdict) -> str:
        lines = []
        for leaf in sorted(
            worse, key=lambda name: -max(one.bits for one in self._of(name))
        ):
            taken = self._of(leaf)
            peak = max(taken, key=lambda one: one.bits)
            points = sorted({one.point for one in taken})
            survey = "  ".join(
                f"point={point:g} "
                f"rel={max(one.relative for one in taken if one.point == point):.3g}"
                for point in points
            )
            lines.append(
                f"  {worse[leaf].name}  {leaf}\n"
                f"    worst {peak.bits:.1f} last bits at {peak.at}, "
                f"point={peak.point:g}\n"
                f"    {survey}"
            )
        counted = len({one.leaf for one in self._taken})
        return (
            f"{self.what}: {len(worse)} of {counted} leaves are worse than "
            f"{expect.name}, worst first:\n" + "\n".join(lines)
        )
```

- [ ] **Step 3: Export it**

Add `from testbench.scoreboard import Scoreboard` and `"Scoreboard"` to `__all__`.

- [ ] **Step 4: Commit and push**

```bash
git add testbench
git commit -m "feat(testbench): keep the worst of a whole run, and judge it once"
git push
```

- [ ] **Step 5: Read the run**

Expected: eight new tests pass, thirty-six in the job.

---

### Task 7: The LRU fixture becomes a module, with a width axis

**Files:**
- Create: `memo/tests/fixtures/__init__.py`
- Create: `memo/tests/fixtures/lru.py`
- Modify: `memo/tests/test_lru_parity.py` (remove lines 73-355, import from the fixture)
- Modify: `memo/tests/test_lru_precision.py:48-60` (import from the fixture)
- Modify: `memo/pyproject.toml` (add the `testbench` path source)

**Interfaces:**
- Consumes: nothing from the package yet.
- Produces, all from `fixtures.lru`:
  - `Shape(hidden: int = 3, features: int = 4)`, a frozen dataclass.
  - `PAPER: dict[str, str]`, `OURS: dict[str, str]`
  - `drawn(seed: int, shape: Shape) -> dict[str, jax.Array]`
  - `inputs(seed: int, shape: Shape, *, steps: int = 5) -> tuple[jax.Array, jax.Array]`
  - `widened(values: dict, dtype) -> dict`
  - `input_gain(values: dict) -> jax.Array`
  - `paper_side(seed, shape, *, dtype=jnp.complex64)` returning `(layer, params, carry)`
  - `paper_step(layer, params, carry, x) -> tuple[Any, Any]`
  - `our_side(seed, shape, *, skip=0.0, cell=LRUCell, dtype=jnp.complex64)` returning `(core, params, carry, sensitivity)`
  - `our_step(core, params, carry, sensitivity, x) -> tuple[Any, Any, Any]`
  - `expected_sensitivity(values, traces, *, correcting=True) -> dict`

- [ ] **Step 1: Add the package as a path dependency**

In `memo/pyproject.toml`, under `[tool.uv.sources]`:

```toml
testbench = { path = "../testbench" }
```

And add `"testbench"` to the `development` dependency group.

- [ ] **Step 2: Move the fixture**

Create `memo/tests/fixtures/__init__.py` as an empty file.

Create `memo/tests/fixtures/lru.py` containing, verbatim from `test_lru_parity.py:73-355`, the constants `PAPER` and `OURS` and the functions `drawn`, `inputs`, `_inject`, `widened`, `paper_side`, `paper_step`, `our_side`, `our_step`, `input_gain`, `expected_sensitivity`, together with the imports they need. Do not move `watch`, `assert_explained`, `READOUT`, `INFLUENCE`, `CREDITED` or `UNROLLED`: the first two are superseded by `Scoreboard` and the four tables are what the conversion removes, so they stay in the case module until each is converted.

Give it this docstring:

```python
"""Mounting our LRU and the published one so the two can be driven together.

Everything specific to this pair lives here: how each is built, how its
parameters are injected, how one transition is taken, and how the reference's
three influence matrices translate into our five sensitivities. What is compared
and what is demanded of the comparison lives in the case modules.

The axes are ``seed``, ``shape`` and ``steps``, plus ``dtype`` for a
double-precision run and ``cell`` for choosing which of the three arms is
mounted. ``skip`` is not an axis: it is the lever one vacuity guard pulls to
establish that injecting zeros for the term the reference lacks was doing work.
"""
```

- [ ] **Step 3: Replace the width constants with the shape**

At the top of `fixtures/lru.py`, in place of `HIDDEN = 3` and `FEATURES = 4`:

```python
@dataclass(frozen=True)
class Shape:
    """How wide the two are mounted.

    ``hidden`` and the output width are the same number and cannot be separated
    here: the published layer reads out with one matrix whose row count is the
    field it was constructed with. Ours can separate them, and here it must not,
    or the two Cs differ in shape before any arithmetic is compared.

    Width is an axis rather than a constant because it is the second independent
    way to lengthen an accumulation. Each contraction reduces over it, so
    rounding grows with it while a multiplicative difference in what is computed
    does not.
    """

    hidden: int = 3
    features: int = 4


NARROW = Shape(3, 4)
WIDE = Shape(6, 8)
```

Then thread `shape` through every function that used the constants:

```python
def drawn(seed: int, shape: Shape) -> dict:
    keys = jax.random.split(jax.random.key(seed), 7)
    return {
        "nu_log": 0.4 * jax.random.normal(keys[0], (shape.hidden,), jnp.float32),
        "theta_log": 0.4 * jax.random.normal(keys[1], (shape.hidden,), jnp.float32),
        "gamma_log": -1.0
        + 0.3 * jax.random.normal(keys[2], (shape.hidden,), jnp.float32),
        "B_real": jax.random.normal(
            keys[3], (shape.hidden, shape.features), jnp.float32
        ),
        "B_imaginary": jax.random.normal(
            keys[4], (shape.hidden, shape.features), jnp.float32
        ),
        "C_real": jax.random.normal(
            keys[5], (shape.hidden, shape.hidden), jnp.float32
        ),
        "C_imaginary": jax.random.normal(
            keys[6], (shape.hidden, shape.hidden), jnp.float32
        ),
    }


def inputs(seed: int, shape: Shape, *, steps: int = 5):
    keys = jax.random.split(jax.random.key(seed + 1000), 2)
    xs = jax.random.normal(keys[0], (steps, shape.features), jnp.float32)
    weights = jax.random.normal(keys[1], (shape.hidden,), jnp.float32)
    return xs, weights
```

`paper_side` and `our_side` take `shape` as their second positional argument and use `shape.hidden` and `shape.features` everywhere the constants appeared, including `OnlineLRULayer(d_hidden=shape.hidden)` and
`LRUConfig(features=shape.features, hidden_dim=shape.hidden, output_dim=shape.hidden)`.

- [ ] **Step 4: Replace the private injector with the package's**

`_inject` is the totality check from Task 4 written for one pair. Delete it and
call the package's, which does the same accounting and reports the offending
path. Flax's `FrozenDict` is a `Mapping`, so `injected` walks it unchanged and
returns plain dicts, which is what the two constructors already accept.

In `fixtures/lru.py`, remove `_inject` and its `traverse_util` import, and have
`paper_side` and `our_side` call:

```python
from testbench import injected

params = injected(layer.init(key, x)["params"], values, PAPER)
```

```python
params = injected(core.init(key, carry, x)["params"], values, OURS)
```

`PAPER` and `OURS` already have exactly the shape `injected` wants for its
`spelling` argument: our name for a parameter mapped to theirs.

- [ ] **Step 5: Point both case modules at the fixture**

At the top of `test_lru_parity.py`, in place of the moved code:

```python
from fixtures.lru import (
    NARROW,
    OURS,
    PAPER,
    Shape,
    drawn,
    expected_sensitivity,
    input_gain,
    inputs,
    our_side,
    our_step,
    paper_side,
    paper_step,
    widened,
)
```

Every call site in the remaining tests gains `NARROW` as its second argument: `paper_side(seed, NARROW)`, `our_side(seed, NARROW)`, `drawn(seed, NARROW)`, `inputs(seed, NARROW)`, `our_side(seed, NARROW, cell=PublishedLRUCell)`, `our_side(seed, NARROW, skip=skip)`. Where a test used `HIDDEN`, `FEATURES` or `STEPS`, use `NARROW.hidden`, `NARROW.features` and the `steps` default.

In `test_lru_precision.py`, change the import block at lines 48-60 to read from `fixtures.lru` instead of `test_lru_parity`, drop `FEATURES` and `HIDDEN` in favour of `NARROW`, and pass `NARROW` at every call site.

- [ ] **Step 6: Commit and push**

```bash
git add memo/tests/fixtures memo/tests/test_lru_parity.py memo/tests/test_lru_precision.py memo/pyproject.toml memo/uv.lock
git commit -m "test(lru): give the two sides a fixture of their own, and a width to vary"
git push
```

- [ ] **Step 7: Read the run**

```bash
gh run list --workflow=memo-ci.yml --limit 1
gh run view <id> --log-failed
```

Expected: the same tests pass as before, with no numbers changed. This task moves code and adds an axis; nothing it does may alter a measurement. If any leaf's gap changes, the move was not verbatim — find the difference rather than adjusting a table.

---

### Task 8: The influence matrices are judged by kind

**Files:**
- Modify: `memo/tests/test_lru_parity.py` — the `INFLUENCE` table and `test_the_influence_matrices_are_the_papers`

**Interfaces:**
- Consumes: `Scoreboard`, `Verdict`, `flatten` from `testbench`; `NARROW`, `WIDE`, `drawn`, `expected_sensitivity`, `inputs`, `our_side`, `our_step`, `paper_side`, `paper_step` from `fixtures.lru`.
- Produces: nothing other tasks rely on.

- [ ] **Step 1: Replace the test**

Delete the `INFLUENCE` table and the eighteen lines of prose above it that justify its entries. That prose contains a contradiction — it calls `theta_log` the widest of the group while the table gives `theta_log` 4.0 and `gamma_log` 8.0 — and there is nothing to carry forward from it.

Replace `test_the_influence_matrices_are_the_papers` with:

```python
# Where each run sits on the axis that lengthens the accumulation, counted as
# the terms the widest leaf sums: one per step, each a contraction over the
# feature width. Four runs spanning twenty to three hundred and twenty, so a
# rounding gap has a factor of four to grow by while a constant relative
# difference has none.
def accumulated(shape, steps: int) -> float:
    return float(steps * shape.features)


@pytest.mark.parametrize("seed", range(4))
def test_the_influence_matrices_are_the_papers(seed):
    """Block: what the real-time credit accumulates.

    This is the quantity RTRL exists for, and the one place the two
    implementations were always going to be hardest to compare: theirs is three
    matrices carried in the cell's own carry, ours is five sensitivities keyed by
    the parameter each credits. ``expected_sensitivity`` is the map between them,
    and it is arithmetic on their matrices rather than a restatement of ours.

    Each run is watched whole because the first step cannot judge two of the
    five: ``nu_log`` and ``theta_log`` are built from the previous carry, which
    is zero there, so they agree at step zero however either side computes them.

    What is demanded is that no leaf be worse than rounding, and rounding is not
    a size here -- it is the requirement that the gap grow as the accumulation
    does. The two implementations chain each parameter's derivative at opposite
    ends of that accumulation, ours before and theirs after, so what separates
    them is a constant distributed over a sum. That is exact in arithmetic, is
    not exact in float32, and piles up per term. A difference in what is
    computed would not pile up, and the published influence matrix for ``B`` is
    the case in point: its missing exponential is a fixed factor that four
    lengths of stream would leave untouched.
    """

    board = Scoreboard(f"influence matrices seed={seed}")
    for shape in (NARROW, WIDE):
        for steps in (5, 40):
            layer, paper_params, paper_carry = paper_side(seed, shape)
            core, our_params, our_carry, sensitivity = our_side(seed, shape)
            values = drawn(seed, shape)
            xs, _ = inputs(seed, shape, steps=steps)

            for step, x in enumerate(xs):
                paper_carry, _ = paper_step(layer, paper_params, paper_carry, x)
                our_carry, _, sensitivity = our_step(
                    core, our_params, our_carry, sensitivity, x
                )
                board.watch(
                    point=accumulated(shape, steps),
                    at=f"{shape.hidden}x{shape.features} step={step}",
                    ours=flatten(jax.tree.map(lambda leaf: leaf[0, 0], sensitivity)),
                    theirs=flatten(expected_sensitivity(values, paper_carry[1])),
                )
    board.close(expect=Verdict.ROUNDING)
```

Add to the imports at the top of the file:

```python
from testbench import Scoreboard, Verdict, flatten
```

and add `WIDE` to the `fixtures.lru` import list.

- [ ] **Step 2: Prove the comparison can still fail**

Add beside it:

```python
def test_the_influence_comparison_would_notice_a_changed_formula():
    """The block above passes; this is what makes that worth something.

    ``expected_sensitivity`` scales the reference's matrix for ``B`` by
    ``exp(gamma_log) / gamma_log`` to undo the exponential its published
    revision is missing. Removing that correction leaves a real difference of a
    known kind -- a fixed relative factor, the class the growth condition exists
    to catch -- and the same comparison has to reach ``ANOMALOUS`` on it.
    """

    board = Scoreboard("influence matrices, correction removed")
    for shape in (NARROW, WIDE):
        for steps in (5, 40):
            layer, paper_params, paper_carry = paper_side(0, shape)
            core, our_params, our_carry, sensitivity = our_side(0, shape)
            values = drawn(0, shape)
            xs, _ = inputs(0, shape, steps=steps)

            for step, x in enumerate(xs):
                paper_carry, _ = paper_step(layer, paper_params, paper_carry, x)
                our_carry, _, sensitivity = our_step(
                    core, our_params, our_carry, sensitivity, x
                )
                board.watch(
                    point=accumulated(shape, steps),
                    at=f"{shape.hidden}x{shape.features} step={step}",
                    ours=flatten(jax.tree.map(lambda leaf: leaf[0, 0], sensitivity)),
                    theirs=flatten(
                        expected_sensitivity(values, paper_carry[1], correcting=False)
                    ),
                )
    with pytest.raises(AssertionError, match="ANOMALOUS  B_real"):
        board.close(expect=Verdict.ROUNDING)
```

- [ ] **Step 3: Commit and push**

```bash
git add memo/tests/test_lru_parity.py
git commit -m "test(lru): ask what kind the influence gap is, not how much is allowed"
git push
```

- [ ] **Step 4: Read the run**

```bash
gh run list --workflow=memo-ci.yml --limit 1
gh run view <id> --log-failed
```

Expected: both tests pass. Two outcomes are informative rather than wrong and neither should be met by loosening a threshold:

If a leaf reaches `ANOMALOUS` because its gap did not grow, that leaf's disagreement is not accumulation. Read the survey line the report prints — the relative gap at each of the four points — and find what is constant about it. That is the finding, not the failure.

If a leaf reaches `ANOMALOUS` on magnitude at the wide shape, the gap grows faster with width than the eight last bits allow. Record the measured value in the failure's own terms before deciding whether the threshold or the arithmetic is what should move.

- [ ] **Step 5: Judge the conversion by reading it**

Open the converted test and check three things. The quantity, the direction and the kind of agreement demanded are all visible at the comparison. The eighteen justifying lines are gone rather than relocated. Nothing in the case module mounts an implementation, injects a parameter, or translates between the two.

If any of the three fails, the boundary between the fixture and the case is in the wrong place, and that is worth fixing before converting `READOUT`, `CREDITED` and `UNROLLED` the same way.
