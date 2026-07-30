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
