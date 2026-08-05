"""The gap metric, on the one input it used to answer zero for.

``last_bits`` widened both sides to float64 before subtracting. numpy discards
the imaginary part of a complex array on that cast and only warns, so every
disagreement that lived in the imaginary part was measured as no disagreement
at all -- across the hidden state and all five sensitivities, which is every
complex quantity a recurrent cell carries.
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
