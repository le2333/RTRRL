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
