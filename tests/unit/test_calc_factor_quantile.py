import unittest

import numpy as np
from numpy.testing import assert_allclose
from tmma.normalisation.tmm import _calc_factor_quantile


class TestCalcFactorQuantile(unittest.TestCase):

    def test_with_standard_parameters(self):
        data = np.array([[1, 4, 7, 10], [2, 5, 8, 11], [3, 6, 9, 12]])
        # Sum of columns
        lib_size = np.array([6, 15, 24, 33])

        # From R
        expected_answer = np.array([0.4166667, 0.3666667, 0.3541667, 0.3484848])
        actual_answer = _calc_factor_quantile(data, lib_sizes=lib_size)
        assert_allclose(expected_answer, actual_answer, rtol=1e-06)

    def test_with_different_libsize(self):

        data = np.array([[1, 4, 7, 10], [2, 5, 8, 11], [3, 6, 9, 12]])
        # Random numbers
        lib_size = np.array([1, 2, 3, 4])

        # From R
        expected_answer = np.array([2.500000, 2.750000, 2.833333, 2.875000])
        actual_answer = _calc_factor_quantile(data, lib_sizes=lib_size)
        assert_allclose(expected_answer, actual_answer, rtol=1e-06)

    def test_with_different_libsize_and_p(self):
        data = np.array([[1, 4, 7, 10], [2, 5, 8, 11], [3, 6, 9, 12]])
        # Random numbers
        lib_size = np.array([1, 2, 3, 4])
        # Non-standard p
        p = 0.21

        # From R
        expected_answer = np.array([1.420000, 2.210000, 2.473333, 2.605000])
        actual_answer = _calc_factor_quantile(data, lib_sizes=lib_size, p=p)
        assert_allclose(expected_answer, actual_answer, rtol=1e-06)