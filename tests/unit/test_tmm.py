import unittest
import numpy as np
import pandas as pd
from tmma.tmm import tmm_normalise, _calc_factor_quantile
from numpy.testing import assert_array_equal, assert_almost_equal, assert_allclose


class TestTMMNormalisation(unittest.TestCase):


    def test_providing_nan_counts_throws_valueerror(self):
        ## This is what EdgeR does with NaNs
        counts = np.array([[1, 2], [np.nan, 4]])
        self.assertRaises(ValueError, tmm_normalise, counts)

        counts = np.array([[1, 2], [None, 4]])
        self.assertRaises(ValueError, tmm_normalise, counts)

        counts = pd.DataFrame([[1, 2], [None, 4]])
        self.assertRaises(ValueError, tmm_normalise, counts)

    def test_providing_inf_counts_throws_valueerror(self):
        ## This is what EdgeR does with NaNs
        counts = np.array([[1, 2], [np.inf, 4]])
        self.assertRaises(ValueError, tmm_normalise, counts)

        counts = np.array([[1, 2], [-np.inf, 4]])
        self.assertRaises(ValueError, tmm_normalise, counts)

    def test_providing_nan_libsize_throws_valueerror(self):
        ## This is what EdgeR does with NaNs
        counts = np.array([[1, 2], [3, 4]])
        lib_sizes = np.array([np.nan, 1])
        self.assertRaises(ValueError, tmm_normalise, counts, lib_sizes=lib_sizes)

    def test_providing_inf_libsize_throws_valueerror(self):
        ## This is what EdgeR does with NaNs
        counts = np.array([[1, 2], [3, 4]])
        lib_sizes = np.array([np.inf, 1])
        self.assertRaises(ValueError, tmm_normalise, counts, lib_sizes=lib_sizes)

        lib_sizes = np.array([-np.inf, 1])
        self.assertRaises(ValueError, tmm_normalise, counts, lib_sizes=lib_sizes)

    def test_providing_wrong_libsize_throws_valueerror(self):

        counts = np.array([[1, 2], [3, 4]])
        # Too small
        lib_sizes = np.array([0])
        self.assertRaises(ValueError, tmm_normalise, counts, lib_sizes=lib_sizes)

        # Too big
        lib_sizes = np.array([0, 1, 2])
        self.assertRaises(ValueError, tmm_normalise, counts, lib_sizes=lib_sizes)

        # Wrong dimensionality
        lib_sizes = np.array([[0, 1], [2, 3]])
        self.assertRaises(ValueError, tmm_normalise, counts, lib_sizes=lib_sizes)

    def test_providing_wrong_refcol_throws_error(self):

        counts = np.array([[1, 2, 3], [3, 4, 5]])

        # Invalid
        self.assertRaises(ValueError, tmm_normalise, counts, ref_column=-1)

        # Also invalid
        # Invalid
        self.assertRaises(ValueError, tmm_normalise, counts, ref_column=0.1)

        # Too large
        self.assertRaises(ValueError, tmm_normalise, counts, ref_column=3)

    def test_degenerate_cases_return_ones(self):

        # For degenerate cases edgeR returns all ones
        expected_answer = np.array([1.0, 1.0, 1.0, 1.0])

        # Degenerate - all zeroes
        counts = np.array([[0, 0, 0, 0]])
        actual_answer = tmm_normalise(counts)
        assert_array_equal(actual_answer, expected_answer)

        # Degenerate - only one sample
        counts = np.array([[1, 2, 3, 4]])
        actual_answer = tmm_normalise(counts)
        assert_array_equal(actual_answer, expected_answer)

        # Degenerate - this actually collapses to only one sample
        # as the zero-row is removed
        counts = np.array([[0, 0, 0, 0], [1, 2, 3, 4]])
        actual_answer = tmm_normalise(counts)
        assert_array_equal(actual_answer, expected_answer)



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
