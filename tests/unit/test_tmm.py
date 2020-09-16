import unittest

import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal, assert_allclose
from tmma.tmm import tmm_normalise, _scale_tmm_factors


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

        # Degenerate - only one column
        counts = np.array([[1], [2], [3]])
        actual_answer = tmm_normalise(counts)
        assert_array_equal(actual_answer, [1.0])

        # Degenerate - this actually collapses to zero samples
        # as the zero-row is removed
        counts = np.array([[0, 0, 0, 0]])
        actual_answer = tmm_normalise(counts)
        assert_array_equal(actual_answer, expected_answer)


    def test_scaling_of_factors(self):

        factors_unscaled = np.array([
            1.0000000, 1.0147745, 0.9489405, 1.0807636, 1.1186484, 0.9771662
        ])

        expected_answer = np.array([
            0.9787380, 0.9931984, 0.9287641, 1.0577844, 1.0948637, 0.9563897
        ])

        actual_answer = _scale_tmm_factors(factors_unscaled)
        assert_allclose(actual_answer, expected_answer)


