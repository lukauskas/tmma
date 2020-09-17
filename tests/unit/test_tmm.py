import unittest

import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal, assert_allclose
from tmma.normalisation.tmm import tmm_normalisation_factors, scale_tmm_factors


class TestTMMNormalisation(unittest.TestCase):

    def test_toy_example_works(self):

        counts = np.array([[1, 4], [2, 5], [3,6]])
        expected_answer = [1.0276039, 0.9731376]

        actual_answer = tmm_normalisation_factors(counts)
        assert_allclose(expected_answer, actual_answer, rtol=1e-6)

    def test_toy_example_with_libsizes_works(self):
        counts = np.array([[1, 4], [2, 5], [3, 6]])
        lib_sizes = np.array([5, 6])

        expected_answer = [0.7276626, 1.3742633]

        actual_answer = tmm_normalisation_factors(counts, lib_sizes=lib_sizes)
        assert_allclose(expected_answer, actual_answer, rtol=1e-6)

    def test_library_sizes_can_be_provided_as_lists(self):
        counts = np.array([[1, 4], [2, 5], [3, 6]])
        lib_sizes = [5, 6]

        expected_answer = [0.7276626, 1.3742633]

        actual_answer = tmm_normalisation_factors(counts, lib_sizes=lib_sizes)
        assert_allclose(expected_answer, actual_answer, rtol=1e-6)

    def test_library_size_of_zero_raises_error(self):
        counts = np.array([[1, 4], [2, 5], [3, 6]])

        lib_sizes = [0, 1]
        self.assertRaises(ValueError, tmm_normalisation_factors, counts, lib_sizes=lib_sizes)

        # Library size of second column is zero here
        counts = np.array([[1, 0], [2, 0], [3, 0]])
        self.assertRaises(ValueError, tmm_normalisation_factors, counts)

    def test_providing_nan_counts_throws_valueerror(self):
        ## This is what EdgeR does with NaNs
        counts = np.array([[1, 2], [np.nan, 4]])
        self.assertRaises(ValueError, tmm_normalisation_factors, counts)

        counts = np.array([[1, 2], [None, 4]])
        self.assertRaises(ValueError, tmm_normalisation_factors, counts)

        counts = pd.DataFrame([[1, 2], [None, 4]])
        self.assertRaises(ValueError, tmm_normalisation_factors, counts)

    def test_providing_inf_counts_throws_valueerror(self):
        ## This is what EdgeR does with NaNs
        counts = np.array([[1, 2], [np.inf, 4]])
        self.assertRaises(ValueError, tmm_normalisation_factors, counts)

        counts = np.array([[1, 2], [-np.inf, 4]])
        self.assertRaises(ValueError, tmm_normalisation_factors, counts)

    def test_providing_nan_libsize_throws_valueerror(self):
        ## This is what EdgeR does with NaNs
        counts = np.array([[1, 2], [3, 4]])
        lib_sizes = np.array([np.nan, 1])
        self.assertRaises(ValueError, tmm_normalisation_factors, counts, lib_sizes=lib_sizes)

    def test_providing_inf_libsize_throws_valueerror(self):
        ## This is what EdgeR does with NaNs
        counts = np.array([[1, 2], [3, 4]])
        lib_sizes = np.array([np.inf, 1])
        self.assertRaises(ValueError, tmm_normalisation_factors, counts, lib_sizes=lib_sizes)

        lib_sizes = np.array([-np.inf, 1])
        self.assertRaises(ValueError, tmm_normalisation_factors, counts, lib_sizes=lib_sizes)

    def test_providing_wrong_libsize_throws_valueerror(self):

        counts = np.array([[1, 2], [3, 4]])
        # Too small
        lib_sizes = np.array([0])
        self.assertRaises(ValueError, tmm_normalisation_factors, counts, lib_sizes=lib_sizes)

        # Too big
        lib_sizes = np.array([0, 1, 2])
        self.assertRaises(ValueError, tmm_normalisation_factors, counts, lib_sizes=lib_sizes)

        # Wrong dimensionality
        lib_sizes = np.array([[0, 1], [2, 3]])
        self.assertRaises(ValueError, tmm_normalisation_factors, counts, lib_sizes=lib_sizes)

    def test_providing_wrong_refcol_throws_error(self):

        counts = np.array([[1, 2, 3], [3, 4, 5]])

        # Invalid
        self.assertRaises(ValueError, tmm_normalisation_factors, counts, ref_column=-1)

        # Also invalid
        # Invalid
        self.assertRaises(ValueError, tmm_normalisation_factors, counts, ref_column=0.1)

        # Too large
        self.assertRaises(ValueError, tmm_normalisation_factors, counts, ref_column=3)

        # Letters not allowed for non-pandas
        self.assertRaises(ValueError, tmm_normalisation_factors, counts, ref_column='a')

        counts_df = pd.DataFrame(counts, columns=['a', 'b', 'c'])

        # Letter not found
        self.assertRaises(ValueError, tmm_normalisation_factors, counts, ref_column='o')

    def test_degenerate_cases_return_ones(self):

        # For degenerate cases edgeR returns all ones
        expected_answer = np.array([1.0, 1.0, 1.0, 1.0])

        # Degenerate - all zeroes
        counts = np.array([[0, 0, 0, 0]])
        actual_answer = tmm_normalisation_factors(counts)
        assert_array_equal(actual_answer, expected_answer)

        # Degenerate - only one column
        counts = np.array([[1], [2], [3]])
        actual_answer = tmm_normalisation_factors(counts)
        assert_array_equal(actual_answer, [1.0])

        # Degenerate - this actually collapses to zero samples
        # as the zero-row is removed
        counts = np.array([[0, 0, 0, 0]])
        actual_answer = tmm_normalisation_factors(counts)
        assert_array_equal(actual_answer, expected_answer)


    def test_scaling_of_factors(self):

        factors_unscaled = np.array([
            1.0000000, 1.0147745, 0.9489405, 1.0807636, 1.1186484, 0.9771662
        ])

        expected_answer = np.array([
            0.9787380, 0.9931984, 0.9287641, 1.0577844, 1.0948637, 0.9563897
        ])

        actual_answer = scale_tmm_factors(factors_unscaled)
        assert_allclose(actual_answer, expected_answer)

    def test_colnames_are_respected_when_providing_a_dataframe(self):
        counts = np.array([[1, 2, 3, 4],
                           [2, 3, 4, 5]])

        counts = pd.DataFrame(counts, columns=['a', 'b', 'c', 'd'])
        counts.columns.name = 'random_letter'

        answer = tmm_normalisation_factors(counts)
        self.assertIsInstance(answer, pd.Series)
        self.assertTrue(counts.columns.equals(answer.index))

    def test_reference_column_can_be_provided_as_string_if_dataframe_given(self):
        counts = np.array([[ 5,  6,  0,  3,  2],
                           [ 3,  8,  4,  4,  8],
                           [ 6,  3,  8,  7,  1],
                           [10,  4,  3,  4,  4],
                           [ 5,  5,  5,  2,  2],
                           [ 4,  2,  5,  1,  5],
                           [ 3,  6,  4,  3,  5],
                           [ 2,  7,  2,  7,  6],
                           [ 6,  5,  5,  6,  8],
                           [ 4,  7,  6,  6,  8]])

        counts = pd.DataFrame(counts, columns=['a', 'b', 'c', 'd', 'e'])
        counts.columns.name = 'random_letter'

        answer_int = tmm_normalisation_factors(counts, ref_column=3)
        answer_letter = tmm_normalisation_factors(counts, ref_column='d')
        assert_array_equal(answer_int, answer_letter)

