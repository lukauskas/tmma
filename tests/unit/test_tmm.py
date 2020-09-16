import unittest

import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal, assert_allclose
from tmma.tmm import tmm_normalise, _calc_factor_quantile, _calc_factor_tmm, _ma_stats, \
    _asymptotic_variance, _tmm_trim, _scale_tmm_factors


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

class TestCalcFactorTMM(unittest.TestCase):

    def test_with_standard_parameters(self):

        # Poisson with lambda=7.2
        obs = np.array([ 9,  3, 11,  3,  1,  6,  7, 11,  8,  9, 12,  7, 10,  8,  7,  9,  6,
                         9,  7,  6,  6,  5, 11, 10, 13,  7,  9,  7,  6,  4,  3,  6,  6,  5,
                        11,  8,  7,  5, 11, 11, 10, 13,  6,  7,  9, 12,  7,  8, 11,  4])

        # Poisson with lambda=4.3
        ref = np.array([8, 6, 0, 5, 8, 7, 7, 6, 2, 3, 2, 2, 4, 5, 4, 4, 4, 6, 6, 5, 7, 2,
                        5, 1, 4, 6, 3, 5, 8, 3, 8, 4, 2, 4, 4, 3, 2, 4, 5, 4, 5, 2, 4, 2,
                        3, 0, 6, 4, 4, 7])

        # From R
        expected_answer = 0.9821526
        actual_answer = _calc_factor_tmm(obs, ref)

        self.assertTrue(np.isclose(expected_answer, actual_answer, rtol=1e-06, atol=0),
                        f'{expected_answer} != {actual_answer}')

    def test_with_custom_lib_sizes(self):

        # Poisson with lambda=7.2
        obs = np.array([ 9,  3, 11,  3,  1,  6,  7, 11,  8,  9, 12,  7, 10,  8,  7,  9,  6,
                         9,  7,  6,  6,  5, 11, 10, 13,  7,  9,  7,  6,  4,  3,  6,  6,  5,
                        11,  8,  7,  5, 11, 11, 10, 13,  6,  7,  9, 12,  7,  8, 11,  4])

        # Poisson with lambda=4.3
        ref = np.array([8, 6, 0, 5, 8, 7, 7, 6, 2, 3, 2, 2, 4, 5, 4, 4, 4, 6, 6, 5, 7, 2,
                        5, 1, 4, 6, 3, 5, 8, 3, 8, 4, 2, 4, 4, 3, 2, 4, 5, 4, 5, 2, 4, 2,
                        3, 0, 6, 4, 4, 7])

        # Equal libsize
        obs_lib_size = 100
        ref_lib_size = 100

        # From R
        expected_answer = 1.768855
        actual_answer = _calc_factor_tmm(obs, ref,
                                         lib_size_obs=obs_lib_size,
                                         lib_size_ref=ref_lib_size)

        self.assertTrue(np.isclose(expected_answer, actual_answer, rtol=1e-06, atol=0),
                        f'{expected_answer} != {actual_answer}')

    def test_with_custom_log_ratio_trim(self):

        # Poisson with lambda=7.2
        obs = np.array([ 9,  3, 11,  3,  1,  6,  7, 11,  8,  9, 12,  7, 10,  8,  7,  9,  6,
                         9,  7,  6,  6,  5, 11, 10, 13,  7,  9,  7,  6,  4,  3,  6,  6,  5,
                        11,  8,  7,  5, 11, 11, 10, 13,  6,  7,  9, 12,  7,  8, 11,  4])

        # Poisson with lambda=4.3
        ref = np.array([8, 6, 0, 5, 8, 7, 7, 6, 2, 3, 2, 2, 4, 5, 4, 4, 4, 6, 6, 5, 7, 2,
                        5, 1, 4, 6, 3, 5, 8, 3, 8, 4, 2, 4, 4, 3, 2, 4, 5, 4, 5, 2, 4, 2,
                        3, 0, 6, 4, 4, 7])

        # Custom trim
        log_ratio_trim = 0.2

        # From R
        expected_answer = 1.023713
        actual_answer = _calc_factor_tmm(obs, ref, log_ratio_trim=log_ratio_trim)

        self.assertTrue(np.isclose(expected_answer, actual_answer, rtol=1e-06, atol=0),
                        f'{expected_answer} != {actual_answer}')

    def test_with_custom_sum_trim(self):

        # Poisson with lambda=7.2
        obs = np.array([ 9,  3, 11,  3,  1,  6,  7, 11,  8,  9, 12,  7, 10,  8,  7,  9,  6,
                         9,  7,  6,  6,  5, 11, 10, 13,  7,  9,  7,  6,  4,  3,  6,  6,  5,
                        11,  8,  7,  5, 11, 11, 10, 13,  6,  7,  9, 12,  7,  8, 11,  4])

        # Poisson with lambda=4.3
        ref = np.array([8, 6, 0, 5, 8, 7, 7, 6, 2, 3, 2, 2, 4, 5, 4, 4, 4, 6, 6, 5, 7, 2,
                        5, 1, 4, 6, 3, 5, 8, 3, 8, 4, 2, 4, 4, 3, 2, 4, 5, 4, 5, 2, 4, 2,
                        3, 0, 6, 4, 4, 7])

        # Custom trim
        sum_trim = 0.1

        # From R
        expected_answer = 0.955578
        actual_answer = _calc_factor_tmm(obs, ref, sum_trim=sum_trim)

        self.assertTrue(np.isclose(expected_answer, actual_answer, rtol=1e-06, atol=0),
                        f'{expected_answer} != {actual_answer}')

    def test_with_custom_a_cutoff(self):

        # Poisson with lambda=7.2
        obs = np.array([ 9,  3, 11,  3,  1,  6,  7, 11,  8,  9, 12,  7, 10,  8,  7,  9,  6,
                         9,  7,  6,  6,  5, 11, 10, 13,  7,  9,  7,  6,  4,  3,  6,  6,  5,
                        11,  8,  7,  5, 11, 11, 10, 13,  6,  7,  9, 12,  7,  8, 11,  4])

        # Poisson with lambda=4.3
        ref = np.array([8, 6, 0, 5, 8, 7, 7, 6, 2, 3, 2, 2, 4, 5, 4, 4, 4, 6, 6, 5, 7, 2,
                        5, 1, 4, 6, 3, 5, 8, 3, 8, 4, 2, 4, 4, 3, 2, 4, 5, 4, 5, 2, 4, 2,
                        3, 0, 6, 4, 4, 7])

        # Custom A cut off
        # Pretty much only value that works with this data
        a_cutoff = -5.5

        # From R
        expected_answer = 0.8056739
        actual_answer = _calc_factor_tmm(obs, ref, a_cutoff=a_cutoff)

        self.assertTrue(np.isclose(expected_answer, actual_answer, rtol=1e-06, atol=0),
                        f'{expected_answer} != {actual_answer}')

    def test_without_weighing(self):

        # Poisson with lambda=7.2
        obs = np.array([ 9,  3, 11,  3,  1,  6,  7, 11,  8,  9, 12,  7, 10,  8,  7,  9,  6,
                         9,  7,  6,  6,  5, 11, 10, 13,  7,  9,  7,  6,  4,  3,  6,  6,  5,
                        11,  8,  7,  5, 11, 11, 10, 13,  6,  7,  9, 12,  7,  8, 11,  4])

        # Poisson with lambda=4.3
        ref = np.array([8, 6, 0, 5, 8, 7, 7, 6, 2, 3, 2, 2, 4, 5, 4, 4, 4, 6, 6, 5, 7, 2,
                        5, 1, 4, 6, 3, 5, 8, 3, 8, 4, 2, 4, 4, 3, 2, 4, 5, 4, 5, 2, 4, 2,
                        3, 0, 6, 4, 4, 7])

        # Do no weighting
        do_weighting = False

        # From R
        expected_answer = 0.9653673
        actual_answer = _calc_factor_tmm(obs, ref, do_weighting=do_weighting)

        self.assertTrue(np.isclose(expected_answer, actual_answer, rtol=1e-06, atol=0),
                        f'{expected_answer} != {actual_answer}')


    def test_tmm_trim_default_params(self):
        m_values = [
            0.1699250, -1.0000000, -0.7369656, -3.0000000, -0.2223924, 0.0000000,
            0.8744691, 2.0000000, 1.5849625, 2.5849625, 1.8073549, 1.3219281,
            0.6780719, 0.8073549, 1.1699250, 0.5849625, 0.5849625, 0.2223924,
            0.2630344, -0.2223924, 1.3219281, 1.1375035, 3.3219281, 1.7004397,
            0.2223924, 1.5849625, 0.4854268, -0.4150375, 0.4150375, -1.4150375,
            0.5849625, 1.5849625, 0.3219281, 1.4594316, 1.4150375, 1.8073549,
            0.3219281, 1.1375035, 1.4594316, 1.0000000, 2.7004397, 0.5849625,
            1.8073549, 1.5849625, 0.2223924, 1.0000000, 1.4594316, -0.8073549,
        ]
        a_values = [
            -3.558894, -4.558894, -4.690411, -5.143856, -3.947697, -3.836501, -3.621659,
            -4.643856, -4.266412, -4.351375, -4.740179, -3.982892, -3.982892, -4.240179,
            -4.058894, -4.351375, -3.766412, -3.947697, -4.190411, -3.947697, -4.982892,
            -3.753176, -4.982892, -3.793636, -3.947697, -4.266412, -4.079215, -3.851375,
            -4.851375, -4.351375, -4.351375, -4.851375, -4.482892, -3.914140, -4.351375,
            -4.740179, -4.482892, -3.753176, -3.914140, -3.821928, -4.293636, -4.351375,
            -4.740179, -4.266412, -3.947697, -4.143856, -3.914140, -4.240179
        ]

        log_ratio_trim = .3
        sum_trim = 0.05

        expected_keep = [
            False, False, False, False, False, False, False, False, False, False, False, True,
            True, True, True, True, True, False, False, False, False, True, False, False,
            False, False, True, False, True, False, True, False, True, False, True, False,
            True, True, False, True, False, True, False, False, False, True, False, False,
        ]

        actual_keep = _tmm_trim(m_values, a_values,
                                log_ratio_trim=log_ratio_trim,
                                sum_trim=sum_trim)

        assert_array_equal(expected_keep, actual_keep)

    def test_tmm_trim_default_params_edge_case(self):

        m_values = [
            0.169925, -1., -0.73696559, -3., -0.22239242, 0.,
            0.87446912, 2., 1.5849625, 2.5849625, 1.80735492, 1.32192809,
            0.67807191, 0.80735492, 1.169925, 0.5849625, 0.5849625, 0.22239242,
            0.26303441, -0.22239242, 1.32192809, 1.13750352, 3.32192809, 1.70043972,
            0.22239242, 1.5849625, 0.48542683, -0.4150375, 0.4150375, -1.4150375,
             0.5849625, 1.5849625, 0.32192809, 1.45943162, 1.4150375, 1.80735492,
            0.32192809, 1.13750352, 1.45943162, 1., 2.70043972, 0.5849625,
            1.80735492, 1.5849625, 0.22239242, 1., 1.45943162, -0.80735492
        ]

        a_values = [
             -3.55889369, -4.55889369, -4.69041089, -5.14385619, -3.94769748,
             -3.83650127, -3.62165913, -4.64385619, -4.26641244, -4.35137494,
             -4.74017873, -3.98289214, -3.98289214, -4.24017873, -4.05889369,
             -4.35137494, -3.76641244, -3.94769748, -4.19041089, -3.94769748,
             -4.98289214, -3.75317633, -4.98289214, -3.79363633, -3.94769748,
             -4.26641244, -4.07921468, -3.85137494, -4.85137494, -4.35137494,
             -4.35137494, -4.85137494, -4.48289214, -3.91414038, -4.35137494,
             -4.74017873, -4.48289214, -3.75317633, -3.91414038, -3.82192809,
             -4.29363633, -4.35137494, -4.74017873, -4.26641244, -3.94769748,
             -4.14385619, -3.91414038, -4.24017873
        ]

        log_ratio_trim = .3
        sum_trim = 0.05

        expected_keep = [
            False, False, False, False, False, False, False, False, False, False, False, True,
            True, True, True, True, True, False, False, False, False, True, False, False,
            False, False, True, False, True, False, True, False, True, False, True, False,
            True, True, False, True, False, True, False, False, False, True, False, False,
        ]

        actual_keep = _tmm_trim(m_values, a_values,
                                log_ratio_trim=log_ratio_trim,
                                sum_trim=sum_trim)

        assert_array_equal(expected_keep, actual_keep)

    def test_tmm_trim_custom_log_ratio_trim(self):
        m_values = [
            0.1699250, -1.0000000, -0.7369656, -3.0000000, -0.2223924, 0.0000000,
            0.8744691, 2.0000000, 1.5849625, 2.5849625, 1.8073549, 1.3219281,
            0.6780719, 0.8073549, 1.1699250, 0.5849625, 0.5849625, 0.2223924,
            0.2630344, -0.2223924, 1.3219281, 1.1375035, 3.3219281, 1.7004397,
            0.2223924, 1.5849625, 0.4854268, -0.4150375, 0.4150375, -1.4150375,
            0.5849625, 1.5849625, 0.3219281, 1.4594316, 1.4150375, 1.8073549,
            0.3219281, 1.1375035, 1.4594316, 1.0000000, 2.7004397, 0.5849625,
            1.8073549, 1.5849625, 0.2223924, 1.0000000, 1.4594316, -0.8073549,
        ]
        a_values = [
            -3.558894, -4.558894, -4.690411, -5.143856, -3.947697, -3.836501, -3.621659,
            -4.643856, -4.266412, -4.351375, -4.740179, -3.982892, -3.982892, -4.240179,
            -4.058894, -4.351375, -3.766412, -3.947697, -4.190411, -3.947697, -4.982892,
            -3.753176, -4.982892, -3.793636, -3.947697, -4.266412, -4.079215, -3.851375,
            -4.851375, -4.351375, -4.351375, -4.851375, -4.482892, -3.914140, -4.351375,
            -4.740179, -4.482892, -3.753176, -3.914140, -3.821928, -4.293636, -4.351375,
            -4.740179, -4.266412, -3.947697, -4.143856, -3.914140, -4.240179
        ]

        log_ratio_trim = .2
        sum_trim = 0.05

        expected_keep = [
            False, False, False, False, False, False, False, False, True, False, False, True,
            True, True, True, True, True, True, True, False, False, True, False, False,
            True, True, True, False, True, False, True, True, True, True, True, False,
            True, True, True, True, False, True, False, True, True, True, True, False,
        ]

        actual_keep = _tmm_trim(m_values, a_values,
                                log_ratio_trim=log_ratio_trim,
                                sum_trim=sum_trim)

        assert_array_equal(expected_keep, actual_keep)

    def test_tmm_trim_custom_sum_trim(self):
        m_values = [
            0.1699250, -1.0000000, -0.7369656, -3.0000000, -0.2223924, 0.0000000,
            0.8744691, 2.0000000, 1.5849625, 2.5849625, 1.8073549, 1.3219281,
            0.6780719, 0.8073549, 1.1699250, 0.5849625, 0.5849625, 0.2223924,
            0.2630344, -0.2223924, 1.3219281, 1.1375035, 3.3219281, 1.7004397,
            0.2223924, 1.5849625, 0.4854268, -0.4150375, 0.4150375, -1.4150375,
            0.5849625, 1.5849625, 0.3219281, 1.4594316, 1.4150375, 1.8073549,
            0.3219281, 1.1375035, 1.4594316, 1.0000000, 2.7004397, 0.5849625,
            1.8073549, 1.5849625, 0.2223924, 1.0000000, 1.4594316, -0.8073549,
        ]
        a_values = [
            -3.558894, -4.558894, -4.690411, -5.143856, -3.947697, -3.836501, -3.621659,
            -4.643856, -4.266412, -4.351375, -4.740179, -3.982892, -3.982892, -4.240179,
            -4.058894, -4.351375, -3.766412, -3.947697, -4.190411, -3.947697, -4.982892,
            -3.753176, -4.982892, -3.793636, -3.947697, -4.266412, -4.079215, -3.851375,
            -4.851375, -4.351375, -4.351375, -4.851375, -4.482892, -3.914140, -4.351375,
            -4.740179, -4.482892, -3.753176, -3.914140, -3.821928, -4.293636, -4.351375,
            -4.740179, -4.266412, -3.947697, -4.143856, -3.914140, -4.240179
        ]

        log_ratio_trim = .3
        sum_trim = 0.1

        expected_keep = [
            False, False, False, False, False, False, False, False, False, False, False, True,
            True, True, True, True, True, False, False, False, False, False, False, False,
            False, False, True, False, False, False, True, False, True, False, True, False,
            True, False, False, True, False, True, False, False, False, True, False, False,
        ]

        actual_keep = _tmm_trim(m_values, a_values,
                                log_ratio_trim=log_ratio_trim,
                                sum_trim=sum_trim)

        assert_array_equal(expected_keep, actual_keep)

    def test_zero_edge_case(self):

        obs = np.array([0, 0])
        ref = np.array([0, 0])

        expected_result = 1.0

        actual_result = _calc_factor_tmm(obs, ref)
        assert_array_equal(expected_result, actual_result)

    def test_almost_zero_edge_case(self):

        obs = np.array([2, 1])
        ref = np.array([1, 1])

        expected_result = 1.036271
        actual_result = _calc_factor_tmm(obs, ref)

        assert_allclose(expected_result, actual_result, rtol=1e-6)

class TestMAStats(unittest.TestCase):

    def test_with_standard_parameters(self):

        # Poisson with lambda=7.2
        obs = np.array([ 9,  3, 11,  3,  1,  6,  7, 11,  8,  9, 12,  7, 10,  8,  7,  9,  6,
                         9,  7,  6,  6,  5, 11, 10, 13,  7,  9,  7,  6,  4,  3,  6,  6,  5,
                        11,  8,  7,  5, 11, 11, 10, 13,  6,  7,  9, 12,  7,  8, 11,  4])

        # Poisson with lambda=4.3
        ref = np.array([8, 6, 0, 5, 8, 7, 7, 6, 2, 3, 2, 2, 4, 5, 4, 4, 4, 6, 6, 5, 7, 2,
                        5, 1, 4, 6, 3, 5, 8, 3, 8, 4, 2, 4, 4, 3, 2, 4, 5, 4, 5, 2, 4, 2,
                        3, 0, 6, 4, 4, 7])

        obs_lib_size = float(np.sum(obs))
        ref_lib_size = float(np.sum(ref))

        # From R
        expected_m = [
            -0.67807191, -1.84799691, np.inf, -1.58496250, -3.84799691, -1.07038933,
            -0.84799691, 0.02647221,  1.15200309,  0.73696559,  1.73696559,  0.95935802,
            0.47393119, -0.16992500, -0.04064198, 0.32192809, -0.26303441, -0.26303441,
            -0.62560449, -0.58496250, -1.07038933, 0.47393119, 0.28950662, 2.47393119,
            0.85244281, -0.62560449,  0.73696559, -0.36257008, -1.26303441, -0.43295941,
            -2.26303441, -0.26303441,  0.73696559, -0.52606881, 0.61143471,  0.56704059,
            0.95935802, -0.52606881,  0.28950662,  0.61143471,  0.15200309,  1.85244281,
            -0.26303441,  0.95935802,  0.73696559,  np.inf, -0.62560449,  0.15200309,
            0.61143471, -1.65535183
        ]

        expected_a = [
             -5.087229, -6.087229, -np.inf, -6.218746, -6.672191, -5.476033, -5.364836,
             -5.149994, -6.172191, -5.794748, -5.879710, -6.268514, -5.511227, -5.511227,
             -5.768514, -5.587229, -5.879710, -5.294748, -5.476033, -5.718746, -5.476033,
             -6.511227, -5.281511, -6.511227, -5.321971, -5.476033, -5.794748, -5.607550,
             -5.379710, -6.379710, -5.879710, -5.879710, -6.379710, -6.011227, -5.442475,
             -5.879710, -6.268514, -6.011227, -5.281511, -5.442475, -5.350263, -5.821971,
             -5.879710, -6.268514, -5.794748, -np.inf, -5.476033, -5.672191, -5.442475,
             -5.768514
        ]

        actual_m, actual_a = _ma_stats(obs, ref,
                                       lib_size_obs=obs_lib_size,
                                       lib_size_ref=ref_lib_size)

        assert_allclose(actual_m, expected_m, rtol=1e-06)
        assert_allclose(actual_a, expected_a, rtol=1e-06)

    def test_with_custom_lib_sizes(self):

        # Poisson with lambda=7.2
        obs = np.array([9, 3, 11, 3, 1, 6, 7, 11, 8, 9, 12, 7, 10, 8, 7, 9, 6,
                        9, 7, 6, 6, 5, 11, 10, 13, 7, 9, 7, 6, 4, 3, 6, 6, 5,
                        11, 8, 7, 5, 11, 11, 10, 13, 6, 7, 9, 12, 7, 8, 11, 4])

        # Poisson with lambda=4.3
        ref = np.array([8, 6, 0, 5, 8, 7, 7, 6, 2, 3, 2, 2, 4, 5, 4, 4, 4, 6, 6, 5, 7, 2,
                        5, 1, 4, 6, 3, 5, 8, 3, 8, 4, 2, 4, 4, 3, 2, 4, 5, 4, 5, 2, 4, 2,
                        3, 0, 6, 4, 4, 7])

        obs_lib_size = 100.0
        ref_lib_size = 100.0

        # From R
        expected_m = [
            0.1699250, -1.0000000, np.inf, -0.7369656, -3.0000000, -0.2223924,
            0.0000000, 0.8744691, 2.0000000, 1.5849625, 2.5849625, 1.8073549,
            1.3219281, 0.6780719, 0.8073549, 1.1699250, 0.5849625, 0.5849625,
            0.2223924, 0.2630344, -0.2223924, 1.3219281, 1.1375035, 3.3219281,
            1.7004397, 0.2223924, 1.5849625, 0.4854268, -0.4150375, 0.4150375,
            -1.4150375, 0.5849625, 1.5849625, 0.3219281, 1.4594316, 1.4150375,
            1.8073549, 0.3219281, 1.1375035, 1.4594316, 1.0000000, 2.7004397,
            0.5849625, 1.8073549, 1.5849625, np.inf, 0.2223924, 1.0000000,
            1.4594316, -0.8073549,
        ]

        expected_a = [
            -3.558894, -4.558894, -np.inf, -4.690411, -5.143856, -3.947697, -3.836501,
            -3.621659, -4.643856, -4.266412, -4.351375, -4.740179, -3.982892, -3.982892,
            -4.240179, -4.058894, -4.351375, -3.766412, -3.947697, -4.190411, -3.947697,
            -4.982892, -3.753176, -4.982892, -3.793636, -3.947697, -4.266412, -4.079215,
            -3.851375, -4.851375, -4.351375, -4.351375, -4.851375, -4.482892, -3.914140,
            -4.351375, -4.740179, -4.482892, -3.753176, -3.914140, -3.821928, -4.293636,
            -4.351375, -4.740179, -4.266412, -np.inf, -3.947697, -4.143856, -3.914140,
            -4.240179
        ]

        actual_m, actual_a = _ma_stats(obs, ref,
                                       lib_size_obs=obs_lib_size,
                                       lib_size_ref=ref_lib_size)

        assert_allclose(actual_m, expected_m, rtol=1e-06)
        assert_allclose(actual_a, expected_a, rtol=1e-06)

    def test_asymptotic_variance_standard_libsize(self):
        # Poisson with lambda=7.2
        obs = np.array([9, 3, 11, 3, 1, 6, 7, 11, 8, 9, 12, 7, 10, 8, 7, 9, 6,
                        9, 7, 6, 6, 5, 11, 10, 13, 7, 9, 7, 6, 4, 3, 6, 6, 5,
                        11, 8, 7, 5, 11, 11, 10, 13, 6, 7, 9, 12, 7, 8, 11, 4])

        # Poisson with lambda=4.3
        ref = np.array([8, 6, 0, 5, 8, 7, 7, 6, 2, 3, 2, 2, 4, 5, 4, 4, 4, 6, 6, 5, 7, 2,
                        5, 1, 4, 6, 3, 5, 8, 3, 8, 4, 2, 4, 4, 3, 2, 4, 5, 4, 5, 2, 4, 2,
                        3, 0, 6, 4, 4, 7])

        obs_lib_size = float(np.sum(obs))
        ref_lib_size = float(np.sum(ref))

        # From R
        expected_v = [
            0.2288760, 0.4927649, np.inf, 0.5260982, 1.1177649, 0.3022887, 0.2784791,
            0.2503406, 0.6177649, 0.4372093, 0.5760982, 0.6356220, 0.3427649, 0.3177649,
            0.3856220, 0.3538760, 0.4094315, 0.2705426, 0.3022887, 0.3594315, 0.3022887,
            0.6927649, 0.2836739, 1.0927649, 0.3196879, 0.3022887, 0.4372093, 0.3356220,
            0.2844315, 0.5760982, 0.4510982, 0.4094315, 0.6594315, 0.4427649, 0.3336739,
            0.4510982, 0.6356220, 0.4427649, 0.2836739, 0.3336739, 0.2927649, 0.5696879,
            0.4094315, 0.6356220, 0.4372093, np.inf, 0.3022887, 0.3677649, 0.3336739,
            0.3856220
        ]

        actual_v = _asymptotic_variance(obs, ref, obs_lib_size, ref_lib_size)
        assert_allclose(actual_v, expected_v, rtol=1e-06)

    def test_asymptotic_variance_custom_libsize(self):
        # Poisson with lambda=7.2
        obs = np.array([9, 3, 11, 3, 1, 6, 7, 11, 8, 9, 12, 7, 10, 8, 7, 9, 6,
                        9, 7, 6, 6, 5, 11, 10, 13, 7, 9, 7, 6, 4, 3, 6, 6, 5,
                        11, 8, 7, 5, 11, 11, 10, 13, 6, 7, 9, 12, 7, 8, 11, 4])

        # Poisson with lambda=4.3
        ref = np.array([8, 6, 0, 5, 8, 7, 7, 6, 2, 3, 2, 2, 4, 5, 4, 4, 4, 6, 6, 5, 7, 2,
                        5, 1, 4, 6, 3, 5, 8, 3, 8, 4, 2, 4, 4, 3, 2, 4, 5, 4, 5, 2, 4, 2,
                        3, 0, 6, 4, 4, 7])

        obs_lib_size = 100.0
        ref_lib_size = 100.0

        # From R
        expected_v = [
            0.2161111, 0.4800000, np.inf, 0.5133333, 1.1050000, 0.2895238, 0.2657143,
            0.2375758, 0.6050000, 0.4244444, 0.5633333, 0.6228571, 0.3300000, 0.3050000,
            0.3728571, 0.3411111, 0.3966667, 0.2577778, 0.2895238, 0.3466667, 0.2895238,
            0.6800000, 0.2709091, 1.0800000, 0.3069231, 0.2895238, 0.4244444, 0.3228571,
            0.2716667, 0.5633333, 0.4383333, 0.3966667, 0.6466667, 0.4300000, 0.3209091,
            0.4383333, 0.6228571, 0.4300000, 0.2709091, 0.3209091, 0.2800000, 0.5569231,
            0.3966667, 0.6228571, 0.4244444, np.inf, 0.2895238, 0.3550000, 0.3209091,
            0.3728571
        ]

        actual_v = _asymptotic_variance(obs, ref, obs_lib_size, ref_lib_size)
        assert_allclose(actual_v, expected_v, rtol=1e-06)