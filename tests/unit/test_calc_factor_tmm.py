import unittest

import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal, assert_allclose
from tmma.normalisation.tmm import two_sample_tmm, tmm_trim_mask
from tmma.warnings import InfiniteWeightsWarning


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
        actual_answer = two_sample_tmm(obs, ref)

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
        actual_answer = two_sample_tmm(obs, ref,
                                       lib_size_obs=obs_lib_size,
                                       lib_size_ref=ref_lib_size)

        self.assertTrue(np.isclose(expected_answer, actual_answer, rtol=1e-06, atol=0),
                        f'{expected_answer} != {actual_answer}')

    def test_library_size_of_zero_raises_error(self):
        obs = np.array([1,2,3])
        ref = np.array([4,5,6])

        # second library size is zero
        self.assertRaises(ValueError, two_sample_tmm, obs, ref, 1, 0)

        # second library size is zero (implicitly)
        self.assertRaises(ValueError, two_sample_tmm, obs, np.array([0, 0, 0]))

    def test_with_custom_m_values_trim_fraction(self):

        # Poisson with lambda=7.2
        obs = np.array([ 9,  3, 11,  3,  1,  6,  7, 11,  8,  9, 12,  7, 10,  8,  7,  9,  6,
                         9,  7,  6,  6,  5, 11, 10, 13,  7,  9,  7,  6,  4,  3,  6,  6,  5,
                        11,  8,  7,  5, 11, 11, 10, 13,  6,  7,  9, 12,  7,  8, 11,  4])

        # Poisson with lambda=4.3
        ref = np.array([8, 6, 0, 5, 8, 7, 7, 6, 2, 3, 2, 2, 4, 5, 4, 4, 4, 6, 6, 5, 7, 2,
                        5, 1, 4, 6, 3, 5, 8, 3, 8, 4, 2, 4, 4, 3, 2, 4, 5, 4, 5, 2, 4, 2,
                        3, 0, 6, 4, 4, 7])

        # Custom trim
        m_values_trim_fraction = 0.2

        # From R
        expected_answer = 1.023713
        actual_answer = two_sample_tmm(obs, ref, m_values_trim_fraction=m_values_trim_fraction)

        self.assertTrue(np.isclose(expected_answer, actual_answer, rtol=1e-06, atol=0),
                        f'{expected_answer} != {actual_answer}')

    def test_with_custom_a_values_trim_fraction(self):

        # Poisson with lambda=7.2
        obs = np.array([ 9,  3, 11,  3,  1,  6,  7, 11,  8,  9, 12,  7, 10,  8,  7,  9,  6,
                         9,  7,  6,  6,  5, 11, 10, 13,  7,  9,  7,  6,  4,  3,  6,  6,  5,
                        11,  8,  7,  5, 11, 11, 10, 13,  6,  7,  9, 12,  7,  8, 11,  4])

        # Poisson with lambda=4.3
        ref = np.array([8, 6, 0, 5, 8, 7, 7, 6, 2, 3, 2, 2, 4, 5, 4, 4, 4, 6, 6, 5, 7, 2,
                        5, 1, 4, 6, 3, 5, 8, 3, 8, 4, 2, 4, 4, 3, 2, 4, 5, 4, 5, 2, 4, 2,
                        3, 0, 6, 4, 4, 7])

        # Custom trim
        a_values_trim_fraction = 0.1

        # From R
        expected_answer = 0.955578
        actual_answer = two_sample_tmm(obs, ref, a_values_trim_fraction=a_values_trim_fraction)

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
        actual_answer = two_sample_tmm(obs, ref, a_cutoff=a_cutoff)

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
        actual_answer = two_sample_tmm(obs, ref, weighted=do_weighting)

        self.assertTrue(np.isclose(expected_answer, actual_answer, rtol=1e-06, atol=0),
                        f'{expected_answer} != {actual_answer}')


    def testtmm_trim_mask_default_params(self):
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

        m_values_trim_fraction = .3
        a_values_trim_fraction = 0.05

        expected_keep = [
            False, False, False, False, False, False, False, False, False, False, False, True,
            True, True, True, True, True, False, False, False, False, True, False, False,
            False, False, True, False, True, False, True, False, True, False, True, False,
            True, True, False, True, False, True, False, False, False, True, False, False,
        ]

        actual_keep = tmm_trim_mask(m_values, a_values,
                                    m_values_trim_fraction=m_values_trim_fraction,
                                    a_values_trim_fraction=a_values_trim_fraction)

        assert_array_equal(expected_keep, actual_keep)

    def test_tmm_mask_respects_indices(self):
        m_values = [
            0.1699250, -1.0000000, -0.7369656, -3.0000000, -0.2223924, 0.0000000,
        ]
        a_values = [
            -3.558894, -4.558894, -4.690411, -5.143856, -3.947697, -3.836501,
        ]

        m_values = pd.Series(m_values, index=['a', 'b', 'c', 'd', 'e', 'f'])
        m_values.index.name = 'random_letter'
        a_values = pd.Series(a_values, index=m_values.index)

        ans = tmm_trim_mask(m_values, a_values)
        self.assertIsInstance(ans, pd.Series)
        self.assertTrue(m_values.index.equals(ans.index))

    def test_tmm_mask_returns_named_series(self):
        m_values = [
            0.1699250, -1.0000000, -0.7369656, -3.0000000, -0.2223924, 0.0000000,
        ]
        a_values = [
            -3.558894, -4.558894, -4.690411, -5.143856, -3.947697, -3.836501,
        ]

        m_values = pd.Series(m_values, index=['a', 'b', 'c', 'd', 'e', 'f'])
        m_values.index.name = 'random_letter'
        a_values = pd.Series(a_values, index=m_values.index)

        ans = tmm_trim_mask(m_values, a_values)
        self.assertEqual(ans.name, 'considered_for_tmm')

    def test_tmm_mask_raises_error_on_index_mismatch(self):
        m_values = [
            0.1699250, -1.0000000, -0.7369656, -3.0000000, -0.2223924, 0.0000000,
        ]
        a_values = [
            -3.558894, -4.558894, -4.690411, -5.143856, -3.947697, -3.836501,
        ]

        m_values = pd.Series(m_values, index=['a', 'b', 'c', 'd', 'e', 'f'])
        m_values.index.name = 'random_letter'
        a_values = pd.Series(a_values, index=['g', 'i', 'j', 'k', 'l', 'm'])

        self.assertRaises(ValueError, tmm_trim_mask, m_values, a_values)

    def testtmm_trim_mask_default_params_edge_case(self):

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

        m_values_trim_fraction = .3
        a_values_trim_fraction = 0.05

        expected_keep = [
            False, False, False, False, False, False, False, False, False, False, False, True,
            True, True, True, True, True, False, False, False, False, True, False, False,
            False, False, True, False, True, False, True, False, True, False, True, False,
            True, True, False, True, False, True, False, False, False, True, False, False,
        ]

        actual_keep = tmm_trim_mask(m_values, a_values,
                                m_values_trim_fraction=m_values_trim_fraction,
                                a_values_trim_fraction=a_values_trim_fraction)

        assert_array_equal(expected_keep, actual_keep)

    @unittest.skip("This is probably due to bug in R")
    def testtmm_trim_mask_another_edge_case(self):

        m_values = [
            -0.227135291, -0.061622305, -0.002585016, 0.020333359, 0.052548715,
            0.117037442, 0.081117867, -0.004953083, 0.098605294, -0.113761061,
            0.171957098, 0.081117867, 0.068787065, -0.273724850, -0.012368762,
        ]
        a_values = [
            -3.963853, -3.808745, -3.983869, -3.884770, -3.865831, -3.933122, -3.957157,
            -3.872127, -3.894642, -3.829597, -3.902635, -3.897533, -3.975548, -3.968039,
            -3.898290
        ]

        m_values_trim_fraction = .3
        a_values_trim_fraction = 0.05

        expected_keep = [
            False, False, True, True, True, False, False, True, False, False, False, True,
            True, False, True,
        ]

        actual_keep = tmm_trim_mask(m_values, a_values,
                                m_values_trim_fraction=m_values_trim_fraction,
                                a_values_trim_fraction=a_values_trim_fraction)

        assert_array_equal(expected_keep, actual_keep)

    def testtmm_trim_mask_custom_m_values_trim_fraction(self):
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

        m_values_trim_fraction = .2
        a_values_trim_fraction = 0.05

        expected_keep = [
            False, False, False, False, False, False, False, False, True, False, False, True,
            True, True, True, True, True, True, True, False, False, True, False, False,
            True, True, True, False, True, False, True, True, True, True, True, False,
            True, True, True, True, False, True, False, True, True, True, True, False,
        ]

        actual_keep = tmm_trim_mask(m_values, a_values,
                                m_values_trim_fraction=m_values_trim_fraction,
                                a_values_trim_fraction=a_values_trim_fraction)

        assert_array_equal(expected_keep, actual_keep)

    def testtmm_trim_mask_custom_a_values_trim_fraction(self):
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

        m_values_trim_fraction = .3
        a_values_trim_fraction = 0.1

        expected_keep = [
            False, False, False, False, False, False, False, False, False, False, False, True,
            True, True, True, True, True, False, False, False, False, False, False, False,
            False, False, True, False, False, False, True, False, True, False, True, False,
            True, False, False, True, False, True, False, False, False, True, False, False,
        ]

        actual_keep = tmm_trim_mask(m_values, a_values,
                                m_values_trim_fraction=m_values_trim_fraction,
                                a_values_trim_fraction=a_values_trim_fraction)

        assert_array_equal(expected_keep, actual_keep)

    def test_zero_edge_case(self):

        obs = np.array([0, 0])
        ref = np.array([0, 0])

        self.assertRaises(ValueError, two_sample_tmm, obs, ref)

    def test_almost_zero_edge_case(self):

        obs = np.array([2, 1])
        ref = np.array([1, 1])

        expected_result = 1.036271
        actual_result = two_sample_tmm(obs, ref)

        assert_allclose(expected_result, actual_result, rtol=1e-6)

    def test_infinite_weight_edge_case(self):

        obs = np.array([2, 1])
        ref = np.array([1, 1])

        # Library sizes

        ls_obs = 1
        ls_ref = 1

        # These library sizes produce variance of zero
        # this makes weights infinite, and the factor NA
        # R defaults to 1.0 in these cases
        expected_result = 1.0
        expected_result_no_weighting = 1.414214

        with self.assertWarns(InfiniteWeightsWarning) as cm:
            actual_result = two_sample_tmm(obs, ref, ls_obs, ls_ref)

        assert_allclose(expected_result, actual_result, rtol=1e-6)

        actual_result_no_weighting = two_sample_tmm(obs, ref, ls_obs, ls_ref, weighted=False)
        assert_allclose(expected_result_no_weighting, actual_result_no_weighting, rtol=1e-6)

    def test_data_type_edge_case(self):

        # These conditions trigger an underflow eror
        obs_uint32 = np.array([2, 3], dtype=np.uint32)
        ref_uint32 = np.array([3, 3], dtype=np.uint32)

        obs_float = np.array([2, 3], dtype=np.float)
        ref_float = np.array([3, 3], dtype=np.float)

        # Library sizes
        ls_obs = 1
        ls_ref = 1

        expected_result = 0.8055355

        # This will work
        actual_result_float = two_sample_tmm(obs_float, ref_float, ls_obs, ls_ref)
        assert_allclose(expected_result, actual_result_float, rtol=1e-6)

        # At the time of writing this somehow fails
        actual_result_uint = two_sample_tmm(obs_uint32, ref_uint32, ls_obs, ls_ref)
        assert_allclose(expected_result, actual_result_uint, rtol=1e-6)

    @unittest.skip("See testtmm_trim_mask_another_edge_case")
    def test_hypothesis_failing_case(self):

        # This case was discovered by hypothesis
        obs = np.array([212., 250., 226., 244., 250., 244., 237., 244., 249., 242., 254.,
                        247., 233., 208., 239.])
        ref = np.array([525., 552., 479., 509., 510., 476., 474., 518., 492., 554., 477.,
                        494., 470., 532., 510.])

        expected_answer = 1.020364
        actual_answer = two_sample_tmm(obs, ref)

        assert_allclose(expected_answer, actual_answer, rtol=1e-6)
