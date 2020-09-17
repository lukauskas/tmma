import unittest

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_array_equal
from tmma.ma.stats import ma_statistics, asymptotic_variance
from tmma.warnings import AsymptoticVarianceWarning

ARRAY_ONE = np.array(
    [9, 3, 11, 3, 1, 6, 7, 11, 8, 9, 12, 7, 10, 8, 7, 9, 6, 9, 7, 6, 6, 5, 11, 10, 13, 7, 9, 7, 6,
     4, 3, 6, 6, 5, 11, 8, 7, 5, 11, 11, 10, 13, 6, 7, 9, 12, 7, 8, 11, 4])

ARRAY_TWO = np.array(
    [8, 6, 0, 5, 8, 7, 7, 6, 2, 3, 2, 2, 4, 5, 4, 4, 4, 6, 6, 5, 7, 2, 5, 1, 4, 6, 3, 5, 8, 3, 8, 4,
     2, 4, 4, 3, 2, 4, 5, 4, 5, 2, 4, 2, 3, 0, 6, 4, 4, 7])


class TestMAStats(unittest.TestCase):

    def test_with_standard_parameters(self):

        # Poisson with lambda=7.2
        obs = ARRAY_ONE

        # Poisson with lambda=4.3
        ref = ARRAY_TWO

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

        actual_m, actual_a = ma_statistics(obs, ref)

        assert_allclose(actual_m, expected_m, rtol=1e-06)
        assert_allclose(actual_a, expected_a, rtol=1e-06)

        actual_m, actual_a = ma_statistics(obs, ref,
                                           lib_size_obs=obs_lib_size,
                                           lib_size_ref=ref_lib_size)

        assert_allclose(actual_m, expected_m, rtol=1e-06)
        assert_allclose(actual_a, expected_a, rtol=1e-06)

    def test_with_custom_lib_sizes(self):

        # Poisson with lambda=7.2
        obs = ARRAY_ONE

        # Poisson with lambda=4.3
        ref = ARRAY_TWO

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

        actual_m, actual_a = ma_statistics(obs, ref,
                                           lib_size_obs=obs_lib_size,
                                           lib_size_ref=ref_lib_size)

        assert_allclose(actual_m, expected_m, rtol=1e-06)
        assert_allclose(actual_a, expected_a, rtol=1e-06)

    def test_asymptotic_variance_standard_libsize(self):
        # Poisson with lambda=7.2
        obs = ARRAY_ONE

        # Poisson with lambda=4.3
        ref = ARRAY_TWO

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

        actual_v = asymptotic_variance(obs, ref)
        assert_allclose(actual_v, expected_v, rtol=1e-06)

        actual_v = asymptotic_variance(obs, ref, obs_lib_size, ref_lib_size)
        assert_allclose(actual_v, expected_v, rtol=1e-06)

    def test_asymptotic_variance_custom_libsize(self):
        # Poisson with lambda=7.2
        obs = ARRAY_ONE

        # Poisson with lambda=4.3
        ref = ARRAY_TWO

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

        actual_v = asymptotic_variance(obs, ref, obs_lib_size, ref_lib_size)
        assert_allclose(actual_v, expected_v, rtol=1e-06)

    def test_library_size_of_zero_raises_error(self):
        obs = np.array([1, 2, 3])
        ref = np.array([4, 5, 6])

        # second library size is zero
        self.assertRaises(ValueError, ma_statistics, obs, ref, 1, 0)

    def test_asymptotic_variance_is_not_influenced_by_dtype(self):

        # This triggers underflow error for uint32
        obs_uint32 = np.array([2, 3], dtype=np.uint32)
        ref_uint32 = np.array([3, 3], dtype=np.uint32)

        obs_float = np.array([2, 3], dtype=np.float)
        ref_float = np.array([3, 3], dtype=np.float)

        # Library sizes
        ls_obs = 1
        ls_ref = 1

        # This will work
        result_float = asymptotic_variance(obs_float, ref_float, ls_obs, ls_ref)
        # At the time of writing this somehow fails
        result_uint = asymptotic_variance(obs_uint32, ref_uint32, ls_obs, ls_ref)
        assert_array_equal(result_float, result_uint)

    def test_asymptotic_variance_warns_when_library_size_smaller_than_observations(self):

        obs_float = np.array([100, 3], dtype=np.float)
        ref_float = np.array([3, 3], dtype=np.float)

        ls_obs = 5
        ls_ref = 10

        with self.assertWarns(AsymptoticVarianceWarning) as cm:
            asymptotic_variance(obs_float, ref_float, ls_obs, ls_ref)

    def test_ma_respect_indices(self):

        obs = ARRAY_ONE
        ref = ARRAY_TWO

        obs = pd.Series(obs,
                        index=[f'obs-item{i}' for i in range(len(obs))])
        obs.index.name = 'some_index'

        ref = pd.Series(ref,
                        index=[f'obs-item{i}' for i in range(len(obs))])

        m, a = ma_statistics(obs, ref)
        self.assertIsInstance(m, pd.Series)
        self.assertTrue(obs.index.equals(m.index))

        self.assertIsInstance(a, pd.Series)
        self.assertTrue(obs.index.equals(a.index))

    def test_ma_returns_named_series(self):

        obs = ARRAY_ONE
        ref = ARRAY_TWO

        obs = pd.Series(obs,
                        index=[f'obs-item{i}' for i in range(len(obs))])

        ref = pd.Series(ref,
                        index=[f'obs-item{i}' for i in range(len(obs))])

        m, a = ma_statistics(obs, ref)
        self.assertEqual(m.name, 'm_values')
        self.assertEqual(a.name, 'a_values')

    def test_ma_raises_error_when_indices_mismatch(self):

        obs = ARRAY_ONE
        ref = ARRAY_TWO

        obs = pd.Series(obs,
                        index=[f'obs-item{i}' for i in range(len(obs))])

        ref = pd.Series(ref,
                        index=[f'ref-item{i}' for i in range(len(obs))])

        self.assertRaises(ValueError, ma_statistics, obs, ref)


    def test_asymptotic_variance_respects_indices(self):

        obs = ARRAY_ONE
        ref = ARRAY_TWO

        obs = pd.Series(obs,
                        index=[f'obs-item{i}' for i in range(len(obs))])
        obs.index.name = 'some_index'

        ref = pd.Series(ref,
                        index=obs.index)

        v = asymptotic_variance(obs, ref)
        self.assertIsInstance(v, pd.Series)
        self.assertTrue(obs.index.equals(v.index))

    def test_asymptotic_variance_returns_named_series(self):

        obs = ARRAY_ONE
        ref = ARRAY_TWO

        obs = pd.Series(obs,
                        index=[f'obs-item{i}' for i in range(len(obs))])

        ref = pd.Series(ref,
                        index=[f'obs-item{i}' for i in range(len(obs))])

        v = asymptotic_variance(obs, ref)
        self.assertEqual(v.name, 'asymptotic_variance')

    def test_ma_raises_error_when_indices_mismatch(self):

        obs = ARRAY_ONE
        ref = ARRAY_TWO

        obs = pd.Series(obs,
                        index=[f'obs-item{i}' for i in range(len(obs))])

        ref = pd.Series(ref,
                        index=[f'ref-item{i}' for i in range(len(obs))])

        self.assertRaises(ValueError, asymptotic_variance, obs, ref)


