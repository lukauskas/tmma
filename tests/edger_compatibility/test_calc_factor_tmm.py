import unittest

import numpy as np
from hypothesis import given, settings, assume
from numpy.testing import assert_allclose
from tmma.normalisation.tmm import two_sample_tmm

from tests.edger_compatibility.r_helpers import r_edger_calcFactorTMM
from tests.edger_compatibility.strategies import uint_counts_array_and_a_lib_size, \
    uint_counts_array, poisson_counts_array


class HypothesisTestCalcFactorTMM(unittest.TestCase):

    @given(uint_counts_array_and_a_lib_size(max_cols=2))
    @settings(max_examples=500, report_multiple_bugs=False)
    def test_two_sample_tmm_uints_and_random_library_size(self, counts_lib_size):
        """
        Given random unsigned integer counts and a random library size
        Check if the output is the same as edgeR's.
        :param counts_lib_size:
        :return:
        """

        counts, lib_size = counts_lib_size

        obs = counts[:, 0].copy()
        ref = counts[:, 1].copy()

        lib_size_obs, lib_size_ref = lib_size

        r_answer = r_edger_calcFactorTMM(obs, ref,
                                         lib_size_obs=lib_size_obs,
                                         lib_size_ref=lib_size_ref)

        # No point testing bugs in R
        assume(not np.isinf(r_answer))
        assume(not np.isnan(r_answer))

        py_answer = two_sample_tmm(obs, ref,
                                   lib_size_obs=lib_size_obs,
                                   lib_size_ref=lib_size_ref)
        assert_allclose(r_answer, py_answer, rtol=1e-6)

    @given(uint_counts_array(max_cols=2).filter(lambda x: (np.sum(x, axis=0) > 0).all()))
    @settings(max_examples=500, report_multiple_bugs=False)
    def testtwo_sample_tmm_uints_only(self, counts):
        """
        Given random unsigned integer counts,
        check if the output is the same as edgeR's.
        :param counts:
        :return:
        """

        obs = counts[:, 0]
        ref = counts[:, 1]

        r_answer = r_edger_calcFactorTMM(obs, ref)
        # No point testing bugs in R
        assume(not np.isinf(r_answer))
        assume(not np.isnan(r_answer))

        py_answer = two_sample_tmm(obs, ref)

        assert_allclose(r_answer, py_answer, rtol=1e-6)

    @given(poisson_counts_array(max_cols=2).filter(lambda x: (np.sum(x, axis=0) > 0).all()))
    @settings(max_examples=500, report_multiple_bugs=False)
    def testtwo_sample_tmm_poisson_only(self, counts):
        """
        Given random poisson counts,
        check if output matches edgeR.

        :param counts:
        :return:
        """
        obs = counts[:, 0]
        ref = counts[:, 1]

        r_answer = r_edger_calcFactorTMM(obs, ref)
        # No point testing bugs in R
        assume(not np.isinf(r_answer))
        assume(not np.isnan(r_answer))

        py_answer = two_sample_tmm(obs, ref)
        # Allow to differ by 2/100th
        assert_allclose(r_answer, py_answer, rtol=0, atol=2e-2)


if __name__ == '__main__':
    unittest.main()
