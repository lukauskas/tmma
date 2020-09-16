"""
Automated tests using hypothesis package
"""

import unittest

import numpy as np
from hypothesis import given, assume, note, settings
from numpy.testing import assert_allclose
from tmma.tmm import tmm_normalise

from .r_helpers import r_edger_calcNormFactors
from .strategies import uint_counts_array, poisson_counts_array, uint_counts_array_and_a_lib_size


class HypothesisTestCalcNormFactors(unittest.TestCase):

    @given(uint_counts_array_and_a_lib_size())
    def test_uints_and_random_library_size(self, counts_lib_size):
        """
        Given random unsigned integer counts and a random library size
        Check if the output is the same as edgeR's.
        :param counts_lib_size:
        :return:
        """

        counts, lib_size = counts_lib_size

        r_answer = r_edger_calcNormFactors(counts, lib_size=lib_size)
        # No point testing bugs in R
        assume(not np.any(np.isinf(r_answer)))
        assume(not np.any(np.isnan(r_answer)))

        py_answer = tmm_normalise(counts, lib_sizes=lib_size)
        assert_allclose(r_answer, py_answer, rtol=1e-6)

    @given(uint_counts_array().filter(lambda x: (np.sum(x, axis=0) > 0).all()))
    def test_uints_only(self, counts):
        """
        Given random unsigned integer counts,
        check if the output is the same as edgeR's.
        :param counts:
        :return:
        """
        r_answer = r_edger_calcNormFactors(counts)

        # No point testing bugs in R
        assume(not np.any(np.isinf(r_answer)))
        assume(not np.any(np.isnan(r_answer)))

        py_answer = tmm_normalise(counts)
        assert_allclose(r_answer, py_answer, rtol=1e-6)

    @given(poisson_counts_array().filter(lambda x: (np.sum(x, axis=0) > 0).all()))
    @settings(max_examples=500)
    def test_poisson_only(self, counts):
        """
        Given random poisson counts,
        check if output matches edgeR.

        :param counts:
        :return:
        """
        r_answer = r_edger_calcNormFactors(counts)
        # No point testing bugs in R
        assume(not np.any(np.isinf(r_answer)))
        assume(not np.any(np.isnan(r_answer)))

        note('Shape = {!r}'.format(counts.shape))
        py_answer = tmm_normalise(counts)

        # Allow to differ by 1/100
        assert_allclose(r_answer, py_answer, rtol=0, atol=1e-2)


if __name__ == '__main__':
    unittest.main()
