"""
Automated tests using hypothesis package
"""

import unittest
from hypothesis import given
from hypothesis import strategies
import hypothesis.extra.numpy as np_strategies
import numpy as np

from numpy.testing import assert_array_equal
from tmma.tmm import tmm_normalise

from tests.edger_compatibility.r_helpers import r_edger_calcNormFactors

@strategies.composite
def uint_counts_strategy(draw, max_rows=5_000, max_cols=10):
    """
    Generates a random `counts` matrix

    :param draw: `Provided by `@strategies.composite`
    :param max_rows: Maximum number of rows in the matrix (minimum is 2)
    :param max_cols: Maximum number of cols in the matrix (minimum is 2)
    :return: counts

    """

    n_rows = draw(strategies.integers(min_value=2, max_value=max_rows))
    n_cols = draw(strategies.integers(min_value=2, max_value=max_cols))

    counts = draw(np_strategies.arrays(dtype=np.uint32,
                                       shape=(n_rows, n_cols)
                                       ))
    return counts

@strategies.composite
def uint_counts_and_libsize_strategy(draw, max_rows=5_000, max_cols=10):
    """
    Generates a random `counts` matrix and appropriate lib size column

    :param draw: `Provided by `@strategies.composite`
    :param max_rows: Maximum number of rows in the matrix (minimum is 2)
    :param max_cols: Maximum number of cols in the matrix (minimum is 2)
    :return: tuple (counts, library_size)

    """

    counts = draw(uint_counts_strategy(max_rows=max_rows,
                                       max_cols=max_cols))
    n_cols = counts.shape[1]
    library_size = draw(
        np_strategies.arrays(dtype=np.uint,
                             shape=(n_cols,),
                             elements=strategies.integers(min_value=1))
    )

    return counts, library_size

@strategies.composite
def poisson_counts_strategy(draw, max_rows=5_000, max_cols=10,
                            min_lambda=0.1,
                            max_lambda=1_000):
    """
    Generates a random `counts` matrix from a poisson distribution.
    This is inspired by the `edgeR` documentation where they
    generate random input data from poisson as well.

    :param draw: `Provided by `@strategies.composite`
    :param max_rows: Maximum number of rows in the matrix (minimum is 2)
    :param max_cols: Maximum number of cols in the matrix (minimum is 2)
    :param min_lambda: Randomised poisson lambda parameter (min)
    :param max_lambda: Randomised poisson lambda parameter (max)
    :return: counts

    """

    n_rows = draw(strategies.integers(min_value=2, max_value=max_rows))
    n_cols = draw(strategies.integers(min_value=2, max_value=max_cols))

    # Not sure if this is the best way to do this but who knows
    random = draw(strategies.randoms(use_true_random=True))
    random_seed = random.randint(1, 10_000_00)
    np_random = np.random.RandomState(random_seed)

    lambdas = np_random.uniform(min_lambda, max_lambda, size=n_cols)
    poisson_counts = np.random.poisson(lam=lambdas, size=(n_rows, n_cols))

    return poisson_counts

class HypothesisAutomatedTests(unittest.TestCase):

    @given(uint_counts_and_libsize_strategy())
    def test_uints_and_random_library_size(self, counts_lib_size):
        """
        Given random unsigned integer counts and a random library size
        Check if the output is the same as edgeR's.
        :param counts_lib_size:
        :return:
        """

        counts, lib_size = counts_lib_size

        r_answer = r_edger_calcNormFactors(counts, lib_size=lib_size)
        py_answer = tmm_normalise(counts, lib_sizes=lib_size)
        assert_array_equal(r_answer, py_answer)

    @given(uint_counts_strategy())
    def test_uints_only(self, counts):
        """
        Given random unsigned integer counts,
        check if the output is the same as edgeR's.
        :param counts:
        :return:
        """
        r_answer = r_edger_calcNormFactors(counts)
        py_answer = tmm_normalise(counts)
        assert_array_equal(r_answer, py_answer)

    @given(poisson_counts_strategy())
    def test_poisson_only(self, counts):
        """
        Given random poisson counts,
        check if output matches edgeR.

        :param counts:
        :return:
        """
        r_answer = r_edger_calcNormFactors(counts)
        py_answer = tmm_normalise(counts)
        assert_array_equal(r_answer, py_answer)

if __name__ == '__main__':
    unittest.main()
