import hypothesis.extra.numpy as np_strategies
import numpy as np
from hypothesis import strategies
from hypothesis.strategies import lists, integers


@strategies.composite
def uint_counts_array(draw, max_rows=5_000, max_cols=10):
    """
    Generates a random `counts` matrix (uint type)

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
def uint_counts_array_and_a_lib_size(draw, max_rows=5_000, max_cols=10):
    """
       Generates a random `counts` matrix (uint type)
       As well as a library size (uint type)

       :param draw: `Provided by `@strategies.composite`
       :param max_rows: Maximum number of rows in the matrix (minimum is 2)
       :param max_cols: Maximum number of cols in the matrix (minimum is 2)
       :return: counts

       """

    counts = draw(uint_counts_array(max_rows=max_rows, max_cols=max_cols))
    lib_size = draw(lists(integers(min_value=1,
                                   max_value=2147483647 # maximum 32 bit int..
                                   ),
                          min_size=counts.shape[1],
                          max_size=counts.shape[1]))

    return counts, lib_size

@strategies.composite
def poisson_counts_array(draw, max_rows=5_000, max_cols=10,
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
