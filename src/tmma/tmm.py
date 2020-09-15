import numpy as np
from typing import Optional

_P_FOR_TMM = 0.75

def _calc_factor_quantile(data, lib_sizes, p: float = 0.75):
    """
    Calculates quantiles for data points.
    Identical to EdgeR `.calcFactorQuantile`
    :param data:
    :param lib_sizes:
    :param p:
    :return:
    """

    # Divide by columns
    normed_data = data / lib_sizes

    # Columnwise quantiles
    ans = np.quantile(normed_data, p, axis=0)
    return ans

def tmm_normalise(counts,
                  lib_sizes=None,
                  ref_column: Optional[int] = None,
                  log_ratio_trim: float = 0.3,
                  sum_trim: float = 0.05,
                  do_weighting: bool = True,
                  a_cutoff: float = -1e10
                  ):
    """
    Calculate normalisation factors using TMM method.
    Identical to edgeR::calcNormFactors.

    :param counts: numpy array of raw (unnormalised) counts for each of the samples.
                   Genes in rows, samples in columns.
    :param lib_sizes: (optional) numpy array of library sizes.
                      Should be in the same order as columns of `counts`
    :param ref_column: (optional) reference column to use
    :param log_ratio_trim: amount of trim to use on M values (log ratios), default: 0.3
    :param sum_trim: amount of trim to use on combined absolute values (A values), default: 0.3
    :param do_weighting: whether to compute asymptotic binomial precision weights, default: True
    :param a_cutoff: cutoff on A values, default: -1e10 (which is equivalent to no cutoff)
    :return:
    """

    counts = np.asarray(counts, dtype=float)

    if np.isnan(counts).any() or np.isinf(counts).any():
        raise ValueError("Your counts contain NaNs and/or infinities. This is not allowed")

    n_rows, n_columns = counts.shape

    if lib_sizes is None:
        lib_sizes = np.sum(counts, 0)
    else:
        if np.isnan(lib_sizes).any() or np.isinf(lib_sizes).any():
            raise ValueError("Your lib size contains NaNs and/or infinities. This is not allowed")

    if lib_sizes.shape != (n_columns,):
        raise ValueError(f"Wrong shape of libsize, was expecting an array of {n_columns} items")

    if ref_column is not None:
        if not isinstance(ref_column, int) or ref_column < 0 or ref_column >= n_columns:
            raise ValueError(f"Wrong ref_column provided {ref_column!r}")

    # Remove all-zero rows
    all_zero = np.all(counts == 0, axis=1)
    counts = counts[~all_zero]

    # Degenerate cases, nothing to normalise here
    if len(counts) < 2:
        return np.ones(counts.shape[1])

    factor_quantiles = _calc_factor_quantile(data=counts,
                                             lib_sizes=lib_sizes,
                                             p=_P_FOR_TMM)
    if ref_column is None:
        ref_column = np.argmin(factor_quantiles - factor_quantiles.mean())

        # TODO: https://rdrr.io/bioc/edgeR/src/R/calcNormFactors.R
        # From `f75 <- .calcFactorQuantile(data=x, lib.size=lib.size, p=0.75)` onwards

    return np.ones(counts.shape[1])
