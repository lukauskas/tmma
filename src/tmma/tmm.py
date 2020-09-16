from typing import Optional

import numpy as np
from scipy.stats import rankdata

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

def _ma_stats(obs,
              ref,
              lib_size_obs: float,
              lib_size_ref: float,
              ):
    """
    Calculates the M and A values for two datasets (obs and ref)
    M (minus) values correspond to log2 fold changes obs/ref
    A (add) values correspond to mean of the two values in log scale

    See more here https://en.wikipedia.org/wiki/MA_plot

    :param obs:
    :param ref:
    :param lib_size_obs: library size for obs array
    :param lib_size_ref: library size for ref array
    :return: tuple (m, a)
    """

    log2_normed_obs = np.log2(obs) - np.log2(lib_size_obs)
    log2_normed_ref = np.log2(ref) - np.log2(lib_size_ref)

    # M
    m = log2_normed_obs - log2_normed_ref
    # A
    a = 0.5 * (log2_normed_obs + log2_normed_ref)

    return m, a

def _asymptotic_variance(obs, ref,
                         lib_size_obs: float,
                         lib_size_ref: float):
    """
    Computes asymptotic variance (weights) for TMM

    :param obs:
    :param ref:
    :param lib_size_obs:
    :param lib_size_ref:
    :return:
    """
    return (lib_size_obs - obs) / lib_size_obs / obs + (lib_size_ref - ref) / lib_size_ref / ref


def _tmm_trim(m_values, a_values,
              log_ratio_trim: float = 0.3,
              sum_trim: float = 0.05,
              ):
    """

    :param m_values: M values (log changes)
    :param a_values: A values (abs intensities)
    :param log_ratio_trim: log ratio trim coefficient
    :param sum_trim: sum trim coefficient
    :return:
    """

    # This is the actual TMM
    n = len(m_values)
    loL = np.floor(n * log_ratio_trim) + 1
    hiL = n + 1 - loL

    loS = np.floor(n * sum_trim) + 1
    hiS = n + 1 - loS

    # This is needed to give the same answers as R
    # In practice it doesn't change much
    # Except in a few edge cases
    m_values = np.round(m_values, 6)
    a_values = np.round(a_values, 6)

    logR_rank = rankdata(m_values, method='average')
    absE_rank = rankdata(a_values, method='average')

    # Trimming

    keep = (logR_rank >= loL) & (logR_rank <= hiL)
    keep &= (absE_rank >= loS) & (absE_rank <= hiS)

    return keep

def _calc_factor_tmm(obs, ref,
                     lib_size_obs: Optional[float] = None,
                     lib_size_ref: Optional[float] = None,
                     log_ratio_trim: float = 0.3,
                     sum_trim: float = 0.05,
                     do_weighting: bool = True,
                     a_cutoff: float = -1e10):
    """
    Function to perform TMM normalisation.
    Identical to `.calcFactorTMM` in edgeR.

    :param obs: counts observed
    :param ref: counts reference
    :param lib_size_obs: library size observed
    :param lib_size_ref: library size reference
    :param log_ratio_trim: amount of trim to use on M values (log ratios), default: 0.3
    :param sum_trim: amount of trim to use on combined absolute values (A values), default: 0.05
    :param do_weighting: whether to compute asymptotic binomial precision weights, default: True
    :param a_cutoff: cutoff on A values, default: -1e10 (which is equivalent to no cutoff)
    :return:
    """

    obs = np.asarray(obs)
    ref = np.asarray(ref)

    if lib_size_obs is None:
        lib_size_obs = np.sum(obs)

    if lib_size_ref is None:
        lib_size_ref = np.sum(ref)

    logR, absE = _ma_stats(obs, ref, lib_size_obs, lib_size_ref)

    # Remove all of the infinities that appear
    mask = np.isinf(logR) | np.isinf(absE)
    # Also remove regions below "A" cutoff
    mask |= absE <= a_cutoff

    logR = logR[~mask]
    absE = absE[~mask]
    obs = obs[~mask]
    ref = ref[~mask]

    # That's a very odd clause in edgeR source, but I guess it is there fore a reason.
    if np.max(np.abs(logR)) < 1e-6:
        return 1.0

    trimmed = _tmm_trim(logR, absE, log_ratio_trim=log_ratio_trim, sum_trim=sum_trim)

    logR = logR[trimmed]
    obs = obs[trimmed]
    ref = ref[trimmed]

    if do_weighting:
        # Estimated asymptotic variance
        weights = 1.0 / _asymptotic_variance(obs, ref, lib_size_obs, lib_size_ref)
    else:
        weights = None

    f = np.average(logR, weights=weights)
    return np.power(2, f)


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
    :param sum_trim: amount of trim to use on combined absolute values (A values), default: 0.05
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
        ref_column = np.argmin(np.abs(factor_quantiles - factor_quantiles.mean()))

    factors = []

    for col in range(n_columns):
        f_col = _calc_factor_tmm(obs=counts[:, col],
                                 ref=counts[:, ref_column],
                                 lib_size_obs=lib_sizes[col],
                                 lib_size_ref=lib_sizes[ref_column],
                                 log_ratio_trim=log_ratio_trim,
                                 sum_trim=sum_trim,
                                 do_weighting=do_weighting,
                                 a_cutoff=a_cutoff)

        factors.append(f_col)
    factors = np.array(factors)

    # Factors should multiply to one
    factors = factors / np.exp(np.mean(np.log(factors)))
    return factors
