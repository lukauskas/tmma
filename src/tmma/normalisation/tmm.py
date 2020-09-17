import warnings
from typing import Optional

import numpy as np
from scipy.stats import rankdata
from tmma.constants import TMMA_ARRAY_DTYPE
from tmma.ma.stats import ma_statistics, asymptotic_variance
from tmma.warnings import InfiniteWeightsWarning

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


def tmm_trim_mask(m_values,
                  a_values,
                  m_values_trim_fraction: float = 0.3,
                  a_values_trim_fraction: float = 0.05,
                  ):
    """

    Note that trimming is done from both ends.

    With the default parameters, the top and bottom 30% of M values.
    And the top and bottom 5% of A values will be trimmed.

    :param m_values: M values (log changes)
    :param a_values: A values (abs intensities)
    :param m_values_trim_fraction: log ratio trim coefficient
    :param a_values_trim_fraction: sum trim coefficient
    :return: A boolean array wose elements are set to:
            True, if value should be kept for mean calculation
            False, if value should not be considered for mean calculation.
    """

    # This is the actual TMM
    n = len(m_values)
    lowest_m_rank = np.floor(n * m_values_trim_fraction) + 1
    highest_m_rank = n + 1 - lowest_m_rank

    lowest_s_rank = np.floor(n * a_values_trim_fraction) + 1
    highest_s_rank = n + 1 - lowest_s_rank

    # This is needed to give the same answers as R
    # In practice it doesn't change much
    # Except in a few edge cases
    # see `test_with_custom_lib_sizes`
    m_values = np.round(m_values, 6)
    a_values = np.round(a_values, 6)

    m_ranks = rankdata(m_values, method='average')
    a_ranks = rankdata(a_values, method='average')

    # Trimming

    keep = (m_ranks >= lowest_m_rank) & (m_ranks <= highest_m_rank)
    keep &= (a_ranks >= lowest_s_rank) & (a_ranks <= highest_s_rank)

    return keep

def two_sample_tmm(obs,
                   ref,
                   lib_size_obs: Optional[float] = None,
                   lib_size_ref: Optional[float] = None,
                   m_values_trim_fraction: float = 0.3,
                   a_values_trim_fraction: float = 0.05,
                   weighted: bool = True,
                   a_cutoff: float = -1e10) -> float:
    """
    Computes the TMM normalisation factor by comparing `obs` array to `ref` array.
    The core function of `tmm_normalise`, equivalent to `.calcFactorTMM` in edgeR.

    :param obs: counts observed
    :param ref: counts reference
    :param lib_size_obs: library size observed
    :param lib_size_ref: library size reference
    :param m_values_trim_fraction: amount of trim to use on M values (log ratios),
           default: 0.3, `logratioTrim` in R
    :param a_values_trim_fraction: amount of trim to use on combined absolute values (A values),
           default: 0.05. `sumTrim` in R
    :param weighted: whether to compute asymptotic binomial precision weights,
           default: True. `doWeighting` in R
    :param a_cutoff: cutoff on A values,
           default: -1e10 (which is equivalent to no cutoff), `Acutoff` in R
    :return: Floating point number of scaling factor.
             Note that scaling factors are estimated in `log2`, but returned in the same scale
             as normal counts (i.e. returned values are *not* logarithms).
    """

    obs = np.asarray(obs, dtype=TMMA_ARRAY_DTYPE)
    ref = np.asarray(ref, dtype=TMMA_ARRAY_DTYPE)

    if lib_size_obs is None:
        lib_size_obs = np.sum(obs)

    if lib_size_ref is None:
        lib_size_ref = np.sum(ref)

    m_values, a_values = ma_statistics(obs, ref, lib_size_obs, lib_size_ref)

    # Remove all of the infinities that appear
    mask = np.isinf(m_values) | np.isnan(m_values)
    mask |= np.isinf(a_values) | np.isnan(a_values)

    # Also remove regions below "A" cutoff
    mask |= a_values <= a_cutoff

    # Nothing wsa left :(
    if (~mask).sum() == 0:
        return 1.0

    m_values = m_values[~mask]
    a_values = a_values[~mask]
    obs = obs[~mask]
    ref = ref[~mask]

    # That's a very odd clause in edgeR source, but I guess it is there fore a reason.
    if np.max(np.abs(m_values)) < 1e-6:
        return 1.0

    trimmed = tmm_trim_mask(m_values, a_values,
                            m_values_trim_fraction=m_values_trim_fraction,
                            a_values_trim_fraction=a_values_trim_fraction)

    if trimmed.sum() == 0:
        return 1.0

    m_values = m_values[trimmed]
    obs = obs[trimmed]
    ref = ref[trimmed]

    if weighted:
        # Estimated asymptotic variance
        variance = asymptotic_variance(obs, ref, lib_size_obs, lib_size_ref)
        if np.any(variance == 0):
            warnings.warn("Some weights became infinite due to zero estimated variance. "
                          "Returning 1.0 to match edgeR behaviour, but consider running "
                          "the normalisation with `do_weighting=False`",
                          InfiniteWeightsWarning)
            # Because sum(weights) will be inf
            # [anything] / inf = 0
            # And 2^0 = 1
            return 1.0

        weights = 1.0 / variance
    else:
        weights = None

    f = np.average(m_values, weights=weights)
    return np.power(2, f)

def scale_tmm_factors(unscaled_factors):
    """
    Ensures that factors returned by TMM multiply by one
    :param unscaled_factors: unscaled factors
    :return: scaled factors
    """
    return unscaled_factors / np.exp(np.mean(np.log(unscaled_factors)))

def _tmm_normalisation_factors_unscaled(counts,
                                        lib_sizes=None,
                                        ref_column: Optional[int] = None,
                                        m_values_trim_fraction: float = 0.3,
                                        a_values_trim_fraction: float = 0.05,
                                        weighted: bool = True,
                                        a_cutoff: float = -1e10):
    """
    Calculate normalisation factors using TMM method.
    Main logic of the `tmm_normalise` function below, however the
    factors are not scaled to multiply by one, like they are in `tmm_normalise`

    Otherwise behaviour is identical to edgeR::calcNormFactors.

    :param counts: numpy array of raw (unnormalised) counts for each of the samples.
                   Genes in rows, samples in columns.
    :param lib_sizes: (optional) numpy array of library sizes.
                      Should be in the same order as columns of `counts`
    :param ref_column: (optional) reference column to use
    :param m_values_trim_fraction: amount of trim to use on M values (log ratios),
           default: 0.3, `logratioTrim` in R
    :param a_values_trim_fraction: amount of trim to use on combined absolute values (A values),
           default: 0.05, `sumTrim` in R
    :param weighted: whether to compute asymptotic binomial precision weights,
           default: True, `doWeighting` in R
    :param a_cutoff: cutoff on A values,
           default: -1e10 (which is equivalent to no cutoff), `Acutoff` in R
    :return: Numpy array of normalisation factors from TMM.
    """

    counts = np.asarray(counts, dtype=TMMA_ARRAY_DTYPE)

    if np.isnan(counts).any() or np.isinf(counts).any():
        raise ValueError("Your counts contain NaNs and/or infinities. This is not allowed")

    n_rows, n_columns = counts.shape

    if lib_sizes is None:
        lib_sizes = np.sum(counts, 0)
    else:
        lib_sizes = np.asarray(lib_sizes, dtype=TMMA_ARRAY_DTYPE)
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
    if len(counts) < 1:
        return np.ones(counts.shape[1])

    if ref_column is None:
        factor_quantiles = _calc_factor_quantile(data=counts,
                                                 lib_sizes=lib_sizes,
                                                 p=_P_FOR_TMM)
        ref_column = np.argmin(np.abs(factor_quantiles - factor_quantiles.mean()))

    factors = []

    for col in range(n_columns):
        f_col = two_sample_tmm(obs=counts[:, col],
                               ref=counts[:, ref_column],
                               lib_size_obs=lib_sizes[col],
                               lib_size_ref=lib_sizes[ref_column],
                               m_values_trim_fraction=m_values_trim_fraction,
                               a_values_trim_fraction=a_values_trim_fraction,
                               weighted=weighted,
                               a_cutoff=a_cutoff)

        factors.append(f_col)
    factors = np.array(factors)
    return factors

def tmm_normalisation_factors(counts,
                              lib_sizes=None,
                              ref_column: Optional[int] = None,
                              m_values_trim_fraction: float = 0.3,
                              a_values_trim_fraction: float = 0.05,
                              weighted: bool = True,
                              a_cutoff: float = -1e10,
                              return_scaled=True
                              ):
    """
    Calculate normalisation factors using TMM method.
    Identical to edgeR::calcNormFactors.

    :param counts: numpy array of raw (unnormalised) counts for each of the samples.
                   Genes in rows, samples in columns.
    :param lib_sizes: (optional) numpy array of library sizes.
                      Should be in the same order as columns of `counts`
    :param ref_column: (optional) reference column to use
    :param m_values_trim_fraction: amount of trim to use on M values (log ratios),
                                  default: 0.3, `logratioTrim` in R
    :param a_values_trim_fraction: amount of trim to use on combined absolute values (A values),
                                  default: 0.05, `sumTrim` in R
    :param weighted: whether to compute asymptotic binomial precision weights, default: True
    :param a_cutoff: cutoff on A values,
                     default: -1e10 (which is equivalent to no cutoff), `Acutoff` in R
    :param return_scaled: whether to return normfactors that multiply to one,
                          default: True (to match R behaviour)
    :return:
    """

    factors = _tmm_normalisation_factors_unscaled(counts,
                                                  lib_sizes=lib_sizes,
                                                  ref_column=ref_column,
                                                  m_values_trim_fraction=m_values_trim_fraction,
                                                  a_values_trim_fraction=a_values_trim_fraction,
                                                  weighted=weighted,
                                                  a_cutoff=a_cutoff)
    if return_scaled:
        # Factors should multiply to one
        factors = scale_tmm_factors(factors)

    return factors
