import warnings

import numpy as np
import pandas as pd
from tmma.common import validate_series_indices
from tmma.constants import TMMA_ARRAY_DTYPE
from tmma.warnings import AsymptoticVarianceWarning


def ma_statistics(obs,
                  ref,
                  lib_size_obs: float = None,
                  lib_size_ref: float = None,
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

    if lib_size_obs is None:
        lib_size_obs = np.sum(obs)
    if lib_size_ref is None:
        lib_size_ref = np.sum(ref)

    lib_size_obs = float(lib_size_obs)
    lib_size_ref = float(lib_size_ref)

    if np.abs(lib_size_obs) <= 1e-6 or np.abs(lib_size_ref) <= 1e-6:
        raise ValueError("One of library sizes is zero")

    index = validate_series_indices(obs, ref)

    log2_normed_obs = np.log2(obs) - np.log2(lib_size_obs)
    log2_normed_ref = np.log2(ref) - np.log2(lib_size_ref)

    # M
    m = log2_normed_obs - log2_normed_ref
    # A
    a = 0.5 * (log2_normed_obs + log2_normed_ref)

    if index is not None:
        m = pd.Series(m, index=index, name='m_values')
        a = pd.Series(a, index=index, name='a_values')

    return m, a


def asymptotic_variance(obs, ref,
                        lib_size_obs: float = None,
                        lib_size_ref: float = None):
    """
    Computes asymptotic variance (weights) for TMM

    :param obs:
    :param ref:
    :param lib_size_obs:
    :param lib_size_ref:
    :return:
    """

    index = validate_series_indices(obs, ref)

    # Cast to float
    obs = np.asarray(obs, dtype=TMMA_ARRAY_DTYPE)
    ref = np.asarray(ref, dtype=TMMA_ARRAY_DTYPE)

    if lib_size_obs is None:
        lib_size_obs = np.sum(obs)
    if lib_size_ref is None:
        lib_size_ref = np.sum(ref)

    lib_size_obs = float(lib_size_obs)
    lib_size_ref = float(lib_size_ref)

    if np.any(obs >= lib_size_obs) or np.any(ref >= lib_size_ref):
        warnings.warn("Some of the observations are greater than library size. "
                      "Asymptotic variance assumptions may be violated.",
                      AsymptoticVarianceWarning)

    av = (lib_size_obs - obs) / lib_size_obs / obs + (lib_size_ref - ref) / lib_size_ref / ref

    if index is not None:
        av = pd.Series(av, index=index, name='asymptotic_variance')

    return av