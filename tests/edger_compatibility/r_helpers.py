"""
Helpers from dealing with edgeR
"""
from typing import Optional

import numpy as np
from rpy2 import robjects
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr

_r_edger = importr('edgeR')

def to_r_matrix(array):
    """
    Convert numpy array to an R matrix (suitable for edgeR)

    :param array:
    :return:
    """

    # Convert to float to avoid warning of dtype
    array = np.asarray(array, dtype=float)

    shape = array.shape
    # For some reason this gives a vector
    r_array_vec = numpy2ri.py2rpy(array)
    r_array_mat = robjects.r.matrix(r_array_vec, nrow=shape[0], ncol=shape[1])

    return r_array_mat

def to_r_vector(vector):
    """
    Convert a numpy vector to R vector (i.e. the one generated with `c`)
    :param vector:
    :return:
    """
    # Convert to float to avoid warning of dtype
    vector = np.asarray(vector, dtype=float)
    return robjects.FloatVector(vector)


def r_edger_calcNormFactors(counts,
                            lib_sizes=None,
                            ref_column: Optional[int] = None,
                            m_values_trim_fraction: float = 0.3,
                            a_values_trim_fraction: float = 0.05,
                            weighted: bool = True,
                            a_cutoff: float = -1e10):
    """
    Calls `edgeR::calcNormFactors` and returns a numpy array back.
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
           default: -1e10 (which is equivalent to no cutoff),
           `Acutoff` in R
    :return:
    """

    r_counts = to_r_matrix(counts)

    kwargs = {
        'logratioTrim': m_values_trim_fraction,
        'sumTrim': a_values_trim_fraction,
        'Acutoff': a_cutoff,
        'doWeighting': weighted
    }
    if lib_sizes is not None:
        kwargs['lib.size'] = to_r_vector(lib_sizes)

    if ref_column is not None:
        # R indexing issues:
        kwargs['refColumn'] = ref_column + 1

    r_answer = _r_edger.calcNormFactors(r_counts, **kwargs)
    return numpy2ri.rpy2py(r_answer)

def r_edger_calcFactorTMM(obs, ref,
                          lib_size_obs=None, lib_size_ref=None,
                          *args, **kwargs):

    r_obs = to_r_vector(obs)
    r_ref = to_r_vector(ref)

    kwargs = kwargs.copy()
    if lib_size_obs is not None:
        kwargs['libsize.obs'] = lib_size_obs
    if lib_size_ref is not None:
        kwargs['libsize.ref'] = lib_size_ref

    r_float = _r_edger._calcFactorTMM(r_obs, r_ref, *args, **kwargs)
    return numpy2ri.rpy2py(r_float)[0]
