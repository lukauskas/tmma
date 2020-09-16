"""
Helpers from dealing with edgeR
"""
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


def r_edger_calcNormFactors(counts, lib_size=None, *args, **kwargs):
    """
    Calls `edgeR::calcNormFactors` and returns a numpy array back.
    :param counts: numpy array of raw counts
    :param lib_size: (optional) numpy array of library sizes

    :param args:
    :param kwargs:
    :return: edgeR scaling factors as numpy array.
    """

    r_counts = to_r_matrix(counts)

    kwargs = kwargs.copy()
    if lib_size is not None:
        kwargs['lib.size'] = to_r_vector(lib_size)

    r_answer = _r_edger.calcNormFactors(r_counts, *args, **kwargs)
    return numpy2ri.rpy2py(r_answer)

def r_edger_calcFactorTMM(obs, ref, lib_size_obs=None, lib_size_ref=None, *args, **kwargs):

    r_obs = to_r_vector(obs)
    r_ref = to_r_vector(ref)

    kwargs = kwargs.copy()
    if lib_size_obs is not None:
        kwargs['libsize.obs'] = lib_size_obs
    if lib_size_ref is not None:
        kwargs['libsize.ref'] = lib_size_ref

    return _r_edger._calcFactorTMM(r_obs, r_ref, *args, **kwargs)
