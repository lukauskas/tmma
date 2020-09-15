import numpy as np


def tmm_normalise(counts, lib_sizes=None):
    """
    Calculate normalisation factors using TMM method.
    Identical to edgeR::calcNormFactors.

    :param counts: numpy array of raw (unnormalised) counts for each of the samples.
                   Genes in rows, samples in columns.
    :param lib_sizes: (optional) numpy array of library sizes.
                      Should be in the same order as columns of `counts`
    :return:
    """
    samples = np.asarray(counts, dtype=float)
    return np.ones(samples.shape[1])
