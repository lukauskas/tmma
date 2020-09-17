import unittest

import numpy as np
import pandas as pd
from hypothesis import given, assume
from hypothesis.strategies import booleans, integers
from numpy.testing import assert_allclose
from tmma.normalisation.tmm import tmm_normalisation_factors

from .r_helpers import r_edger_calcNormFactors
from .strategies import reasonable_floats

import os
_here = os.path.dirname(__file__)
DATASET = os.path.join(_here, '../..', 'data/from-edger-user-guide/arabidopsis/arab.csv')

def load_arabidopsis():
    return pd.read_csv(DATASET, index_col=0)

# Allow to differ by 0.01
ABS_TOL = 1e-2
REL_TOL = 0

class TestAgainstArabidopsisDataset(unittest.TestCase):
    """
    Tests against Arabidopsis dataset from edgeR user guide section 4.2.
    See `data/from-edger-user-guide/arabidopsis/`
    """
    @given(reasonable_floats(min_value=0.0, max_value=0.499), booleans())
    def test_different_m_values_trim_fraction(self, m_values_trim_fraction, do_weighting):
        df = load_arabidopsis()

        r_answer = r_edger_calcNormFactors(df,
                                           m_values_trim_fraction=m_values_trim_fraction,
                                           weighted=do_weighting
                                           )

        # No point testing bugs in R
        assume(not np.any(np.isinf(r_answer)))
        assume(not np.any(np.isnan(r_answer)))

        py_answer = tmm_normalisation_factors(df,
                                              m_values_trim_fraction=m_values_trim_fraction,
                                              weighted=do_weighting)
        assert_allclose(r_answer, py_answer, rtol=REL_TOL, atol=ABS_TOL)

    @given(reasonable_floats(min_value=0.0, max_value=0.499), booleans())
    def test_different_a_values_trim_fraction(self, a_values_trim_fraction, do_weighting):
        df = load_arabidopsis()

        r_answer = r_edger_calcNormFactors(df,
                                           a_values_trim_fraction=a_values_trim_fraction,
                                           weighted=do_weighting)

        # No point testing bugs in R
        assume(not np.any(np.isinf(r_answer)))
        assume(not np.any(np.isnan(r_answer)))

        py_answer = tmm_normalisation_factors(df, a_values_trim_fraction=a_values_trim_fraction,
                                              weighted=do_weighting)
        assert_allclose(r_answer, py_answer, rtol=REL_TOL, atol=ABS_TOL)

    @given(reasonable_floats(min_value=-18, max_value=-10), booleans())
    def test_different_a_cutoff(self, a_cutoff, do_weighting):
        df = load_arabidopsis()

        r_answer = r_edger_calcNormFactors(df,
                                           a_cutoff=a_cutoff,
                                           weighted=do_weighting)

        # No point testing bugs in R
        assume(not np.any(np.isinf(r_answer)))
        assume(not np.any(np.isnan(r_answer)))

        py_answer = tmm_normalisation_factors(df, a_cutoff=a_cutoff,
                                              weighted=do_weighting)
        assert_allclose(r_answer, py_answer, rtol=REL_TOL, atol=ABS_TOL)


    @given(integers(min_value=0, max_value=5), booleans())
    def test_different_ref_column(self, ref_col, do_weighting):
        df = load_arabidopsis()
        r_answer = r_edger_calcNormFactors(df, ref_column=ref_col,
                                           weighted=do_weighting)

        # No point testing bugs in R
        assume(not np.any(np.isinf(r_answer)))
        assume(not np.any(np.isnan(r_answer)))

        py_answer = tmm_normalisation_factors(df, ref_column=ref_col, weighted=do_weighting)
        assert_allclose(r_answer, py_answer, rtol=REL_TOL, atol=ABS_TOL)

    @given(
        reasonable_floats(min_value=0, max_value=0.449),
        reasonable_floats(min_value=0, max_value=0.449),
        reasonable_floats(min_value=-2, max_value=2),
        integers(min_value=0, max_value=5),
        booleans()
    )
    def test_all_at_once(self,
                         m_values_trim_fraction,
                         a_values_trim_fraction,
                         a_cutoff,
                         ref_column,
                         do_weighting
                         ):

        df = load_arabidopsis()
        r_answer = r_edger_calcNormFactors(df,
                                           m_values_trim_fraction=m_values_trim_fraction,
                                           a_values_trim_fraction=a_values_trim_fraction,
                                           a_cutoff=a_cutoff,
                                           ref_column=ref_column,
                                           weighted=do_weighting)

        # No point testing bugs in R
        assume(not np.any(np.isinf(r_answer)))
        assume(not np.any(np.isnan(r_answer)))

        py_answer = tmm_normalisation_factors(df,
                                              m_values_trim_fraction=m_values_trim_fraction,
                                              a_values_trim_fraction=a_values_trim_fraction,
                                              a_cutoff=a_cutoff,
                                              ref_column=ref_column,
                                              weighted=do_weighting)
        assert_allclose(r_answer, py_answer, rtol=REL_TOL, atol=ABS_TOL)
