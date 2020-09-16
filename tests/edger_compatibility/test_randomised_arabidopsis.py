import unittest

import numpy as np
import pandas as pd
from hypothesis import given, assume
from hypothesis.strategies import floats, booleans, integers
from numpy.testing import assert_allclose
from tmma.tmm import tmm_normalise

from tests.edger_compatibility.r_helpers import r_edger_calcNormFactors

DATASET = '../data/from-edger-user-guide/arabidopsis/arab.csv'

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
    @given(floats(min_value=0.0, max_value=1.0), booleans())
    def test_different_log_ratio_trim(self, log_ratio_trim, do_weighting):
        df = load_arabidopsis()

        r_answer = r_edger_calcNormFactors(df,
                                           log_ratio_trim=log_ratio_trim,
                                           do_weighting=do_weighting)

        # No point testing bugs in R
        assume(not np.any(np.isinf(r_answer)))
        assume(not np.any(np.isnan(r_answer)))

        py_answer = tmm_normalise(df, log_ratio_trim=log_ratio_trim, do_weighting=do_weighting)
        assert_allclose(r_answer, py_answer, rtol=REL_TOL, atol=ABS_TOL)

    @given(floats(min_value=0.0, max_value=1.0), booleans())
    def test_different_sum_trim(self, sum_trim, do_weighting):
        df = load_arabidopsis()

        r_answer = r_edger_calcNormFactors(df, sum_trim=sum_trim,
                                           do_weighting=do_weighting)

        # No point testing bugs in R
        assume(not np.any(np.isinf(r_answer)))
        assume(not np.any(np.isnan(r_answer)))

        py_answer = tmm_normalise(df, sum_trim=sum_trim, do_weighting=do_weighting)
        assert_allclose(r_answer, py_answer, rtol=REL_TOL, atol=ABS_TOL)

    @given(floats(min_value=-2, max_value=2), booleans())
    def test_different_a_cutoff(self, a_cutoff, do_weighting):
        df = load_arabidopsis()

        r_answer = r_edger_calcNormFactors(df, sum_trim=a_cutoff,
                                           do_weighting=do_weighting)

        # No point testing bugs in R
        assume(not np.any(np.isinf(r_answer)))
        assume(not np.any(np.isnan(r_answer)))

        py_answer = tmm_normalise(df, sum_trim=a_cutoff, do_weighting=do_weighting)
        assert_allclose(r_answer, py_answer, rtol=REL_TOL, atol=ABS_TOL)


    @given(integers(min_value=0, max_value=5), booleans())
    def test_different_ref_column(self, ref_col, do_weighting):
        df = load_arabidopsis()
        r_answer = r_edger_calcNormFactors(df, ref_column=ref_col,
                                           do_weighting=do_weighting)

        # No point testing bugs in R
        assume(not np.any(np.isinf(r_answer)))
        assume(not np.any(np.isnan(r_answer)))

        py_answer = tmm_normalise(df, ref_column=ref_col, do_weighting=do_weighting)
        assert_allclose(r_answer, py_answer, rtol=REL_TOL, atol=ABS_TOL)

    @given(
        floats(min_value=0, max_value=1),
        floats(min_value=0, max_value=1),
        floats(min_value=-2, max_value=2),
        integers(min_value=0, max_value=5),
        booleans()
    )
    def test_all_at_once(self,
                         log_ratio_trim,
                         sum_trim,
                         a_cutoff,
                         ref_column,
                         do_weighting
                         ):

        df = load_arabidopsis()
        r_answer = r_edger_calcNormFactors(df,
                                           log_ratio_trim=log_ratio_trim,
                                           sum_trim=sum_trim,
                                           a_cutoff=a_cutoff,
                                           ref_column=ref_column,
                                           do_weighting=do_weighting)

        # No point testing bugs in R
        assume(not np.any(np.isinf(r_answer)))
        assume(not np.any(np.isnan(r_answer)))

        py_answer = tmm_normalise(df,
                                  log_ratio_trim=log_ratio_trim,
                                  sum_trim=sum_trim,
                                  a_cutoff=a_cutoff,
                                  ref_column=ref_column,
                                  do_weighting=do_weighting)
        assert_allclose(r_answer, py_answer, rtol=REL_TOL, atol=ABS_TOL)
