import unittest

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from tmma.tmm import tmm_normalise, _tmm_normalise_unscaled

DATASET = '../data/from-edger-user-guide/arabidopsis/arab.csv'

def load_arabidopsis():
    return pd.read_csv(DATASET, index_col=0)


ABS_TOL = 1e-3
REL_TOL = 0

class TestAgainstArabidopsisDataset(unittest.TestCase):
    """
    Tests against Arabidopsis dataset from edgeR user guide section 4.2.
    See `data/from-edger-user-guide/arabidopsis/`
    """

    def test_arabidopsis_default_parameters(self):
        df = load_arabidopsis()

        # See `data/from-edger-user-guide/README.md`
        expected_answer = np.array([
            0.9771684,
            1.0228448,
            0.9142136,
            1.0584492,
            1.0828426,
            0.9548559
        ])

        actual_answer = tmm_normalise(df)
        assert_allclose(actual_answer, expected_answer, atol=ABS_TOL, rtol=REL_TOL)

    def test_arabidopsis_specific_ref_column(self):
        df = load_arabidopsis()

        # This is equivalent to ref_column=1 in R due to indexing differences
        ref_column = 0

        # > calcNormFactors(counts, refColumn=1)
        #     mock1     mock2     mock3     hrcc1     hrcc2     hrcc3
        # 0.9787380 0.9931984 0.9287641 1.0577844 1.0948637 0.9563897
        expected_answer = np.array([
            0.9787380, 0.9931984, 0.9287641, 1.0577844, 1.0948637, 0.9563897
        ])

        actual_answer = tmm_normalise(df, ref_column=ref_column)
        assert_allclose(actual_answer, expected_answer, atol=ABS_TOL, rtol=REL_TOL)

    def test_arabidopsis_specific_sum_trim(self):
        df = load_arabidopsis()

        # Failure case discovered by hypothesis
        sum_trim = 0.4997824194952133

        expected_answer_weighted = np.array([
            0.9288730, 1.1990624, 0.8014742, 1.0889198, 1.1075418, 0.9288730
        ])

        expected_answer_unweighted = np.array([
            0.9297549, 1.1952989, 0.8021529, 1.0885250, 1.1083857, 0.9297549
        ])

        actual_answer_weighted = tmm_normalise(df, sum_trim=sum_trim, do_weighting=True)
        assert_allclose(actual_answer_weighted, expected_answer_weighted, atol=ABS_TOL, rtol=REL_TOL)

        actual_answer_unweighted = tmm_normalise(df, sum_trim=sum_trim, do_weighting=False)
        assert_allclose(actual_answer_unweighted, expected_answer_unweighted, atol=ABS_TOL,
                        rtol=REL_TOL)

    def test_arabidopsis_unscaled(self):

        df = load_arabidopsis()
        expected_answer = np.array([
            1.0233674, 1.0712034, 0.9574362, 1.1084911, 1.1340377, 1.0000000,
        ])

        actual_answer = _tmm_normalise_unscaled(df)

        assert_allclose(actual_answer, expected_answer, atol=ABS_TOL, rtol=REL_TOL)

    def test_arabidopsis_unscaled_specific_ref_column(self):

        df = load_arabidopsis()
        expected_answer = np.array([
            1.0000000, 1.0147745, 0.9489405, 1.0807636, 1.1186484, 0.9771662
        ])

        actual_answer = _tmm_normalise_unscaled(df, ref_column=0)

        assert_allclose(actual_answer, expected_answer, atol=ABS_TOL, rtol=REL_TOL)




