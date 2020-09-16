import unittest

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from tmma.tmm import tmm_normalise

DATASET = '../data/from-edger-user-guide/arabidopsis/arab.csv'

def load_arabidopsis():
    return pd.read_csv(DATASET, index_col=0)

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
        assert_allclose(actual_answer, expected_answer, rtol=1e-6)

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
        assert_allclose(actual_answer, expected_answer, rtol=1e-6)





