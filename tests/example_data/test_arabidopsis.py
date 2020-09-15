import unittest
import pandas as pd
from tmma.tmm import tmm_normalise
import numpy as np
from numpy.testing import assert_array_equal

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
        assert_array_equal(actual_answer, expected_answer)





