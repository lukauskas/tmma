from hypothesis import given
from tmma.tmm import _calc_factor_tmm

from tests.edger_compatibility.r_helpers import r_edger_calcFactorTMM
from tests.edger_compatibility.strategies import uint_counts_array_and_a_lib_size, \
    uint_counts_array, poisson_counts_array


class HypothesisTestCalcFactorTMM(unittest.TestCase):

    @given(uint_counts_array_and_a_lib_size(max_cols=2))
    def test_uints_and_random_library_size(self, counts_lib_size):
        """
        Given random unsigned integer counts and a random library size
        Check if the output is the same as edgeR's.
        :param counts_lib_size:
        :return:
        """

        counts, lib_size = counts_lib_size

        obs = counts[:, 0]
        ref = counts[:, 1]

        lib_size_obs, lib_size_ref = lib_size

        r_answer = r_edger_calcFactorTMM(obs, ref,
                                         lib_size_obs=lib_size_obs,
                                         lib_size_ref=lib_size_ref)

        py_answer = _calc_factor_tmm(obs, ref,
                                     lib_size_obs=lib_size_obs,
                                     lib_size_ref=lib_size_ref)

        assert_allclose(r_answer, py_answer, rtol=1e-6)

    @given(uint_counts_array(max_cols=2))
    def test_uints_only(self, counts):
        """
        Given random unsigned integer counts,
        check if the output is the same as edgeR's.
        :param counts:
        :return:
        """

        obs = counts[:, 0]
        ref = counts[:, 1]

        r_answer = r_edger_calcFactorTMM(obs, ref)
        py_answer = _calc_factor_tmm(obs, ref)

        assert_allclose(r_answer, py_answer, rtol=1e-6)

    @given(poisson_counts_array(max_cols=2))
    def test_poisson_only(self, counts):
        """
        Given random poisson counts,
        check if output matches edgeR.

        :param counts:
        :return:
        """
        obs = counts[:, 0]
        ref = counts[:, 1]

        r_answer = r_edger_calcFactorTMM(obs, ref)
        py_answer = _calc_factor_tmm(obs, ref)

        assert_allclose(r_answer, py_answer, rtol=1e-6)


if __name__ == '__main__':
    unittest.main()
