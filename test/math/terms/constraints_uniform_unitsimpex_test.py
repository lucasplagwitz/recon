

from recon.math.terms.constraints_uniform_unitsimplex import ConstraintsUniformUnitsimplex

from numpy.testing import assert_almost_equal

import unittest
import numpy as np

class ConstraintsUniformUnitsimplexTest(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def test_project_unitsimplex(self):
        """
        Example of V. N. Malozemov and G. Sh. Tamasyan
        Two Fast Algorithms for Projecting a Point onto the Canonical Simplex
        :return:
        """
        c = np.array([[-1, 1, 0, -1, 0, 2/3]])

        x = ConstraintsUniformUnitsimplex.project_unitsimplex(c)

        assert_almost_equal(x, np.array([[0, 2/3, 0, 0, 0, 1/3]]), decimal=10)