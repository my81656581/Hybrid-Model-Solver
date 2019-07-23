import unittest
import numpy as np

# local test only
# import sys
# sys.path.append('./../../')

from hmsolver.material.material2d import Material2d
from hmsolver.material.pd_material2d import PdMaterial2d


class TestPdMaterial2d(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_material2d_lame(self):
        a = Material2d(3e11, 1.0 / 3)
        result = np.array(
            [a.youngs_modulus, a.poissons_ratio, a.lame_lambda, a.lame_mu])
        answer = np.array([3e11, 1.0 / 3, 2.25e11, 1.125e11])
        ratio = result / answer
        runs = (1 - 0.005 < ratio).all() and (1 + 0.005 > ratio).all()
        self.assertEqual(runs, True)

    def test_material2d_constitutive(self):
        a = Material2d(3e11, 1.0 / 3)
        result = np.reshape(a.constitutive, (-1)) + 1e-9
        answer = np.reshape(
            np.array([[3.375e+11, 1.125e+11, 0], [1.125e+11, 3.375e+11, 0],
                      [0, 0, 1.125e+11]]), (-1)) + 1e-9
        ratio = result / answer
        runs = (1 - 0.005 < ratio).all() and (1 + 0.005 > ratio).all()
        self.assertEqual(runs, True)

    def test_pd_material2d(self):
        b = PdMaterial2d(3e11, 1.0 / 3)
        b.setIsotropic(0.06, 0.02, 0.015)
        result = b.coefficients
        answer = np.array([1.24834178e+21, 1.24834178e+21, 9.78995500e+20])
        ratio = result / answer
        runs = (1 - 0.005 < ratio).all() and (1 + 0.005 > ratio).all()
        self.assertEqual(runs, True)


if __name__ == "__main__":
    unittest.main()
