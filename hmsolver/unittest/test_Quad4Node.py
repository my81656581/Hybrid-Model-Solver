import unittest
import numpy as np

# local test only
# import sys
# sys.path.append('./../../')

from hmsolver.basis.Quad4Node import Quad4Node


class TestQuad4Node(unittest.TestCase):
    def setUp(self):
        self._a = np.array([1, 2, 3, 4, 5])
        self._b = np.array([1, 2, 3, 4, 5])
        self._basis = Quad4Node()

    def tearDown(self):
        pass

    def test_reference_basis_0_0_0(self):
        result = self._basis.reference_basis(self._a, self._b, 0, (0, 0))
        answer = np.array([0, 0.25, 1, 2.25, 4])
        runs = (np.abs((result - answer)) < 1e-9).all()
        self.assertEqual(runs, True)

    def test_reference_basis_0_0_1(self):
        result = self._basis.reference_basis(self._a, self._b, 0, (0, 1))
        answer = np.array([0, 0.25, 0.5, 0.75, 1])
        runs = (np.abs((result - answer)) < 1e-9).all()
        self.assertEqual(runs, True)

    def test_reference_basis_0_1_0(self):
        result = self._basis.reference_basis(self._a, self._b, 0, (1, 0))
        answer = np.array([0, 0.25, 0.5, 0.75, 1])
        runs = (np.abs((result - answer)) < 1e-9).all()
        self.assertEqual(runs, True)

    def test_reference_basis_0_1_1(self):
        result = self._basis.reference_basis(self._a, self._b, 0, (1, 1))
        answer = np.array([0.25, 0.25, 0.25, 0.25, 0.25])
        runs = (np.abs((result - answer)) < 1e-9).all()
        self.assertEqual(runs, True)

    def test_reference_basis_0_1_2(self):
        result = self._basis.reference_basis(self._a, self._b, 0, (1, 2))
        answer = np.array([0, 0, 0, 0, 0])
        runs = (np.abs((result - answer)) < 1e-9).all()
        self.assertEqual(runs, True)

    def test_reference_basis_0_2_1(self):
        result = self._basis.reference_basis(self._a, self._b, 0, (2, 1))
        answer = np.array([0, 0, 0, 0, 0])
        runs = (np.abs((result - answer)) < 1e-9).all()
        self.assertEqual(runs, True)


if __name__ == "__main__":
    unittest.main()
