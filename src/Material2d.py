import numpy as np

from generate_mesh import build_mesh
from Mesh2d import PrototypePdMesh2d
import pd_stiffness


class Material2d(object):
    def __init__(self,
                 youngs_modulus: float = 3e11,
                 poissons_ratio: float = 1.0 / 3):
        c11 = youngs_modulus / (1 - poissons_ratio * poissons_ratio)
        c12 = c11 * poissons_ratio
        c33 = youngs_modulus / (1 + poissons_ratio) / 2.0
        self.__youngs_modulus_ = youngs_modulus
        self.__poissons_ratio_ = poissons_ratio
        self.__lame_mu_ = c33
        self.__lame_lambda_ = c33 * 2.0 * poissons_ratio / (1 -
                                                            2 * poissons_ratio)
        self.__constructive_ = np.array([[c11, c12, 0], [c12, c11, 0],
                                         [0, 0, c33]])

    @property
    def youngs_modulus(self):
        return self.__youngs_modulus_

    @property
    def poissons_ratio(self):
        return self.__poissons_ratio_

    @property
    def shear_modulus(self):
        return self.__lame_mu_

    @property
    def lame_lambda(self):
        return self.__lame_lambda_

    @property
    def lame_mu(self):
        return self.__lame_mu_

    @property
    def constructive(self):
        return self.__constructive_


class PdMaterial2d(Material2d):
    def __init__(self,
                 continuum: Material2d = Material2d(),
                 coefficients=np.array([1.0, 0.0, 0.0])):
        super().__init__(continuum.youngs_modulus, continuum.poissons_ratio)
        self.coefficients = coefficients

    def __init_std_meshgrid(self, horizon_radius, grid_size):
        scale = horizon_radius // grid_size
        assert scale > 0
        grids = 2 * scale + 1
        _l, _r, _d = 0, grids * grid_size, grid_size
        nodes, elements = build_mesh((_l, _r, _d), (_l, _r, _d))
        n_nodes, n_elements = len(nodes), len(elements)
        self.__mesh_ = PrototypePdMesh2d(n_nodes, n_elements)
        self.__mesh_.manually_construct(np.array(nodes), np.array(elements))
        self.__mesh_.prototype_construct(horizon_radius)

    def setIsotropic(self, horizon_radius, grid_size=0):
        self.__grid_vol_ = (horizon_radius / 3.0)**2
        # e = self.youngs_modulus
        # v = self.poissons_ratio
        # pid6 = np.pi * horizon_radius ** 6 # \pi \times d^6
        # self.__coefficients_[0] = 12 * e / (1 - v) / pid6
        # self.__coefficients_[1] = 0
        # self.__coefficients_[2] = 0
        # print(self.__coefficients_)

    def syncIsotropic(self, horizon_radius, grid_size, inst_len):
        self.__init_std_meshgrid(horizon_radius, grid_size)
        grid_vol = grid_size**2
        c0, c1, c2 = self.coefficients

        def helper(x, y):
            phi, xi = np.arctan2(y, x), np.hypot(x, y)
            scale = c0 + c1 * np.cos(2 * phi) + c2 * np.cos(4 * phi)
            return scale * np.exp(-xi / inst_len)

        coef_fun = np.vectorize(helper)
        pd_constructive = pd_stiffness.estimate_stiffness_matrix_isotropic(
            self.__mesh_, coef_fun)
        pd_constructive /= grid_vol
        print(pd_constructive)

        # pd_constructive = np.reshape(pd_constructive, (-1, 1))
        # con_vec = np.reshape(
        #     np.array([self.constructive[_, _] for _ in range(3)]), (-1, 1))
        # constructive = np.hstack((con_vec, con_vec, con_vec))
        # coeff = np.linalg.pinv(constructive) / pd_constructive
        # self.coefficients = np.reshape(coeff, (-1))
        self.coefficients[0] = self.constructive[0, 0] / pd_constructive[0]
        self.coefficients[1] = self.constructive[1, 1] / pd_constructive[1]
        self.coefficients[2] = self.constructive[2, 2] / pd_constructive[2]

    def testIsotropic(self, horizon_radius, grid_size, inst_len):
        self.__init_std_meshgrid(horizon_radius, grid_size)
        grid_vol = grid_size**2
        c0, c1, c2 = self.coefficients

        def helper(x, y):
            phi, xi = np.arctan2(y, x), np.hypot(x, y)
            scale = c0 + c1 * np.cos(2 * phi) + c2 * np.cos(4 * phi)
            return scale * np.exp(-xi / inst_len)

        coef_fun = np.vectorize(helper)
        pd_constructive = pd_stiffness.estimate_stiffness_matrix_isotropic(
            self.__mesh_, coef_fun)
        pd_constructive /= grid_vol
        print(pd_constructive)


if __name__ == "__main__":
    a = Material2d(3e11, 1.0 / 3)
    print(a.youngs_modulus, a.poissons_ratio, a.lame_lambda, a.lame_mu)
    print(a.constructive)
    b = PdMaterial2d(Material2d(3e11, 1.0 / 3))
    print(b.youngs_modulus, b.poissons_ratio, b.lame_lambda, b.lame_mu)
    b.syncIsotropic(0.06, 0.02, 0.015)
    print("b.coefficients=", b.coefficients)
    b.testIsotropic(0.06, 0.02, 0.015)