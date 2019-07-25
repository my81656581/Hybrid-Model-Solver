import numpy as np

from hmsolver.basis.quad4 import Quad4Node
from hmsolver.material.material2d import Material2d
from hmsolver.meshgrid.prototype_pd_mesh2d import PrototypePdMesh2d
from hmsolver.meshgrid.generate_mesh import build_mesh
from hmsolver.femcore.pd_stiffness import estimate_stiffness_matrix

__all__ = ['PdMaterial2d']


class PdMaterial2d(Material2d):
    def __init__(self, E, v, coefficients=np.array([1.0, 1.0, 1.0])):
        super().__init__(E, v)
        self.__coefficients_ = coefficients
        self.__pdready_ = False

    def is_pdready(self):
        return self.__pdready_

    def __init_std_meshgrid(self):
        scale = self.horizon_radius // self.grid_size
        assert scale > 0
        grids = 2 * scale + 1
        _l, _r, _d = 0, grids * self.grid_size, self.grid_size
        nodes, elements = build_mesh((_l, _r, _d), (_l, _r, _d))
        n_nodes, n_elements = len(nodes), len(elements)
        self.__mesh_ = PrototypePdMesh2d(n_nodes, n_elements)
        self.__mesh_.manually_construct(np.array(nodes), np.array(elements))
        self.__mesh_.prototype_construct(self.horizon_radius)

    def generate_coef(self):
        c0, c1, c2 = self.coefficients
        inst_len = self.inst_len

        def helper(x, y):
            # phi = np.arctan2(y, x)
            xi = np.hypot(x, y)
            scale = np.array([c0, c1, c2])
            # scale = np.array([c0, c1 * np.cos(2 * phi), c2 * np.cos(4 * phi)])
            return scale * np.exp(-xi / inst_len)

        return np.vectorize(helper)

    def setIsotropic(self, horizon_radius, grid_size, inst_len):
        self.__horizon_radius_ = horizon_radius
        self.__grid_size_ = grid_size
        self.__inst_len_ = inst_len
        self.__syncIsotropic()

    def __syncIsotropic(self):
        self.__init_std_meshgrid()
        grid_vol = self.grid_size**2
        basis = Quad4Node()
        coef_fun = self.generate_coef()
        pd = estimate_stiffness_matrix(self.__mesh_, basis, coef_fun)
        pd /= grid_vol
        self.coefficients = np.diag(self.constitutive) / pd
        coef_fun = self.generate_coef()
        pd = estimate_stiffness_matrix(self.__mesh_, basis, coef_fun)
        pd /= grid_vol
        ratio = np.diag(self.constitutive) / pd
        print("Synchronize Complete. Ratio=", ratio)
        print("Constitutive: (C11, C22, C33)=", np.diag(self.constitutive))
        print("Peridynamic:  (C11, C22, C33)=", pd)
        self.__pdready_ = True

    @property
    def coefficients(self):
        return self.__coefficients_

    @coefficients.setter
    def coefficients(self, var):
        self.__coefficients_ = var

    @property
    def horizon_radius(self):
        return self.__horizon_radius_

    @property
    def grid_size(self):
        return self.__grid_size_

    @property
    def inst_len(self):
        return self.__inst_len_


# if __name__ == "__main__":
#     a = Material2d(3e11, 1.0 / 3)
#     print(1, a.youngs_modulus, a.poissons_ratio, a.lame_lambda, a.lame_mu)
#     print(2, a.constitutive)
#     b = PdMaterial2d(Material2d(3e11, 1.0 / 3))
#     print(3, b.youngs_modulus, b.poissons_ratio, b.lame_lambda, b.lame_mu)
#     b.setIsotropic(0.06, 0.02, 0.015)
#     print("b.coefficients=", b.coefficients)