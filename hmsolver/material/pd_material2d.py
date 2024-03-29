import numpy as np

from hmsolver.basis.quad4 import Quad4Node
from hmsolver.material.material2d import Material2d
from hmsolver.meshgrid.zone2d import Zone2d
from hmsolver.meshgrid.prototype_pd_mesh2d import PrototypePdMesh2d
from hmsolver.femcore.pd_stiffness import estimate_stiffness_matrix

__all__ = ['PdMaterial2d']


class PdMaterial2d(Material2d):
    def __init__(self, E, v, coefficients=np.array([1.0, 1.0, 1.0])):
        super().__init__(E, v)
        self.__coefficients_ = coefficients
        self.__pdready_ = False
        self.__stretch_crit_ = None

    def is_pdready(self):
        return self.__pdready_

    def can_break(self):
        return not (self.stretch_crit is None)

    def __init_std_meshgrid(self):
        scale = self.horizon_radius // self.grid_size
        assert scale > 0
        grids = 2 * scale + 1
        l, r, d = 0, grids * self.grid_size, self.grid_size
        self.__mesh_ = Zone2d(l, r, l, r).meshgrid_zone(PrototypePdMesh2d, d)
        self.__mesh_.peridynamic_construct(self.horizon_radius)

    def generate_coef(self):
        c0, c1, c2 = self.coefficients
        inst_len = self.inst_len

        def helper(x, y):
            # phi = np.arctan2(y, x)
            xi = np.hypot(x, y).reshape((-1, 1))
            scale = np.array([c0, c1, c2]).reshape((1, -1))
            # scale = np.array([c0, c1 * np.cos(2 * phi), c2 * np.cos(4 * phi)])
            return np.exp(-xi / inst_len) @ scale

        return helper
        # return np.vectorize(helper)

    def setOrthotropic(self, E1, E2, v12, v21, G_eff):
        super().setOrthotropic(E1, E2, v12, v21, G_eff)
        self.__pdready_ = False
        self.__stretch_crit_ = None

    def setPeridynamic(self, horizon_radius, grid_size, inst_len):
        self.__horizon_radius_ = horizon_radius
        self.__grid_size_ = grid_size
        self.__inst_len_ = inst_len
        self.__sync_stiffness()

    def __sync_stiffness(self):
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

    @property
    def stretch_crit(self):
        return self.__stretch_crit_

    @stretch_crit.setter
    def stretch_crit(self, s_crit=1.1):
        self.__stretch_crit_ = s_crit


# if __name__ == "__main__":
#     a = Material2d(3e11, 1.0 / 3)
#     print(1, a.youngs_modulus, a.poissons_ratio, a.lame_lambda, a.lame_mu)
#     print(2, a.constitutive)
#     b = PdMaterial2d(Material2d(3e11, 1.0 / 3))
#     print(3, b.youngs_modulus, b.poissons_ratio, b.lame_lambda, b.lame_mu)
#     b.setIsotropic(0.06, 0.02, 0.015)
#     print("b.coefficients=", b.coefficients)