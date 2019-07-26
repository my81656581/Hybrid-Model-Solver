from hmsolver.meshgrid.mesh2d import Mesh2d
from hmsolver.meshgrid.hybrid_mesh2d import HybridMesh2d
from hmsolver.material.material2d import Material2d
from hmsolver.material.pd_material2d import PdMaterial2d
from hmsolver.femcore.treat_boundary import BoundaryConds2d

__all__ = ['Problem2d', 'PdProblem2d']


class Problem2d(object):
    def __init__(self, mesh2d, material2d, bconds):
        self.__mesh_ = mesh2d
        self.__material_ = material2d
        self.__boundary_conds_ = bconds
        self.__type_check_result_ = self.type_check()

    def type_check(self):
        mesh_ret = isinstance(self.mesh, Mesh2d)
        mtrl_ret = isinstance(self.material, Material2d)
        bcnd_ret = isinstance(self.boundary_conds, BoundaryConds2d)
        msg1 = ["Mesh", "Material", "Boundary Conds"]
        msg2 = ["Mesh2d", "Material2d", "BoundaryConds2d"]
        flags = [mesh_ret, mtrl_ret, bcnd_ret]
        for s1, s2, flag in zip(msg1, msg2, flags):
            if not flag:
                print(f"{s1} instance type ERROR.")
                print(f"SHOULD BE {s2} object/subclass object.")
        self.__type_check_result_ = all(flags)
        return all(flags)

    @property
    def mesh(self):
        return self.__mesh_

    @property
    def material(self):
        return self.__material_

    @property
    def boundary_conds(self):
        return self.__boundary_conds_

    def is_defined(self):
        return self.__type_check_result_


class PdProblem2d(Problem2d):
    def type_check(self):
        mesh_ret = isinstance(self.mesh, HybridMesh2d)
        mtrl_ret = isinstance(self.material, PdMaterial2d)
        bcnd_ret = isinstance(self.boundary_conds, BoundaryConds2d)
        if not mesh_ret:
            print("Mesh instance type ERROR.")
            print("SHOULD BE Mesh2d object/subclass object.")
        if not mtrl_ret:
            print("Material instance type ERROR.")
            print("SHOULD BE Material2d object/subclass object.")
        if not bcnd_ret:
            print("Boundary Conds instance type ERROR.")
            print("SHOULD BE Boundary_Conds2d object/subclass object.")
        return mesh_ret & mtrl_ret & bcnd_ret