from hmsolver.meshgrid.mesh2d import Mesh2d
from hmsolver.meshgrid.hybrid_mesh2d import HybridMesh2d
from hmsolver.material.material2d import Material2d
from hmsolver.material.pd_material2d import PdMaterial2d
from hmsolver.femcore.treat_boundary import BoundaryConds2d

__all__ = ['Problem2d', 'PdProblem2d']


class Problem2d(object):
    def __init__(self, mesh2d, material2d, bconds):
        self._mesh_ = mesh2d
        self._material_ = material2d
        self._boundary_conds_ = bconds
        self._msg1_ = ["Mesh", "Material", "Boundary Conds"]
        self._msg2_ = ["Mesh2d", "Material2d", "BoundaryConds2d"]
        self._type_check_result_ = self.type_check()

    def type_check(self):
        mesh_ret = self.is_mesh_match()
        mtrl_ret = self.is_material_match()
        bcnd_ret = self.is_boundary_conds_match()
        flags = [mesh_ret, mtrl_ret, bcnd_ret]
        for s1, s2, flag in zip(self._msg1_, self._msg2_, flags):
            if not flag:
                print(f"{s1} instance type ERROR.")
                print(f"SHOULD BE {s2} object/subclass object.")
        self._type_check_result_ = all(flags)
        return all(flags)

    @property
    def mesh(self):
        return self._mesh_

    @property
    def material(self):
        return self._material_

    @property
    def boundary_conds(self):
        return self._boundary_conds_

    def is_defined(self):
        return self._type_check_result_

    def is_mesh_match(self):
        return isinstance(self.mesh, Mesh2d)

    def is_material_match(self):
        return isinstance(self.material, Material2d)

    def is_boundary_conds_match(self):
        return isinstance(self.boundary_conds, BoundaryConds2d)


class PdProblem2d(Problem2d):
    def __init__(self, mesh2d, material2d, bconds):
        self._mesh_ = mesh2d
        self._material_ = material2d
        self._boundary_conds_ = bconds
        self._msg1_ = ["Mesh", "Material", "Boundary Conds"]
        self._msg2_ = ["HybridMesh2d", "PdMaterial2d", "BoundaryConds2d"]
        self._type_check_result_ = self.type_check()

    def is_mesh_match(self):
        return isinstance(self.mesh, HybridMesh2d)

    def is_material_match(self):
        return isinstance(self.material, PdMaterial2d)
