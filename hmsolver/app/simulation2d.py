from hmsolver.meshgrid.hybrid_mesh2d import HybridMesh2d
from hmsolver.material.pd_material2d import PdMaterial2d
from hmsolver.app.problem2d import Problem2d, PdProblem2d

from hmsolver.basis.basisinfo import is_suitable_for_mesh
from hmsolver.basis.infrastructures import IShapeable

import hmsolver.femcore.main_procedure as main_procedure

__all__ = ['Simulation2d', 'PdSimulation2d']


class Simulation2d(Problem2d):
    def __init__(self, mesh2d, material2d, bconds):
        super().__init__(mesh2d, material2d, bconds)
        self.__basis_ = None
        self._state_ = [False, False, False, False]
        self.__app_name_ = None

    def _check_mesh(self):
        self._state_[0] = self.mesh.is_ready()
        return self._state_[0]

    def _check_material(self):
        self._state_[1] = self.material.is_ready()
        return self._state_[1]

    def _check_boundary_conds(self):
        self._state_[2] = self.boundary_conds.is_ready()
        return self._state_[2]

    def _check_basis(self):
        if isinstance(self.basis, IShapeable):
            self._state_[3] = is_suitable_for_mesh(self.basis.typeid,
                                                   self.mesh.e_basistype)
        return self._state_[3]

    def apply_basis(self, basis: IShapeable):
        if is_suitable_for_mesh(basis.typeid, self.mesh.e_basistype):
            self.__basis_ = basis
        else:
            print("Basis is NOT suitable for mesh grid.")

    def check_engine(self):
        print("*" * 32 + "\nSimulation Manual Checking:")
        msgs = ["Mesh", "Material", "Boundary Conds", "Basis"]
        funcs = [
            self._check_mesh, self._check_material, self._check_boundary_conds,
            self._check_basis
        ]
        for msg, func in zip(msgs, funcs):
            print(f"{msg} is", " NOT" if not func() else "", " ready.", sep='')
        print("OK." if self.ready() else "Failed.")
        print("*" * 32)

    @property
    def app_name(self):
        return self.__app_name_

    @app_name.setter
    def app_name(self, name):
        self.__app_name_ = name

    @property
    def basis(self):
        return self.__basis_

    def ready(self) -> bool:
        return all(self._state_)

    def u_solution(self):
        if not self.ready():
            print("Engine is NOT ready. Pls check engine first.")
            return False, None
        return True, main_procedure.elasticity(self.mesh, self.material,
                                               self.boundary_conds, self.basis)


class PdSimulation2d(Simulation2d, PdProblem2d):
    def _check_mesh(self):
        self._state_[0] = self.mesh.is_pdready()
        return self._state_[0]

    def _check_material(self):
        self._state_[1] = self.material.is_pdready()
        return self._state_[1]
