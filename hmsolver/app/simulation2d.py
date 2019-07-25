from hmsolver.meshgrid.hybrid_mesh2d import HybridMesh2d
from hmsolver.material.pd_material2d import PdMaterial2d
from hmsolver.app.problem2d import Problem2d, PdProblem2d

__all__ = ['Simulation2d', 'PdSimulation2d']


class Simulation2d(Problem2d):
    def startup(self, name):
        self.__app_name_ = name
        self._state_ = [False, False, False]

    def _check_mesh(self):
        self._state_[0] = self.mesh.is_ready()
        return self._state_[0]

    def _check_material(self):
        self._state_[1] = self.material.is_ready()
        return self._state_[1]

    def _check_boundary_conds(self):
        self._state_[2] = self.boundary_conds.is_ready()
        return self._state_[2]

    def check_engine(self):
        print("*" * 32 + "\nSimulation Manual Checking:")
        if self._check_mesh():
            # if self.mesh2d.is_pdready():
            print("Mesh Ready.")
        else:
            print("Mesh is NOT Ready.")
        if self._check_material():
            print("Material Ready.")
        else:
            print("Material is NOT Ready.")
        if self._check_boundary_conds():
            print("Boundary Conds Ready.")
        else:
            print("Boundary Conds is NOT Ready.")
        if self.ready():
            print("OK.")
        else:
            print("Failed.")
        print("*" * 32)

    @property
    def app_name(self):
        return self.__app_name_

    def ready(self) -> bool:
        return all(self._state_)


class PdSimulation2d(Simulation2d, PdProblem2d):
    def _check_mesh(self):
        self._state_[0] = self.mesh.is_pdready()
        return self._state_[0]

    def _check_material(self):
        self._state_[1] = self.material.is_pdready()
        return self._state_[1]
