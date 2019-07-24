import sys
sys.path.append('./../../')

__all__ = ['PdSimulation2d']

from hmsolver.meshgrid.hybrid_mesh2d import HybridMesh2d
from hmsolver.material.pd_material2d import PdMaterial2d
from hmsolver.meshgrid.hybrid_mesh2d import HybridMesh2d


class PdSimulation2d(object):
    def __init__(self, mesh2d, material2d, bconds):
        self.__mesh2d_ = mesh2d
        self.__material2d_ = material2d
        self.__boundary_conds_ = bconds
        self.__state_ = [False, False, False]

    def check_engine(self):
        print("*" * 32 + "\nSimulation Manual Checking:")
        if self.mesh2d != None:
            # if self.mesh2d.pd_ready():
            self.__state_[0] = True
            print("Mesh Ready.")
        else:
            print("Mesh is NOT Ready.")
        if self.material2d.ready():
            self.__state_[1] = True
            print("Material Ready.")
        else:
            print("Material is NOT Ready.")
        if self.boundary_conds != None:
            self.__state_[2] = True
            print("Boundary Conds Ready.")
        else:
            print("Boundary Conds is NOT Ready.")
        if self.ready():
            print("OK.")
        else:
            print("Failed.")
        print("*" * 32)

    @property
    def mesh2d(self):
        return self.__mesh2d_

    @property
    def material2d(self):
        return self.__material2d_

    @property
    def boundary_conds(self):
        return self.__boundary_conds_

    def ready(self) -> bool:
        return all(self.__state_)


if __name__ == '__main__':
    mesh2d = None
    material2d = PdMaterial2d(3e11, 1.0 / 3)
    a = PdSimulation2d(mesh2d, material2d, None)
    a.check_engine()
    a.material2d.setIsotropic(0.06, 0.02, 0.015)
    a.check_engine()