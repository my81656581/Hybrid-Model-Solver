import numpy as np
from scipy.spatial.distance import pdist, squareform

from hmsolver.meshgrid.mesh2d import Mesh2d

__all__ = ['PrototypePdMesh2d']


class PrototypePdMesh2d(Mesh2d):
    def __init__(self, n_nodes: int, n_elements: int, e_basistype: int = 2401):
        super().__init__(n_nodes, n_elements, e_basistype)
        self._pdready_ = False

    def peridynamic_construct(self, horizon_radius: float) -> None:
        self._startup(horizon_radius)
        self._build_dist()
        self._build_bonds()
        self._pdready_ = True

    def _startup(self, horizon_radius) -> None:
        """Startup the peridynamic config.
        
        Args:
            horizon_radius (float, optional): Maximum radius of a bond. Defaults to 3e-2.
        Returns:
            None: By runing this function, the containers will be setted for further use.
        """
        self.__k_horizon_radius_ = horizon_radius
        self.bonds = [list() for _ in range(self.n_elements)]

    def _build_dist(self) -> None:
        self.centers = np.mean(self.nodes[self.elements], 1)
        self._dist_ = squareform(pdist(self.centers))

    def _build_bonds(self) -> None:
        ngrid_in_line = int(np.sqrt(self.n_elements))
        assert ngrid_in_line**2 == self.n_elements
        ngrid_in_halfline = ngrid_in_line // 2
        assert ngrid_in_halfline**2 < self.n_elements
        i = self.n_elements // 2
        for j in range(i + 1, self.n_elements):
            dist_ij = self._dist_[i, j]
            if dist_ij >= self.horizon_radius: continue
            self.bonds[i].append(j)
            self.bonds[j].append(i)

    def is_pdready(self) -> bool:
        return self.__pdready_

    @property
    def bonds(self):
        return self.__bonds_

    @property
    def related(self):
        return self.__bonds_

    @bonds.setter
    def bonds(self, val):
        self.__bonds_ = val

    @property
    def centers(self):
        return self.__centers_

    @centers.setter
    def centers(self, val):
        self.__centers_ = val

    @property
    def horizon_radius(self):
        return self.__k_horizon_radius_

    @horizon_radius.setter
    def horizon_radius(self, val):
        self.__k_horizon_radius_ = val

    @property
    def inner_radius(self):
        return self.__k_inner_radius_

    @inner_radius.setter
    def inner_radius(self, val):
        self.__k_inner_radius_ = val

    @property
    def outer_radius(self):
        return self.__k_outer_radius_

    @outer_radius.setter
    def outer_radius(self, val):
        self.__k_outer_radius_ = val

    @property
    def _pdready_(self) -> bool:
        return self.__pdready_

    @_pdready_.setter
    def _pdready_(self, val):
        self.__pdready_ = val
