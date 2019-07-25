import time
import numpy as np
from scipy.spatial.distance import pdist, squareform
# from typing import Callable, List, Tuple

from hmsolver.basis.basisinfo import get_localnodes_by_id

__all__ = ['Mesh2d']


class Mesh2d(object):
    """Storage the required data container of 2d mesh.
    
    Args:
        n_nodes (int): Number of nodes
        n_elements (int): Number of finite elements
        e_basistype (int, optional): Basis typeid, ref: BasisInfo. Defaults to 2401.
    
    Returns:
        Mesh2d: Package of nodes' coordinates and elements' vertices.
    """

    def __init__(self, n_nodes: int, n_elements: int, e_basistype: int = 2401):
        self.__n_nodes_ = n_nodes
        self.__n_elements_ = n_elements
        self.__e_basistype_ = e_basistype
        self.__n_localnodes_ = get_localnodes_by_id(e_basistype)
        self.__nodes_ = np.zeros(shape=(n_nodes, 2))
        self.__adjoint_ = [[] for _ in range(n_nodes)]
        self.__x_, self.__y_ = [self.__nodes_[:, _] for _ in range(2)]
        self.__elements_ = np.zeros(shape=(n_elements, 4), dtype=np.int32)
        self.__ready_ = False

    def manually_construct(self, nodes: np.ndarray,
                           elements: np.ndarray) -> bool:
        if not self.is_ready():
            self.__nodes_ = np.array(nodes)
            self.__elements_ = np.array(elements)
            self.__consturct_adjoint()
            self.__ready_ = True
            return True
        else:
            return False

    def __consturct_adjoint(self):
        for i, element in enumerate(self.__elements_):
            for j in element:
                self.__adjoint_[j].append(i)
        self.__frequent_ = np.reshape(
            np.array([len(_) for _ in self.__adjoint_]), (self.n_nodes, 1))

    def update_frequent(self):
        self.__frequent_ = np.reshape(
            np.array([len(_) for _ in self.__adjoint_]), (self.n_nodes, 1))

    def is_ready(self):
        return self.__ready_

    @property
    def n_nodes(self):
        return self.__n_nodes_

    @n_nodes.setter
    def n_nodes(self, val):
        self.__n_nodes_ = val

    @property
    def n_elements(self):
        return self.__n_elements_

    @property
    def n_localnodes(self):
        return self.__n_localnodes_

    @property
    def e_basistype(self):
        return self.__e_basistype_

    @property
    def x(self):
        return self.__x_

    @property
    def y(self):
        return self.__y_

    @property
    def nodes(self):
        return self.__nodes_

    @nodes.setter
    def nodes(self, val):
        self.__nodes_ = val

    @property
    def elements(self):
        return self.__elements_

    @property
    def adjoint(self):
        return self.__adjoint_

    @property
    def frequent(self):
        return self.__frequent_
