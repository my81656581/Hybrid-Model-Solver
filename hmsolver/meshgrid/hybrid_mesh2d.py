import time
import numpy as np
from scipy.spatial.distance import pdist, squareform
from typing import Callable, List, Tuple

import hmsolver.utils as utils
import hmsolver.geometry as geometry

from hmsolver.meshgrid.prototype_pd_mesh2d import PrototypePdMesh2d

__all__ = ['HybridMesh2d', 'HybridCrackMesh2d']


class HybridMesh2d(PrototypePdMesh2d):
    def peridynamic_construct(self,
                              horizon_radius: float = 3e-2,
                              inner_radius: float = 6e-2,
                              outer_radius: float = 12e-2) -> None:
        self._startup(horizon_radius, inner_radius, outer_radius)
        self._startup_weight_function()
        self._build_dist()
        self._build_bonds()
        self._pdready_ = True

    def _startup(self, horizon_radius, inner_radius, outer_radius) -> None:
        """Startup the peridynamic config.
        
        Args:
            horizon_radius (float, optional): Maximum radius of a bond. Defaults to 3e-2.
            inner_radius (float, optional): Radius of non-local domain. Defaults to 6e-2.
            outer_radius (float, optional): Radius of morphing domain. Defaults to 12e-2.
        
        Returns:
            None: By runing this function, the containers will be setted for further use.
        """
        self.horizon_radius = horizon_radius
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self._pdready_ = False
        self.bonds = [list() for _ in range(self.n_elements)]

    def _startup_weight_function(self) -> None:
        # deal with weight function
        self._inner_circle_ = [set() for _ in range(self.n_elements)]
        self._interface_ring_ = [set() for _ in range(self.n_elements)]
        self._is_critical_ = [False for _ in range(self.n_elements)]
        self._is_dgfem_ = [False for _ in range(self.n_elements)]
        self._ruler_ = [-1 for _ in range(self.n_elements)]
        self._alpha_rules_ = []
        d, r0, r1 = self.horizon_radius, self.horizon_radius, self.outer_radius
        self._default_r0 = r0
        self._default_r1 = r1
        self._default_slope_ = 1.0 / (r0 - r1 + d)
        self._default_yintercept_ = (r1 - d) / (r1 - d - r0)

    def _build_bonds(self) -> None:
        flag, flag_0 = [17.0 / 100 * self.n_elements for _ in range(2)]
        t0 = time.time()
        for i in range(self.n_elements):
            if i > flag:
                flag = utils.display_progress(
                    msg="building bonds",
                    current=flag,
                    display_sep=flag_0,
                    current_id=int(i * (2 - i / self.n_elements)),
                    total=self.n_elements,
                    start_time=t0)
            for j in range(i + 1, self.n_elements):
                dist_ij = self._dist_[i, j]
                if dist_ij >= self.outer_radius: continue
                if dist_ij >= self.inner_radius:
                    self._interface_ring_[i].add(j)
                    self._interface_ring_[j].add(i)
                    continue
                self._inner_circle_[i].add(j)
                self._inner_circle_[j].add(i)
                if dist_ij >= self.horizon_radius: continue
                self.bonds[i].append(j)
                self.bonds[j].append(i)
        tot = time.time() - t0
        print(f"building bond completed. Total {utils.formatting_time(tot)}")

    def namelist_of_dgfem(self):
        return [
            _ for _ in range(self.n_elements)
            if self._is_critical_[_] and (not self._is_dgfem_[_])
        ]

    def convert_mesh_into_DGFEM(self, todolist):
        for e_idx in todolist:
            if self._is_dgfem_[e_idx]: continue
            vertices = self.elements[e_idx, :]
            for vertex in vertices:
                adjoint_elements = self.adjoint[vertex]
                for i, e_manipulating in enumerate(adjoint_elements):
                    if i == 0:  # use former node
                        pass
                    else:  # build a new node
                        v_manipulating = self.elements[e_manipulating, :]
                        new_node = np.array(
                            [self.nodes[vertex, 0], self.nodes[vertex, 1]])
                        new_node_id = self.n_nodes
                        self.n_nodes += 1
                        self.nodes = np.vstack((self.nodes, new_node))
                        v_manipulating[v_manipulating == vertex] = new_node_id
                        self.adjoint.append([e_manipulating])
                self.adjoint[vertex] = [self.adjoint[vertex][0]]
            self.is_dgfem[e_idx] = True
        self.update_frequent()
        return len(todolist)

    def manual_set_rule_at_element(self, element_idx: int, *args) -> bool:
        return False if not self.is_pdready() else self._put_ruler_at_element(
            element_idx, *args)

    def manual_set_rule_at_point(self, x_source: float, y_source: float,
                                 *args) -> bool:
        return False if not self.is_pdready() else self._put_ruler_at_point(
            x_source, y_source, *args)

    def weight_function_builder(self, x_source, y_source,
                                *args) -> Callable[[np.ndarray], float]:
        # #1: generic_weight_function_builder(self, x_source, y_source)
        # #2: generic_weight_function_builder(self, x_source, y_source, radius0, radius1)
        assert len(args) == 0 or len(args) == 2
        if len(args) == 2:
            radius0, radius1 = args
            if radius0 == self._default_r0 and radius1 == self._default_r1:
                slope, y_intercept = self._default_slope_, self._default_yintercept_
            else:
                hoge = radius1 - self.horizon_radius
                piyo = hoge - radius0
                slope, y_intercept = -1.0 / piyo, hoge / piyo
        else:
            radius0, radius1 = self._default_r0, self._default_r1
            slope, y_intercept = self._default_slope_, self._default_yintercept_

        def weight_function(x: np.ndarray, y: np.ndarray) -> float:
            dist = np.sqrt((x - x_source)**2 + (y - y_source)**2)
            ret = np.zeros(shape=dist.shape)
            ret[dist <= radius0] = 1
            ret[(dist > radius0) & (dist < radius1)] = dist[
                (dist > radius0) & (dist < radius1)] * slope + y_intercept
            ret[ret > 1] = 1
            ret[ret < 0] = 0
            return ret

        return weight_function

    def _maintain_ruler(self, ruler_id, rule, bonds, interface_ring) -> bool:
        for element in bonds:
            if self._is_critical_[element]: continue
            self._is_critical_[element] = True
            self._ruler_[element] = ruler_id
        for element in interface_ring:
            if self._is_critical_[element]: continue
            if self._ruler_[element] == -1:
                self._ruler_[element] = ruler_id
                continue
            alpha_old = self._alpha_rules_[self._ruler_[element]](
                *self.centers[element])
            alpha_new = rule(*self.centers[element])
            if alpha_old < alpha_new:
                self._ruler_[element] = ruler_id
        return True

    def _put_ruler_at_point(self, x_source, y_source, *args) -> bool:
        weight_function = self.weight_function_builder(x_source, y_source,
                                                       *args)
        ruler_id = len(self._alpha_rules_)
        self._alpha_rules_.append(weight_function)
        bonds, interface_ring = [], []
        if len(args) == 2:
            radius0, radius1 = args
        else:
            radius0, radius1 = self.horizon_radius, self.outer_radius
        for element in range(self.n_elements):
            dist = pdist(
                np.array([[x_source, y_source], self.centers[element]]))
            if dist >= radius1: continue
            if dist >= radius0:
                interface_ring.append(element)
                continue
            bonds.append(element)
        self._maintain_ruler(ruler_id, weight_function, bonds, interface_ring)
        return True

    def _put_ruler_at_element(self, element_idx: int, *args) -> bool:
        # weight_function = self.weight_function_builder(x_source, y_source, *args)
        weight_function = self.weight_function_builder(
            *self.centers[element_idx], *args)
        ruler_id = len(self._alpha_rules_)
        self._alpha_rules_.append(weight_function)
        self._is_critical_[element_idx] = True
        self._ruler_[element_idx] = ruler_id
        self._maintain_ruler(ruler_id, weight_function,
                             self.bonds[element_idx],
                             (self._interface_ring_[element_idx]
                              | self._inner_circle_[element_idx]))
        return True

    def query_alpha(self, element_idx: int
                    ) -> (int, Callable[[np.ndarray, np.ndarray], float]):
        if self._is_critical_[element_idx]:
            return 1, np.vectorize(lambda x, y: 1)
        if self._ruler_[element_idx] == -1:
            return 0, np.vectorize(lambda x, y: 0)
        return -1, self._alpha_rules_[self._ruler_[element_idx]]

    def get_weight_function_value_roughly(self):
        ret = np.zeros(shape=(self.n_nodes, 1))
        for i in range(self.n_elements):
            weight_function = self.query_alpha(i)[-1]
            for j in self.elements[i, :]:
                ret[j, 0] += weight_function(*self.nodes[j])
        return ret / self.frequent

    def get_weight_function_value_exactly(self, gauss_points, basis_config):
        w_, x_, y_ = gauss_points
        n_gauss, w_ = len(w_), np.reshape(w_, (-1))
        ret = np.zeros(shape=(self.n_nodes, 1))
        xy_local = [
            basis_config.transform(x_, y_, self.nodes[self.elements[i, :], :],
                                   (0, 0)) for i in range(self.n_elements)
        ]
        for i in range(self.n_elements):
            weight_function = self.query_alpha(i)[-1]
            weight = np.sum(w_ * weight_function(*xy_local[i])) / n_gauss
            for j in self.elements[i, :]:
                ret[j, 0] += weight
        return ret / self.frequent

    def debug_element(self, element_idx: int) -> None:
        print("element {}:".format(element_idx))
        print("\tvertices id: [{}]".format(", ".join(
            map(str, self.elements[element_idx, :]))))
        print("\tbond with: {}".format(", ".join(
            map(str, self.bonds[element_idx]))))
        print("\tinner range: {}".format(", ".join(
            map(str, self._inner_circle_[element_idx]))))
        print("\touter rings: {}\n".format(", ".join(
            map(str, self._interface_ring_[element_idx]))))

    def debug_element_weight_function(self, element_idx: int) -> None:
        for element in (self._interface_ring_[element_idx]
                        | self._inner_circle_[element_idx]):
            weight_function = self.query_alpha(element)[-1]
            vertices = self.nodes[self.elements[element]]
            for _1, _2 in zip(vertices, self.elements[element]):
                print(element, _2, weight_function(*_1))

    def debug_all_elements(self) -> None:
        for i, _ in enumerate(self.elements):
            self.debug_element(i)

    @property
    def is_dgfem(self):
        return self._is_dgfem_


class HybridCrackMesh2d(HybridMesh2d):
    def peridynamic_construct(self,
                              horizon_radius: float = 3e-2,
                              inner_radius: float = 6e-2,
                              outer_radius: float = 12e-2) -> None:
        self._startup(horizon_radius, inner_radius, outer_radius)
        self._startup_weight_function()
        self._build_dist()
        self._build_bonds()
        self._cracks = []
        self._pdready_ = True

    def initCrack(self, cracks=None):
        self.cracks = []
        if not (cracks is None):
            self.cracks.extend(cracks)

    def addCracks(self, cracks):
        self.cracks.extend(cracks)

    def _build_bonds(self) -> None:
        self.__bond_horizon_ = [set() for _ in range(self.n_elements)]
        flag, flag_0 = [17.0 / 100 * self.n_elements for _ in range(2)]
        t0 = time.time()
        for i in range(self.n_elements):
            if i > flag:
                flag = utils.display_progress(
                    msg="building bonds",
                    current=flag,
                    display_sep=flag_0,
                    current_id=int(i * (2 - i / self.n_elements)),
                    total=self.n_elements,
                    start_time=t0)
            for j in range(i + 1, self.n_elements):
                dist_ij = self._dist_[i, j]
                if dist_ij >= self.outer_radius: continue
                if dist_ij >= self.inner_radius:
                    self._interface_ring_[i].add(j)
                    self._interface_ring_[j].add(i)
                    continue
                self._inner_circle_[i].add(j)
                self._inner_circle_[j].add(i)
                if dist_ij >= self.horizon_radius: continue
                self.__bond_horizon_[i].add(j)
                self.__bond_horizon_[j].add(i)
                validation = True
                for crack in self.cracks:
                    if geometry.intersect_in(*crack, self.centers[i],
                                             self.centers[j]):
                        validation = False
                        break
                if validation:
                    self.bonds[i].append(j)
                    self.bonds[j].append(i)
        tot = time.time() - t0
        print(f"building bond completed. Total {utils.formatting_time(tot)}")

    def _put_ruler_at_element(self, element_idx: int, *args) -> bool:
        # weight_function = self.weight_function_builder(x_source, y_source, *args)
        weight_function = self.weight_function_builder(
            *self.centers[element_idx], *args)
        ruler_id = len(self._alpha_rules_)
        self._alpha_rules_.append(weight_function)
        self._is_critical_[element_idx] = True
        self._ruler_[element_idx] = ruler_id
        self._maintain_ruler(ruler_id, weight_function,
                             self.__bond_horizon_[element_idx],
                             (self._interface_ring_[element_idx]
                              | self._inner_circle_[element_idx]))
        return True


# def main(input_file):
#     n_nodes, n_elements, nodes, elements = preprocessing.read_mesh(input_file)
#     print(n_nodes, n_elements)
#     meshdata = HybridMesh2d(n_nodes, n_elements)
#     meshdata.manually_construct(np.array(nodes), np.array(elements))
#     meshdata.peridynamic_construct(0.06, 0.12, 0.24)
#     # meshdata.debug_all_elements()
#     # meshdata.debug_element(2000)
#     print(1, meshdata.ready())
#     print(2, meshdata.is_pdready())
#     print(3, meshdata.x.shape, meshdata.y.shape)
#     print(4, meshdata.nodes.shape, meshdata.elements.shape)
#     print(input_file)
#     if input_file == "data_2601_2500.msh":
#         # meshdata.manual_set_rule_at_element(2000)
#         # meshdata.manual_set_rule_at_element(2001)
#         # meshdata.debug_element_weight_function(24)
#         meshdata.manual_set_rule_at_point(0.98, 0)
#         meshdata.debug_element_weight_function(49)

#     if input_file == "data_30_20.msh":
#         for i in range(meshdata.n_elements):
#             print(i, meshdata.elements[i])
#         print("------")
#         for i in range(meshdata.n_nodes):
#             print(i, meshdata.adjoint[i])
#         print("======")
#         meshdata.convert_mesh_into_DGFEM(todolist=[7])
#         print(f'meshdata.n_nodes= {meshdata.n_nodes}')
#         for i in range(meshdata.n_elements):
#             print(i, meshdata.elements[i])
#         print("------")
#         for i in range(meshdata.n_nodes):
#             print(i, meshdata.adjoint[i])
#         print("======")
#         meshdata.convert_mesh_into_DGFEM(todolist=[8, 12, 13])
#         print(f'meshdata.n_nodes= {meshdata.n_nodes}')

# if __name__ == "__main__":
#     if len(sys.argv) < 2:
#         print("Input Meshdata Needed.")
#         exit()
#     main(sys.argv[1])