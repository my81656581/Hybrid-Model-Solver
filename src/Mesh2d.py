import numpy as np
from scipy.spatial.distance import pdist, squareform
import sys
import time
from typing import Callable, List, Tuple

import utils

import BasisConfig
import preprocessing


class Zone2d(object):
    def __init__(self, zone_xl, zone_xr, zone_yl, zone_yr):
        self.zone_xl = zone_xl
        self.zone_xr = zone_xr
        self.zone_yl = zone_yl
        self.zone_yr = zone_yr


class Mesh2d(object):
    """Storage the required data container of 2d mesh.
    
    Args:
        n_nodes (int): Number of nodes
        n_elements (int): Number of finite elements
        e_basistype (int, optional): Basis type, ref: BasisConfig. Defaults to 2401.
    
    Returns:
        Mesh2d: Package of nodes' coordinates and elements' vertices.
    """

    def __init__(self, n_nodes: int, n_elements: int, e_basistype: int = 2401):
        self.__n_nodes_ = n_nodes
        self.__n_elements_ = n_elements
        self.__e_basistype_ = e_basistype
        self.__n_localnodes_ = BasisConfig.BasisConfig(e_basistype)
        self.__nodes_ = np.zeros(shape=(n_nodes, 2))
        self.__adjoint_ = [[] for _ in range(n_nodes)]
        self.__x_, self.__y_ = [self.__nodes_[:, _] for _ in range(2)]
        self.__elements_ = np.zeros(shape=(n_elements, 4), dtype=np.int32)
        self.__is_ready_ = False

    def manually_construct(self, nodes: np.ndarray,
                           elements: np.ndarray) -> bool:
        if not self.__is_ready_:
            self.__nodes_ = np.array(nodes)
            self.__elements_ = np.array(elements)
            self.__consturct_adjoint()
            self.__is_ready_ = True
            return True
        else:
            return False

    def __consturct_adjoint(self):
        for i, element in enumerate(self.__elements_):
            for j in element:
                self.__adjoint_[j].append(i)
        self.__frequent_ = np.reshape(np.array([len(_) for _ in self.__adjoint_]), (self.n_nodes, 1))

    def update_frequent(self):
        self.__frequent_ = np.reshape(np.array([len(_) for _ in self.__adjoint_]), (self.n_nodes, 1))

    def ready(self):
        return self.__is_ready_

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
    


class PrototypePdMesh2d(Mesh2d):
    def prototype_construct(self, horizon_radius: float) -> None:
        self.__startup(horizon_radius)
        self.__build_dist()
        self.__build_bonds()
        self.is_pd_ready = True


    def __startup(self, horizon_radius) -> None:
        """Startup the peridynamic config.
        
        Args:
            horizon_radius (float, optional): Maximum radius of a bond. Defaults to 3e-2.
        Returns:
            None: By runing this function, the containers will be setted for further use.
        """
        self.__k_horizon_radius_ = horizon_radius
        self.is_pd_ready = False
        self.bonds = [list() for _ in range(self.n_elements)]


    def __build_dist(self) -> None:
        self.centers = np.mean(self.nodes[self.elements], 1)
        self.__dist_ = squareform(pdist(self.centers))

    
    def __build_bonds(self) -> None:
        ngrid_in_line = int(np.sqrt(self.n_elements))
        assert ngrid_in_line ** 2 == self.n_elements
        ngrid_in_halfline = ngrid_in_line // 2
        assert ngrid_in_halfline ** 2 < self.n_elements
        i = self.n_elements // 2
        for j in range(i + 1, self.n_elements):
            dist_ij = self.__dist_[i, j]
            if dist_ij >= self.horizon_radius: continue
            self.bonds[i].append(j)
            self.bonds[j].append(i)

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

    def pd_ready(self) -> bool:
        return self.__is_pd_ready_

    @property
    def is_pd_ready(self) -> bool:
        return self.__is_pd_ready_

    @is_pd_ready.setter
    def is_pd_ready(self, val):
        self.__is_pd_ready_ = val


class HybridMesh2d(PrototypePdMesh2d):
    def peridynamic_construct(self,
                              horizon_radius: float = 3e-2,
                              inner_radius: float = 6e-2,
                              outer_radius: float = 12e-2) -> None:
        self.__startup(horizon_radius, inner_radius, outer_radius)
        self.__startup_weight_function()
        self.__build_dist()
        self.__build_bonds()
        self.is_pd_ready = True

    
    def __startup(self, horizon_radius, inner_radius, outer_radius) -> None:
        """Startup the peridynamic config.
        
        Args:
            horizon_radius (float, optional): Maximum radius of a bond. Defaults to 3e-2.
            inner_radius (float, optional): Radius of non-local domain. Defaults to 6e-2.
            outer_radius (float, optional): Radius of morphing domain. Defaults to 12e-2.
        
        Returns:
            None: By runing this function, the containers will be setted for further use.
        """
        self.k_horizon_radius_ = horizon_radius
        self.k_inner_radius_ = inner_radius
        self.k_outer_radius_ = outer_radius
        self.is_pd_ready = False
        self.bonds = [list() for _ in range(self.n_elements)]
    

    def __startup_weight_function(self) -> None:
        # deal with weight function
        self.__inner_circle_ = [set() for _ in range(self.n_elements)]
        self.__interface_ring_ = [set() for _ in range(self.n_elements)]
        self.__is_critical_ = [False for _ in range(self.n_elements)]
        self.__is_dgfem_ = [False for _ in range(self.n_elements)]
        self.__ruler_ = [-1 for _ in range(self.n_elements)]
        self.__alpha_rules_ = []
        d, r0, r1 = self.k_horizon_radius_, self.k_horizon_radius_, self.k_outer_radius_
        self.__default_slope_ = 1.0 / (r0 - r1 + d)
        self.__default_yintercept_ = (r1 - d) / (r1 - d - r0)


    @property
    def is_dgfem(self):
        return self.__is_dgfem_


    def namelist_of_dgfem(self):
        return [_ for _ in range(self.n_elements) if self.__is_critical_[_] and (not self.__is_dgfem_[_])]

    
    def convert_mesh_into_DGFEM(self, todolist):
        for e_idx in todolist:
            vertices = self.elements[e_idx, :]
            for vertex in vertices:
                adjoint_elements = self.adjoint[vertex]
                for i, e_manipulating in enumerate(adjoint_elements):
                    if i == 0: # use former node
                        pass
                    else: # build a new node
                        v_manipulating = self.elements[e_manipulating, :]
                        new_node = np.array([self.nodes[vertex, 0], self.nodes[vertex, 1]])
                        new_node_id = self.n_nodes
                        self.n_nodes += 1
                        self.nodes = np.vstack((self.nodes, new_node))
                        v_manipulating[v_manipulating == vertex] = new_node_id
                        self.adjoint.append([new_node_id])
                self.adjoint[vertex] = [self.adjoint[vertex][0]]
            self.is_dgfem[e_idx] = True
        self.update_frequent()
        return len(todolist)


    def manual_set_rule_at_element(self, element_idx: int, *args) -> bool:
        return False if not self.pd_ready() else self.__put_ruler_at_element(element_idx, *args)


    def manual_set_rule_at_point(self, x_source: float, y_source: float, *args) -> bool:
        return False if not self.pd_ready() else self.__put_ruler_at_point(x_source, y_source, *args)

    
    def weight_function_builder(self, x_source, y_source, *args) -> Callable[[np.ndarray], float]:
        # #1: generic_weight_function_builder(self, x_source, y_source)
        # #2: generic_weight_function_builder(self, x_source, y_source, radius0, radius1)
        assert len(args) == 0 or len(args) == 2
        if len(args) == 2:
            radius0, radius1 = args
            if radius0 == self.k_horizon_radius_ and radius1 == self.k_outer_radius_:
                slope, y_intercept = self.__default_slope_, self.__default_yintercept_
            else:
                hoge = radius1 - self.k_horizon_radius_
                piyo = hoge - radius0
                slope, y_intercept = -1.0 / piyo, hoge / piyo
        else:
            radius0, radius1 = self.k_horizon_radius_, self.k_outer_radius_
            slope, y_intercept = self.__default_slope_, self.__default_yintercept_
        def weight_function(x: np.ndarray, y: np.ndarray) -> float:
            dist = np.sqrt((x - x_source) ** 2 + (y - y_source) ** 2)
            ret = np.zeros(shape=dist.shape)
            ret[dist <= radius0] = 1
            ret[(dist > radius0) & (dist < radius1)] = dist[(dist > radius0) & (dist < radius1)] * slope + y_intercept
            ret[ret > 1] = 1
            ret[ret < 0] = 0
            return ret
        return weight_function

    
    def __maintain_ruler(self, ruler_id, rule, bonds, interface_ring) -> bool:
        for element in bonds:
            if self.__is_critical_[element]: continue
            self.__is_critical_[element] = True
            self.__ruler_[element] = ruler_id
        for element in interface_ring:
            if self.__is_critical_[element]: continue
            if self.__ruler_[element] == -1:
                self.__ruler_[element] = ruler_id
                continue
            alpha_old = self.__alpha_rules_[self.__ruler_[element]](*self.centers[element])
            alpha_new = rule(*self.centers[element])
            if alpha_old < alpha_new:
                self.__ruler_[element] = ruler_id
        return True

    
    def __put_ruler_at_point(self, x_source, y_source, *args) -> bool:
        weight_function = self.weight_function_builder(x_source, y_source, *args)
        ruler_id = len(self.__alpha_rules_)
        self.__alpha_rules_.append(weight_function)
        bonds, interface_ring = [], []
        if len(args) == 2:
            radius0, radius1 = args
        else:
            radius0, radius1 = self.k_horizon_radius_, self.k_outer_radius_
        for element in range(self.n_elements):
            dist = pdist(np.array([[x_source, y_source], self.centers[element]]))
            if dist >= radius1: continue
            if dist >= radius0:
                interface_ring.append(element)
                continue
            bonds.append(element)
        self.__maintain_ruler(ruler_id, weight_function, bonds, interface_ring)
        return True

    
    def __put_ruler_at_element(self, element_idx: int, *args) -> bool:
        # weight_function = self.weight_function_builder(x_source, y_source, *args)
        weight_function = self.weight_function_builder(*self.centers[element_idx], *args)
        ruler_id = len(self.__alpha_rules_)
        self.__alpha_rules_.append(weight_function)
        self.__is_critical_[element_idx] = True
        self.__ruler_[element_idx] = ruler_id
        # todo
        self.__maintain_ruler(ruler_id, weight_function,
                              self.bonds[element_idx],
                              (self.__interface_ring_[element_idx] | self.__inner_circle_[element_idx]))
                              # (self.__inner_circle_[element_idx]))
                              # (self.__interface_ring_[element_idx]))
        return True

    
    def query_alpha(self, element_idx: int) -> (int, Callable[[np.ndarray, np.ndarray], float]):
        if self.__is_critical_[element_idx]:
            return 1, np.vectorize(lambda x, y: 1)
        if self.__ruler_[element_idx] == -1:
            return 0, np.vectorize(lambda x, y: 0)
        return -1, self.__alpha_rules_[self.__ruler_[element_idx]]


    def get_weight_function_value_roughly(self):
        ret = np.zeros(shape=(self.n_nodes, 1))
        for i in range(self.n_elements):
            flag, weight_function = self.query_alpha(i)
            for j in self.elements[i, :]:
                ret[j, 0] += weight_function(*self.nodes[j])
        return ret / self.frequent


    def get_weight_function_value_exactly(self, gauss_points, basis_config):
        w_, x_, y_ = gauss_points
        n_gauss, w_ = len(w_), np.reshape(w_, (-1))
        ret = np.zeros(shape=(self.n_nodes, 1))
        xy_local = [basis_config.transform(x_, y_, self.nodes[self.elements[i, :], :], (0, 0)) for i in range(self.n_elements)]
        for i in range(self.n_elements):
            flag, weight_function = self.query_alpha(i)
            weight = np.sum(w_ * weight_function(*xy_local[i])) / n_gauss
            for j in self.elements[i, :]:
                ret[j, 0] += weight
        return ret / self.frequent

    
    def __build_dist(self) -> None:
        self.centers = np.mean(self.nodes[self.elements], 1)
        self.__dist_ = squareform(pdist(self.centers))

    
    def __build_bonds(self) -> None:
        flag, flag_0 = [17.0 / 100 * self.n_elements for _ in range(2)]
        t0 = time.time()
        for i in range(self.n_elements):
            if i > flag:
                flag = utils.display_progress(msg="building bonds",
                                              current=flag,
                                              display_sep=flag_0,
                                              current_id=int(i * (2 - i / self.n_elements)),
                                              total=self.n_elements,
                                              start_time=t0)
            for j in range(i + 1, self.n_elements):
                dist_ij = self.__dist_[i, j]
                if dist_ij >= self.k_outer_radius_: continue
                if dist_ij >= self.k_inner_radius_:
                    self.__interface_ring_[i].add(j)
                    self.__interface_ring_[j].add(i)
                    continue
                self.__inner_circle_[i].add(j)
                self.__inner_circle_[j].add(i)
                if dist_ij >= self.k_horizon_radius_: continue
                self.bonds[i].append(j)
                self.bonds[j].append(i)
        tot = time.time() - t0
        print(f"building bond completed. Total {utils.formatting_time(tot)}")

    
    def debug_element(self, element_idx: int) -> None:
        print("element {}:".format(element_idx))
        print("\tvertices id: [{}]".format(", ".join(
            map(str, self.elements[element_idx, :]))))
        print("\tbond with: {}".format(", ".join(
            map(str, self.bonds[element_idx]))))
        print("\tinner range: {}".format(", ".join(
            map(str, self.__inner_circle_[element_idx]))))
        print("\touter rings: {}\n".format(", ".join(
            map(str, self.__interface_ring_[element_idx]))))

    
    def debug_element_weight_function(self, element_idx: int) -> None:
        for element in (self.__interface_ring_[element_idx]
                        | self.__inner_circle_[element_idx]):
            flag, weight_function = self.query_alpha(element)
            vertices = self.nodes[self.elements[element]]
            for _1, _2 in zip(vertices, self.elements[element]):
                print(element, _2, weight_function(*_1))

    
    def debug_all_elements(self) -> None:
        for i, _ in enumerate(self.elements):
            self.debug_element(i)

    
    



def main(input_file):
    n_nodes, n_elements, nodes, elements = preprocessing.read_mesh(input_file)
    print(n_nodes, n_elements)
    meshdata = HybridMesh2d(n_nodes, n_elements)
    meshdata.manually_construct(np.array(nodes), np.array(elements))
    meshdata.peridynamic_construct(0.06, 0.12, 0.24)
    # meshdata.debug_all_elements()
    # meshdata.debug_element(2000)
    print(1, meshdata.ready())
    print(2, meshdata.pd_ready())
    print(3, meshdata.x.shape, meshdata.y.shape)
    print(4, meshdata.nodes.shape, meshdata.elements.shape)
    print(input_file)
    if input_file == "data_2601_2500.msh":
        # meshdata.manual_set_rule_at_element(2000)
        # meshdata.manual_set_rule_at_element(2001)
        # meshdata.debug_element_weight_function(24)
        meshdata.manual_set_rule_at_point(0.98, 0)
        meshdata.debug_element_weight_function(49)

    if input_file == "data_30_20.msh":
        for i in range(n_elements):
            print(i, meshdata.elements[i])
        print("------")
        for i in range(n_nodes):
            print(i, meshdata.adjoint[i])
        print("======")
        p, t = meshdata.nodes, meshdata.elements
        adj, flags = meshdata.adjoint, meshdata.is_dgfem
        meshdata.convert_mesh_into_DGFEM(todolist=[7])
        print(f'meshdata.n_nodes= {meshdata.n_nodes}')
        for i in range(n_elements):
            print(i, meshdata.elements[i])
        print("------")
        for i in range(n_nodes):
            print(i, meshdata.adjoint[i])
        print("======")
        meshdata.convert_mesh_into_DGFEM(todolist=[8, 12, 13])
        print(f'meshdata.n_nodes= {meshdata.n_nodes}')



if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Input Meshdata Needed.")
        exit()
    main(sys.argv[1])
