import time

from hmsolver.meshgrid.hybrid_mesh2d import HybridMesh2d
from hmsolver.material.pd_material2d import PdMaterial2d
from hmsolver.basis.basisinfo import is_suitable_for_mesh
from hmsolver.basis.infrastructures import IShapeable
from hmsolver.app.problem2d import Problem2d, PdProblem2d
from hmsolver.app.failureprone import ProneRing2d

import hmsolver.femcore.main_procedure as main_procedure
import hmsolver.femcore.postprocessing as postprocessing

__all__ = ['Simulation2d', 'PdSimulation2d']

HEADER = {
    "displace_field": "Ux, Uy",
    "absolute_displace": "Uabs",
    "strain_field": "epsilon_x, epsilon_y, epsilon_xy",
    "stress_field": "sigma_x, sigma_y, sigma_xy",
    "distortion_energy": "w_distortion"
}


class Simulation2d(Problem2d):
    def __init__(self, mesh2d, material2d, bconds):
        super().__init__(mesh2d, material2d, bconds)
        self.__basis_ = None
        self._state_ = [False, False, False, False]
        self.__app_name_ = None
        self.__provied_solutions_ = [
            "displace_field",
            "absolute_displace",
            "strain_field",
            "stress_field",
            "distortion_energy",
        ]

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
        if self.ready():
            self._u_, self._eps_, self._sigma_ = None, None, None
            self._u_abs_, self._w_dis_ = None, None

    @property
    def app_name(self):
        return self.__app_name_

    @property
    def provied_solutions(self):
        return self.__provied_solutions_

    @app_name.setter
    def app_name(self, name):
        self.__app_name_ = name

    @property
    def basis(self):
        return self.__basis_

    def ready(self) -> bool:
        return all(self._state_)

    @property
    def displace_field(self):
        return self.u

    @property
    def absolute_u(self):
        return self.u_abs

    @property
    def strain_field(self):
        return self.epsilon

    @property
    def stress_field(self):
        return self.sigma

    @property
    def distortion_energy(self):
        return self.w_dis

    def _selfcheck(self):
        if not self.ready():
            print("Engine is NOT ready. Pls check engine first.")
            return False
        return True

    @property
    def u(self):
        if not self._selfcheck():
            return None
        if self._u_ is None:
            self._u_ = main_procedure.elasticity(self.mesh, self.material,
                                                 self.boundary_conds,
                                                 self.basis)
        return self._u_

    @property
    def u_abs(self):
        if not self._selfcheck():
            return None
        if self._u_abs_ is None:
            self._u_abs_ = postprocessing.get_absolute_displace(self.u)
        return self._u_abs_

    @property
    def epsilon(self):
        if not self._selfcheck():
            return None
        if self._eps_ is None:
            self._eps_ = postprocessing.get_strain_field(
                self.mesh.nodes, self.mesh.elements, self.basis, self.u)
        return self._eps_

    @property
    def sigma(self):
        if not self._selfcheck():
            return None
        if self._sigma_ is None:
            self._sigma_ = postprocessing.get_stress_field(
                self.material.constitutive, self.epsilon)
        return self._sigma_

    @property
    def w_dis(self):
        if not self._selfcheck():
            return None
        if self._w_dis_ is None:
            self._w_dis_ = postprocessing.get_distortion_energy(
                self.material.youngs_modulus, self.material.poissons_ratio,
                self.sigma)
        return self._w_dis_

    def export_to_tecplot(self, export_filename, *orders):
        cfg = postprocessing.generate_tecplot_config(self.mesh.n_nodes,
                                                     self.mesh.n_elements,
                                                     self.mesh.n_localnodes,
                                                     self.mesh.e_zonetype)
        header = ["X, Y"]
        header.extend([HEADER[_] for _ in orders])
        cfg["variables"] = ", ".join(header)
        __DATA_MAPPING_ = {
            "displace_field": self.u,
            "absolute_displace": self.u_abs,
            "strain_field": self.epsilon,
            "stress_field": self.sigma,
            "distortion_energy": self.w_dis
        }
        data = [__DATA_MAPPING_[_] for _ in orders]
        postprocessing.export_tecplot_data(
            f"{self.app_name}-{export_filename}.dat", cfg, self.mesh.nodes,
            self.mesh.elements, *data)


class PdSimulation2d(Simulation2d, PdProblem2d):
    def _check_mesh(self):
        self._state_[0] = self.mesh.is_pdready()
        return self._state_[0]

    def _check_material(self):
        self._state_[1] = self.material.is_pdready()
        return self._state_[1]

    @property
    def u(self):
        if not self._selfcheck():
            return None
        if self._u_ is None:
            self._u_ = main_procedure.peridynamic(self.mesh, self.material,
                                                  self.boundary_conds,
                                                  self.basis)
        return self._u_


class CrackSimulation2d(PdSimulation2d):
    def __init__(self, mesh2d, material2d, bconds):
        super().__init__(mesh2d, material2d, bconds)
        self._contains_weight_function_ = []
        self._n_dgfe_ = 0

    def _selfcheck(self):
        if not self.ready():
            print("Engine is NOT ready. Pls check engine first.")
            return False
        if not self.material.can_break():
            print("Crack Simulation SHOULD allow the bond break.")
            print("Pls set maximun bond stretch first.")
            print("eg: app.material.stretch_crit = 1.1")
            return False
        if not len(self._contains_weight_function_) == self.mesh.n_elements:
            self._contains_weight_function_ = [
                False for _ in range(self.mesh.n_elements)
            ]
        return True

    @property
    def u(self):
        if not self._selfcheck():
            return None
        if self._u_ is None:
            self._u_ = main_procedure.elasticity(self.mesh, self.material,
                                                 self.boundary_conds,
                                                 self.basis)
        return self._u_

    def detect_failure(self, max_distortion_energy):
        if not self._selfcheck():
            return None
        critical_indices = postprocessing.maximum_distortion_energy_criterion(
            self.w_dis, max_distortion_energy)
        element_indices = []
        for c_index in critical_indices:
            element_indices.extend(self.mesh.adjoint[c_index])
        for e_index in set(element_indices):
            self.mesh.manual_set_rule_at_element(e_index)
            self._contains_weight_function_[e_index] = True
        return True

    def manual_set_failureprone_zone(self, zone: ProneRing2d):
        if not self._selfcheck():
            return None
        r0, r1 = zone.inner_radius, zone.outer_radius
        self.mesh.manual_set_rule_at_point(zone.x, zone.y, r0, r1)

    def update_mesh(self):
        if not self._selfcheck():
            return None
        critical = self.mesh.namelist_of_dgfem()
        self._n_dgfe_ += self.mesh.convert_mesh_into_DGFEM(todolist=critical)

    def run_simulation(self, total_phases=1, max_iter=100):
        if not self._selfcheck():
            return None
        self._u_, self._n_dgfe_ = main_procedure.simulate(
            self.mesh, self.material, self.boundary_conds, self.basis,
            (self.app_name, self._n_dgfe_, self._contains_weight_function_),
            (total_phases, max_iter))
