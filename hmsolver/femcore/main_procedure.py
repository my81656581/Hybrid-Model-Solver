import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import pickle
import time
import os

from functools import reduce

import hmsolver.utils as utils

from hmsolver.femcore.gaussint import gauss_point_quadrature_standard
from hmsolver.femcore.stiffness import preprocessing_all_jacobi
from hmsolver.femcore.stiffness import generate_stiffness_matrix_k0
from hmsolver.femcore.stiffness import mapping_element_stiffness_matrix
from hmsolver.femcore.pd_stiffness import generate_stiffness_matrix_k1
from hmsolver.femcore.pd_stiffness import generate_stiffness_matrix_k1_with_connection
from hmsolver.femcore.pd_stiffness import assemble_stiffness_matrix
from hmsolver.femcore.pd_stiffness import assemble_stiffness_matrix_with_weight
from hmsolver.femcore.pd_stiffness import assemble_stiffness_matrix_with_weight_and_connection
from hmsolver.femcore.pd_stiffness import deal_bond_stretch
from hmsolver.femcore.treat_boundary import CompiledBoundaryConds2d
from hmsolver.femcore.treat_boundary import BoundaryConds2d
from hmsolver.femcore.postprocessing import get_absolute_displace
from hmsolver.femcore.postprocessing import get_strain_field
from hmsolver.femcore.postprocessing import get_stress_field
from hmsolver.femcore.postprocessing import get_distortion_energy_density
from hmsolver.femcore.postprocessing import convert_distortion_energy_for_element
from hmsolver.femcore.postprocessing import maximum_distortion_energy_criterion
from hmsolver.femcore.postprocessing import get_deform_mesh
from hmsolver.femcore.postprocessing import get_local_damage
from hmsolver.femcore.postprocessing import generate_tecplot_config
from hmsolver.femcore.postprocessing import export_tecplot_data

__all__ = [
    'accumulate_stiffness_matrix', 'assemble_load_vector',
    'solve_linear_system', 'elasticity', 'simulate_phase', 'simulate'
]

MAX_DISTORTION_ENERGY = 5.5e7
TOTAL_PHASES = 1
MAX_ITER = 100


def accumulate_stiffness_matrix(*stiff_mats):
    return reduce(lambda a, b: a + b, stiff_mats)


def assemble_load_vector(n_nodes):
    return np.zeros((n_nodes * 2, 1))


def solve_linear_system(a, b):
    t0 = time.time()
    print(f"Solving Linear System: DOF={len(b)}.")
    if sp.isspmatrix(a):
        u = np.reshape(spla.spsolve(a.tocsr(), b), (2, -1)).T
    else:
        u = np.reshape(spla.spsolve(sp.csr_matrix(a), b), (2, -1)).T
    print(
        f"Linear System Solved. Time cost= {utils.formatting_time(time.time() - t0)}"
    )
    return u


def elasticity(mesh2D, material2D, bconds, basis, boundary_scale=1.0):
    n_nodes, n_elements = mesh2D.n_nodes, mesh2D.n_elements
    p, t = mesh2D.nodes, mesh2D.elements
    constitutive = material2D.constitutive
    jacobis = preprocessing_all_jacobi(p, t, basis)
    # just cache for test
    ks0 = generate_stiffness_matrix_k0(p, t, constitutive, basis, jacobis)
    pickle.dump(ks0, open(f"ESM-{n_elements}-elements.bin", "wb"))
    as0 = pickle.load(open(f'ESM-{n_elements}-elements.bin', "rb"))
    a0 = mapping_element_stiffness_matrix(p, t, basis, as0)
    b = assemble_load_vector(n_nodes)
    pickle.dump((a0, b), open(f"ESM-{n_elements}-elements-before.bin", "wb"))
    if isinstance(bconds, BoundaryConds2d):
        a0, b = bconds.compile(p).apply(a0.tolil(), b, p, boundary_scale)
    elif isinstance(bconds, CompiledBoundaryConds2d):
        a0, b = bconds.apply(a0.tolil(), b, p, boundary_scale)
    pickle.dump((a0, b), open(f"ESM-{n_elements}-elements-final.bin", "wb"))
    # a0, b = pickle.load(open(f"ESM-{n_elements}-elements-final.bin", "rb"))
    u = solve_linear_system(a0, b)
    return u


def peridynamic(mesh2D, material2D, bconds, basis, boundary_scale=1.0):
    n_nodes, n_elements = mesh2D.n_nodes, mesh2D.n_elements
    p, t, related = mesh2D.nodes, mesh2D.elements, mesh2D.bonds
    # just cache for test
    k0 = assemble_stiffness_matrix(p, t, related, material2D.generate_coef(),
                                   basis)
    pickle.dump(k0, open(f"PDSM-{n_elements}-elements.bin", "wb"))
    a0 = pickle.load(open(f'PDSM-{n_elements}-elements.bin', "rb"))
    b = assemble_load_vector(n_nodes)
    if isinstance(bconds, BoundaryConds2d):
        a0, b = bconds.compile(p).apply(a0.tolil(), b, p, boundary_scale)
    elif isinstance(bconds, CompiledBoundaryConds2d):
        a0, b = bconds.apply(a0.tolil(), b, p, boundary_scale)
    u = solve_linear_system(a0, b)
    return u


def hybrid(mesh2D, material2D, bconds, basis, boundary_scale=1.0):
    n_nodes, n_elements = mesh2D.n_nodes, mesh2D.n_elements
    p, t, related = mesh2D.nodes, mesh2D.elements, mesh2D.bonds
    weight_handle = mesh2D.query_alpha
    constitutive = material2D.constitutive
    jacobis = preprocessing_all_jacobi(p, t, basis)
    coeff = material2D.generate_coef()
    # just cache for test
    if not os.path.exists(f"ESM-{n_elements}-elements.bin"):
        ks0 = generate_stiffness_matrix_k0(p, t, constitutive, basis, jacobis)
        pickle.dump(ks0, open(f"ESM-{n_elements}-elements.bin", "wb"))
    as0 = pickle.load(open(f'ESM-{n_elements}-elements.bin', "rb"))
    as1 = generate_stiffness_matrix_k1(p, t, related, weight_handle, coeff,
                                       basis, jacobis)
    a0 = mapping_element_stiffness_matrix(p, t, basis, as0)
    a1 = mapping_element_stiffness_matrix(p, t, basis, as1)
    apd = assemble_stiffness_matrix_with_weight(p, t, related, weight_handle,
                                                coeff, basis)
    a0 = accumulate_stiffness_matrix(a0, -a1)
    a = accumulate_stiffness_matrix(a0, apd)
    b = assemble_load_vector(n_nodes)
    if isinstance(bconds, BoundaryConds2d):
        a, b = bconds.compile(p).apply(a.tolil(), b, p, boundary_scale)
    elif isinstance(bconds, CompiledBoundaryConds2d):
        a, b = bconds.apply(a.tolil(), b, p, boundary_scale)
    u = solve_linear_system(a, b)
    return u


def simulate_phase(max_iter: int,
                   phase_id: int,
                   runtime_id: int,
                   mesh2D,
                   material2D,
                   compiled_bconds,
                   basis,
                   connection,
                   boundary_scale=1.0):
    n_nodes, n_elements = mesh2D.n_nodes, mesh2D.n_elements
    p, t, related = mesh2D.nodes, mesh2D.elements, mesh2D.bonds
    weight_handle = mesh2D.query_alpha
    constitutive = material2D.constitutive
    jacobis = preprocessing_all_jacobi(p, t, basis)
    coeff, s_crit = material2D.generate_coef(), material2D.stretch_crit
    # just cache for test
    if not os.path.exists(f"ESM-{n_elements}-elements.bin"):
        ks0 = generate_stiffness_matrix_k0(p, t, constitutive, basis, jacobis)
        pickle.dump(ks0, open(f"ESM-{n_elements}-elements.bin", "wb"))
    as0 = pickle.load(open(f'ESM-{n_elements}-elements.bin', "rb"))
    as1 = generate_stiffness_matrix_k1_with_connection(p, t, related,
                                                       connection,
                                                       weight_handle, coeff,
                                                       basis, jacobis)
    a0 = mapping_element_stiffness_matrix(p, t, basis, as0)
    a1 = mapping_element_stiffness_matrix(p, t, basis, as1)
    apd = assemble_stiffness_matrix_with_weight_and_connection(
        p, t, related, connection, weight_handle, coeff, basis)
    a0 = accumulate_stiffness_matrix(a0, -a1)
    is_first = True
    broken_ep = []  # broken_endpoint
    all_broken_ep = []  # all_broken_endpoint
    for ii in range(max_iter):
        if (not is_first) and len(broken_ep) <= 0:
            break
        t0 = time.time()
        a = accumulate_stiffness_matrix(a0, apd)
        b = assemble_load_vector(n_nodes)
        a, b = compiled_bconds.apply(a.tolil(), b, p, boundary_scale)
        u = solve_linear_system(a, b)
        broken_ep, cnt, apd, connection = deal_bond_stretch(
            p, t, related, weight_handle, connection, u, apd, coeff, basis,
            s_crit)
        all_broken_ep.extend(broken_ep)
        is_first = False
        tot = time.time() - t0
        print(f"    Phase {phase_id:2d}, Runtime {runtime_id:2d}", end="")
        print(f", Iteration {ii:4d}: total broken bonds {cnt:8d}", end="")
        print(f", total time cost {utils.formatting_time(tot)}")
    all_broken_ep = list(set(all_broken_ep))
    return u, all_broken_ep, connection


def simulate(mesh2D, material2D, bconds, basis, app_data, simulate_configs):
    n_dgfe, contains_weight_function, connection = app_data
    app_name, total_phases, max_iter = simulate_configs
    constitutive = material2D.constitutive
    w_, xg, yg = gauss_point_quadrature_standard()
    n_elements = mesh2D.n_elements
    contains_weight_function = [False for _ in range(n_elements)]
    start_timestamp = time.time()
    for phase in range(total_phases):
        t0 = time.time()
        print(f"\nPhase {phase:4d}, with DGFE/tot= ({n_dgfe}/{n_elements})")
        boundary_scale = (phase + 1) / total_phases
        flag, runtime_id = True, 1
        while flag:
            flag = False
            cbonds = bconds.compile(mesh2D.nodes)
            u, all_broken_ep, connection = simulate_phase(
                max_iter, phase, runtime_id, mesh2D, material2D, cbonds, basis,
                connection, boundary_scale)
            for c_index in all_broken_ep:
                if contains_weight_function[c_index]: continue
                flag = True
                mesh2D.manual_set_rule_at_element(c_index)
                contains_weight_function[c_index] = True
            critical = mesh2D.namelist_of_dgfem()
            n_dgfe += mesh2D.convert_mesh_into_DGFEM(todolist=critical)
            print(f"    update mesh, with DGFE/tot=({n_dgfe}/{n_elements})")
            runtime_id += 1
        print(
            f"Phase {phase:4d} end, total time cost {utils.formatting_time(time.time() - t0)}"
        )
        p, t = mesh2D.nodes, mesh2D.elements
        n_nodes, n_elements = mesh2D.n_nodes, mesh2D.n_elements
        # postprocessing
        u_abs = get_absolute_displace(u)
        p_deform = get_deform_mesh(p, u)
        epsilon = get_strain_field(p, t, basis, u)
        sigma = get_stress_field(constitutive, epsilon)
        w_dis = get_distortion_energy_density(sigma, epsilon)
        local_damage = get_local_damage(p, t, basis, mesh2D.bonds, connection)
        # w_rough = mesh2D.get_weight_function_value_roughly()
        w_exact = mesh2D.get_weight_function_value_exactly((w_, xg, yg), basis)
        # without deformation
        plot_cfg = generate_tecplot_config(n_nodes, n_elements, basis.length)
        plot_cfg[
            "variables"] = "X, Y, Ux, Uy, Uabs, alpha, damage, epsilon_x, epsilon_y, epsilon_xy, sigma_x, sigma_y, sigma_xy, w_distortion"
        export_tecplot_data(
            f"{app_name}-simulate-phase-{phase:04d}-{n_elements}-elements.dat",
            plot_cfg, p, t, u, u_abs, w_exact, local_damage, epsilon, sigma,
            w_dis)
        # containing deformation
        deform_cfg = generate_tecplot_config(n_nodes, n_elements, basis.length)
        deform_cfg["variables"] = "X, Y, Ux, Uy, Uabs, alpha, damage"
        export_tecplot_data(
            f"{app_name}-simulate-phase-{phase:04d}-{n_elements}-elements-deform.dat",
            deform_cfg, p_deform, t, u, u_abs, w_exact, local_damage)
    print(
        f"total phase {total_phases}, total time cost {utils.formatting_time(time.time() - start_timestamp)}"
    )
    return u, n_dgfe, connection
