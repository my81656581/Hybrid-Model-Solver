import numpy as np
import pickle
import time
import os

import utils

import Material2d
import Mesh2d
import preprocessing
import reference_basis
import gaussint
import stiffness
import pd_stiffness
import treat_boundary
import postprocessing

MAX_DISTORTION_ENERGY = 5.5e7
TOTAL_PHASES = 1
MAX_ITER = 100
EXAMPLE_NAME = "example-05"


def assemble_stiffness_matrix(k0, kpd):
    return k0 + kpd


def assemble_load_vector(n_nodes):
    return np.zeros((n_nodes * 2, 1))


def solve_linear_system(a, b):
    u_ = np.linalg.solve(a, b)
    u = np.reshape(u_, (2, -1)).T
    return u


def elasticity(mesh2D, material2D, boundarys, boundary_scale=1.0):
    n_nodes, n_elements = mesh2D.n_nodes, mesh2D.n_elements
    p, t = mesh2D.nodes, mesh2D.elements
    print(f"n_nodes= {n_nodes}, n_elements= {n_elements}")
    print(f"len(p)= {len(p)}, len(t)= {len(t)}")

    lame_mu = material2D.lame_mu
    lame_lambda = material2D.lame_lambda
    constitutive = material2D.constitutive
    print(f"lame parameters: lambda={lame_lambda:.4e}, mu={lame_mu:.4e}")

    basis = reference_basis.Quadrilateral4Node()
    jacobis = stiffness.preprocessing_all_jacobi(p, t, basis)

    # just cache for test
    ks0 = stiffness.generate_stiffness_matrix_k0(p, t, constitutive, basis,
                                                 jacobis)
    pickle.dump(ks0, open(f"stiff-e-k0s-{n_elements}-elements.bin", "wb"))
    as0 = pickle.load(open(f'stiff-e-k0s-{n_elements}-elements.bin', "rb"))
    # if not os.path.exists(f"stiff-e-k0-{n_elements}-elements.bin"):
    #     k0 = stiffness.mapping_element_stiffness_matrix(p, t, basis, as0)
    #     pickle.dump(k0, open(f"stiff-e-k0-{n_elements}-elements.bin", "wb"))
    # a0 = pickle.load(open(f"stiff-e-k0-{n_elements}-elements.bin", "rb"))

    # or don't cache stiffness matrix
    # as0 = stiffness.generate_stiffness_matrix_k0(p, t, constitutive, basis, jacobis)
    a0 = stiffness.mapping_element_stiffness_matrix(p, t, basis, as0)
    b = assemble_load_vector(n_nodes)
    compiled_boundary = treat_boundary.complie_boundary(p, boundarys)
    treat_boundary.apply_boundary(a0, b, p, compiled_boundary, boundary_scale)
    u = solve_linear_system(a0, b)
    return u


def morphing_phase(phase_id: int,
                   runtime_id: int,
                   mesh2D,
                   material2D,
                   boundarys,
                   boundary_scale=1.0,
                   s_crit=1.1,
                   l_inst=0.015):
    n_nodes, n_elements = mesh2D.n_nodes, mesh2D.n_elements
    p, t = mesh2D.nodes, mesh2D.elements
    related = mesh2D.bonds
    weight_function_handle = mesh2D.query_alpha

    constitutive = material2D.constitutive

    basis = reference_basis.Quadrilateral4Node()
    n_gauss = len(gaussint.gauss_point_quadrature_standard()[0])
    jacobis = stiffness.preprocessing_all_jacobi(p, t, basis)
    coeff = material2D.generate_coef()

    # just cache for test
    if not os.path.exists(f"stiff-e-k0s-{n_elements}-elements.bin"):
        ks0 = stiffness.generate_stiffness_matrix_k0(p, t, constitutive, basis,
                                                     jacobis)
        pickle.dump(ks0, open(f"stiff-e-k0s-{n_elements}-elements.bin", "wb"))
    as0 = pickle.load(open(f'stiff-e-k0s-{n_elements}-elements.bin', "rb"))

    # or don't cache stiffness matrix
    # as0 = stiffness.generate_stiffness_matrix_k0(p, t, constitutive, basis, jacobis)
    # as1 = pd_stiffness.generate_stiffness_matrix_k1(p, t, related, weight_function_handle, coef_c0, basis, jacobis)
    # a0 = stiffness.mapping_element_stiffness_matrix_with_weight(p, t, centers, weight_function_handle, basis, as0)
    # a1 = stiffness.mapping_element_stiffness_matrix(p, t, basis, as1)
    # apd = pd_stiffness.assemble_stiffness_matrix_with_weight(p, t, related, weight_function_handle, coef_c0, basis)

    as1 = pd_stiffness.generate_stiffness_matrix_k1(p, t, related,
                                                    weight_function_handle,
                                                    coeff, basis, jacobis)

    a0 = stiffness.mapping_element_stiffness_matrix(p, t, basis, as0)
    a1 = stiffness.mapping_element_stiffness_matrix(p, t, basis, as1)
    apd = pd_stiffness.assemble_stiffness_matrix_with_weight(
        p, t, related, weight_function_handle, coeff, basis)

    a0 = assemble_stiffness_matrix(a0, -a1)

    is_first = True
    connection = np.ones(shape=(n_elements, n_elements, n_gauss, n_gauss),
                         dtype=np.bool)
    broken_endpoint = []
    all_broken_endpoint = []
    for ii in range(MAX_ITER):
        if (not is_first) and len(broken_endpoint) <= 0:
            break
        t0 = time.time()
        a = assemble_stiffness_matrix(a0, apd)
        b = assemble_load_vector(n_nodes)
        compiled_boundary = treat_boundary.complie_boundary(p, boundarys)
        treat_boundary.apply_boundary(a, b, p, compiled_boundary,
                                      boundary_scale)
        # print(a[0:5, 0:5])
        u = solve_linear_system(a, b)
        broken_endpoint, cnt = pd_stiffness.deal_bond_stretch(
            p, t, related, weight_function_handle, connection, u, apd, coeff,
            basis, s_crit)
        all_broken_endpoint.extend(broken_endpoint)
        is_first = False
        tot = time.time() - t0
        print(
            f"    Phase {phase_id:2d}, Runtime {runtime_id:2d}, Iteration {ii:4d}: ",
            end="")
        print(
            f"total broken bonds {cnt:8d}, total time cost {utils.formatting_time(tot)}"
        )
        # input()
    all_broken_endpoint = list(set(all_broken_endpoint))
    return u, all_broken_endpoint


def morphing_setup(mesh2D, material2D, boundarys):
    youngs_modulus = material2D.youngs_modulus
    poissons_ratio = material2D.poissons_ratio
    constitutive = material2D.constitutive
    p, t = mesh2D.nodes, mesh2D.elements
    basis = reference_basis.Quadrilateral4Node()
    contains_weight_function = [False for _ in range(len(t))]
    u_elasticity = elasticity(mesh2D, material2D, boundarys)
    epsilon = postprocessing.get_strain_field(p, t, basis, u_elasticity)
    sigma = postprocessing.get_stress_field(constitutive, epsilon)
    w_distortion = postprocessing.get_distortion_energy(
        youngs_modulus, poissons_ratio, sigma)
    w_element = postprocessing.convert_distortion_energy_for_element(
        w_distortion, t)
    critical_indices = postprocessing.maximum_distortion_energy_criterion(
        w_element, MAX_DISTORTION_ENERGY)
    for c_index in critical_indices:
        print(c_index)
        mesh2D.manual_set_rule_at_element(c_index)
        contains_weight_function[c_index] = True
    critical = mesh2D.namelist_of_dgfem()
    mesh2D.convert_mesh_into_DGFEM(todolist=critical)


def morphing(mesh2D, material2D, boundarys):
    youngs_modulus = material2D.youngs_modulus
    poissons_ratio = material2D.poissons_ratio
    constitutive = material2D.constitutive
    basis = reference_basis.Quadrilateral4Node()
    w_, x_gauss, y_gauss = gaussint.gauss_point_quadrature_standard()
    contains_weight_function = [False for _ in range(len(t))]
    start_timestamp = time.time()
    for phase in range(TOTAL_PHASES):
        t0 = time.time()
        print(
            f"\nPhase {phase:4d}, with DGFE/tot= ({n_dg_elements}/{n_elements})"
        )
        boundary_scale = (phase + 1) / TOTAL_PHASES
        flag, runtime_id = True, 1
        while flag:
            flag = False
            u, all_broken_endpoint = morphing_phase(phase, runtime_id, mesh2D,
                                                    material2D, boundarys,
                                                    boundary_scale)
            for c_index in all_broken_endpoint:
                if contains_weight_function[c_index]: continue
                flag = True
                mesh2D.manual_set_rule_at_element(c_index)
                contains_weight_function[c_index] = True
            critical = mesh2D.namelist_of_dgfem()
            n_dg_elements += mesh2D.convert_mesh_into_DGFEM(todolist=critical)
            print(
                f"    update mesh, with DGFE/tot=({n_dg_elements}/{n_elements})"
            )
            runtime_id += 1
        print(
            f"Phase {phase:4d} end, total time cost {utils.formatting_time(time.time() - t0)}"
        )

        # postprocessing
        p, t = mesh2D.nodes, mesh2D.elements
        n_nodes, n_elements = mesh2D.n_nodes, mesh2D.n_elements
        u_abs = postprocessing.get_absolute_displace(u)
        p_deform = postprocessing.get_deform_mesh(p, u)
        epsilon = postprocessing.get_strain_field(p, t, basis, u)
        sigma = postprocessing.get_stress_field(constitutive, epsilon)
        w_distortion = postprocessing.get_distortion_energy(
            youngs_modulus, poissons_ratio, sigma)

        # without deformation
        tecplot_config = postprocessing.generate_tecplot_config(
            n_nodes, n_elements, basis.length)
        tecplot_config[
            "variables"] = "X, Y, Ux, Uy, Uabs, epsilon_x, epsilon_y, epsilon_xy, sigma_x, sigma_y, sigma_xy, w_distortion"
        postprocessing.export_tecplot_data(
            f"{EXAMPLE_NAME}-morphing-phase-{phase:02d}-{n_elements}-elements.dat",
            tecplot_config, p, t, u, u_abs, epsilon, sigma, w_distortion)
        # tecplot_config["variables"] = "X, Y, Ux, Uy, Uabs"
        # postprocessing.export_tecplot_data(f"{EXAMPLE_NAME}-morphing-phase-{phase:02d}-{n_elements}-elements.dat", tecplot_config, p, t, u, u_abs)

        # containing deformation
        deform_config = postprocessing.generate_tecplot_config(
            n_nodes, n_elements, basis.length)
        deform_config["variables"] = "X, Y, Ux, Uy, Uabs"
        postprocessing.export_tecplot_data(
            f"{EXAMPLE_NAME}-morphing-phase-{phase:02d}-{n_elements}-elements-deform.dat",
            deform_config, p_deform, t, u, u_abs)

        w_rough = meshdata.get_weight_function_value_roughly()
        alpha_config = postprocessing.generate_tecplot_config(
            n_nodes, n_elements, basis.length)
        alpha_config["variables"] = "X, Y, alpha"
        postprocessing.export_tecplot_data(
            f"{EXAMPLE_NAME}-morphing-phase-{phase:02d}-{n_elements}-elements-alpha-roughly.dat",
            alpha_config, p, t, w_rough)

        w_exact = meshdata.get_weight_function_value_exactly(
            (w_, x_gauss, y_gauss), basis)
        alpha_config = postprocessing.generate_tecplot_config(
            n_nodes, n_elements, basis.length)
        alpha_config["variables"] = "X, Y, alpha"
        postprocessing.export_tecplot_data(
            f"{EXAMPLE_NAME}-morphing-phase-{phase:02d}-{n_elements}-elements-alpha-exactly.dat",
            alpha_config, p, t, w_exact)

    print(
        f"total phase {TOTAL_PHASES}, total time cost {utils.formatting_time(time.time() - start_timestamp)}"
    )
