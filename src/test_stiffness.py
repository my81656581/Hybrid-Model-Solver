import numpy as np
import pickle

import utils

import material2D
import preprocessing
import reference_basis
import stiffness
import treat_boundary
import postprocessing


if __name__ == "__main__":
    # example config
    # input_file = 'data_121_100.msh'
    input_file = 'data_2601_2500.msh'
    example_name = 'example_01'
    # example_name = 'example_02'

    # zone boundary
    zone_xl, zone_xr = 0, 1
    zone_yl, zone_yr = 0, 1
    zone_xmid = 0.5 * (zone_xl + zone_xr)
    zone_ymid = 0.5 * (zone_yl + zone_yr)

    # node & element
    n_nodes, n_elements, nodes, elements = preprocessing.read_mesh(input_file)
    p, t = np.array(nodes), np.array(elements)
    
    # material
    material = material2D.Material2D(youngs_modulus=3e11, poissons_ratio=1.0 / 3)
    youngs_modulus = material.youngs_modulus
    poissons_ratio = material.poissons_ratio
    shear_modulus = material.shear_modulus
    lame_mu = material.lame_mu
    lame_lambda = material.lame_lambda
    constructive = material.constructive
    c11, c12 = constructive[0, 0], constructive[0, 1]
    print(f"lame parameters: lambda={lame_lambda:.4e}, mu={lame_mu:.4e}")

    coef_c11 = np.vectorize(lambda x, y: c11)
    coef_c12 = np.vectorize(lambda x, y: c12)
    coef_mu = np.vectorize(lambda x, y: shear_modulus)

    basis = reference_basis.Quadrilateral4Node()
    jacobis = stiffness.preprocessing_all_jacobi(p, t, basis)

    # method 1
    # k0s = stiffness.generate_stiffness_matrix_k0(p, t, constructive, basis, jacobis)
    # pickle.dump(k0s, open(f"stiff-{n_nodes}-nodes-k0s.bin", "wb"))
    # a0s = pickle.load(open(f'stiff-{n_nodes}-nodes-k0s.bin', "rb"))
    # k = stiffness.mapping_element_stiffness_matrix(p, t, basis, a0s)
    # pickle.dump(k, open(f"stiff-{n_nodes}-nodes-k-method-1.bin", "wb"))
    a = pickle.load(open(f'stiff-{n_nodes}-nodes-k-method-1.bin', "rb"))

    # method 2
    # a1 = stiffness.assemble_stiffness_matrix(p, t, coef_c11, basis, jacobis, [1, 0, 1, 0])
    # a2 = stiffness.assemble_stiffness_matrix(p, t, coef_c12, basis, jacobis, [1, 0, 0, 1])
    # a3 = stiffness.assemble_stiffness_matrix(p, t, coef_c12, basis, jacobis, [0 ,1, 1, 0])
    # a4 = stiffness.assemble_stiffness_matrix(p, t, coef_c11, basis, jacobis, [0, 1, 0, 1])
    # b1 = stiffness.assemble_stiffness_matrix(p, t, coef_mu, basis, jacobis, [0, 1, 0, 1])
    # b2 = stiffness.assemble_stiffness_matrix(p, t, coef_mu, basis, jacobis, [0 ,1, 1, 0])
    # b3 = stiffness.assemble_stiffness_matrix(p, t, coef_mu, basis, jacobis, [1, 0, 0, 1])
    # b4 = stiffness.assemble_stiffness_matrix(p, t, coef_mu, basis, jacobis, [1, 0, 1, 0])
    # k1 = a1 + b1
    # k2 = a2 + b2
    # k3 = a3 + b3
    # k4 = a4 + b4
    # k = np.vstack((np.hstack((k1, k2)), np.hstack((k3, k4))))
    # pickle.dump(k, open(f"stiff-{n_nodes}-nodes-k-method-2.bin", "wb"))
    # a = pickle.load(open(f'stiff-{n_nodes}-nodes-k-method-2.bin', "rb"))

    b = np.zeros((n_nodes * 2, 1))

    # example 01, using data_121_100.msh
    # a[5, :] = 0
    # a[:, 5] = 0
    # a[5, 5] = 1
    # for i in range(11):
    #     for j in [i + n_nodes]:
    #         a[j, :] = 0
    #         a[:, j] = 0
    #         a[j, j] = 1
    # for i in range(110, n_nodes):
    #     for j in [i + n_nodes]:
    #         a[j, j] *= 1e10
    #         b[j] = 0.1 * a[j, j]

    # example 01, using data_2601_2500.msh
    a[25, :] = 0
    a[:, 25] = 0
    a[25, 25] = 1
    for i in range(51):
        for j in [i + n_nodes]:
            a[j, :] = 0
            a[:, j] = 0
            a[j, j] = 1
    for i in range(51 * 50, n_nodes):
        for j in [i + n_nodes]:
            a[j, j] *= 1e10
            b[j] = 0.1 * a[j, j]

    # example 02, using data_2601_2500.msh
    # <  <  <  -  >  >  >
    # +  +  +  +  +  +  +
    # +  +  +  +  +  +  +
    # +  +  +  +  +  +  +
    # +  +  +  +  +  +  +
    # +  +  +  +  +  +  +
    # -  -  -  x  -  -  -
    # let stretch = 0.1
    # boundary 0: fixed point, ux = uy = 0
    # @x = 1 / 2 * (zone_xl + zone_xr)
    # @y = zone_yl
    # boundary 1: segment, uy = 0
    # @x \in [zone_xl, zone_xr]
    # @y = zone_yl
    # boundary 2: segment, ux = -stretch
    # @x \in [zone_xl, 1 / 2 * (zone_xl + zone_xr))
    # @y = zone_yr
    # boundary 3: segment, ux = +stretch
    # @x \in (1 / 2 * (zone_xl + zone_xr), zone_xr]
    # @y = zone_yr
    # stretch = 0.1
    # boundary_0 = treat_boundary.point_criteria(zone_xmid, zone_yl)
    # boundary_1 = treat_boundary.segment_criteria(zone_xl, zone_yl, zone_xr, zone_yl)
    # boundary_2 = treat_boundary.segment_criteria(zone_xl, zone_yr, zone_xmid - utils.SPACING, zone_yr)
    # boundary_3 = treat_boundary.segment_criteria(zone_xmid + utils.SPACING, zone_yr, zone_xr, zone_yr)
    # boundarys = [
    #     ("point", "fixed", "constant", boundary_0),
    #     ("segment", "set_uy", "constant", boundary_1, 0),
    #     ("segment", "set_ux", "constant", boundary_2, -stretch),
    #     ("segment", "set_ux", "constant", boundary_3, +stretch),
    # ]
    # compiled_boundary = treat_boundary.complie_boundary(p, boundarys)
    # treat_boundary.apply_boundary(a, b, p, compiled_boundary)

    print(a[0:5, 0:5])

    u_ = np.linalg.solve(a, b)
    u = np.reshape(u_, (2, n_nodes)).T

    # postprocessing
    u_abs = postprocessing.get_absolute_displace(u)
    p_deform = postprocessing.get_deform_mesh(p, u)
    epsilon = postprocessing.get_strain_field(p, t, basis, u)
    sigma = postprocessing.get_stress_field(constructive, epsilon)
    w_distortion = postprocessing.get_distortion_energy(youngs_modulus, poissons_ratio, sigma)

    # without deformation
    tecplot_config = postprocessing.generate_tecplot_config(n_nodes, n_elements, basis.length)
    tecplot_config["variables"] = "X, Y, Ux, Uy, Uabs, epsilon_x, epsilon_y, epsilon_xy, sigma_x, sigma_y, sigma_xy, w_distortion"
    postprocessing.export_tecplot_data(f"{example_name}-{n_nodes}-nodes.dat", tecplot_config, p, t, u, u_abs, epsilon, sigma, w_distortion)

    # containing deformation
    deform_config = postprocessing.generate_tecplot_config(n_nodes, n_elements, basis.length)
    deform_config["variables"] = "X, Y, Ux, Uy, Uabs"
    postprocessing.export_tecplot_data(f"{example_name}-{n_nodes}-nodes-deform.dat", deform_config, p_deform, t, u, u_abs)
