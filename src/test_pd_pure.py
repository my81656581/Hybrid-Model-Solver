import numpy as np
import pickle

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

if __name__ == "__main__":
    # example config
    # input_file = 'data_121_100.msh'
    input_file = 'data_2601_2500.msh'
    example_name = 'example_01'
    # example_name = 'example_02'


    # input_file = 'data_2100_2000.msh'
    # example_name = 'example-05'

    # zone boundary
    zone_xl, zone_xr = 0, 0.2
    zone_yl, zone_yr = 0, 0.1
    zone_xmid = 0.5 * (zone_xl + zone_xr)
    zone_ymid = 0.5 * (zone_yl + zone_yr)

    # node & element
    n_nodes, n_elements, nodes, elements = preprocessing.read_mesh(input_file)
    p, t = np.array(nodes), np.array(elements)

    # mesh
    # horizon_radius = 0.3
    horizon_radius = 0.06
    # horizon_radius = 0.0045
    meshdata = Mesh2d.HybridMesh2d(n_nodes, n_elements)
    meshdata.manually_construct(np.array(nodes), np.array(elements))
    meshdata.peridynamic_construct(horizon_radius, 2 * horizon_radius, 4 * horizon_radius)
    related = meshdata.bonds
    meshdata.debug_element(50)

    # material
    continuum = Material2d.Material2d(youngs_modulus=3e11, poissons_ratio=1.0 / 3)
    material = Material2d.PdMaterial2d(continuum=continuum)
    material.setIsotropic(horizon_radius=horizon_radius)
    # continuum
    youngs_modulus = material.youngs_modulus
    poissons_ratio = material.poissons_ratio
    shear_modulus = material.shear_modulus
    lame_mu = material.lame_mu
    lame_lambda = material.lame_lambda
    constructive = material.constructive
    c11, c12 = constructive[0, 0], constructive[0, 1]
    # non-local
    c0 = material.coefficients[0]
    # c0 = 1

    print(material.coefficients)
    print(f"lame parameters: lambda={lame_lambda:.4e}, mu={lame_mu:.4e}")

    coef_c0 = np.vectorize(lambda xi2: c0 * np.exp(-xi2**0.5 / 0.015))
    # coef_c0 = np.vectorize(lambda xi2: c0 * np.exp(-xi2**0.5 / 0.0005))

    basis = reference_basis.Quadrilateral4Node()

    print(c0)
    material.syncIsotropic(horizon_radius, horizon_radius / 3.0, 0.015)
    print(material.coefficients)
    print(c0)

    k = pd_stiffness.assemble_stiffness_matrix(p, t, related, coef_c0, basis)

    pickle.dump(k, open(f"stiff-pd-{n_nodes}-nodes.bin", "wb"))
    a = pickle.load(open(f"stiff-pd-{n_nodes}-nodes.bin", "rb"))

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
    # <  <  <  x  >  >  >
    # +  +  +  +  +  +  +
    # +  +  +  +  +  +  +
    # +  +  +  +  +  +  +
    # +  +  +  +  +  +  +
    # +  +  +  +  +  +  +
    # -  -  -  x  -  -  -
    # boundary 0: fixed point, ux = uy = 0
    # @x = 1 / 2 * (zone_xl + zone_xr)
    # @y = zone_yl
    # boundary 1: segment, uy = 0
    # @x \in [zone_xl, zone_xr]
    # @y = zone_yl
    # boundary 2: segment, ux = -0.1
    # @x \in [zone_xl, 1 / 2 * (zone_xl + zone_xr))
    # @y = zone_yr
    # boundary 3: segment, ux = +0.1
    # @x \in (1 / 2 * (zone_xl + zone_xr), zone_xr]
    # @y = zone_yr
    # boundary 4: point, ux = 0
    # @x = 1 / 2 * (zone_xl + zone_xr)
    # @y = zone_yr

    # zone_xmid = 0.5 * (zone_xl + zone_xr)
    # zone_ymid = 0.5 * (zone_yl + zone_yr)

    # boundary_0 = treat_boundary.point_criteria(zone_xmid, zone_yl)
    # boundary_1 = treat_boundary.segment_criteria(zone_xl, zone_yl, zone_xr, zone_yl)
    # boundary_2 = treat_boundary.segment_criteria(zone_xl, zone_yr, zone_xmid - treat_boundary.SPACING, zone_yr)
    # boundary_3 = treat_boundary.segment_criteria(zone_xmid + treat_boundary.SPACING, zone_yr, zone_xr, zone_yr)
    # # boundary_4 = treat_boundary.point_criteria(zone_xmid, zone_yr)

    # treat_boundary.fixed_point(a, b, p, boundary_0)
    # treat_boundary.segment_set_uy(a, b, p, boundary_1, 0)
    # treat_boundary.segment_set_ux(a, b, p, boundary_2, -0.1)
    # treat_boundary.segment_set_ux(a, b, p, boundary_3, +0.1)
    # # treat_boundary.point_set_ux(a, b, p, boundary_4, 0)


    # example 05, using data_2100_2000.msh
    # |  +  +  +  +  +  >
    # |  +  +  +  +  +  >
    # |  +  +  +  +  +  >
    # x  +  +  +  +  +  >
    # |  +  +  +  +  +  >
    # |  +  +  +  +  +  >
    # |  +  +  +  +  +  >
    # let tension = 0.02
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

    # tension = 0.002
    # boundary_0 = treat_boundary.point_criteria(zone_xl, zone_ymid)
    # boundary_1 = treat_boundary.segment_criteria(zone_xl, zone_yl, zone_xl, zone_yr)
    # boundary_2 = treat_boundary.segment_criteria(zone_xr, zone_yl, zone_xr, zone_yr)
    # boundarys = [
    #     ("point", "fixed", "constant", boundary_0),
    #     ("segment", "set_ux", "constant", boundary_1, 0),
    #     ("segment", "set_ux", "constant", boundary_2, tension),
    # ]


    # compiled_boundary = treat_boundary.complie_boundary(p, boundarys)
    # treat_boundary.apply_boundary(a, b, p, compiled_boundary)

    print(a[0:5, 0:5])

    # u_ = np.linalg.pinv(a) @ b

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
    postprocessing.export_tecplot_data(f"{example_name}-pd-{n_nodes}-nodes.dat", tecplot_config, p, t, u, u_abs, epsilon, sigma, w_distortion)

    # containing deformation
    deform_config = postprocessing.generate_tecplot_config(n_nodes, n_elements, basis.length)
    deform_config["variables"] = "X, Y, Ux, Uy, Uabs"
    postprocessing.export_tecplot_data(f"{example_name}-pd-{n_nodes}-nodes-deform.dat", deform_config, p_deform, t, u, u_abs)

    # deal_bond_stretch(p, t, related, u, coef_c0, basis)
