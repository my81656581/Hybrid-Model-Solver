import numpy as np
import pickle

import utils

import material2D
import HMSMeshData2D
import preprocessing
import reference_basis
import gaussint
import stiffness
import pd_stiffness
import morphing
import treat_boundary
import postprocessing

if __name__ == "__main__":
    # example config
    input_file = 'data_2601_2500.msh'
    example_name = 'example-03'
    # example_name = 'example-04'

    # zone boundary
    zone = HMSMeshData2D.Zone2d(0, 1, 0, 1)
    zone_xl, zone_xr = 0, 1
    zone_yl, zone_yr = 0, 1
    zone_xmid = 0.5 * (zone_xl + zone_xr)
    zone_ymid = 0.5 * (zone_yl + zone_yr)

    # node & element
    n_nodes, n_elements, nodes, elements = preprocessing.read_mesh(input_file)
    p, t = np.array(nodes), np.array(elements)

    # mesh
    horizon_radius = 0.06
    meshdata = HMSMeshData2D.HybridMesh2d(n_nodes, n_elements)
    meshdata.manually_construct(np.array(nodes), np.array(elements))
    meshdata.peridynamic_construct(horizon_radius, 2 * horizon_radius, 4 * horizon_radius)
    meshdata.debug_element(50)

    # material
    continuum = material2D.Material2D(youngs_modulus=3e11, poissons_ratio=1.0 / 3)
    material = material2D.PdMaterial2D(continuum=continuum)
    material.setIsotropic(horizon_radius=horizon_radius)


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

    # example 03, using data_2601_2500.msh
    # <  <  <  x  >  >  >
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

    stretch = 0.1
    boundary_0 = treat_boundary.point_criteria(zone_xmid, zone_yl)
    boundary_1 = treat_boundary.segment_criteria(zone_xl, zone_yl, zone_xr, zone_yl)
    boundary_2 = treat_boundary.segment_criteria(zone_xl, zone_yr, zone_xmid - utils.SPACING, zone_yr)
    boundary_3 = treat_boundary.segment_criteria(zone_xmid + utils.SPACING, zone_yr, zone_xr, zone_yr)

    boundarys = [
        ("point", "fixed", boundary_0),
        ("segment", "set_uy", boundary_1, 0),
        ("segment", "set_ux", boundary_2, -stretch),
        ("segment", "set_ux", boundary_3, +stretch),
    ]

    # idx_0 = treat_boundary.point_setting(p, boundary_0)
    # idx_1 = treat_boundary.segment_setting(p, boundary_1)
    # idx_2 = treat_boundary.segment_setting(p, boundary_2)
    # idx_3 = treat_boundary.segment_setting(p, boundary_3)
    # treat_boundary.just_fixed_point(a, b, p, idx_0)
    # treat_boundary.just_set_segment_uy(a, b, p, idx_1, 0)
    # treat_boundary.just_set_segment_ux(a, b, p, idx_2, -stretch)
    # treat_boundary.just_set_segment_ux(a, b, p, idx_3, +stretch)
    complied_boundary = treat_boundary.complie_boundary(p, boundarys)
    treat_boundary.apply_boundary(a, b, p, complied_boundary)



    # example 04, using data_2601_2500.msh
    # ^  ^  ^  ^  ^  ^  ^
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
    # boundary 2: segment, uy = -0.1
    # @x \in [zone_xl, zone_xr)
    # @y = zone_yr

    # boundary_0 = treat_boundary.point_criteria(zone_xmid, zone_yl)
    # boundary_1 = treat_boundary.segment_criteria(zone_xl, zone_yl, zone_xr, zone_yl)
    # boundary_2 = treat_boundary.segment_criteria(zone_xl, zone_yr, zone_xr, zone_yr)
    # treat_boundary.fixed_point(a, b, p, boundary_0)
    # treat_boundary.segment_set_uy(a, b, p, boundary_1, 0)
    # treat_boundary.segment_set_uy(a, b, p, boundary_2, +0.1)

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
    # tecplot_config["variables"] = "X, Y, Ux, Uy, Uabs, epsilon_x, epsilon_y, epsilon_xy, sigma_x, sigma_y, sigma_xy, w_distortion"
    # postprocessing.export_tecplot_data(f"{example_name}-pd-morphing-{n_elements}-elements.dat", tecplot_config, p, t, u, u_abs, epsilon, sigma, w_distortion)
    tecplot_config["variables"] = "X, Y, Ux, Uy, Uabs"
    postprocessing.export_tecplot_data(f"{example_name}-pd-morphing-{n_elements}-elements.dat", tecplot_config, p, t, u, u_abs)

    # containing deformation
    deform_config = postprocessing.generate_tecplot_config(n_nodes, n_elements, basis.length)
    deform_config["variables"] = "X, Y, Ux, Uy, Uabs"
    postprocessing.export_tecplot_data(f"{example_name}-pd-morphing-{n_elements}-elements-deform.dat", deform_config, p_deform, t, u, u_abs)


    w_rough = meshdata.get_weight_function_value_roughly()
    alpha_config = postprocessing.generate_tecplot_config(n_nodes, n_elements, basis.length)
    alpha_config["variables"] = "X, Y, alpha"
    postprocessing.export_tecplot_data(f"{example_name}-pd-morphing-{n_elements}-elements-alpha-roughly.dat", alpha_config, p, t, w_rough)


    w_exact = meshdata.get_weight_function_value_exactly((w_, x_gauss, y_gauss), basis)
    alpha_config = postprocessing.generate_tecplot_config(n_nodes, n_elements, basis.length)
    alpha_config["variables"] = "X, Y, alpha"
    postprocessing.export_tecplot_data(f"{example_name}-pd-morphing-{n_elements}-elements-alpha-exactly.dat", alpha_config, p, t, w_exact)