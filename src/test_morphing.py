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
    


    # material
    continuum = material2D.Material2D(youngs_modulus=3e11, poissons_ratio=1.0 / 3)
    material = material2D.PdMaterial2D(continuum=continuum)
    material.setIsotropic(horizon_radius=horizon_radius)
    # continuum
    youngs_modulus = material.youngs_modulus
    poissons_ratio = material.poissons_ratio
    shear_modulus = material.shear_modulus
    lame_mu = material.lame_mu
    lame_lambda = material.lame_lambda
    constructive = material.constructive
    # non-local
    c0 = material.coefficients[0]
    coef_c0 = np.vectorize(lambda xi2: c0 * xi2 * np.exp(-xi2**0.5 / 0.015))

    # basis & gauss integral
    basis = reference_basis.Quadrilateral4Node()
    w_, x_gauss, y_gauss = gaussint.gauss_point_quadrature_standard()
    n_gauss = len(w_)

    # example 03, using data_2601_2500.msh
    # <  <  <  x  >  >  >
    # +  +  +  +  +  +  +
    # +  +  +  +  +  +  +
    # +  +  +  +  +  +  +
    # +  +  +  +  +  +  +
    # +  +  +  +  +  +  +
    # -  -  -  x  -  -  -
    # let stretch = 0.1
    # let slope = stretch / (zone_xr - zone_xl)
    # boundary 0: fixed point, ux = uy = 0
    # @x = 1 / 2 * (zone_xl + zone_xr)
    # @y = zone_yl
    # boundary 1: segment, uy = 0
    # @x \in [zone_xl, zone_xr]
    # @y = zone_yl
    # boundary 2: segment, ux = -stretch + slope * x
    # @x \in [zone_xl, 1 / 2 * (zone_xl + zone_xr))
    # @y = zone_yr
    # boundary 3: segment, ux = slope * x
    # @x \in (1 / 2 * (zone_xl + zone_xr), zone_xr]
    # @y = zone_yr
    stretch = 0.1
    slope = stretch / (zone_xr - zone_xl)
    boundary_0 = treat_boundary.point_criteria(zone_xmid, zone_yl)
    boundary_1 = treat_boundary.segment_criteria(zone_xl, zone_yl, zone_xr, zone_yl)
    boundary_2 = treat_boundary.segment_criteria(zone_xl, zone_yr, zone_xmid - utils.SPACING, zone_yr)
    boundary_3 = treat_boundary.segment_criteria(zone_xmid + utils.SPACING, zone_yr, zone_xr, zone_yr)
    boundarys = [
        ("point", "fixed", "constant", boundary_0),
        ("segment", "set_uy", "constant", boundary_1, 0),
        ("segment", "set_ux", "lambda", boundary_2, lambda x, y: -stretch + stretch * x),
        ("segment", "set_ux", "lambda", boundary_3, lambda x, y: slope * x),
    ]
    compiled_boundary = treat_boundary.complie_boundary(p, boundarys)

    u = morphing.morphing(zone, meshdata, material, compiled_boundary)

    n_nodes, n_elements = meshdata.n_nodes, meshdata.n_elements
    p, t, adj = meshdata.nodes, meshdata.elements, meshdata.adjoint
    related = meshdata.bonds
    centers = meshdata.centers
    weight_function_handle = meshdata.query_alpha

    # postprocessing
    u_abs = postprocessing.get_absolute_displace(u)
    p_deform = postprocessing.get_deform_mesh(p, u)
    epsilon = postprocessing.get_strain_field(p, t, basis, u)
    sigma = postprocessing.get_stress_field(constructive, epsilon)
    w_distortion = postprocessing.get_distortion_energy(youngs_modulus, poissons_ratio, sigma)

    # without deformation
    tecplot_config = postprocessing.generate_tecplot_config(n_nodes, n_elements, basis.length)
    tecplot_config["variables"] = "X, Y, Ux, Uy, Uabs, epsilon_x, epsilon_y, epsilon_xy, sigma_x, sigma_y, sigma_xy, w_distortion"
    postprocessing.export_tecplot_data(f"{example_name}-pd-morphing-{n_elements}-elements.dat", tecplot_config, p, t, u, u_abs, epsilon, sigma, w_distortion)
    # tecplot_config["variables"] = "X, Y, Ux, Uy, Uabs"
    # postprocessing.export_tecplot_data(f"{example_name}-pd-morphing-{n_elements}-elements.dat", tecplot_config, p, t, u, u_abs)

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