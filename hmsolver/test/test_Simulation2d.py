import sys
sys.path.append('./../../')

import numpy as np

import hmsolver.geometry as geometry

from hmsolver.meshgrid.zone2d import Zone2d
from hmsolver.femcore.preprocessing import read_mesh
from hmsolver.femcore.treat_boundary import point_criteria, segment_criteria
from hmsolver.femcore.treat_boundary import boundary_cond2d, BoundaryConds2d
from hmsolver.app.simulation2d import Simulation2d, PdSimulation2d
from hmsolver.meshgrid.hybrid_mesh2d import HybridMesh2d
from hmsolver.material.pd_material2d import PdMaterial2d

from hmsolver.basis.quad4 import Quad4Node

if __name__ == '__main__':
    input_file = 'data_2601_2500.msh'
    example_name = 'example-05'

    horizon_radius = 0.06
    grid_size = 0.02
    l_inst = 0.015

    n_nodes, n_elements, nodes, elements = read_mesh(input_file)
    mesh2d = HybridMesh2d(n_nodes, n_elements)
    mesh2d.manually_construct(np.array(nodes), np.array(elements))

    material2d = PdMaterial2d(3e11, 1.0 / 3)

    zone = Zone2d(0, 1, 0, 1)
    zone_xl, zone_xr = 0, 1
    zone_yl, zone_yr = 0, 1
    zone_xmid = 0.5 * (zone_xl + zone_xr)
    zone_ymid = 0.5 * (zone_yl + zone_yr)
    stretch, tension = 0.04, 0.02
    slope = stretch / (zone_xl - zone_xr) / 2
    boundary_0 = point_criteria(zone_xmid, zone_yl)
    boundary_1 = segment_criteria(zone_xl, zone_yl, zone_xr, zone_yl)
    boundary_2 = segment_criteria(zone_xl, zone_yr,
                                  zone_xmid - geometry.SPACING, zone_yr)
    boundary_3 = segment_criteria(zone_xmid + geometry.SPACING, zone_yr,
                                  zone_xr, zone_yr)
    boundary_4 = segment_criteria(zone_xl, zone_yr, zone_xr, zone_yr)

    _bc_ = boundary_cond2d  # abbreviate the word for type & read
    boundarys = BoundaryConds2d(
        _bc_("point", boundary_0, "fixed", None, None),
        _bc_("segment", boundary_1, "set_uy", "constant", 0),
        # _bc_("segment", boundary_2, "set_ux", "constant", -stretch),
        # _bc_("segment", boundary_3, "set_ux", "constant", +stretch),
        # _bc_("segment", boundary_2, "set_ux",
        #  "lambda", lambda x, y: slope * x - 0.5 * stretch),
        # _bc_("segment", boundary_3, "set_ux",
        #  "lambda", lambda x, y: slope * x + 1.5 * stretch),
        _bc_("segment", boundary_4, "set_uy", "constant", tension),
    )
    del _bc_  # delete the abbreviation

    a = Simulation2d(mesh2d, material2d, boundarys)
    a.app_name = example_name
    print(1, end='')
    a.check_engine()
    print("\n\n")
    b = PdSimulation2d(mesh2d, material2d, boundarys)
    b.app_name = example_name
    print(2, end='')
    b.check_engine()
    b.material.setIsotropic(horizon_radius, grid_size, l_inst)
    print(3, end='')
    b.check_engine()
    b.mesh.peridynamic_construct(horizon_radius, 2 * horizon_radius,
                                 4 * horizon_radius)
    print(4, end='')
    b.check_engine()

    b.boundary_conds.manually_verify()
    print(5, end='')
    b.check_engine()

    c = Simulation2d(None, None, None)
    d = Simulation2d(None, material2d, boundarys)
    d = Simulation2d(mesh2d, None, boundarys)
    d = Simulation2d(mesh2d, material2d, None)

    print('\n' * 3)
    a.apply_basis(Quad4Node())
    a.check_engine()
    u = a.u_solution()
