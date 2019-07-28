# local test only
import sys
sys.path.append('./../../')
import numpy as np

import hmsolver.geometry as geometry

from hmsolver.meshgrid.zone2d import Zone2d
from hmsolver.femcore.preprocessing import read_mesh
from hmsolver.femcore.treat_boundary import point_criteria, segment_criteria
from hmsolver.femcore.treat_boundary import boundary_cond2d, BoundaryConds2d
from hmsolver.app.simulation2d import CrackSimulation2d
from hmsolver.meshgrid.hybrid_mesh2d import HybridMesh2d
from hmsolver.material.pd_material2d import PdMaterial2d
from hmsolver.basis.quad4 import Quad4Node

if __name__ == '__main__':
    # zone config
    zone_xl, zone_xr = 0, 0.2
    zone_yl, zone_yr = 0, 0.1
    zone_xmid = 0.5 * (zone_xl + zone_xr)
    zone_ymid = 0.5 * (zone_yl + zone_yr)

    grid_size = 0.0015
    horizon_radius = 0.0045
    inst_len = 0.0005

    zone = Zone2d(zone_xl, zone_xr, zone_yl, zone_yr)

    input_file = 'data_2100_2000.msh'

    n_nodes, n_elements, nodes, elements = read_mesh(input_file)

    mesh2d = HybridMesh2d(n_nodes, n_elements)
    mesh2d.manually_construct(np.array(nodes), np.array(elements))

    material2d = PdMaterial2d(3e11, 1.0 / 3)

    stretch = 0.002
    boundary_0 = point_criteria(zone_xl, zone_ymid)
    boundary_1 = segment_criteria(zone_xl, zone_yl, zone_xl, zone_yr)
    boundary_2 = segment_criteria(zone_xr, zone_yl, zone_xr, zone_yr)

    _bc_ = boundary_cond2d  # abbreviate the word for type & read
    boundarys = BoundaryConds2d(
        _bc_("point", boundary_0, "fixed", None, None),
        _bc_("segment", boundary_1, "set_ux", "constant", 0),
        _bc_("segment", boundary_2, "set_ux", "constant", stretch))
    del _bc_  # delete the abbreviation
    boundarys.manually_verify()
    a = CrackSimulation2d(mesh2d, material2d, boundarys)
    a.app_name = "plate"
    a.material.setIsotropic(horizon_radius, grid_size, inst_len)
    a.material.stretch_crit = 20
    a.mesh.peridynamic_construct(horizon_radius, 2 * horizon_radius,
                                 4 * horizon_radius)
    a.apply_basis(Quad4Node())
    a.check_engine()

    a.detect_failure(0.95 * np.max(a.w_dis))
    a.update_mesh()
    a.run_simulation(1)

    # a.export_to_tecplot("peridynamic", *a.provied_solutions)
