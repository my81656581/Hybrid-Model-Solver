# local test only
import sys
sys.path.append('./../../')
import numpy as np

import hmsolver.geometry as geometry

from hmsolver.meshgrid.zone2d import Zone2d
from hmsolver.femcore.preprocessing import read_mesh
from hmsolver.femcore.treat_boundary import point_criteria, segment_criteria
from hmsolver.femcore.treat_boundary import boundary_cond2d, BoundaryConds2d
from hmsolver.app.simulation2d import Simulation2d
from hmsolver.meshgrid.mesh2d import Mesh2d
from hmsolver.material.material2d import Material2d
from hmsolver.basis.quad4 import Quad4Node

if __name__ == '__main__':
    # zone config
    zone_xl, zone_xr = 0, 1
    zone_yl, zone_yr = 0, 1
    zone_xmid = 0.5 * (zone_xl + zone_xr)
    zone_ymid = 0.5 * (zone_yl + zone_yr)

    grid_size = 0.02

    zone = Zone2d(zone_xl, zone_xr, zone_yl, zone_yr)

    mesh2d = zone.meshgrid_zone(Mesh2d, grid_size)

    material2d = Material2d(3e11, 1.0 / 3)

    stretch, tension = 0.04, 0.02
    slope = stretch / (zone_xl - zone_xr) / 2
    boundary_0 = point_criteria(zone_xmid, zone_yl)
    boundary_1 = segment_criteria(zone_xl, zone_yl, zone_xr, zone_yl)
    boundary_2 = segment_criteria(zone_xl, zone_yr,
                                  zone_xmid - geometry.SPACING, zone_yr)
    boundary_3 = segment_criteria(zone_xmid + geometry.SPACING, zone_yr,
                                  zone_xr, zone_yr)
    # boundary_4 = segment_criteria(zone_xl, zone_yr, zone_xr, zone_yr)

    _bc_ = boundary_cond2d  # abbreviate the word for type & read
    boundarys = BoundaryConds2d(
        _bc_("point", boundary_0, "fixed", None, None),
        _bc_("segment", boundary_1, "set_uy", "constant", 0),
        _bc_("segment", boundary_2, "set_ux", "constant", -stretch),
        _bc_("segment", boundary_3, "set_ux", "constant", +stretch),
        # _bc_("segment", boundary_2, "set_ux",
        #  "lambda", lambda x, y: slope * x - 0.5 * stretch),
        # _bc_("segment", boundary_3, "set_ux",
        #  "lambda", lambda x, y: slope * x + 1.5 * stretch),
        # _bc_("segment", boundary_4, "set_uy", "constant", tension),
    )
    del _bc_  # delete the abbreviation
    boundarys.manually_verify()

    a = Simulation2d(mesh2d, material2d, boundarys)
    a.app_name = "plate"
    a.apply_basis(Quad4Node())
    a.check_engine()
    a.export_to_tecplot("elasticity", *a.provied_solutions)
