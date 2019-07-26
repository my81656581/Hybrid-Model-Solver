import numpy as np

from hmsolver.meshgrid.generate_mesh import build_mesh

__all__ = ['Zone2d']


class Zone2d(object):
    def __init__(self, zone_xl, zone_xr, zone_yl, zone_yr):
        self.zone_xl = zone_xl
        self.zone_xr = zone_xr
        self.zone_yl = zone_yl
        self.zone_yr = zone_yr

    def meshgrid_zone(self, mesh_type, grid_size):
        nodes, elements = build_mesh((self.zone_xl, self.zone_xr, grid_size),
                                     (self.zone_yl, self.zone_yr, grid_size))
        n_nodes, n_elements = len(nodes), len(elements)
        mesh = mesh_type(n_nodes, n_elements)
        mesh.manually_construct(np.array(nodes), np.array(elements))
        return mesh
