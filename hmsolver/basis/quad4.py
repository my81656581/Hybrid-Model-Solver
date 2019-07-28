import numpy as np

from hmsolver.basis.infrastructures import IShapeable, IShapeDxable, IShapeDxDyable, IShapeDyable

__all__ = ['Quad4Node']


class Quad4Node(IShapeable, IShapeDxable, IShapeDyable, IShapeDxDyable):
    def __init__(self):
        super().__init__()
        n1 = np.vectorize(lambda a, b: 0.25 * (1 - a) * (1 - b))
        n2 = np.vectorize(lambda a, b: 0.25 * (1 + a) * (1 - b))
        n3 = np.vectorize(lambda a, b: 0.25 * (1 + a) * (1 + b))
        n4 = np.vectorize(lambda a, b: 0.25 * (1 - a) * (1 + b))
        dx1 = np.vectorize(lambda a, b: -0.25 * (1 - b))
        dx2 = np.vectorize(lambda a, b: 0.25 * (1 - b))
        dx3 = np.vectorize(lambda a, b: 0.25 * (1 + b))
        dx4 = np.vectorize(lambda a, b: -0.25 * (1 + b))
        dy1 = np.vectorize(lambda a, b: -0.25 * (1 - a))
        dy2 = np.vectorize(lambda a, b: -0.25 * (1 + a))
        dy3 = np.vectorize(lambda a, b: 0.25 * (1 + a))
        dy4 = np.vectorize(lambda a, b: 0.25 * (1 - a))
        dxdy1 = np.vectorize(lambda a, b: 0.25)
        dxdy2 = np.vectorize(lambda a, b: -0.25)
        high = np.vectorize(lambda a, b: 0)
        self.typeid = 2401
        self.length = 4
        self.shapes = [n1, n2, n3, n4]
        self.shapes_dx = [dx1, dx2, dx3, dx4]
        self.shapes_dy = [dy1, dy2, dy3, dy4]
        self.shapes_dxdy = [dxdy1, dxdy2, dxdy1, dxdy2]
        self.highorder = [high, high, high, high]

    def reference_basis(self, x, y, basis_index, diff_order):
        dx, dy = diff_order
        assert dx >= 0
        assert dy >= 0
        if dx == 0 and dy == 0:
            return self.shapes[basis_index](x, y)
        elif dx == 1 and dy == 0:
            return self.shapes_dx[basis_index](x, y)
        elif dx == 0 and dy == 1:
            return self.shapes_dy[basis_index](x, y)
        elif dx == 1 and dy == 1:
            return self.shapes_dxdy[basis_index](x, y)
        else:
            return self.highorder[basis_index](x, y)

    def transform(self, x, y, vertices, diff_order):
        dx, dy = diff_order
        assert dx >= 0
        assert dy >= 0
        if dx == 0 and dy == 0:
            transform_function = self.shapes
        elif dx == 1 and dy == 0:
            transform_function = self.shapes_dx
        elif dx == 0 and dy == 1:
            transform_function = self.shapes_dy
        elif dx == 1 and dy == 1:
            transform_function = self.shapes_dxdy
        else:
            transform_function = self.highorder
        x_, y_ = [vertices[:, _] for _ in (0, 1)]
        shape = np.hstack(
            [transform_function[_](x, y) for _ in range(self.length)])
        return shape @ x_, shape @ y_

    def shape_vector(self, x, y):
        return np.hstack([self.shapes[_](x, y) for _ in range(self.length)])
