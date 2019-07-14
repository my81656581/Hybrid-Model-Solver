import numpy as np
import abc


class __IShapeable(metaclass=abc.ABCMeta):
    @property
    def shapes(self):
        return self.__shapes_

    @shapes.setter
    def shapes(self, _):
        self.__shapes_ = _

    @property
    def length(self):
        return self.__basis_num_

    @length.setter
    def length(self, _):
        self.__basis_num_ = _

    @property
    def highorder(self):
        return self.__highorder_

    @highorder.setter
    def highorder(self, _):
        self.__highorder_ = _

    @abc.abstractmethod
    def reference_basis(self, x, y, basis_index, diff_order):
        pass

    @abc.abstractmethod
    def transform(self, x, y, vertices, diff_order):
        pass

    @abc.abstractmethod
    def shape_vector(self, x, y, vertices):
        pass


class __IShapeDxable(object):
    @property
    def shapes_dx(self):
        return self.__shapes_dx_

    @shapes_dx.setter
    def shapes_dx(self, _):
        self.__shapes_dx_ = _


class __IShapeDyable(object):
    @property
    def shapes_dy(self):
        return self.__shapes_dy_

    @shapes_dy.setter
    def shapes_dy(self, _):
        self.__shapes_dy_ = _


class __IShapeDxDyable(object):
    @property
    def shapes_dxdy(self):
        return self.__shapes_dxdy_

    @shapes_dxdy.setter
    def shapes_dxdy(self, _):
        self.__shapes_dxdy_ = _


class Quadrilateral4Node(__IShapeable, __IShapeDxable, __IShapeDyable,
                         __IShapeDxDyable):
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
        shape = np.hstack([transform_function[_](x, y) for _ in range(self.length)])
        return shape @ x_, shape @ y_

    def shape_vector(self, x, y):
        return np.hstack([self.shapes[_](x, y) for _ in range(self.length)])




if __name__ == "__main__":
    a = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    b = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    print(0, Quadrilateral4Node().reference_basis(a, b, 0, (0, 0)))
    print(1, Quadrilateral4Node().reference_basis(a, b, 0, (1, 0)))
    print(2, Quadrilateral4Node().reference_basis(a, b, 0, (0, 1)))
    print(3, Quadrilateral4Node().reference_basis(a, b, 0, (1, 1)))
    print(4, Quadrilateral4Node().reference_basis(a, b, 0, (2, 1)))
    print(5, Quadrilateral4Node().reference_basis(a, b, 0, (1, 2)))
    # 0 [ 0.2025  0.16    0.1225  0.09    0.0625]
    # 1 [-0.225 -0.2   -0.175 -0.15  -0.125]
    # 2 [-0.225 -0.2   -0.175 -0.15  -0.125]
    # 3 [ 0.25  0.25  0.25  0.25  0.25]
    # 4 [ 0.  0.  0.  0.  0.]
    # 5 [ 0.  0.  0.  0.  0.]