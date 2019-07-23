import abc

__all__ = ['IShapeable', 'IShapeDxable', 'IShapeDxDyable', 'IShapeDyable']


class IShapeable(metaclass=abc.ABCMeta):
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


class IShapeDxable(object):
    @property
    def shapes_dx(self):
        return self.__shapes_dx_

    @shapes_dx.setter
    def shapes_dx(self, _):
        self.__shapes_dx_ = _


class IShapeDyable(object):
    @property
    def shapes_dy(self):
        return self.__shapes_dy_

    @shapes_dy.setter
    def shapes_dy(self, _):
        self.__shapes_dy_ = _


class IShapeDxDyable(object):
    @property
    def shapes_dxdy(self):
        return self.__shapes_dxdy_

    @shapes_dxdy.setter
    def shapes_dxdy(self, _):
        self.__shapes_dxdy_ = _
