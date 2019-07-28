import numpy as np

__all__ = ['Material2d']


class Material2d(object):
    def __init__(self, E: float = 3e11, v: float = 1.0 / 3):
        c11 = E / (1 - v * v)
        c12 = c11 * v
        c33 = E / (1 + v) / 2.0
        self.__youngs_modulus_ = E
        self.__poissons_ratio_ = v
        self.__lame_mu_ = c33
        self.__lame_lambda_ = c33 * 2.0 * v / (1 - 2 * v)
        self.__constitutive_ = np.array([[c11, c12, 0], [c12, c11, 0],
                                         [0, 0, c33]])
        self.__ready_ = True

    @property
    def youngs_modulus(self):
        return self.__youngs_modulus_

    @property
    def poissons_ratio(self):
        return self.__poissons_ratio_

    @property
    def shear_modulus(self):
        return self.__lame_mu_

    @property
    def lame_lambda(self):
        return self.__lame_lambda_

    @property
    def lame_mu(self):
        return self.__lame_mu_

    @property
    def constitutive(self):
        return self.__constitutive_

    def is_ready(self):
        return self.__ready_