import numpy as np

__all__ = ['Material2d']


class Material2d(object):
    def __init__(self,
                 youngs_modulus: float = 3e11,
                 poissons_ratio: float = 1.0 / 3):
        c11 = youngs_modulus / (1 - poissons_ratio * poissons_ratio)
        c12 = c11 * poissons_ratio
        c33 = youngs_modulus / (1 + poissons_ratio) / 2.0
        self.__youngs_modulus_ = youngs_modulus
        self.__poissons_ratio_ = poissons_ratio
        self.__lame_mu_ = c33
        self.__lame_lambda_ = c33 * 2.0 * poissons_ratio / (1 -
                                                            2 * poissons_ratio)
        self.__constitutive_ = np.array([[c11, c12, 0], [c12, c11, 0],
                                         [0, 0, c33]])

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
