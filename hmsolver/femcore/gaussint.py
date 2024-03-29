import numpy as np

__all__ = [
    'gauss_point_linear',
    'gauss_point_linear_standard',
    'gauss_point_quadrature',
    'gauss_point_quadrature_standard'
]


__GAUSS_XW_DICT_ = {
    1: (np.array([[-0.5773502692], [+0.5773502692]]),
        np.array([[+1.0000000000], [+1.0000000000]])),
    2: (np.array([[-0.7745966692], [+0.0000000000], [+0.7745966692]]),
        np.array([[+0.5555555556], [+0.8888888888], [+0.5555555556]])),
    3: (np.array([[-0.8611363116], [-0.3399810436], [+0.3399810436],
                  [+0.8611363116]]),
        np.array([[+0.3478548461], [+0.6521451549], [+0.6521451549],
                  [+0.3478548461]])),
    4: (np.array([[-0.9061798459], [-0.5384693101], [+0.0000000000],
                  [+0.5384693101], [+0.9061798459]]),
        np.array([[+0.2369268851], [+0.4786286705], [+0.5688888889],
                  [+0.4786286705], [+0.2369268851]])),
    5: (np.array([[-0.9324695142], [-0.6612093865], [-0.2386191761],
                  [+0.2386191761], [+0.6612093865], [+0.9324695142]]),
        np.array([[+0.1713244924], [+0.3607615730], [+0.4679139346],
                  [+0.4679139346], [+0.3607615730], [+0.1713244924]])),
    6: (np.array([[-0.9491079123], [-0.7415311856], [-0.4058451514],
                  [+0.0000000000], [+0.4058451514], [+0.7415311856],
                  [+0.9491079123]]),
        np.array([[+0.1294849662], [+0.2797053915], [+0.3818300505],
                  [+0.4179591837], [+0.3818300505], [+0.2797053915],
                  [+0.1294849662]])),
    7: (np.array([[-0.9602898566], [-0.7966664774], [-0.5255324099],
                  [-0.1834346425], [+0.1834346425], [+0.5255324099],
                  [+0.7966664774], [+0.9602898566]]),
        np.array([[+0.1012285363], [+0.2223810345], [+0.3137066459],
                  [+0.3626837834], [+0.3626837834], [+0.3137066459],
                  [+0.2223810345], [+0.1012285363]])),
}


def gauss_point_linear(a, b, order: int = 1):
    assert 1 <= order <= 7
    x, w = __GAUSS_XW_DICT_[order]
    f = np.vectorize(lambda t: (b - a) / 2 * t)
    g = np.vectorize(lambda t: f(t) + (b + a) / 2)
    return (f(w), g(x))


def gauss_point_linear_standard(order: int = 1):
    assert 1 <= order <= 7
    x, w = __GAUSS_XW_DICT_[order]
    x_ = np.reshape(x, (-1, 1))
    w_ = np.reshape(w, (-1, 1))
    return (w_, x_)


def gauss_point_quadrature(x1, x2, y1, y2, order: int = 1):
    assert 1 <= order <= 7
    u, x = gauss_point_linear(x1, x2, order)
    v, y = gauss_point_linear(y1, y2, order)
    x_, y_ = [np.reshape(_, (-1, 1)) for _ in np.meshgrid(x, y)]
    w_ = np.reshape(v @ u.T, (-1, 1))
    return (w_, x_, y_)


def gauss_point_quadrature_standard(order: int = 1):
    assert 1 <= order <= 7
    x, u = __GAUSS_XW_DICT_[order]
    y, v = __GAUSS_XW_DICT_[order]
    x_, y_ = [np.reshape(_, (-1, 1)) for _ in np.meshgrid(x, y)]
    w_ = np.reshape(v @ u.T, (-1, 1))
    return (w_, x_, y_)
