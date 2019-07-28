# import math
import numpy as np

# EPS = np.spacing(10)
EPS = 1e-8
SPACING = 10 * EPS
BIG = 1e10


def is_zero(x):
    return np.abs(x) < EPS


def xmult(A, O, C):
    # (O -> A) x (O -> C)
    return (A[0] - O[0]) * (C[1] - O[1]) - (C[0] - O[0]) * (A[1] - O[1])