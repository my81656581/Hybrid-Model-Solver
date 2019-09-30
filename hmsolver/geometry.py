# import math
import numpy as np

# EPS = np.spacing(10)
EPS = 1e-8
SPACING = 10 * EPS
BIG = 1e10


def is0(x):
    return np.abs(x) < EPS


def not0(x):
    return not is0(x)


def gt0(x):
    return x > EPS


def lt0(x):
    return x < -EPS


def geq0(x):
    return not lt0(x)


def leq0(x):
    return not gt0(x)


def xmult(A, O, C):
    # (O -> A) x (O -> C)
    return (A[0] - O[0]) * (C[1] - O[1]) - (C[0] - O[0]) * (A[1] - O[1])


def dots_inline(A, B, C):
    # A, B, C belong to one line
    return is0(xmult(A, B, C))


def dot_online_in(P, A, B):
    # P belongs to segment [A, B]
    # consider endpoint A & B
    return is0(xmult(P, A, B)) and leq0(
        (A[0] - P[0]) * (B[0] - P[0])) and leq0(A[1] - P[1]) * (B[1] - P[1])


def dot_online_ex(P, A, B):
    # P belongs to segment (A, B)
    # except endpoint A & B
    return dot_online_in(
        P, A, B) and (not0(P[0] - A[0])
                      or not0(P[1] - A[1])) and (not0(P[0] - B[0])
                                                 or not0(P[1] - B[1]))


def same_side(P, Q, A, B):
    # P and Q are in the same side of line AB
    return gt0(xmult(A, B, P) * xmult(A, B, Q))


def opposite_side(P, Q, A, B):
    # P and Q are not in the same side of line AB
    return lt0(xmult(A, B, P) * xmult(A, B, Q))


def intersect_in(P, Q, A, B):
    # segments [P, Q] and [A, B] intersect
    if (not dots_inline(P, Q, A)) or (not dots_inline(P, Q, B)):
        return (not same_side(P, Q, A, B)) and (not same_side(A, B, P, Q))
    return dot_online_in(P, A, B) or dot_online_in(Q, A, B) or dot_online_in(
        A, P, Q) or dot_online_in(B, P, Q)


def intersect_ex(P, Q, A, B):
    # segments (P, Q) and (A, B) intersect
    return opposite_side(P, Q, A, B) or opposite_side(A, B, P, Q)
