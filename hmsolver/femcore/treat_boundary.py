import numpy as np
from collections import namedtuple

import hmsolver.geometry as geometry

__all__ = [
    'point_criteria', 'segment_criteria', 'find_point', 'find_segment',
    'just_set_point', 'just_fixed_point', 'just_set_point_ux',
    'just_set_point_uy', 'just_set_point_ux_uy', 'just_set_point_ux_by_lambda',
    'just_set_point_uy_by_lambda', 'just_set_point_ux_uy_by_lambda',
    'just_set_segment', 'just_fixed_segment', 'just_set_segment_ux',
    'just_set_segment_uy', 'just_set_segment_ux_uy',
    'just_set_segment_ux_by_lambda', 'just_set_segment_uy_by_lambda',
    'just_set_segment_ux_uy_by_lambda', 'Boundary_Func', 'Boundary_Cond',
    'boundary_cond2d', 'CompiledBoundaryConds2d', 'BoundaryConds2d'
]

EPS, BIG = geometry.EPS, geometry.BIG


def __set_zero(stiff, load, target, val):
    stiff[target, :] = 0
    stiff[:, target] = 0
    stiff[target, target] = 1
    load[target, 0] = 0
    return True


def __set_value(stiff, load, target, val):
    # stiff[target, target] *= BIG
    # load[target] = val * stiff[target, target]
    stiff[target, :] = 0
    stiff[target, target] = 1
    load[target] = val
    return True


def __nop(a, b, c, d):
    pass


def point_criteria(x0, y0):
    def criteria(x, y):
        return geometry.is_zero(x - x0) and geometry.is_zero(y - y0)

    return criteria


def segment_criteria(x_start, y_start, x_end, y_end):
    def criteria(x, y):
        return geometry.is_zero(
            geometry.xmult(
                (x, y), (x_start, y_start),
                (x_end, y_end))) and (x_start - x) * (x_end - x) < EPS and (
                    y_start - y) * (y_end - y) < EPS

    return criteria


def find_point(nodes, criteria):
    n_nodes = len(nodes)
    for idx in range(n_nodes):
        if criteria(*nodes[idx, :]):
            return (idx, True)
    return (-1, False)


def find_segment(nodes, criteria):
    n_nodes, ret, flag = len(nodes), [], False
    for idx in range(n_nodes):
        if criteria(*nodes[idx, :]):
            ret.append(idx)
            if not flag: flag = True
    return (ret, flag)


def just_set_point(stiff, load, nodes, target, fixed_ux, fixed_uy):
    n_nodes = len(nodes)
    apply_ux, apply_uy = [(__set_zero if geometry.is_zero(_) else __set_value)
                          if _ != None else __nop
                          for _ in [fixed_ux, fixed_uy]]
    apply_ux(stiff, load, target, fixed_ux)
    apply_uy(stiff, load, target + n_nodes, fixed_uy)
    return True


def just_fixed_point(stiff, load, nodes, target):
    return just_set_point(stiff, load, nodes, target, 0, 0)


def just_set_point_ux(stiff, load, nodes, target, fixed_ux):
    return just_set_point(stiff, load, nodes, target, fixed_ux, None)


def just_set_point_uy(stiff, load, nodes, target, fixed_uy):
    return just_set_point(stiff, load, nodes, target, None, fixed_uy)


def just_set_point_ux_uy(stiff, load, nodes, target, fixed_ux_uy):
    fixed_ux, fixed_uy = fixed_ux_uy
    return just_set_point(stiff, load, nodes, target, fixed_ux, fixed_uy)


def just_set_point_ux_by_lambda(stiff, load, nodes, target, fixed_ux):
    fixed_ux = fixed_ux(*nodes[target, :])
    return just_set_point(stiff, load, nodes, target, fixed_ux, None)


def just_set_point_uy_by_lambda(stiff, load, nodes, target, fixed_uy):
    fixed_uy = fixed_uy(*nodes[target, :])
    return just_set_point(stiff, load, nodes, target, None, fixed_uy)


def just_set_point_ux_uy_by_lambda(stiff, load, nodes, target, fixed_ux_uy):
    fixed_ux, fixed_uy = fixed_ux_uy(*nodes[target, :])
    return just_set_point(stiff, load, nodes, target, fixed_ux, fixed_uy)


def just_set_segment(stiff, load, nodes, targets, fixed_ux, fixed_uy):
    n_nodes, ret = len(nodes), False
    apply_ux, apply_uy = [(__set_zero if geometry.is_zero(_) else __set_value)
                          if _ != None else __nop
                          for _ in [fixed_ux, fixed_uy]]
    for idx in targets:
        apply_ux(stiff, load, idx, fixed_ux)
        apply_uy(stiff, load, idx + n_nodes, fixed_uy)
        if not ret: ret = True
    return ret


def just_fixed_segment(stiff, load, nodes, targets):
    return just_set_segment(stiff, load, nodes, targets, 0, 0)


def just_set_segment_ux(stiff, load, nodes, targets, fixed_ux):
    return just_set_segment(stiff, load, nodes, targets, fixed_ux, None)


def just_set_segment_uy(stiff, load, nodes, targets, fixed_uy):
    return just_set_segment(stiff, load, nodes, targets, None, fixed_uy)


def just_set_segment_ux_uy(stiff, load, nodes, targets, fixed_ux_uy):
    fixed_ux, fixed_uy = fixed_ux_uy
    return just_set_segment(stiff, load, nodes, targets, fixed_ux, fixed_uy)


def just_set_segment_ux_by_lambda(stiff, load, nodes, targets, fixed_ux):
    ret = False
    for idx in targets:
        __set_value(stiff, load, idx, fixed_ux(*nodes[idx, :]))
        if not ret: ret = True
    return ret


def just_set_segment_uy_by_lambda(stiff, load, nodes, targets, fixed_uy):
    n_nodes, ret = len(nodes), False
    for idx in targets:
        __set_value(stiff, load, idx + n_nodes, fixed_uy(*nodes[idx, :]))
        if not ret: ret = True
    return ret


def just_set_segment_ux_uy_by_lambda(stiff, load, nodes, targets, fixed_ux_uy):
    n_nodes, ret = len(nodes), False
    for idx in targets:
        fixed_ux, fixed_uy = fixed_ux_uy(*nodes[idx, :])
        __set_value(stiff, load, idx, fixed_ux)
        __set_value(stiff, load, idx + n_nodes, fixed_uy)
        if not ret: ret = True
    return ret


__SAPC_ = {
    "set_ux": just_set_point_ux,
    "set_uy": just_set_point_uy,
    "set_ux_uy": just_set_point_ux_uy
}  # __SWITCH_APPLY_POINT_CONSTANT_

__SAPL_ = {
    "set_ux": just_set_point_ux_by_lambda,
    "set_uy": just_set_point_uy_by_lambda,
    "set_ux_uy": just_set_point_ux_uy_by_lambda
}  # __SWITCH_APPLY_POINT_LAMBDA_

__SASC_ = {
    "set_ux": just_set_segment_ux,
    "set_uy": just_set_segment_uy,
    "set_ux_uy": just_set_segment_ux_uy
}  # __SWITCH_APPLY_SEGMENT_CONSTANT_

__SASL_ = {
    "set_ux": just_set_segment_ux_by_lambda,
    "set_uy": just_set_segment_uy_by_lambda,
    "set_ux_uy": just_set_segment_ux_uy_by_lambda
}  # __SWITCH_APPLY_SEGMENT_LAMBDA_

SBT = {
    "default": [
        lambda s:
        f"Apply Failed. \nWrong boundary configs. No such entry named {s}"
    ],
    "point": [
        "Wrong boundary configs. Apply Point Config Failed.", just_fixed_point,
        [__SAPC_, __SAPL_]
    ],
    "segment": [
        "Wrong boundary configs. Apply Segment Config Failed.",
        just_fixed_segment, [__SASC_, __SASL_]
    ]
}  # __SWITCH_BOUNDARY_TYPE_

Boundary_Func = namedtuple('Boundary_Func', ['type', 'method', 'value'])
Boundary_Cond = namedtuple('Boundary_Cond', ['type', 'criteria', 'app'])


def boundary_cond2d(boundary_type, criteria, app_type, method, value):
    app = Boundary_Func(app_type, method, value)
    return Boundary_Cond(boundary_type, criteria, app)


class CompiledBoundaryConds2d(list):
    def __init__(self, *cbconds):
        list.__init__([])
        self.extend(cbconds)

    def apply(self, stiff, loads, nodes, scale=1.0):
        for cond, idx in self:
            routine = SBT.get(cond.type, SBT["default"])
            if len(routine) == len(SBT["default"]):
                print(routine[0](cond.type))
            else:
                if cond.app.type == "fixed":
                    routine[1](stiff, loads, nodes, idx)
                elif cond.app.method == "constant":
                    routine[2][0][cond.app.type](stiff, loads, nodes, idx,
                                                 scale * cond.app.value)
                elif cond.app.method == "lambda":
                    routine[2][1][cond.app.type](
                        stiff, loads, nodes,
                        idx, lambda x, y: scale * cond.app.value(x, y))
                else:
                    print(routine[0])
        return stiff, loads


class BoundaryConds2d(list):
    def __init__(self, *bconds):
        list.__init__([])
        self.extend(bconds)
        self.manually_halt()

    def compile(self, nodes):
        if not self.is_ready():
            return None
        indices = []
        for cond in self:
            if cond.type == "point":
                indices.append(find_point(nodes, cond.criteria))
            elif cond.type == "segment":
                indices.append(find_segment(nodes, cond.criteria))
            else:
                print(
                    f"Wrong boundary configs. No such entry named {cond.type}")
                print("Compile Failed.")
        ret = [(config, idx[0]) for config, idx in zip(self, indices)
               if idx[-1]]
        return CompiledBoundaryConds2d(*ret)

    def manually_verify(self):
        self.__ready_ = True

    def manually_halt(self):
        self.__ready_ = False

    def is_ready(self):
        return self.__ready_


# def compile_boundary(nodes, conditions):
#     indices = []
#     for cond in conditions:
#         if cond[0] == "point":
#             indices.append(find_point(nodes, cond[3]))
#         elif cond[0] == "segment":
#             indices.append(find_segment(nodes, cond[3]))
#         else:
#             print(f"Wrong boundary configs. No such entry named {cond[0]}")
#             print("Compile Failed.")
#     ret = [(config, idx[0]) for config, idx in zip(conditions, indices)
#            if idx[-1]]
#     return ret

# def apply_boundary(stiff, loads, nodes, compiled_boundary, scale=1.0):
#     for cond, idx in compiled_boundary:
#         routine = __SBT_.get(cond[0], __SBT_["default"])
#         if len(routine) == len(__SBT_["default"]):
#             print(routine[0](cond[0]))
#         else:
#             if cond[1] == "fixed":
#                 routine[1](stiff, loads, nodes, idx)
#             elif cond.app.method == "constant":
#                 routine[2][0][cond[1]](stiff, loads, nodes, idx,
#                                        scale * cond[4])
#             elif cond[2] == "lambda":
#                 routine[2][1][cond[1]](stiff, loads, nodes,
#                                        idx, lambda x, y: scale * cond[4](x, y))
#             else:
#                 print(routine[0])
# if cond[0] == "point":
#     if cond[1] == "fixed":
#         just_fixed_point(stiff, loads, nodes, idx)
#     elif cond[2] == "constant":
#         __SAPC_[cond[1]](stiff, loads, nodes, idx, scale * cond[4])
#     elif cond[2] == "lambda":
#         __SAPL_[cond[1]](stiff, loads, nodes,
#                          idx, lambda x, y: scale * cond[4](x, y))
#     else:
#         print("Wrong boundary configs. Apply Point Config Failed.")
# elif cond[0] == "segment":
#     if cond[1] == "fixed":
#         just_fixed_segment(stiff, loads, nodes, idx)
#     elif cond[2] == "constant":
#         __SASC_[cond[1]](stiff, loads, nodes, idx, scale * cond[4])
#     elif cond[2] == "lambda":
#         __SASL_[cond[1]](stiff, loads, nodes,
#                          idx, lambda x, y: scale * cond[4](x, y))
#     else:
#         print("Wrong boundary configs. Apply Segment Config Failed.")
# else:
#     print(f"Wrong boundary configs. No such entry named {cond[0]}")
#     print("Apply Failed.")