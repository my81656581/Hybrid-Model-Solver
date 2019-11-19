import numpy as np
from collections import namedtuple

from hmsolver.geometry import is0, onrange_in, dot_online_in
from hmsolver.geometry import EPS, BIG

__all__ = [
    'point_criteria', 'circle_criteria', 'segment_criteria', 'find_point',
    'find_segment', 'just_fixed', 'just_set_ux', 'just_set_uy',
    'just_set_ux_uy', 'just_set_ux_by_rule', 'just_set_uy_by_rule',
    'just_set_ux_uy_by_rule', 'just_set_by_trans', 'Boundary_Func',
    'Boundary_Cond', 'fixed_boundary_cond2d', 'boundary_cond2d',
    'CompiledBoundaryConds2d', 'BoundaryConds2d'
]


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
    load[target, 0] = val
    return True


def __nop(a, b, c, d):
    pass


def __nop2(a, b):
    return None


def point_criteria(x0, y0):
    def criteria(x, y):
        return is0(x - x0) and is0(y - y0)

    return criteria


def circle_criteria(x0, y0, r, theta_start, theta_end):
    r2 = r**2

    def criteria(x, y):
        dx, dy = x - x0, y - y0
        if not is0(dx**2 + dy**2 - r2): return False
        t = np.arctan2(dy, dx)
        return onrange_in(t, theta_start, theta_end)

    return criteria


def segment_criteria(x_start, y_start, x_end, y_end):
    def criteria(x, y):
        return dot_online_in((x, y), (x_start, y_start), (x_end, y_end))

    return criteria


def __just_set(stiff, load, nodes, targets, fixed_ux, fixed_uy):
    offset = len(nodes)  # offset := n_nodes = len(nodes)
    targets = np.array(targets)
    apply_ux, apply_uy = [
        (__set_zero if is0(_) else __set_value) if _ != None else __nop
        for _ in [fixed_ux, fixed_uy]
    ]
    apply_ux(stiff, load, targets, fixed_ux)
    apply_uy(stiff, load, targets + offset, fixed_uy)
    return True


def __just_set_by_rule_(stiff, load, nodes, targets, ux_rule, uy_rule):
    # rule means that each rule yields single value
    # for example: the Dirichlet boundary condition of point (x, y) is (ux, uy),
    # ux = ux_rule(x, y), and uy = uy_rule(x, y)
    offset = len(nodes)  # offset := n_nodes = len(nodes)
    targets = np.array(targets)
    (apply_ux, ux_rule), (apply_uy, uy_rule) = [
        (__set_value, _) if _ != None else (__nop, __nop2)
        for _ in [ux_rule, uy_rule]
    ]
    xs, ys = nodes[targets, 0], nodes[targets, 1]
    apply_ux(stiff, load, targets, ux_rule(xs, ys))
    apply_uy(stiff, load, targets + offset, uy_rule(xs, ys))
    return True


def __just_set_by_trans_(stiff, load, nodes, targets, uv_trans):
    # trans means that each transformation yields a pair of values
    # for example: the Dirichlet boundary condition of point (x, y) is (ux, uy),
    # (ux, uy) = uv_trans(x, y)
    offset = len(nodes)  # offset := n_nodes = len(nodes)
    targets = np.array(targets)
    apply_ux, apply_uy = __set_value, __set_value
    xs, ys = nodes[targets, 0], nodes[targets, 1]
    uv = uv_trans(xs, ys)
    ux_value, uy_value = uv[:, 0], uv[:, 1]
    apply_ux(stiff, load, targets, ux_value)
    apply_uy(stiff, load, targets + offset, uy_value)
    return True


# readable alias
just_set_ux_uy_by_rule = __just_set_by_rule_
just_set_by_trans = __just_set_by_trans_


def find_point(nodes, criteria):
    n_nodes, ret, flag = len(nodes), [], False
    for idx in range(n_nodes):
        if criteria(*nodes[idx, :]):
            ret.append(idx)
            if not flag: flag = True
            break
    return (ret, flag)


def find_segment(nodes, criteria):
    n_nodes, ret, flag = len(nodes), [], False
    for idx in range(n_nodes):
        if criteria(*nodes[idx, :]):
            ret.append(idx)
            if not flag: flag = True
    return (ret, flag)


def just_fixed(stiff, load, nodes, targets):
    return __just_set(stiff, load, nodes, targets, 0, 0)


def just_set_ux(stiff, load, nodes, targets, fixed_ux):
    return __just_set(stiff, load, nodes, targets, fixed_ux, None)


def just_set_uy(stiff, load, nodes, targets, fixed_uy):
    return __just_set(stiff, load, nodes, targets, None, fixed_uy)


def just_set_ux_uy(stiff, load, nodes, targets, fixed_ux_uy):
    # fixed_ux_uy := (fixed_ux, fixed_uy)
    return __just_set(stiff, load, nodes, targets, *fixed_ux_uy)


def just_set_ux_by_rule(stiff, load, nodes, targets, ux_rule):
    return __just_set_by_rule_(stiff, load, nodes, targets, ux_rule, None)


def just_set_uy_by_rule(stiff, load, nodes, targets, uy_rule):
    return __just_set_by_rule_(stiff, load, nodes, targets, None, uy_rule)


__SAC_ = {
    "set_ux": just_set_ux,
    "set_uy": just_set_uy,
    "set_ux_uy": just_set_ux_uy
}  # __SWITCH_APPLY_CONSTANT_

__SAR_ = {
    "set_ux": just_set_ux_by_rule,
    "set_uy": just_set_uy_by_rule,
    "set_ux_uy": just_set_ux_uy_by_rule
}  # __SWITCH_APPLY_RULE_

ARBCR = [
    lambda s: f"Wrong boundary configs. Apply {s} config failed.", just_fixed,
    [__SAC_, __SAR_], just_set_by_trans
]  # __APPLY_RECOGNIZED_BOUNDARY_CONDITION_ROUTINE_

Boundary_Func = namedtuple('Boundary_Func', ['type', 'method', 'value'])
Boundary_Cond = namedtuple('Boundary_Cond', ['type', 'criteria', 'func'])


def fixed_boundary_cond2d(boundary_type, criteria, app_type):
    return Boundary_Cond(boundary_type, criteria,
                         Boundary_Func(app_type, None, None))


def boundary_cond2d(boundary_type, criteria, app_type, method, value):
    func = Boundary_Func(app_type, method, value)
    return Boundary_Cond(boundary_type, criteria, func)


class CompiledBoundaryConds2d(list):
    def __init__(self, *cbconds):
        list.__init__([])
        self.extend(cbconds)

    def apply(self, stiff, loads, nodes, scale=1.0):
        memo = set(["body", "point", "segment"])
        for cond, idx in self:
            recognized = cond.type in memo
            if not recognized:
                print(
                    f"Apply Failed. \nWrong boundary configs. No such entry named {cond.type}"
                )
                continue
            if cond.func.type == "load":
                continue
            elif cond.func.type == "fixed":
                ARBCR[1](stiff, loads, nodes, idx)
            elif cond.func.method == "constant":
                ARBCR[2][0][cond.func.type](stiff, loads, nodes, idx,
                                            scale * cond.func.value)
            elif cond.func.method == "rule" or cond.func.method == "lambda":
                if isinstance(cond.func.value, (list, tuple)):
                    wrapped = cond.func.value
                else:
                    wrapped = [cond.func.value]
                ARBCR[2][1][cond.func.type](stiff, loads, nodes, idx,
                                            *[(lambda x, y: scale * _(x, y))
                                              for _ in wrapped])
            elif cond.func.method == "transformation":
                ARBCR[3](stiff, loads, nodes,
                         idx, lambda x, y: scale * cond.func.value(x, y))
            else:
                print(ARBCR[0](cond.type))
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
            if cond.type == "body":
                indices.append((None, True))
            elif cond.type == "point":
                indices.append(find_point(nodes, cond.criteria))
            elif cond.type == "segment":
                indices.append(find_segment(nodes, cond.criteria))
            else:
                print(
                    f"Wrong boundary configs. No such entry named {cond.type}")
                print("Compile Failed.")
                indices.append((None, False))
        ret = [(config, idx[0]) for config, idx in zip(self, indices)
               if idx[-1]]
        return CompiledBoundaryConds2d(*ret)

    def manually_verify(self):
        self.__ready_ = True

    def manually_halt(self):
        self.__ready_ = False

    def is_ready(self):
        return self.__ready_
