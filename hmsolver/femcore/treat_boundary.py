import numpy as np

import hmsolver.geometry as geometry

__all__ = [
    'point_criteria',
    'segment_criteria',
    'point_setting',
    'segment_setting',
    'just_set_point',
    'just_fixed_point',
    'just_set_point_ux',
    'just_set_point_uy',
    'just_set_point_ux_by_lambda',
    'just_set_point_uy_by_lambda',
    'just_set_segment',
    'just_fixed_segment',
    'just_set_segment_ux',
    'just_set_segment_uy',
    'just_set_segment_ux_by_lambda',
    'just_set_segment_uy_by_lambda',
    'complie_boundary',
    'apply_boundary',
]

EPS = geometry.EPS
SPACING = geometry.SPACING
BIG = geometry.BIG


def __set_zero(stiffness, load, idx, val):
    stiffness[idx, :] = 0
    stiffness[:, idx] = 0
    stiffness[idx, idx] = 1
    load[idx, 0] = 0
    return True


def __set_value(stiffness, load, idx, val):
    stiffness[idx, idx] *= BIG
    load[idx] = val * stiffness[idx, idx]
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


def point_setting(nodes, criteria):
    n_nodes = len(nodes)
    for idx in range(n_nodes):
        if criteria(*nodes[idx, :]):
            return (idx, True)
    return (-1, False)


def segment_setting(nodes, criteria):
    n_nodes, ret, flag = len(nodes), [], False
    for idx in range(n_nodes):
        if criteria(*nodes[idx, :]):
            ret.append(idx)
            if not flag: flag = True
    return (ret, flag)


def just_set_point(stiffness, load, nodes, target, fixed_ux, fixed_uy):
    n_nodes = len(nodes)
    apply_ux, apply_uy = [(__set_zero if geometry.is_zero(_) else __set_value)
                          if _ != None else __nop
                          for _ in [fixed_ux, fixed_uy]]
    apply_ux(stiffness, load, target, fixed_ux)
    apply_uy(stiffness, load, target + n_nodes, fixed_uy)
    return True


def just_fixed_point(stiffness, load, nodes, target):
    return just_set_point(stiffness, load, nodes, target, 0, 0)


def just_set_point_ux(stiffness, load, nodes, target, fixed_ux):
    return just_set_point(stiffness, load, nodes, target, fixed_ux, None)


def just_set_point_uy(stiffness, load, nodes, target, fixed_uy):
    return just_set_point(stiffness, load, nodes, target, None, fixed_uy)


def just_set_point_ux_by_lambda(stiffness, load, nodes, target, fixed_ux):
    __set_value(stiffness, load, target, fixed_ux(*nodes[target, :]))
    return True


def just_set_point_uy_by_lambda(stiffness, load, nodes, target, fixed_uy):
    __set_value(stiffness, load, target + len(nodes),
                fixed_uy(*nodes[target, :]))
    return True


def just_set_segment(stiffness, load, nodes, targets, fixed_ux, fixed_uy):
    n_nodes, ret = len(nodes), False
    apply_ux, apply_uy = [(__set_zero if geometry.is_zero(_) else __set_value)
                          if _ != None else __nop
                          for _ in [fixed_ux, fixed_uy]]
    for idx in targets:
        apply_ux(stiffness, load, idx, fixed_ux)
        apply_uy(stiffness, load, idx + n_nodes, fixed_uy)
        if not ret: ret = True
    return ret


def just_fixed_segment(stiffness, load, nodes, targets):
    return just_set_segment(stiffness, load, nodes, targets, 0, 0)


def just_set_segment_ux(stiffness, load, nodes, targets, fixed_ux):
    return just_set_segment(stiffness, load, nodes, targets, fixed_ux, None)


def just_set_segment_uy(stiffness, load, nodes, targets, fixed_uy):
    return just_set_segment(stiffness, load, nodes, targets, None, fixed_uy)


def just_set_segment_ux_by_lambda(stiffness, load, nodes, targets, fixed_ux):
    ret = False
    for idx in targets:
        __set_value(stiffness, load, idx, fixed_ux(*nodes[idx, :]))
        if not ret: ret = True
    return ret


def just_set_segment_uy_by_lambda(stiffness, load, nodes, targets, fixed_uy):
    n_nodes, ret = len(nodes), False
    for idx in targets:
        __set_value(stiffness, load, idx + n_nodes, fixed_uy(*nodes[idx, :]))
        if not ret: ret = True
    return ret


def complie_boundary(nodes, conditions):
    indices = []
    for cond in conditions:
        if cond[0] == "point":
            indices.append(point_setting(nodes, cond[3]))
        elif cond[0] == "segment":
            indices.append(segment_setting(nodes, cond[3]))
        else:
            print(f"Wrong boundary configs. No such entry named {cond[0]}")
            print("Compile Failed.")
    ret = [(config, index[0]) for config, index in zip(conditions, indices)
           if index[-1]]
    return ret


def apply_boundary(stiffness, loads, nodes, complied_boundary, scale=1.0):
    for cond, index in complied_boundary:
        # print(cond, index)
        if cond[0] == "point":
            if cond[1] == "fixed":
                just_fixed_point(stiffness, loads, nodes, index)
            elif cond[1] == "set_ux":
                if cond[2] == "constant":
                    just_set_point_ux(stiffness, loads, nodes, index,
                                      scale * cond[4])
                elif cond[2] == "lambda":
                    just_set_point_ux_by_lambda(
                        stiffness, loads, nodes,
                        index, lambda x, y: scale * cond[4](x, y))
            elif cond[1] == "set_uy":
                if cond[2] == "constant":
                    just_set_point_uy(stiffness, loads, nodes, index,
                                      scale * cond[4])
                elif cond[2] == "lambda":
                    just_set_point_uy_by_lambda(
                        stiffness, loads, nodes,
                        index, lambda x, y: scale * cond[4](x, y))
            else:
                print("Wrong boundary configs.")
                print("Apply Point Config Failed.")
        elif cond[0] == "segment":
            if cond[1] == "fixed":
                just_fixed_segment(stiffness, loads, nodes, index)
            elif cond[1] == "set_ux":
                if cond[2] == "constant":
                    just_set_segment_ux(stiffness, loads, nodes, index,
                                        scale * cond[4])
                elif cond[2] == "lambda":
                    just_set_segment_ux_by_lambda(
                        stiffness, loads, nodes,
                        index, lambda x, y: scale * cond[4](x, y))
            elif cond[1] == "set_uy":
                if cond[2] == "constant":
                    just_set_segment_uy(stiffness, loads, nodes, index,
                                        scale * cond[4])
                elif cond[2] == "lambda":
                    just_set_segment_uy_by_lambda(
                        stiffness, loads, nodes,
                        index, lambda x, y: scale * cond[4](x, y))
            else:
                print("Wrong boundary configs.")
                print("Apply Segment Config Failed.")
        else:
            print(f"Wrong boundary configs. No such entry named {cond[0]}")
            print("Apply Failed.")
