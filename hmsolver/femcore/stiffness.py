import numpy as np
import scipy.sparse as sp
import time

import hmsolver.utils as utils

from hmsolver.femcore.gaussint import gauss_point_linear_standard
from hmsolver.femcore.gaussint import gauss_point_quadrature_standard

__all__ = [
    'local_basis', 'element_sitffness_matrix', 'preprocessing_jacobi',
    'preprocessing_all_jacobi', 'preprocessing_all_jacobi_det',
    'generate_stiffness_matrix_k0', 'mapping_element_stiffness_matrix',
    'mapping_element_stiffness_matrix_with_weight',
    'generate_body_load_vector', 'generate_boundary_load_vector'
]


def local_basis(x_gauss, y_gauss, vertices, local_jacobi: np.ndarray, basis,
                basis_idx, diff_order):
    dx, dy = diff_order
    assert dx >= 0 & dy >= 0
    # jacobi = [[pxpxg, pxpyg],
    #           [pypxg, pypyg]]
    # wanted = [[pxgpx, pygpx],
    #           [pxgpy, pygpy]]
    # helper = jacobi^{-1} = [[pxgpx, pxgpy], [pygpx, pygpy]]
    # pApBg := \frac{\partial A}{\partial B_{gauss}}
    # pAgpB := \frac{\partial A_{gauss}}{\partial B}
    helper = np.array(
        [np.linalg.inv(local_jacobi[_, :, :]) for _ in range(len(x_gauss))])
    if dx == 0 and dy == 0:
        return basis.shapes[basis_idx](x_gauss, y_gauss)
    elif dx == 1 and dy == 0:
        hoge = helper[:, 0, 0] * basis.shapes_dx[basis_idx](x_gauss, y_gauss).T
        piyo = helper[:, 1, 0] * basis.shapes_dy[basis_idx](x_gauss, y_gauss).T
        return (hoge + piyo)
    elif dx == 0 and dy == 1:
        hoge = helper[:, 0, 1] * basis.shapes_dx[basis_idx](x_gauss, y_gauss)
        piyo = helper[:, 1, 1] * basis.shapes_dy[basis_idx](x_gauss, y_gauss)
        return (hoge + piyo)
    elif dx == 1 and dy == 1:
        pxgpx = helper[:, 0, 0]
        pygpx = helper[:, 0, 1]
        pxgpy = helper[:, 1, 0]
        pygpy = helper[:, 1, 1]
        hoge = pxgpy * pxgpx * basis.highorder[basis_idx](x_gauss, y_gauss)
        piyo = (pxgpy * pygpx + pxgpx * pygpy) * basis.shapes_dxdy[basis_idx](
            x_gauss, y_gauss)
        fuga = pygpy * pygpx * basis.highorder[basis_idx](x_gauss, y_gauss)
        return (hoge + piyo + fuga)
    else:
        return basis.highorder[basis_idx](x_gauss, y_gauss)


def preprocessing_jacobi(x_gauss, y_gauss, vertices, basis):
    dxdxg, dydxg = basis.transform(x_gauss, y_gauss, vertices, (1, 0))
    dxdyg, dydyg = basis.transform(x_gauss, y_gauss, vertices, (0, 1))
    dxdxg, dxdyg, dydxg, dydyg = [
        _.reshape(-1) for _ in [dxdxg, dxdyg, dydxg, dydyg]
    ]
    jacobis = np.array([
        np.array([[_11, _12], [_21, _22]])
        for _11, _12, _21, _22 in zip(dxdxg, dxdyg, dydxg, dydyg)
    ])
    # jacobis.shape= (4, 2, 2), because of 4 gauss points
    return jacobis


def preprocessing_all_jacobi(nodes, elements, basis):
    x_gauss, y_gauss = gauss_point_quadrature_standard()[1:]
    n_elements = len(elements)
    return [
        preprocessing_jacobi(x_gauss, y_gauss, nodes[elements[_, :], :], basis)
        for _ in range(n_elements)
    ]


def preprocessing_all_jacobi_det(n_elements, n_gauss, jacobis):
    det_jacobi = np.zeros(shape=(n_elements, n_gauss))
    for i in range(n_elements):
        det_jacobi[i, :] = jacobis[i][:, 0, 0] * jacobis[i][:, 1, 1] - jacobis[
            i][:, 0, 1] * jacobis[i][:, 1, 0]
    return det_jacobi


def element_sitffness_matrix(local_stiff, vertices, local_jacobi, n_stiffsize,
                             gauss_points, constitutive, basis):
    # elastic matrix D := [[d_11, d_12,    0],
    #                      [d_21, d_22,    0],
    #                      [   0,    0, d_33]]
    # constitutive := [d_11, d_22, d_12, d_33]
    w_, x_, y_ = gauss_points
    d_11, d_22, d_12, d_33 = constitutive
    pxgpxpygpy = local_jacobi[:, 0, 0] * local_jacobi[:, 1, 1]
    pygpxpxgpy = local_jacobi[:, 0, 1] * local_jacobi[:, 1, 0]
    w_jcb = w_ * (pxgpxpygpy - pygpxpxgpy)
    pxg = [
        local_basis(x_, y_, vertices, local_jacobi, basis, _, (1, 0))
        for _ in range(n_stiffsize)
    ]
    pyg = [
        local_basis(x_, y_, vertices, local_jacobi, basis, _, (0, 1))
        for _ in range(n_stiffsize)
    ]
    for ii in range(n_stiffsize):
        for jj in range(n_stiffsize):
            pxgipxgj = pxg[ii] * pxg[jj]
            pygipygj = pyg[ii] * pyg[jj]
            pxgipygj = pxg[ii] * pyg[jj]
            pygipxgj = pyg[ii] * pxg[jj]
            a1 = np.sum(w_jcb * d_11(x_, y_) * pxgipxgj)
            a3 = np.sum(w_jcb * d_12(x_, y_) * pxgipygj)
            a5 = np.sum(w_jcb * d_12(x_, y_) * pygipxgj)
            a7 = np.sum(w_jcb * d_22(x_, y_) * pygipygj)
            a2 = np.sum(w_jcb * d_33(x_, y_) * pygipygj)
            a4 = np.sum(w_jcb * d_33(x_, y_) * pygipxgj)
            a6 = np.sum(w_jcb * d_33(x_, y_) * pxgipygj)
            a8 = np.sum(w_jcb * d_33(x_, y_) * pxgipxgj)
            local_stiff[ii, jj] = a1 + a2  # k11
            local_stiff[ii, jj + n_stiffsize] = a3 + a4  # k12
            local_stiff[ii + n_stiffsize, jj] = a5 + a6  # k21
            local_stiff[ii + n_stiffsize, jj + n_stiffsize] = a7 + a8  # k22
    return True


def generate_stiffness_matrix_k0(nodes, elements, constitutive, basis,
                                 jacobis):
    w_, x_, y_ = gauss_point_quadrature_standard()
    n_elements, n_stiffsize = len(elements), basis.length
    ret = np.zeros((n_elements, 2 * n_stiffsize, 2 * n_stiffsize))
    c11, c12 = constitutive[0, 0], constitutive[0, 1]
    c22, shear_modulus = constitutive[1, 1], constitutive[2, 2]
    d_11 = np.vectorize(lambda x, y: c11)
    d_22 = np.vectorize(lambda x, y: c22)
    d_12 = np.vectorize(lambda x, y: c12)
    d_33 = np.vectorize(lambda x, y: shear_modulus)
    # time counter init begin
    flag, flag_0 = [0.17 * n_elements for _ in range(2)]
    t0 = time.time()
    # time counter init end
    for i in range(n_elements):
        # time counter runs begin
        if i > flag:
            flag = utils.display_progress(
                msg="        build stiffness martix k0",
                current=flag,
                display_sep=flag_0,
                current_id=i,
                total=n_elements,
                start_time=t0)
        # time counter runs end
        element_sitffness_matrix(local_stiff=ret[i, :, :],
                                 vertices=nodes[elements[i, :], :],
                                 local_jacobi=jacobis[i],
                                 n_stiffsize=n_stiffsize,
                                 gauss_points=(w_, x_, y_),
                                 constitutive=(d_11, d_22, d_12, d_33),
                                 basis=basis)
    # time counter summary begin
    tot = time.time() - t0
    print(f"        generating completed. Total {utils.formatting_time(tot)}")
    # time counter summary end
    return ret


# Body force density := rho(mass density) * a(accelerate)
def generate_body_load_vector(nodes, elements, load, basis, jacobis):
    w_, x_, y_ = gauss_point_quadrature_standard()
    n_nodes, n_elements, n_stiffsize = len(nodes), len(elements), basis.length
    dof = 2 * n_nodes
    _is, _js, _ks = [], [], []
    load_xvalue, load_yvalue = load
    load_x = np.vectorize(lambda x, y: load_xvalue)
    load_y = np.vectorize(lambda x, y: load_yvalue)
    # time counter init begin
    flag, flag_0 = [0.17 * n_elements for _ in range(2)]
    t0 = time.time()
    # time counter init end
    for i in range(n_elements):
        # time counter runs begin
        if i > flag:
            flag = utils.display_progress(
                msg="        generate boundary vector b_body",
                current=flag,
                display_sep=flag_0,
                current_id=i,
                total=n_elements,
                start_time=t0)
        # time counter runs end
        vertices = nodes[elements[i, :], :]
        local_jacobi = jacobis[i]
        pxgpxpygpy = local_jacobi[:, 0, 0] * local_jacobi[:, 1, 1]
        pygpxpxgpy = local_jacobi[:, 0, 1] * local_jacobi[:, 1, 0]
        w_jcb = w_ * (pxgpxpygpy - pygpxpxgpy)
        vg = [
            local_basis(x_, y_, vertices, local_jacobi, basis, _, (0, 0))
            for _ in range(n_stiffsize)
        ]
        for j in range(n_stiffsize):
            idx = elements[i, j]
            integral_rx = np.sum(w_jcb * load_x(x_, y_) * vg[j])
            integral_ry = np.sum(w_jcb * load_y(x_, y_) * vg[j])
            _is.extend([idx, idx + n_nodes])
            _js.extend([0, 0])
            _ks.extend([integral_rx, integral_ry])
    ret = sp.coo_matrix((np.array(_ks), (np.array(_is), np.array(_js))),
                        shape=(dof, 1)).tocsr()
    # time counter summary begin
    tot = time.time() - t0
    print(f"        generating completed. Total {utils.formatting_time(tot)}")
    # time counter summary end
    return ret


# face force density := pressure(force density)
def generate_boundary_load_vector(nodes, elements, adjoint, load_config, basis,
                                  jacobis):
    w_, x_gauss = gauss_point_linear_standard()
    x_gauss2, y_gauss2 = gauss_point_quadrature_standard()[1:]
    x_Lfixed = np.array([-1, -1]).reshape((-1, 1))
    x_Rfixed = np.array([1, 1]).reshape((-1, 1))
    n_nodes, n_stiffsize = len(nodes), basis.length
    dof = 2 * n_nodes
    _is, _js, _ks = [], [], []
    load_cond, affected_nodes = load_config
    b_criteria = load_cond.criteria
    load_xvalue, load_yvalue = load_cond.func.value
    if load_cond.func.method == "constant":
        load_x = np.vectorize(lambda x, y: load_xvalue)
        load_y = np.vectorize(lambda x, y: load_yvalue)
    elif load_cond.func.method == "rule":
        load_x, load_y = load_xvalue, load_yvalue
    affected_nodes_set = set(affected_nodes)
    real_affected_elements = set()
    for idx in affected_nodes:
        for affected_elem in adjoint[idx]:
            if affected_elem in real_affected_elements:
                continue
            cnt = sum([
                1 for _ in elements[affected_elem, :]
                if _ in affected_nodes_set
            ])
            if cnt == 2:
                real_affected_elements.add(affected_elem)
    workload = len(real_affected_elements)
    print(f"This face load affects {workload} elements.")
    # time counter init begin
    flag, flag_0 = [0.17 * workload for _ in range(2)]
    t0 = time.time()
    # time counter init end
    for _i_, i in enumerate(real_affected_elements):
        # time counter runs begin
        if _i_ > flag:
            flag = utils.display_progress(
                msg="        generate boundary vector b_boundary",
                current=flag,
                display_sep=flag_0,
                current_id=_i_,
                total=workload,
                start_time=t0)
        # time counter runs end
        element_nodes = elements[i, :]
        vertices = nodes[element_nodes, :]
        local_jacobi = jacobis[i]
        dxdxg, dydxg = basis.transform(x_gauss2, y_gauss2, vertices, (1, 0))
        dxdyg, dydyg = basis.transform(x_gauss2, y_gauss2, vertices, (0, 1))
        dXG = np.sqrt(dxdxg * dxdxg + dydxg * dydxg)
        dYG = np.sqrt(dxdyg * dxdyg + dydyg * dydyg)
        which_line = [
            j for j, idx in enumerate(element_nodes)
            if b_criteria(*nodes[idx, :])
        ]
        line_identification = which_line[0] * 10 + which_line[1]
        if line_identification == 1:
            # line: 0-1 <=> \eta = -1 <=> button
            x_real, y_real, w_dA = x_gauss, x_Lfixed, w_ * dXG
        elif line_identification == 12:
            # line: 1-2 <=> \xi = 1 <=> right
            x_real, y_real, w_dA = x_Rfixed, x_gauss, w_ * dYG
        elif line_identification == 23:
            # line: 2-3 <=> \eta = 1 <=> top
            x_real, y_real, w_dA = x_gauss, x_Rfixed, w_ * dXG
        elif line_identification == 3:
            # line: 0-3 <=> \xi = -1 <=> left
            x_real, y_real, w_dA = x_Lfixed, x_gauss, w_ * dYG
        else:
            # should be error
            x_real, y_real, w_dA = x_gauss, x_gauss, w_
        vg = [
            local_basis(x_real, y_real, vertices, local_jacobi, basis, _,
                        (0, 0)) for _ in range(n_stiffsize)
        ]
        for j in range(n_stiffsize):
            idx = elements[i, j]
            integral_rx = np.sum(w_dA * load_x(x_real, y_real) * vg[j])
            integral_ry = np.sum(w_dA * load_y(x_real, y_real) * vg[j])
            _is.extend([idx, idx + n_nodes])
            _js.extend([0, 0])
            _ks.extend([integral_rx, integral_ry])
    ret = sp.coo_matrix((np.array(_ks), (np.array(_is), np.array(_js))),
                        shape=(dof, 1)).tocsr()
    # time counter summary begin
    tot = time.time() - t0
    print(f"        generating completed. Total {utils.formatting_time(tot)}")
    # time counter summary end
    return ret


def mapping_element_stiffness_matrix(nodes, elements, basis, ks):
    n_nodes, n_elements, n_stiffsize = len(nodes), len(elements), basis.length
    dof = 2 * n_nodes
    _is, _js, _ks = [], [], []
    ij = [(i, j) for i in range(n_stiffsize) for j in range(n_stiffsize)]
    # time counter init begin
    flag, flag_0 = [0.17 * n_elements for _ in range(2)]
    t0 = time.time()
    # time counter init end
    for i in range(n_elements):
        # time counter runs begin
        if i > flag:
            flag = utils.display_progress(
                msg="        mapping stiffness martix",
                current=flag,
                display_sep=flag_0,
                current_id=i,
                total=n_elements,
                start_time=t0)
        # time counter runs end
        ki = ks[i, :, :]
        for ii, jj in ij:
            _i, _j = elements[i, ii], elements[i, jj]
            _is.extend([_i, _i, _i + n_nodes, _i + n_nodes])
            _js.extend([_j, _j + n_nodes, _j, _j + n_nodes])
            _ks.extend([
                ki[ii, jj], ki[ii, jj + n_stiffsize], ki[ii + n_stiffsize, jj],
                ki[ii + n_stiffsize, jj + n_stiffsize]
            ])
    ret = sp.coo_matrix((np.array(_ks), (np.array(_is), np.array(_js))),
                        shape=(dof, dof)).tocsr()
    # time counter summary begin
    tot = time.time() - t0
    print(f"        assembling completed. Total {utils.formatting_time(tot)}")
    # time counter summary end
    return ret


def mapping_element_stiffness_matrix_with_weight(nodes, elements, centers,
                                                 weight_handle, basis, ks):
    n_nodes, n_elements, n_stiffsize = len(nodes), len(elements), basis.length
    dof = 2 * n_nodes
    _is, _js, _ks = [], [], []
    ij = [(i, j) for i in range(n_stiffsize) for j in range(n_stiffsize)]
    # time counter init begin
    flag, flag_0 = [0.17 * n_elements for _ in range(2)]
    t0 = time.time()
    # time counter init end
    for i in range(n_elements):
        # time counter runs begin
        if i > flag:
            flag = utils.display_progress(
                msg="        mapping stiffness martix with weight",
                current=flag,
                display_sep=flag_0,
                current_id=i,
                total=n_elements,
                start_time=t0)
        # time counter runs end
        flag_i, weight_i = weight_handle(i)
        if flag_i == 1: continue
        di = (1 - weight_i(*centers[i])) * ks[i, :, :]
        for ii, jj in ij:
            _i, _j = elements[i, ii], elements[i, jj]
            _is.extend([_i, _i, _i + n_nodes, _i + n_nodes])
            _js.extend([_j, _j + n_nodes, _j, _j + n_nodes])
            _ks.extend([
                di[ii, jj], di[ii, jj + n_stiffsize], di[ii + n_stiffsize, jj],
                di[ii + n_stiffsize, jj + n_stiffsize]
            ])
    ret = sp.coo_matrix((np.array(_ks), (np.array(_is), np.array(_js))),
                        shape=(dof, dof)).tocsr()
    # time counter summary begin
    tot = time.time() - t0
    print(
        f"        assembling with weight completed. Total {utils.formatting_time(tot)}"
    )
    # time counter summary end
    return ret
