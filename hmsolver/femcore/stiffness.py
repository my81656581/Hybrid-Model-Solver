import numpy as np
import scipy.sparse as sp
import time

import hmsolver.utils as utils

from hmsolver.femcore.gaussint import gauss_point_quadrature_standard

__all__ = [
    'local_basis', 'element_sitffness_matrix', 'preprocessing_jacobi',
    'preprocessing_all_jacobi', 'preprocessing_all_jacobi_det',
    'generate_stiffness_matrix_k0', 'mapping_element_stiffness_matrix',
    'mapping_element_stiffness_matrix_with_weight'
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
        for _ in range(basis.length)
    ]
    pyg = [
        local_basis(x_, y_, vertices, local_jacobi, basis, _, (0, 1))
        for _ in range(basis.length)
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
