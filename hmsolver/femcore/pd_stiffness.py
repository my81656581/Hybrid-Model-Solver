import numpy as np
import scipy.sparse as sp
import time

import hmsolver.utils as utils

from hmsolver.femcore.gaussint import gauss_point_quadrature_standard
from hmsolver.femcore.stiffness import preprocessing_all_jacobi
from hmsolver.femcore.stiffness import preprocessing_all_jacobi_det
from hmsolver.femcore.stiffness import element_sitffness_matrix

__all__ = [
    'xi2', 'pd_constitutive_core', 'estimate_stiffness_matrix',
    'generate_stiffness_matrix_k1', 'assemble_stiffness_matrix',
    'assemble_stiffness_matrix_with_weight', 'deal_bond_stretch',
    'generate_stiffness_matrix_k1_with_connection',
    'assemble_stiffness_matrix_with_weight_and_connection'
]


@utils.SingletonDecorator
class FakeFullConnection(object):
    def __getitem__(self, _):
        return True


@utils.SingletonDecorator
class FakeWeightFunction(object):
    def __call__(self, _):
        return 1, np.vectorize(lambda x, y: 1)


def xi2(xi, yi, xj, yj, coef_fun):
    dx, dy = xi - xj, yi - yj
    dx2, dxdy, dy2 = dx * dx, dx * dy, dy * dy
    _ = coef_fun(dx, dy) * np.array([dx2, dy2, dxdy])
    return np.array([[_[0], _[2]], [_[2], _[1]]])


def pd_constitutive_core(xi, yi, xj, yj, coef_fun):
    dx, dy = xi - xj, yi - yj
    dx2, dy2 = dx**2, dy**2
    pd_constitutive_ = np.array([dx2**2, dy2**2, dx2 * dy2])
    return coef_fun(dx, dy) * pd_constitutive_


def estimate_stiffness_matrix(mesh, basis, coef_fun):
    nodes, elements, related = mesh.nodes, mesh.elements, mesh.related
    w_, x_, y_ = gauss_point_quadrature_standard()
    n_elements, n_gauss = len(elements), len(w_)
    jacobis = preprocessing_all_jacobi(nodes, elements, basis)
    xy_local = [
        basis.transform(x_, y_, nodes[elements[i, :], :], (0, 0))
        for i in range(n_elements)
    ]
    det_jcb = preprocessing_all_jacobi_det(n_elements, n_gauss, jacobis)
    kl = [(k, l) for k in range(n_gauss) for l in range(n_gauss)]
    i = n_elements // 2
    xi_local, yi_local = xy_local[i]
    pd_constitutive = np.array([0.0, 0.0, 0.0])
    for j in related[i]:
        xj_local, yj_local = xy_local[j]
        for k, l in kl:
            # scale = w_[k] * w_[l] * det_jcb[i][k] * det_jcb[j][l]
            # because of w_[_] == 1
            scale = det_jcb[i][k] * det_jcb[j][l]
            pd_constitutive += scale * pd_constitutive_core(
                xi_local[k], yi_local[k], xj_local[l], yj_local[l], coef_fun)
    return pd_constitutive


def generate_stiffness_matrix_k1(nodes, elements, related, weight_handle,
                                 coef_fun, basis, jacobis):
    return _generate_stiffness_matrix_k1_core(
        nodes, elements, related, FakeFullConnection(), weight_handle,
        coef_fun, basis, jacobis, "        build stiffness martix k1")


def generate_stiffness_matrix_k1_with_connection(nodes, elements, related,
                                                 connection, weight_handle,
                                                 coef_fun, basis, jacobis):
    return _generate_stiffness_matrix_k1_core(
        nodes, elements, related, connection, weight_handle, coef_fun, basis,
        jacobis, "        build stiffness martix k1 with connection")


def _generate_stiffness_matrix_k1_core(nodes, elements, related, connection,
                                       weight_handle, coef_fun, basis, jacobis,
                                       log_msg):
    w_, x_, y_ = gauss_point_quadrature_standard()
    n_elements, n_stiffsize, n_gauss = len(elements), basis.length, len(w_)
    ret = np.zeros((n_elements, 2 * n_stiffsize, 2 * n_stiffsize))
    kl = [(k, l) for k in range(n_gauss) for l in range(n_gauss)]
    xy_local = [
        basis.transform(x_, y_, nodes[elements[i, :], :], (0, 0))
        for i in range(n_elements)
    ]
    det_jcb = preprocessing_all_jacobi_det(n_elements, n_gauss, jacobis)
    # time counter init begin
    flag, flag_0 = [0.17 * n_elements for _ in range(2)]
    t0 = time.time()
    # time counter init end
    for i in range(n_elements):
        # time counter runs begin
        if i > flag:
            flag = utils.display_progress(msg=log_msg,
                                          current=flag,
                                          display_sep=flag_0,
                                          current_id=i,
                                          total=n_elements,
                                          start_time=t0)
        # time counter runs end
        xi_local, yi_local = xy_local[i]
        flag_i, weight_i = weight_handle(i)
        if flag_i:
            aks = np.array(
                [weight_i(xi_local[k], yi_local[k]) for k in range(n_gauss)])
        else:
            aks = np.zeros(shape=(n_gauss))
        pd_constitutive_ij = np.array([0.0, 0.0, 0.0])
        for j in related[i]:
            xj_local, yj_local = xy_local[j]
            flag_j, weight_j = weight_handle(j)
            if flag_i == 0 and flag_j == 0: continue
            als = np.array(
                [weight_j(xj_local[l], yj_local[l]) for l in range(n_gauss)])
            for k, l in kl:
                if not connection[i, j, k, l]: continue
                ak, al = aks[k], als[l]
                scale = (ak + al) / 2 * det_jcb[i][k] * det_jcb[j][l]
                pd_constitutive_ij += scale * pd_constitutive_core(
                    xi_local[k], yi_local[k], xj_local[l], yj_local[l],
                    coef_fun)
        d_11 = np.vectorize(lambda x, y: pd_constitutive_ij[0])
        d_22 = np.vectorize(lambda x, y: pd_constitutive_ij[1])
        d_12 = np.vectorize(lambda x, y: pd_constitutive_ij[2])
        d_33 = np.vectorize(lambda x, y: pd_constitutive_ij[2])
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


def assemble_stiffness_matrix(nodes, elements, related, coef_fun, basis):
    return _assemble_stiffness_matrix_core(
        nodes, elements, related, FakeFullConnection(), FakeWeightFunction(),
        coef_fun, basis, "        assembling PD-stiffness martix")


def assemble_stiffness_matrix_with_weight(nodes, elements, related,
                                          weight_handle, coef_fun, basis):
    return _assemble_stiffness_matrix_core(
        nodes, elements, related, FakeFullConnection(), weight_handle,
        coef_fun, basis, "        assembling PD-stiffness martix with weight")


def assemble_stiffness_matrix_with_weight_and_connection(
        nodes, elements, related, connection, weight_handle, coef_fun, basis):
    return _assemble_stiffness_matrix_core(
        nodes, elements, related, connection, weight_handle, coef_fun, basis,
        "        assembling PD-stiffness martix with weight & connection")


def _assemble_stiffness_matrix_core(nodes, elements, related, connection,
                                    weight_handle, coef_fun, basis, log_msg):
    w_, x_gauss, y_gauss = gauss_point_quadrature_standard()
    n_nodes, n_elements, n_gauss = len(nodes), len(elements), len(w_)
    n_stiffsize = basis.length
    dof = 2 * n_nodes
    ret = np.zeros(shape=(dof, dof))
    kl = [(k, l) for k in range(n_gauss) for l in range(n_gauss)]
    zero1row = np.zeros(shape=(1, n_stiffsize))
    xy_local = [
        basis.transform(x_gauss, y_gauss, nodes[elements[i, :], :], (0, 0))
        for i in range(n_elements)
    ]
    mapp = [
        np.reshape(np.hstack((elements[i, :], elements[i, :] + n_nodes)), (-1))
        for i in range(n_elements)
    ]
    jacobis = preprocessing_all_jacobi(nodes, elements, basis)
    det_jcb = preprocessing_all_jacobi_det(n_elements, n_gauss, jacobis)
    shape0s = [
        np.reshape(basis.shape_vector(x_gauss[_], y_gauss[_]),
                   (1, n_stiffsize)) for _ in range(n_gauss)
    ]
    shapes = [
        np.vstack((np.hstack(
            (shape0s[_], zero1row)), np.hstack((zero1row, shape0s[_]))))
        for _ in range(n_gauss)
    ]
    # time counter init begin
    flag, flag_0 = [0.17 * n_elements for _ in range(2)]
    t0 = time.time()
    # time counter init end
    for i in range(n_elements):
        # time counter runs begin
        if i > flag:
            flag = utils.display_progress(msg=log_msg,
                                          current=flag,
                                          display_sep=flag_0,
                                          current_id=int(i *
                                                         (2 - i / n_elements)),
                                          total=n_elements,
                                          start_time=t0)
        # time counter runs end
        xi_local, yi_local = xy_local[i]
        mapp_i = mapp[i]
        ii_col, ii_row = [
            np.reshape(_, (-1)) for _ in np.meshgrid(mapp_i, mapp_i)
        ]
        flag_i, weight_i = weight_handle(i)
        if flag_i:
            aks = np.array(
                [weight_i(xi_local[k], yi_local[k]) for k in range(n_gauss)])
        else:
            aks = np.zeros(shape=(n_gauss))
        for j in related[i]:
            if i > j: continue
            mapp_j = mapp[j]
            xj_local, yj_local = xy_local[j]
            jj_col, jj_row = [
                np.reshape(_, (-1)) for _ in np.meshgrid(mapp_j, mapp_j)
            ]
            stiff_ii, stiff_ij, stiff_jj = [
                np.zeros(shape=(2 * n_gauss, 2 * n_gauss)) for _ in range(3)
            ]
            flag_j, weight_j = weight_handle(j)
            if flag_i == 0 and flag_j == 0: continue
            als = np.array(
                [weight_j(xj_local[l], yj_local[l]) for l in range(n_gauss)])
            for k, l in kl:
                if not connection[i, j, k, l]: continue
                shape_k, shape_l = shapes[k], shapes[l]
                ak, al = aks[k], als[l]
                # scale = (ak + al) / 2 * w_[k] * w_[l] * det_jcb[i][k] * det_jcb[j][l]
                # because of w_[_] == 1
                scale = (ak + al) / 2 * det_jcb[i][k] * det_jcb[j][l]
                core = scale * xi2(xi_local[k], yi_local[k], xj_local[l],
                                   yj_local[l], coef_fun)
                stiff_ii += shape_k.T @ core @ shape_k
                stiff_ij += shape_k.T @ core @ shape_l
                stiff_jj += shape_l.T @ core @ shape_l
            ret[ii_row, ii_col] += np.reshape(stiff_ii, (-1))
            ret[ii_row, jj_col] -= np.reshape(stiff_ij, (-1))
            ret[jj_row, ii_col] -= np.reshape(stiff_ij.T, (-1))
            ret[jj_row, jj_col] += np.reshape(stiff_jj, (-1))
    # time counter summary begin
    tot = time.time() - t0
    print(f"        assembling completed. Total {utils.formatting_time(tot)}")
    # time counter summary end
    return sp.csr_matrix(ret)


def deal_bond_stretch(nodes,
                      elements,
                      related,
                      weight_handle,
                      connection,
                      displace_field,
                      old_stiffness,
                      coef_fun,
                      basis,
                      s_crit=1.1):
    w_, x_gauss, y_gauss = gauss_point_quadrature_standard()
    n_nodes, n_elements, n_gauss = len(nodes), len(elements), len(w_)
    n_stiffsize = basis.length
    old_stiffness = old_stiffness.todense()
    # kl = [(k, l) for k in range(n_gauss) for l in range(n_gauss)]
    zero1row = np.zeros(shape=(1, n_stiffsize))
    endpoints, broken_bond_cnt = [], 0
    xy_local = [
        basis.transform(x_gauss, y_gauss, nodes[elements[i, :], :], (0, 0))
        for i in range(n_elements)
    ]
    uxuy_local = [
        basis.transform(x_gauss, y_gauss, displace_field[elements[i, :], :],
                        (0, 0)) for i in range(n_elements)
    ]
    mapp = [
        np.reshape(np.hstack((elements[i, :], elements[i, :] + n_nodes)), (-1))
        for i in range(n_elements)
    ]
    jacobis = preprocessing_all_jacobi(nodes, elements, basis)
    det_jcb = preprocessing_all_jacobi_det(n_elements, n_gauss, jacobis)
    shape0s = [
        np.reshape(basis.shape_vector(x_gauss[_], y_gauss[_]),
                   (1, n_stiffsize)) for _ in range(n_gauss)
    ]
    shapes = [
        np.vstack((np.hstack(
            (shape0s[_], zero1row)), np.hstack((zero1row, shape0s[_]))))
        for _ in range(n_gauss)
    ]
    # time counter init begin
    flag, flag_0 = [0.17 * n_elements for _ in range(2)]
    t0 = time.time()
    # time counter init end
    stretch_max = 0
    for i in range(n_elements):
        # time counter runs begin
        if i > flag:
            flag = utils.display_progress(
                msg="        dealing with bond stretch",
                current=flag,
                display_sep=flag_0,
                current_id=int(i * (2 - i / n_elements)),
                total=n_elements,
                start_time=t0)
        # time counter runs end
        xi, yi = xy_local[i]
        uxi, uyi = uxuy_local[i]
        mapp_i = mapp[i]
        ii_col, ii_row = [
            np.reshape(_, (-1)) for _ in np.meshgrid(mapp_i, mapp_i)
        ]
        flag_i, weight_i = weight_handle(i)
        # if flag_i:
        # aks = np.array([weight_i(xi[k], yi[k]) for k in range(n_gauss)])
        # else:
        # aks = np.zeros(shape=(n_gauss))
        for j in related[i]:
            if i > j: continue
            bond_break = False
            xj, yj = xy_local[j]
            uxj, uyj = uxuy_local[j]
            mapp_j = mapp[j]
            jj_col, jj_row = [
                np.reshape(_, (-1)) for _ in np.meshgrid(mapp_j, mapp_j)
            ]
            # jj_col, ii_row = [np.reshape(_, (-1)) for _ in np.meshgrid(mapp[j], mapp_i)]
            stiff_ii, stiff_ij, stiff_jj = [
                np.zeros(shape=(2 * n_gauss, 2 * n_gauss)) for _ in range(3)
            ]
            # stiff_ii, stiff_ij = [np.zeros(shape=(2 * n_gauss, 2 * n_gauss)) for _ in range(2)]
            flag_j, weight_j = weight_handle(j)
            if flag_i == 0 and flag_j == 0: continue
            # als = np.array([weight_j(xj_local[l], yj_local[l]) for l in range(n_gauss)])
            for k in range(n_gauss):
                shape_k = shapes[k]
                ak = weight_i(xi[k], yi[k])
                for l in range(n_gauss):
                    if not connection[i, j, k, l]: continue
                    b_x, b_y = xi[k] - xj[l], yi[k] - yj[l]
                    b_nx, b_ny = uxi[k] - uxj[l] + b_x, uyi[k] - uyj[l] + b_y
                    b_abs = np.sqrt(b_x * b_x + b_y * b_y)
                    b_nabs = np.sqrt(b_nx * b_nx + b_ny * b_ny)
                    stretch_ijkl = (b_nabs - b_abs) / b_abs
                    stretch_max = max(stretch_max, stretch_ijkl)
                    if stretch_ijkl <= s_crit:
                        continue
                    shape_l = shapes[l]
                    al = weight_j(xj[l], yj[l])
                    # scale = (ak + al) / 2 * w_[k] * w_[l] * det_jcb[i][k] * det_jcb[j][l]
                    # because of w_[_] == 1
                    # scale = (ak + al) * 0.25
                    scale = (ak + al) / 2 * det_jcb[i][k] * det_jcb[j][l]
                    core = scale * xi2(xi[k], yi[k], xj[l], yj[l], coef_fun)
                    stiff_ii -= shape_k.T @ core @ shape_k
                    stiff_ij -= shape_k.T @ core @ shape_l
                    stiff_jj -= shape_l.T @ core @ shape_l
                    broken_bond_cnt += 1
                    bond_break = True
                    connection[i, j, k, l] = False
                    connection[j, i, l, k] = False
            if bond_break:
                endpoints.append(i)
                endpoints.append(j)
            old_stiffness[ii_row, ii_col] += np.reshape(stiff_ii, (-1))
            old_stiffness[ii_row, jj_col] -= np.reshape(stiff_ij, (-1))
            old_stiffness[jj_row, ii_col] -= np.reshape(stiff_ij.T, (-1))
            old_stiffness[jj_row, jj_col] += np.reshape(stiff_jj, (-1))
    # time counter summary begin
    tot = time.time() - t0
    print(
        f"        dealing with bond stretch completed. Total {utils.formatting_time(tot)}"
    )
    # time counter summary end
    if len(endpoints) > 0:
        endpoints = list(set(endpoints))
    print(f"        stretch_max={stretch_max}")
    return endpoints, broken_bond_cnt, sp.csr_matrix(old_stiffness), connection
