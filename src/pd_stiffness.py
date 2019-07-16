import numpy as np
import time

import utils

import gaussint
import stiffness
import reference_basis


def core_xi2(xi, yi, xj, yj, coef_fun):
    # dx, dy = xi - xj, yi - yj
    # ds2 = dx**2 + dy**2
    # dx2, dxdy, dy2 = dx * dx, dx * dy, dy * dy
    # core_ = np.array([[dx2, dxdy], [dxdy, dy2]])
    # return coef_fun(ds2) * core_
    dx, dy = xi - xj, yi - yj
    dx2, dxdy, dy2 = dx * dx, dx * dy, dy * dy
    _ = coef_fun(dx, dy) * np.array([dx2, dy2, dxdy])
    return np.array([[_[0], _[2]], [_[2], _[1]]])





def pd_constitutive_core(xi, yi, xj, yj, coef_fun):
    # dx2, dy2 = (xi - xj)**2, (yi - yj)**2
    # ds2 = dx2 + dy2
    # pd_constitutive_ = np.array([dx2**2, dy2**2, dx2 * dy2])
    # return coef_fun(ds2) * pd_constitutive_
    dx, dy = xi - xj, yi - yj
    dx2, dy2 = dx**2, dy**2
    pd_constitutive_ = np.array([dx2**2, dy2**2, dx2 * dy2])
    return coef_fun(dx, dy) * pd_constitutive_


def pd_constitutive_core_est(xi, yi, xj, yj, coef_fun):
    dx, dy = xi - xj, yi - yj
    dx2, dy2 = dx**2, dy**2
    pd_constitutive_ = np.array([dx2**2, dy2**2, dx2 * dy2])
    return coef_fun(dx, dy) * pd_constitutive_


def estimate_stiffness_matrix_isotropic(mesh, coef_fun):
    nodes, elements, related = mesh.nodes, mesh.elements, mesh.related
    w_, x_, y_ = gaussint.gauss_point_quadrature_standard()
    basis = reference_basis.Quadrilateral4Node()
    n_elements, n_gauss = len(elements), len(w_)
    jacobis = stiffness.preprocessing_all_jacobi(nodes, elements, basis)
    xy_local = [
        basis.transform(x_, y_, nodes[elements[i, :], :], (0, 0))
        for i in range(n_elements)
    ]
    det_jacobi = stiffness.preprocessing_all_jacobi_det(
        n_elements, n_gauss, jacobis)
    i = n_elements // 2
    xi_local, yi_local = xy_local[i]
    pd_constitutive_ij = np.array([0.0, 0.0, 0.0])
    for j in related[i]:
        xj_local, yj_local = xy_local[j]
        for k in range(n_gauss):
            for l in range(n_gauss):
                # scale = w_[k] * w_[l] * det_jacobi[i][k] * det_jacobi[j][l]
                # because of w_[_] == 1
                scale = det_jacobi[i][k] * det_jacobi[j][l]
                pd_constitutive_ij += scale * pd_constitutive_core_est(
                    xi_local[k], yi_local[k], xj_local[l], yj_local[l],
                    coef_fun)
    return pd_constitutive_ij


def estimate_stiffness_matrix(nodes, elements, related, coef_fun, basis,
                              jacobis, material):
    w_, x_, y_ = gaussint.gauss_point_quadrature_standard()
    n_elements, n_gauss = len(elements), len(w_)
    xy_local = [
        basis.transform(x_, y_, nodes[elements[i, :], :], (0, 0))
        for i in range(n_elements)
    ]
    det_jacobi = stiffness.preprocessing_all_jacobi_det(
        n_elements, n_gauss, jacobis)
    # time counter init begin
    t0 = time.time()
    # time counter init end
    i = (n_elements - 1) // 2
    xi_local, yi_local = xy_local[i]
    pd_constitutive_ij = np.array([0.0, 0.0, 0.0])
    for j in related[i]:
        xj_local, yj_local = xy_local[j]
        for k in range(n_gauss):
            for l in range(n_gauss):
                # scale = w_[k] * w_[l] * det_jacobi[i][k] * det_jacobi[j][l]
                # because of w_[_] == 1
                scale = det_jacobi[i][k] * det_jacobi[j][l]
                pd_constitutive_ij += scale * pd_constitutive_core(
                    xi_local[k], yi_local[k], xj_local[l], yj_local[l],
                    coef_fun)
    pd_constitutive_ij /= material.grid_vol
    print("i=", i)
    print("vertex_i=", nodes[elements[i, :], :])
    print("center_i=", np.mean(nodes[elements[i, :], :], 0))
    print("constitutive=", (pd_constitutive_ij[0], pd_constitutive_ij[1],
                            pd_constitutive_ij[2], pd_constitutive_ij[2]))
    material.coefficients[
        0] = material.constitutive[0, 0] / pd_constitutive_ij[0]
    # time counter summary begin
    tot = time.time() - t0
    print(f"        estimate completed. Total {utils.formatting_time(tot)}")
    # time counter summary end
    print(material.coefficients)
    return material.coefficients[0]


def generate_stiffness_matrix_k1(nodes, elements, related, weight_handle,
                                 coef_fun, basis, jacobis):
    w_, x_, y_ = gaussint.gauss_point_quadrature_standard()
    n_elements, n_stiffsize, n_gauss = len(elements), basis.length, len(w_)
    ret = np.zeros((n_elements, 2 * n_stiffsize, 2 * n_stiffsize))
    xy_local = [
        basis.transform(x_, y_, nodes[elements[i, :], :], (0, 0))
        for i in range(n_elements)
    ]
    det_jacobi = stiffness.preprocessing_all_jacobi_det(
        n_elements, n_gauss, jacobis)
    # time counter init begin
    flag, flag_0 = [0.17 * n_elements for _ in range(2)]
    t0 = time.time()
    # time counter init end
    for i in range(n_elements):
        # time counter runs begin
        if i > flag:
            flag = utils.display_progress(
                msg="        build stiffness martix k1",
                current=flag,
                display_sep=flag_0,
                current_id=i,
                total=n_elements,
                start_time=t0)
        # time counter runs end
        xi_local, yi_local = xy_local[i]
        flag_i, weight_i = weight_handle(i)
        pd_constitutive_ij = np.array([0.0, 0.0, 0.0])
        for j in related[i]:
            xj_local, yj_local = xy_local[j]
            flag_j, weight_j = weight_handle(j)
            if flag_i == 0 and flag_j == 0: continue
            for k in range(n_gauss):
                alpha_k = weight_i(xi_local[k], yi_local[k])
                for l in range(n_gauss):
                    alpha_l = weight_j(xj_local[l], yj_local[l])
                    # scale = (alpha_k - alpha_l) * 0.5 * w_[k] * w_[l] * det_jacobi[i][k] * det_jacobi[j][l]
                    # because of w_[_] == 1
                    scale = (alpha_k + alpha_l
                             ) * 0.5 * det_jacobi[i][k] * det_jacobi[j][l]
                    pd_constitutive_ij += scale * pd_constitutive_core(
                        xi_local[k], yi_local[k], xj_local[l], yj_local[l],
                        coef_fun)
        d_11 = np.vectorize(lambda x, y: pd_constitutive_ij[0])
        d_22 = np.vectorize(lambda x, y: pd_constitutive_ij[1])
        d_12 = np.vectorize(lambda x, y: pd_constitutive_ij[2])
        d_33 = np.vectorize(lambda x, y: pd_constitutive_ij[2])
        # if weight_i(*np.mean(nodes[elements[i, :], :], 0)) > 0.99:
        #     print("i=", i)
        #     print("vertex_i=", nodes[elements[i, :], :])
        #     print("center_i=", np.mean(nodes[elements[i, :], :], 0))
        #     print("aphla(i)=", weight_i(*np.mean(nodes[elements[i, :], :], 0)))
        #     print("constitutive=", (pd_constitutive_ij[0], pd_constitutive_ij[1], pd_constitutive_ij[2], pd_constitutive_ij[2]))
        stiffness.generate_element_sitffness_matrix_base(
            local_stiff=ret[i, :, :],
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
    w_, x_gauss, y_gauss = gaussint.gauss_point_quadrature_standard()
    n_nodes, n_elements, n_gauss = len(nodes), len(elements), len(w_)
    ret = np.zeros(shape=(2 * n_nodes, 2 * n_nodes))
    zero1x4 = np.zeros(shape=(1, 4))
    xy_local = [
        basis.transform(x_gauss, y_gauss, nodes[elements[i, :], :], (0, 0))
        for i in range(n_elements)
    ]
    mapping = [
        np.reshape(np.hstack((elements[i, :], elements[i, :] + n_nodes)), (-1))
        for i in range(n_elements)
    ]
    jacobis = stiffness.preprocessing_all_jacobi(nodes, elements, basis)
    det_jacobi = stiffness.preprocessing_all_jacobi_det(
        n_elements, n_gauss, jacobis)
    shape0s = [
        np.reshape(basis.shape_vector(x_gauss[_], y_gauss[_]), (1, 4))
        for _ in range(n_gauss)
    ]
    shapes = [
        np.vstack((np.hstack(
            (shape0s[_], zero1x4)), np.hstack((zero1x4, shape0s[_]))))
        for _ in range(n_gauss)
    ]
    # time counter init begin
    flag, flag_0 = [0.17 * n_elements for _ in range(2)]
    t0 = time.time()
    # time counter init end
    for i in range(n_elements):
        # time counter runs begin
        if i > flag:
            flag = utils.display_progress(
                msg="        assembling stiffness martix pd",
                current=flag,
                display_sep=flag_0,
                current_id=i,
                total=n_elements,
                start_time=t0)
        # time counter runs end
        xi_local, yi_local = xy_local[i]
        mapping_i = mapping[i]
        ii_col, ii_row = [
            np.reshape(_, (-1)) for _ in np.meshgrid(mapping_i, mapping_i)
        ]
        # print("xi_local", xi_local)
        # print("yi_local", yi_local)
        # print("mapping[i]", mapping[i])
        for j in related[i]:
            xj_local, yj_local = xy_local[j]
            jj_col, ii_row = [
                np.reshape(_, (-1))
                for _ in np.meshgrid(mapping[j], mapping_i)
            ]
            # print("xj_local", xj_local)
            # print("yj_local", yj_local)
            # print("mapping[j]", mapping[j])
            stiff_ii, stiff_ij = [
                np.zeros(shape=(2 * n_gauss, 2 * n_gauss)) for _ in range(2)
            ]
            for k in range(n_gauss):
                shape_k = shapes[k]
                for l in range(n_gauss):
                    shape_l = shapes[l]
                    # scale = w_[k] * w_[l] * det_jacobi[i][k] * det_jacobi[j][l]
                    scale = det_jacobi[i][k] * det_jacobi[j][l]
                    # scale = 1
                    # because of w_[_] == 1
                    core = scale * core_xi2(xi_local[k], yi_local[k],
                                            xj_local[l], yj_local[l], coef_fun)
                    stiff_ii += shape_k.T @ core @ shape_k
                    stiff_ij += shape_k.T @ core @ shape_l
                    # code without optim
                    # stiff_kk = shape_k.T @ core @ shape_k
                    # stiff_kl = shape_k.T @ core @ shape_l
                    # stiff_lk = shape_l.T @ core @ shape_k
                    # stiff_ll = shape_l.T @ core @ shape_l
                    # ii_col, jj_row = [np.reshape(_, (-1)) for _ in np.meshgrid(mapping_i, mapping_j)]
                    # jj_col, ii_row = [np.reshape(_, (-1)) for _ in np.meshgrid(mapping_j, mapping_i)]
                    # ret[ii_row, ii_col] += np.reshape(stiff_kk, (-1))
                    # ret[ii_row, jj_col] -= np.reshape(stiff_kl, (-1))
                    # ret[jj_row, ii_col] -= np.reshape(stiff_lk, (-1))
                    # ret[jj_row, jj_col] += np.reshape(stiff_ll, (-1))
                    # print(f"bond ({i} -> {j}): gauss point pair ({k}) - ({l})")
                    # print("core", core)
                    # print("shape_k", shape_k)
                    # print("shape_l", shape_l)
                    # print("stiff_kk", stiff_kk)
                    # print("stiff_kl", stiff_kl)
                    # print("stiff_lk", stiff_lk)
                    # print("stiff_ll", stiff_ll)
                    # input()
            ret[ii_row, ii_col] += np.reshape(stiff_ii, (-1))
            ret[ii_row, jj_col] -= np.reshape(stiff_ij, (-1))
            # print(stiff_ii)
            # input()
    # time counter summary begin
    tot = time.time() - t0
    print(f"        assembling completed. Total {utils.formatting_time(tot)}")
    # time counter summary end
    return ret
    # return ret / 2


def assemble_stiffness_matrix_with_weight(nodes, elements, related,
                                          weight_handle, coef_fun, basis):
    w_, x_gauss, y_gauss = gaussint.gauss_point_quadrature_standard()
    n_nodes, n_elements, n_gauss = len(nodes), len(elements), len(w_)
    ret = np.zeros(shape=(2 * n_nodes, 2 * n_nodes))
    zero1x4 = np.zeros(shape=(1, 4))
    xy_local = [
        basis.transform(x_gauss, y_gauss, nodes[elements[i, :], :], (0, 0))
        for i in range(n_elements)
    ]
    mapping = [
        np.reshape(np.hstack((elements[i, :], elements[i, :] + n_nodes)), (-1))
        for i in range(n_elements)
    ]
    jacobis = stiffness.preprocessing_all_jacobi(nodes, elements, basis)
    det_jacobi = stiffness.preprocessing_all_jacobi_det(
        n_elements, n_gauss, jacobis)
    shape0s = [
        np.reshape(basis.shape_vector(x_gauss[_], y_gauss[_]), (1, 4))
        for _ in range(n_gauss)
    ]
    shapes = [
        np.vstack((np.hstack(
            (shape0s[_], zero1x4)), np.hstack((zero1x4, shape0s[_]))))
        for _ in range(n_gauss)
    ]
    # time counter init begin
    flag, flag_0 = [0.17 * n_elements for _ in range(2)]
    t0 = time.time()
    # time counter init end
    for i in range(n_elements):
        # time counter runs begin
        if i > flag:
            flag = utils.display_progress(
                msg="        assembling stiffness martix pd",
                current=flag,
                display_sep=flag_0,
                current_id=i,
                total=n_elements,
                start_time=t0)
        # time counter runs end
        xi_local, yi_local = xy_local[i]
        mapping_i = mapping[i]
        ii_col, ii_row = [
            np.reshape(_, (-1)) for _ in np.meshgrid(mapping_i, mapping_i)
        ]
        flag_i, weight_i = weight_handle(i)
        # debug_flag = True
        for j in related[i]:
            xj_local, yj_local = xy_local[j]
            jj_col, ii_row = [
                np.reshape(_, (-1))
                for _ in np.meshgrid(mapping[j], mapping_i)
            ]
            stiff_ii, stiff_ij = [
                np.zeros(shape=(2 * n_gauss, 2 * n_gauss)) for _ in range(2)
            ]
            flag_j, weight_j = weight_handle(j)
            if flag_i == 0 and flag_j == 0: continue
            for k in range(n_gauss):
                shape_k = shapes[k]
                alpha_k = weight_i(xi_local[k], yi_local[k])
                # if debug_flag:
                #     debug_flag = False
                #     print("i=", i)
                #     print("j=", j)
                #     print("det_jacobi[i]=", det_jacobi[i])
                #     print("det_jacobi[j]=", det_jacobi[j])
                #     print("(xi_local, yi_local)=", (xi_local, yi_local))
                #     print("(xj_local, yj_local)=", (xj_local, yj_local))
                for l in range(n_gauss):
                    shape_l = shapes[l]
                    alpha_l = weight_j(xj_local[l], yj_local[l])
                    # scale = (alpha_k + alpha_l) * 0.5 * w_[k] * w_[l] * det_jacobi[i][k] * det_jacobi[j][l]
                    # because of w_[_] == 1
                    # scale = (alpha_k + alpha_l) * 0.5
                    scale = (alpha_k + alpha_l
                             ) * 0.5 * det_jacobi[i][k] * det_jacobi[j][l]
                    core = scale * core_xi2(xi_local[k], yi_local[k],
                                            xj_local[l], yj_local[l], coef_fun)
                    stiff_ii += shape_k.T @ core @ shape_k
                    stiff_ij += shape_k.T @ core @ shape_l
            ret[ii_row, ii_col] += np.reshape(stiff_ii, (-1))
            ret[ii_row, jj_col] -= np.reshape(stiff_ij, (-1))
    # time counter summary begin
    tot = time.time() - t0
    print(f"        assembling completed. Total {utils.formatting_time(tot)}")
    # time counter summary end
    return ret


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
    w_, x_gauss, y_gauss = gaussint.gauss_point_quadrature_standard()
    n_nodes, n_elements, n_gauss = len(nodes), len(elements), len(w_)
    zero1x4 = np.zeros(shape=(1, 4))
    endpoints, broken_bond_cnt = [], 0
    xy_local = [
        basis.transform(x_gauss, y_gauss, nodes[elements[i, :], :], (0, 0))
        for i in range(n_elements)
    ]
    uxuy_local = [
        basis.transform(x_gauss, y_gauss, displace_field[elements[i, :], :],
                        (0, 0)) for i in range(n_elements)
    ]
    mapping = [
        np.reshape(np.hstack((elements[i, :], elements[i, :] + n_nodes)), (-1))
        for i in range(n_elements)
    ]
    jacobis = stiffness.preprocessing_all_jacobi(nodes, elements, basis)
    det_jacobi = stiffness.preprocessing_all_jacobi_det(
        n_elements, n_gauss, jacobis)
    shape0s = [
        np.reshape(basis.shape_vector(x_gauss[_], y_gauss[_]), (1, 4))
        for _ in range(n_gauss)
    ]
    shapes = [
        np.vstack((np.hstack(
            (shape0s[_], zero1x4)), np.hstack((zero1x4, shape0s[_]))))
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
                current_id=i,
                total=n_elements,
                start_time=t0)
        # time counter runs end
        xi_local, yi_local = xy_local[i]
        uxi_local, uyi_local = uxuy_local[i]
        mapping_i = mapping[i]
        ii_col, ii_row = [
            np.reshape(_, (-1)) for _ in np.meshgrid(mapping_i, mapping_i)
        ]
        flag_i, weight_i = weight_handle(i)
        for j in related[i]:
            bond_break = False
            xj_local, yj_local = xy_local[j]
            uxj_local, uyj_local = uxuy_local[j]
            mapping_j = mapping[j]
            jj_col, jj_row = [
                np.reshape(_, (-1)) for _ in np.meshgrid(mapping_j, mapping_j)
            ]
            # jj_col, ii_row = [np.reshape(_, (-1)) for _ in np.meshgrid(mapping[j], mapping_i)]
            stiff_ii, stiff_ij, stiff_jj = [
                np.zeros(shape=(2 * n_gauss, 2 * n_gauss)) for _ in range(3)
            ]
            # stiff_ii, stiff_ij = [np.zeros(shape=(2 * n_gauss, 2 * n_gauss)) for _ in range(2)]
            flag_j, weight_j = weight_handle(j)
            if flag_i == 0 and flag_j == 0: continue
            for k in range(n_gauss):
                shape_k = shapes[k]
                alpha_k = weight_i(xi_local[k], yi_local[k])
                for l in range(n_gauss):
                    if not connection[i, j, k, l]: continue
                    xi_old_x, xi_old_y = xi_local[k] - xj_local[l], yi_local[
                        k] - yj_local[l]
                    xi_new_x, xi_new_y = uxi_local[k] - uxj_local[
                        l] + xi_old_x, uyi_local[k] - uyj_local[l] + xi_old_y
                    xi_old_abs = np.sqrt(xi_old_x * xi_old_x +
                                         xi_old_y * xi_old_y)
                    xi_new_abs = np.sqrt(xi_new_x * xi_new_x +
                                         xi_new_y * xi_new_y)
                    stretch_ijkl = (xi_new_abs - xi_old_abs) / xi_old_abs
                    stretch_max = max(stretch_max, stretch_ijkl)
                    if stretch_ijkl <= s_crit:
                        continue
                    # print(f"            element ({i}, {j}) bond ({k}, {l}): stretch= {stretch_ijkl:.2f}, break!!")
                    shape_l = shapes[l]
                    alpha_l = weight_j(xj_local[l], yj_local[l])
                    # scale = (alpha_k + alpha_l) * 0.5 * w_[k] * w_[l] * det_jacobi[i][k] * det_jacobi[j][l]
                    # because of w_[_] == 1
                    # scale = (alpha_k + alpha_l) * 0.25
                    scale = (
                        alpha_k + alpha_l
                    ) * 0.5 * det_jacobi[i][k] * det_jacobi[j][l] * 0.5
                    core = scale * core_xi2(xi_local[k], yi_local[k],
                                            xj_local[l], yj_local[l], coef_fun)
                    stiff_ii -= shape_k.T @ core @ shape_k
                    stiff_ij -= shape_k.T @ core @ shape_l
                    stiff_jj -= shape_l.T @ core @ shape_l
                    broken_bond_cnt += 1
                    connection[i, j, k, l] = False
                    # connection[j, i, l, k] = False
                    bond_break = True
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
    return endpoints, broken_bond_cnt
