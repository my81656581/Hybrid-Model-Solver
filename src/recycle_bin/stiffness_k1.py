
# def pd_constructive_core(xi, yi, xj, yj, coef_fun):
#     dx2, dy2 = (xi - xj) ** 2, (yi - yj) ** 2
#     ds2 = dx2 + dy2
#     pd_constructive_ = np.array([dx2 ** 2, dy2 ** 2, dx2 * dy2])
#     return coef_fun(ds2) * pd_constructive_ / 2.0


# def generate_stiffness_matrix_k1(nodes, elements, related, weight_handle, coef_fun, basis_config, jacobis):
#     print(6, nodes.shape, elements.shape)
#     w_, x_, y_ = gaussint.gauss_point_quadrature_standard()
#     n_elements, n_stiffsize, n_gauss = len(elements), basis_config.length, len(w_)
#     ret = np.zeros((n_elements, 2 * n_stiffsize, 2 * n_stiffsize))
#     xy_local = [basis_config.transform(x_, y_, nodes[elements[i, :], :], (0, 0)) for i in range(n_elements)]
#     det_jacobi = stiffness.preprocessing_all_jacobi_det(n_elements, n_gauss, jacobis)
#     # time counter init begin
#     flag, flag_0 = [0.07 * n_elements for _ in range(2)]
#     t0 = time.time()
#     # time counter init end
#     for i in range(n_elements):
#         # time counter runs begin
#         if i > flag:
#             flag = utils.display_progress(msg="build stiffness martix k1",
#                                           current=flag,
#                                           display_sep=flag_0,
#                                           current_id=i,
#                                           total=n_elements,
#                                           start_time=t0)
#         # time counter runs end
#         xi_local, yi_local = xy_local[i]
#         flag_i, weight_i = weight_handle(i)
#         pd_constructive_ij = np.array([0.0, 0.0, 0.0])
#         for j in related[i]:
#             xj_local, yj_local = xy_local[j]
#             flag_j, weight_j = weight_handle(j)
#             if flag_i == 0 and flag_j == 0: continue
#             for k in range(n_gauss):
#                 alpha_k = weight_i(xi_local[k], yi_local[k])
#                 for l in range(n_gauss):
#                     alpha_l = weight_j(xj_local[l], yj_local[l])
#                     # scale = (alpha_k - alpha_l) * 0.5 * w_[k] * w_[l] * det_jacobi[i][k] * det_jacobi[j][l]
#                     # because of w_[_] == 1
#                     scale = (alpha_k - alpha_l) * 0.5 * det_jacobi[i][k] * det_jacobi[j][l]
#                     pd_constructive_ij += scale * pd_constructive_core(xi_local[k], yi_local[k], xj_local[l], yj_local[l], coef_fun)
#         d_11 = np.vectorize(lambda x, y: pd_constructive_ij[0])
#         d_22 = np.vectorize(lambda x, y: pd_constructive_ij[1])
#         d_12 = np.vectorize(lambda x, y: pd_constructive_ij[2])
#         d_33 = np.vectorize(lambda x, y: pd_constructive_ij[2])
#         stiffness.generate_element_sitffness_matrix_base(
#             local_stiff=ret[i, :, :],
#             vertices=nodes[elements[i, :], :],
#             local_jacobi=jacobis[i],
#             n_stiffsize=n_stiffsize,
#             gauss_points=(w_, x_, y_),
#             constructive=(d_11, d_22, d_12, d_33),
#             basis_config=basis_config)
#     # time counter summary begin
#     tot = time.time() - t0
#     print(f"generating completed. Total {utils.formatting_time(tot)}")
#     # time counter summary end
#     return ret
