def preprocessing_jacobi_old_version(
        x_gauss, y_gauss, vertices, basis: reference_basis.Quadrilateral4Node):
    x_local, y_local = [vertices[:, _] for _ in (0, 1)]
    dndxg = np.hstack([_(x_gauss, y_gauss) for _ in basis.shapes_dx])
    dndyg = np.hstack([_(x_gauss, y_gauss) for _ in basis.shapes_dy])
    dxdxg = dndxg @ x_local
    dydxg = dndxg @ y_local
    dxdyg = dndyg @ x_local
    dydyg = dndyg @ y_local
    jacobis = np.array([
        np.array([[_11, _12], [_21, _22]])
        for _11, _12, _21, _22 in zip(dxdxg, dxdyg, dydxg, dydyg)
    ])
    # print(0, "vertices", vertices.shape, vertices)
    # print(0, "x_local", x_local.shape, x_local)
    # print(0, "y_local", y_local.shape, y_local)
    # print(0, "x_gauss", x_gauss.shape, x_gauss)
    # print(0, "y_gauss", y_gauss.shape, y_gauss)
    # print(0, "dndxg", dndxg.shape, dndxg)
    # print(0, "dndyg", dndyg.shape, dndyg)
    # print(0, "dxdxg", dxdxg.shape, dxdxg)
    # print(0, "dydxg", dydxg.shape, dydxg)
    # print(0, "dxdyg", dxdyg.shape, dxdyg)
    # print(0, "dydyg", dydyg.shape, dydyg)
    # print(0, "jacobis", jacobis.shape, jacobis)
    # jacobis.shape= (4, 2, 2), because of 4 gauss points
    return jacobis


def intgt(vertices, coef_fun, basis_indices, basis, local_jacobi, diff_orders):
    r, s, p, q = diff_orders
    ii, jj = basis_indices
    w_, x_, y_ = gaussint.gauss_point_quadrature_standard()
    # local_jacobi = preprocessing_jacobi(x_, y_, vertices, basis)
    phi_a = local_basis(x_, y_, vertices, local_jacobi, basis, ii, (r, s))
    phi_b = local_basis(x_, y_, vertices, local_jacobi, basis, jj, (p, q))
    # print(w_.shape, x_.shape, y_.shape)
    # print(2, phi_a.shape, phi_a)
    # print(2, phi_b.shape, phi_b)
    pxgpxpygpy = local_jacobi[:, 0, 0] * local_jacobi[:, 1, 1]
    pygpxpxgpy = local_jacobi[:, 0, 1] * local_jacobi[:, 1, 0]
    jcb = pxgpxpygpy - pygpxpxgpy
    return np.sum(w_ * jcb * coef_fun(x_, y_) * phi_a * phi_b)


def assemble_stiffness_matrix(nodes, elements, coef_fun, basis, jacobis,
                              diff_orders):
    print(4, nodes.shape, elements.shape)
    n_nodes, n_elements = len(nodes), len(elements)
    ret = np.zeros(shape=(n_nodes, n_nodes))
    # time counter init begin
    flag, flag_0 = [0.13 * n_elements for _ in range(2)]
    t0 = time.time()
    # time counter init end
    for i in range(n_elements):
        # time counter runs begin
        if i > flag:
            flag = utils.display_progress(msg="assemble stiffness martix",
                                          current=flag,
                                          display_sep=flag_0,
                                          current_id=i,
                                          total=n_elements,
                                          start_time=t0)
        # time counter runs end
        for ii in range(basis.length):
            for jj in range(basis.length):
                integral_r = intgt(vertices=nodes[elements[i, :], :],
                                   coef_fun=coef_fun,
                                   basis_indices=(ii, jj),
                                   basis=basis,
                                   local_jacobi=jacobis[i],
                                   diff_orders=diff_orders)
                iii, jjj = elements[i, ii], elements[i, jj]
                ret[iii, jjj] += integral_r
    # time counter summary begin
    tot = time.time() - t0
    print(f"assembling completed. Total {utils.formatting_time(tot)}")
    # time counter summary end
    return ret
