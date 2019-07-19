import numpy as np
from typing import Callable, Dict, List, Tuple
import sys

import utils

import reference_basis
from stiffness import preprocessing_jacobi
from geometry import xmult

REFERENCE_VERTICES = np.array([[-1, -1], [+1, -1], [-1, +1], [+1, +1]])


def get_absolute_displace(displace_field):
    u_abs = (displace_field[:, 0]**2 + displace_field[:, 1]**2)**0.5
    print(f"{sys._getframe().f_code.co_name} done.")
    return u_abs


def get_deform_mesh(nodes, displace_field):
    node_deform = nodes + displace_field
    print(f"{sys._getframe().f_code.co_name} done.")
    return node_deform


def get_strain_field(nodes, elements, basis, displace_field):
    # jacobi = [[pxpxg, pxpyg],
    #           [pypxg, pypyg]]
    # wanted = [[pxgpx, pygpx],
    #           [pxgpy, pygpy]]
    # helper = jacobi^{-1} = [[pxgpx, pxgpy], [pygpx, pygpy]]
    # pApBg := \frac{\partial A}{\partial B_{gauss}}
    # pAgpB := \frac{\partial A_{gauss}}{\partial B}
    n_nodes, n_elements, n_basis = len(nodes), len(elements), basis.length
    strain_field = np.zeros(shape=(n_nodes, 3))
    frequent = np.zeros(shape=(n_nodes, 1))
    xg, yg = REFERENCE_VERTICES.T
    xg, yg = [np.reshape(_, (-1, 1)) for _ in [xg, yg]]
    for i in range(n_elements):
        jacobi = preprocessing_jacobi(xg, yg, nodes[elements[i, :], :], basis)
        helper = [np.linalg.inv(jacobi[_, :, :]) for _ in range(n_basis)]
        a, b, c, d = nodes[elements[i, :], :]
        displace_i = displace_field[elements[i, :], :]
        aera = np.abs(xmult(b, a, c)) + np.abs(xmult(c, a, d))
        for j in range(n_basis):
            node_index = elements[i, j]
            uxg, vxg = basis.transform(xg[j], yg[j], displace_i, (1, 0))
            uyg, vyg = basis.transform(xg[j], yg[j], displace_i, (0, 1))
            ex = helper[j][0, 0] * uxg + helper[j][1, 0] * uyg
            ey = helper[j][0, 1] * vxg + helper[j][1, 1] * vyg
            exy = helper[j][0, 0] * vxg + helper[j][1, 0] * vyg
            eyx = helper[j][0, 1] * uxg + helper[j][1, 1] * uyg
            frequent[node_index] += aera
            strain_field[node_index, :] += aera * np.array([ex, ey, exy + eyx])
    strain_field /= frequent
    print(f"{sys._getframe().f_code.co_name} done.")
    return strain_field


def get_stress_field(constructive, strain_field):
    stress_field = constructive @ strain_field.T
    print(f"{sys._getframe().f_code.co_name} done.")
    return stress_field.T


def get_distortion_energy(youngs_modulus, poissons_ratio, stress_field):
    # Distortion Energy
    # W = \frac{1 + \mu}{6E} \bigl[ (sigma_1 - sigma_2)^2 + (sigma_2 - sigma_3)^2 + (sigma_3 - sigma_1)^2 \bigr]
    # w= 2 * a^2 + 2 * b^2 - 2 * a * b + 6 * c^2
    a, b, c = [stress_field[:, _] for _ in range(3)]
    coefficient = (1 + poissons_ratio) / 6.0 / youngs_modulus
    distortion_energy = coefficient * (2 * a**2 + 2 * b**2 - 2 * a * b +
                                       6 * c**2)
    print(f"{sys._getframe().f_code.co_name} done.")
    return distortion_energy


def convert_distortion_energy_for_element(distortion_energy, elements):
    w_element = np.array([
        np.mean(distortion_energy[elements[_, :]])
        for _ in range(len(elements))
    ])
    print(f"{sys._getframe().f_code.co_name} done.")
    return w_element


def maximum_distortion_energy_criterion(distortion_energy, w_max):
    w_distortion_index = np.reshape(np.where(distortion_energy > w_max), (-1))
    print(f"{sys._getframe().f_code.co_name} done.")
    return w_distortion_index


def generate_tecplot_config(n_nodes: int,
                            n_elements: int,
                            n_localnodes: int = 4,
                            e_type: str = "QUADRILATERAL"):
    return {
        "title": "Solution",
        "variables": "X, Y, U_x, U_y",
        "n_nodes": n_nodes,
        "n_elements": n_elements,
        "n_localnodes": n_localnodes,
        "e_type": e_type
    }


def export_tecplot_data(export: str, config: Dict, nodes, elements,
                        *raw_solutions):
    solutions = [
        utils.refine_vector(_) if len(_.shape) == 1 else _
        for _ in raw_solutions
    ]
    with open(export, 'w') as fout:
        print(f"TITLE= {config['title']}", file=fout)
        print(f"VARIABLES= {config['variables']}", file=fout)
        print(
            f"ZONE N= {config['n_nodes']}, E= {config['n_elements']}, F= FEPOINT, ET= {config['e_type']}",
            file=fout)
        for i in range(config["n_nodes"]):
            print(f"{nodes[i][0]:6f}",
                  f"{nodes[i][1]:6f}",
                  end='',
                  sep='\t',
                  file=fout)
            for sol in solutions:
                for j in range(sol.shape[1]):
                    print('\t', sol[i][j], end='', sep='', file=fout)
            print(file=fout)
        for i in range(config["n_elements"]):
            for j in range(config["n_localnodes"]):
                print('\t', elements[i][j] + 1, end='', sep='', file=fout)
            print(file=fout)
