import numpy as np
from typing import Dict
import sys

import hmsolver.utils as utils

from hmsolver.femcore.stiffness import preprocessing_jacobi
from hmsolver.geometry import xmult
from hmsolver.algebra import quadratic_equation

__all__ = [
    'get_absolute_displace', 'get_deform_mesh', 'get_strain_field',
    'get_stress_field', 'get_distortion_energy_density',
    'convert_distortion_energy_for_element',
    'maximum_distortion_energy_criterion', 'generate_tecplot_config',
    'export_tecplot_data'
]

__REFERENCE_VERTICES_ = np.array([[-1, -1], [+1, -1], [-1, +1], [+1, +1]])


def get_absolute_displace(displace_field):
    u_abs = (displace_field[:, 0]**2 + displace_field[:, 1]**2)**0.5
    print(f"{sys._getframe().f_code.co_name} done.")
    return u_abs


def get_deform_mesh(nodes, displace_field):
    node_deform = nodes + displace_field
    print(f"{sys._getframe().f_code.co_name} done.")
    return node_deform


def get_local_damage(nodes, elements, basis, related, connection):
    n_nodes, n_elements, n_basis = len(nodes), len(elements), basis.length
    n_gauss = connection.shape[-1]
    n_fullconnect = n_gauss**2
    node_damage = np.zeros(shape=(n_nodes, 1))
    elem_damage = np.zeros(shape=(n_elements, 1))
    frequent = np.zeros(shape=(n_nodes, 1))
    bond_cnt = np.sum(np.sum(connection, axis=3), axis=2)
    for i in range(n_elements):
        a, b, c, d = nodes[elements[i, :], :]
        aera = np.abs(xmult(b, a, c)) + np.abs(xmult(c, a, d))
        res, tot = 0, 0
        res = np.sum([bond_cnt[i, _] for _ in related[i]])
        tot = n_fullconnect * len(related[i])
        if tot == 0:
            elem_damage = 0
        else:
            elem_damage = 1 - res / tot
        for j in range(n_basis):
            node_index = elements[i, j]
            frequent[node_index] += aera
            node_damage[node_index] += aera * elem_damage
    node_damage /= frequent
    print(f"{sys._getframe().f_code.co_name} done.")
    return node_damage


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
    xg, yg = __REFERENCE_VERTICES_.T
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


def get_distortion_energy_density(stress_field, strain_field):
    # nu_epsilon: strain energy density
    # nu_v: dilatational strain energy density
    # nu_d: distortional strain energy density
    # average strain, epsilon_m := 1/3 (epsilon_1 + epsilon_2 + epsilon_3)
    # average stress,   sigma_m := 1/3 (sigma_1 + sigma_2 + sigma_3)
    # nu_epsilon == nu_v + nu_d
    # nu_epsilon := 1/2 * sigma_ij epsilon_ij
    #       nu_v := 3/2 * sigma_m epsilon_m
    # =>    nu_d := nu_epsilon - nu_v
    _1, _2, _3 = [stress_field[:, _] for _ in range(3)]
    _4, _5, _6 = [strain_field[:, _] for _ in range(3)]
    s1, s2 = quadratic_equation(1, -(_1 + _2), _1 * _2 - _3**2)
    e1, e2 = quadratic_equation(1, -(_4 + _5), _4 * _5 - _6**2)
    # nu_epsilon = 0.5 * (s1 * e1 + s2 * e2)
    # sigma_m, epsilon_m = (s1 + s2) / 3.0, (e1 + e2) / 3.0
    # nu_v = 1.5 * sigma_m * epsilon_m
    # nu_d = nu_epsilon - nu_v
    nu_d = (s1 * e1 + s2 * e2 + (s1 - s2) * (e1 - e2)) / 6.0
    print(f"{sys._getframe().f_code.co_name} done.")
    return nu_d


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
                            e_zonetype: str = "FEQUADRILATERAL"):
    return {
        "title": "Solution",
        "variables": "X, Y, U_x, U_y",
        "n_nodes": n_nodes,
        "n_elements": n_elements,
        "n_localnodes": n_localnodes,
        "e_zonetype": e_zonetype
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
        print(f"ZONE N= {config['n_nodes']}, ", end="", file=fout)
        print(f"E= {config['n_elements']}, ", end="", file=fout)
        print(f"DATAPACKING=POINT, ZONETYPE={config['e_zonetype']}", file=fout)
        for i in range(config["n_nodes"]):
            print(f"{nodes[i][0]:6f}\t{nodes[i][1]:6f}", end='', file=fout)
            for sol in solutions:
                for j in range(sol.shape[1]):
                    print('\t', sol[i][j], end='', sep='', file=fout)
            print(file=fout)
        for i in range(config["n_elements"]):
            for j in range(config["n_localnodes"]):
                print('\t', elements[i][j] + 1, end='', sep='', file=fout)
            print(file=fout)
