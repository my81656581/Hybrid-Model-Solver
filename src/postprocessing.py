import numpy as np
from typing import Callable, Dict, List, Tuple
import sys

import utils

import reference_basis


def get_absolute_displace(displace_field):
    u_abs = (displace_field[:, 0] ** 2 + displace_field[:, 1] ** 2) ** 0.5
    print(f"{sys._getframe().f_code.co_name} done.")
    return u_abs


def get_deform_mesh(nodes, displace_field):
    node_deform = nodes + displace_field
    print(f"{sys._getframe().f_code.co_name} done.")
    return node_deform


def get_strain_field(nodes, elements, basis_config, displace_field):
    strain_field = np.zeros((len(nodes), 3))
    frequent = np.zeros((len(nodes), 1))
    # for i in range(len(elements)):
    #     xi, yi = [nodes[elements[i, :], _] for _ in range(2)]
    #     ui, vi = [displace_field[elements[i, :], _] for _ in range(2)]
    #     for j in range(basis_config.length):
    #         node_index = elements[i, j]
    #         epsilon_x, epsilon_y, epsilon_xy = 0, 0, 0
    #         for k in range(basis_config.length):
    #             epsilon_x += basis_config.shapes_dx[k](xi[j], yi[j]) * ui[k]
    #             epsilon_y += basis_config.shapes_dy[k](xi[j], yi[j]) * vi[k]
    #             epsilon_xy += basis_config.shapes_dx[k](xi[j], yi[j]) * vi[k]
    #             epsilon_xy += basis_config.shapes_dy[k](xi[j], yi[j]) * ui[k]
    #         frequent[node_index] += 1
    #         strain_field[node_index, :] = [epsilon_x, epsilon_y, epsilon_xy]
    for i in range(len(elements)):
        xi, yi = [nodes[elements[i, :], _] for _ in range(2)]
        ui, vi = [displace_field[elements[i, :], _] for _ in range(2)]
        displace_i = displace_field[elements[i, :], :]
        for j in range(basis_config.length):
            epsilon_x, epsilon_xy = basis_config.transform(
                xi[j], yi[j], displace_i, (1, 0))
            epsilon_y, epsilon_yx = basis_config.transform(
                xi[j], yi[j], displace_i, (0, 1))
            node_index = elements[i, j]
            frequent[node_index] += 1
            strain_field[node_index, :] += [
                epsilon_x, epsilon_y, epsilon_xy + epsilon_yx
            ]
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
    distortion_energy = coefficient * (2 * a * a + 2 * b * b - 2 * a * b + 6 * c * c)
    print(f"{sys._getframe().f_code.co_name} done.")
    return distortion_energy


def convert_distortion_energy_for_element(distortion_energy, elements):
    w_element = np.array([np.mean(distortion_energy[elements[_, :]]) for _ in range(len(elements))])
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
            print(f"{nodes[i][0]:6f}", f"{nodes[i][1]:6f}", end='', sep='\t', file=fout)
            for sol in solutions:
                for j in range(sol.shape[1]):
                    print('\t', sol[i][j], end='', sep='', file=fout)
            print(file=fout)
        for i in range(config["n_elements"]):
            for j in range(config["n_localnodes"]):
                print('\t', elements[i][j] + 1, end='', sep='', file=fout)
            print(file=fout)
