import numpy as np
from typing import List, Tuple

__all__ = ['read_mesh', 'read_mesh_to_MeshObject', 'convert_gmsh_into_msh']


def read_mesh(mesh_data_file: str) -> Tuple[int, int, List, List]:
    n_nodes, n_elements = 0, 0
    nodes, elements = [], []
    with open(mesh_data_file, 'r') as fin:
        n_nodes = int(fin.readline())
        for _ in range(n_nodes):
            nodes.append(list(map(float, fin.readline().split())))
        n_elements = int(fin.readline())
        for _ in range(n_elements):
            elements.append(list(map(int, fin.readline().split())))
    return n_nodes, n_elements, nodes, elements


def read_mesh_to_MeshObject(mesh_data_file: str, mesh_type):
    n_nodes, n_elements, nodes, elements = read_mesh(mesh_data_file)
    mesh = mesh_type(n_nodes, n_elements)
    mesh.manually_construct(np.array(nodes), np.array(elements))
    return mesh


def convert_gmsh_into_msh(gmsh_data_file: str, export_file: str):
    n_nodes, n_elements = 0, 0
    nodes, elements = [], []
    with open(gmsh_data_file, 'r') as fin:
        while True:
            line = fin.readline().strip()
            if line == "$Nodes":
                break
        n_nodes = int(fin.readline())
        for _ in range(n_nodes):
            hoge = list(map(float, fin.readline().strip().split()))
            nodes.append(hoge[1:-1])
        while True:
            line = fin.readline().strip()
            if line == "$Elements":
                break
        n_elements = int(fin.readline())
        for _ in range(n_elements):
            hoge = list(map(int, fin.readline().strip().split()))
            if hoge[1] != 3:
                continue
            elements.append([_ - 1 for _ in hoge[-4:]])
        n_elements = len(elements)
    visit = [False for i in range(n_nodes)]
    for element in elements:
        for node in element:
            visit[node] = True
    nodes_without_toolmen, toolmen = [], []
    for node, flag, idx in zip(nodes, visit, range(n_nodes)):
        if flag:
            nodes_without_toolmen.append(node)
        else:
            toolmen.append(idx)
    toolmen = sorted(toolmen, reverse=True)
    for i in range(len(elements)):
        for j in range(len(element)):
            current = elements[i][j]
            for toolman in toolmen:
                current -= 1 if current > toolman else 0
            elements[i][j] = current
    n_nodes = len(nodes_without_toolmen)
    with open(export_file, 'w') as fout:
        for _ in (nodes_without_toolmen, elements):
            print(len(_), file=fout)
            [print("\t".join(map(str, __)), file=fout) for __ in _]
    return n_nodes, n_elements
