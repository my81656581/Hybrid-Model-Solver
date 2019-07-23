from typing import List, Tuple

__all__ = ['read_mesh', 'convert_gmsh_into_msh']


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
    with open(export_file, 'w') as fout:
        for _ in (nodes, elements):
            print(len(_), file=fout)
            [print("\t".join(map(str, __)), file=fout) for __ in _]
    return n_nodes, n_elements
