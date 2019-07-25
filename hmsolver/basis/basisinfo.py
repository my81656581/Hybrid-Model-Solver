from collections import namedtuple

__all__ = ['BasisInfo', 'get_localnodes_by_id', 'get_zonetype_by_id']

BasisInfo = namedtuple('BasisInfo', ['id', 'localnodes', 'zonetype'])

__BASIS_MAP_ = {
    2301: BasisInfo(2301, 3, 'FETRIANGLE'),  # 二维三角形单元 3 节点
    # 2302: BasisInfo(2302, 6, 'FETRIANGLE'),  # 二维三角形单元 6 节点
    2401: BasisInfo(2401, 4, 'FEQUADRILATERAL'),  # 二维四边形单元 4 节点
    # 2402: BasisInfo(2402, 9, 'FEQUADRILATERAL'),  # 二维四边形单元 9 节点
}


def get_localnodes_by_id(e_basistype: int) -> int:
    return __BASIS_MAP_[e_basistype].localnodes


def get_zonetype_by_id(e_basistype: int) -> str:
    return __BASIS_MAP_[e_basistype].zonetype