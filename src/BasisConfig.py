class BasisConfig(object):
    __n_localnodes_map_ = {
        2301: 3,  # 二维三角形单元 3 节点
        2302: 6,  # 二维三角形单元 6 节点
        2401: 4,  # 二维四边形单元 4 节点
        2402: 9,  # 二维四边形单元 9 节点
    }

    def __init__(self, e_basistype):
        self.e_basistype_ = e_basistype
        self.n_localnodes_ = self.__n_localnodes_map_[e_basistype]

    @property
    def e_basistype(self):
        return self.e_basistype_

    @property
    def n_localnodes(self):
        return self.n_localnodes_

    @staticmethod
    def localnodes(e_basistype: int) -> int:
        return BasisConfig.__n_localnodes_map_[e_basistype]


if __name__ == "__main__":
    for e_type in [2301, 2302, 2401, 2402]:
        print(BasisConfig.localnodes(e_type))