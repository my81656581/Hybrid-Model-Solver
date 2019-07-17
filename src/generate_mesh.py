import numpy as np
import sys
import utils


# @utils.log
def build_mesh(x_config, y_config):
    nx, ny = [int((_r - _l) / _d + 1) for _l, _r, _d in (x_config, y_config)]
    xv, yv = [
        np.linspace(_l, _r, _n)
        for (_l, _r, _), _n in zip([x_config, y_config], [nx, ny])
    ]
    mX, mY = np.meshgrid(xv, yv)
    p = [_ for _ in zip(mX.flat, mY.flat)]
    i = np.array(range(nx * ny)).reshape((ny, nx))
    e = [_.flat for _ in zip(i[:-1, :-1], i[:-1, 1:], i[1:, 1:], i[1:, :-1])]
    return p, e


# @utils.log
def save_mesh(filename: str, mesh):
    with open(filename, 'w') as fout:
        for _ in mesh:
            print(len(_), file=fout)
            [print("\t".join(map(str, __)), file=fout) for __ in _]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Output filename Needed.')
        exit()
    # p, e = build_mesh((0, 3.024e-6, 0.168e-6), (0, 2.016e-6, 0.112e-6))
    p, e = build_mesh((0, 3.024e-6, 0.084e-6), (0, 2.016e-6, 0.084e-6))
    save_mesh(sys.argv[1], (p, e))