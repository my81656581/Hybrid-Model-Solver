import numpy as np

__all__ = ['build_mesh', 'save_mesh']


def build_mesh(x_config, y_config):
    # example: p, e = build_mesh((0, 1, 0.02), (0, 1, 0.02))
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


def save_mesh(filename: str, mesh):
    # example: save_mesh(f'data_{len(p)}_{len(e)}.msh', (p, e))
    with open(filename, 'w') as fout:
        for _ in mesh:
            print(len(_), file=fout)
            [print("\t".join(map(str, __)), file=fout) for __ in _]
