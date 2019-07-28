from collections import namedtuple

__all__ = ['ProneRing2d']

ProneRing2d = namedtuple('ProneRing2d',
                         ['x', 'y', 'inner_radius', 'outer_radius'])
