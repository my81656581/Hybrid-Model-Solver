__all__ = ['Zone2d']


class Zone2d(object):
    def __init__(self, zone_xl, zone_xr, zone_yl, zone_yr):
        self.zone_xl = zone_xl
        self.zone_xr = zone_xr
        self.zone_yl = zone_yl
        self.zone_yr = zone_yr