import Configuration as cfg
import numpy as np


class Distance:
    """
    class Distance is used to save every measured length in the unit pixels and Angstrom
    Helps to use same parameters for calculations in Angstrom and Visualitation in pixeln
    """

    def __init__(self, useAng, arg, pxPerAng=None):
        """
        Initializes new Distance
        :param useAng: True: arg has the unit Angstrom, False: arg has the unit pixels
        :param arg: length in angstrom or pixel
        """
        if pxPerAng is None:
            px_durch_ang = cfg.get_px_per_angstrom()
        else:
            px_durch_ang = pxPerAng
        if (useAng):
            self.px = px_durch_ang * arg
            self.ang = arg
        else:
            self.px = arg
            self.ang = arg / px_durch_ang

    def __str__(self):
        return "{:.3f}A".format(self.ang)

    def __repr__(self):
        return "Distance({:.2f}px, {:.2f}Ang)".format(self.px, self.ang)

    def __mul__(self, other):
        if type(other) is not type(self):
            return Distance(True, self.ang * other)
        return Distance(True, self.ang * other.ang)

    def __rmul__(self, other):
        if type(other) is not type(self):
            return Distance(True, self.ang * other)
        return Distance(True, self.ang * other.ang)

    def __truediv__(self, other):
        if type(other) is not type(self):
            return Distance(True, self.ang / other)
        #return Distance(True, self.ang / other.ang)
        return self.ang / other.ang

    def __add__(self, other):
        if type(other) is not type(self):
            raise TypeError
        return Distance(True, self.ang + other.ang)

    def __sub__(self, other):
        if type(other) is not type(self):
            raise TypeError
        return Distance(True, self.ang - other.ang)

    def __lt__(self, other):
        if type(other) is not type(self):
            if other == 0:
                return self.px < 0
            raise TypeError
        return self.ang < other.ang

    def __abs__(self):
        return Distance(True, abs(self.ang))

    def __gt__(self, other):
        if type(other) is not type(self):
            if other == 0:
                return self.px > 0
            raise TypeError
        return self.ang > other.ang

    def __le__(self, other):
        if type(other) is not type(self):
            if other == 0:
                return self.px <= 0
            raise TypeError
        return self.ang <= other.ang

    def __ge__(self, other):
        if type(other) is not type(self):
            if other == 0:
                return self.px >= 0
            raise TypeError
        return self.ang >= other.ang

    def __eq__(self, other):
        if type(other) is not type(self):
            raise TypeError
        return self.ang == other.ang

    def __neg__(self):
        return Distance(True, -self.ang)

    @staticmethod
    def px_vec(vec):
        """
        translates a np.array from Angstrom to pixel
        :param vec:
        :return:
        """
        ret = []
        for elem in vec:
            if isinstance(elem, Distance):
                ret.append(elem.px)
            else:
                ret.append(elem)
        return np.array(ret)
