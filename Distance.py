import Configuration as cfg
import numpy as np


class Distance:

    def __init__(self, useAng, arg):
        px_durch_ang = cfg.get_px_per_angstrom()
        if (useAng):
            self.px = px_durch_ang * arg
            self.ang = arg
        else:
            self.px = arg
            self.ang = arg / px_durch_ang

    def __str__(self):
        return "{}px".format(self.px)

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
        return Distance(True, self.ang / other.ang)

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
            raise TypeError
        return self.ang < other.ang

    def __abs__(self):
        return Distance(True, abs(self.ang))

    def __gt__(self, other):
        if type(other) is not type(self):
            raise TypeError
        return self.ang > other.ang

    def __le__(self, other):
        if type(other) is not type(self):
            raise TypeError
        return self.ang <= other.ang

    def __ge__(self, other):
        if type(other) is not type(self):
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
        ret = []
        for elem in vec:
            if isinstance(elem, Distance):
                ret.append(elem.px)
            else:
                ret.append(elem)
        return np.array(ret)
