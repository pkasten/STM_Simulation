import numpy as np
import functools

class Charge:

    def __init__(self, x, y, q):
        #Absolute Positions
        self.x = x
        self.y = y
        self.q = q
        self.maxpotential = 1
        self.r = 1

    @functools.lru_cache
    def calc_Potential(self, x, y):
        #if np.abs(self.x - x) > 100 or np.abs(self.y - y) > 100: #ToDo: Remove
        #    return 0
        if self.x == x and self.y == y:
            return self.q * self.maxpotential
        d = np.sqrt(np.square(self.x - x) + np.square(self.y - y))
        if d < self.r:
            return self.q * self.maxpotential
        return self.q / d

    def has_negative_index(self):
        return self.x < 0 or self.y < 0

