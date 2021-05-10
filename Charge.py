import numpy as np

class Charge:

    def __init__(self, x, y, q):
        #Absolute Positions
        self.x = x
        self.y = y
        self.q = q
        self.maxpotential = 1
        self.r = 5

    def calc_Potential(self, x, y):
        if self.x == x and self.y == y:
            return self.q * self.maxpotential
        d = np.sqrt(np.square(self.x - x) + np.square(self.y - y))
        if d < self.r:
            return self.q * self.maxpotential
        return self.q / d

