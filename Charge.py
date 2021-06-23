import numpy as np
import functools

class Charge:
    """
    DEPRECATED. Used as Charges when electric interactions for potential and energy minimization have been used
    """

    def __init__(self, x, y, q):
        """
        DEPRECATED. Initializes new Charge
        :param x: absloute position
        :param y: absolute position
        :param q: charge
        """
        #Absolute Positions
        self.x = x
        self.y = y
        self.q = q
        self.maxpotential = 1
        self.r = 1

    @functools.lru_cache
    def calc_Potential(self, x, y):
        """
        DEPRECATED. Calculates Potential impact on position xy
        :param x: positon
        :param y: position
        :return: potential
        """
        if np.abs(self.x - x) > 50 or np.abs(self.y - y) > 50: #ToDo: Remove
            return 0
        if self.x == x and self.y == y:
            return self.q * self.maxpotential
        d = np.sqrt(np.square(self.x - x) + np.square(self.y - y))
        if d < self.r:
            return self.q * self.maxpotential
        return self.q / d

    def has_negative_index(self):
        """
        Returns true if x or y position is negative
        :return:
        """
        return self.x < 0 or self.y < 0

