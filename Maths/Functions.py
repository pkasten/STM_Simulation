import numpy as np

class Functions:
    @staticmethod
    def dist2D(a, b):
        return np.sqrt(np.square(a[0] - b[0]) + np.square(a[1] - b[1]))

    @staticmethod
    def normalDistribution(center, point, sigma):
        d = Functions.dist2D(center, point)
        # return np.power(2 * np.pi * np.square(sigma), -0.5) * np.exp(-(np.square(d)) / (2 * np.square(sigma)))
        return np.exp(-(np.square(d)) / (2 * np.square(sigma)))
