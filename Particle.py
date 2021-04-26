import math, random
import Configuration as cfg
import numpy as np

class Particle:

    def __init__(self):
        self.x = random.randint(0 - cfg.get_px_overlap(), cfg.get_width() + cfg.get_px_overlap())
        self.y = random.randint(0 - cfg.get_px_overlap(), cfg.get_height() + cfg.get_px_overlap())
        self.theta = 2 * math.pi * random.random()
        self.width = cfg.get_part_width()
        self.height = cfg.get_part_height()
        self.length = cfg.get_part_length()
        self.img_width = cfg.get_width()
        self.img_height = cfg.get_height()
        self.effect_range = 10 #ToDO: calculate

    def toMatrix(self):
        matrix = np.zeros((self.img_width, self.img_height))
        for i in range(0, self.img_width):
            for j in range(0, self.img_height):
                matrix[i, j] = 255 if math.dist((i, j), (self.x, self.y)) < 5 else 0 #ToDo: Stick

        return matrix

    def efficient_Matrix(self):
        eff_matrix = np.zeros((2 * self.effect_range, 2 * self.effect_range))
        for i in range(-self.effect_range, self.effect_range):
            for j in range(-self.effect_range, self.effect_range):
                eff_matrix[i + self.effect_range, j + self.effect_range] = \
                    255 if math.dist((i, j), (0, 0)) < 5 else 0 #ToDo: Stick

        return eff_matrix, self.x, self.y

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    @staticmethod
    def str_Header():
        return "x, y, theta, width, height, length\n"

    def __str__(self):
        args = [self.x, self.y, self.theta, self.width, self.height, self.length]
        return ", ".join(str(arg) for arg in args)
