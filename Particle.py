import math, random
import Configuration as cfg
import numpy as np
from Functions import turnMatrix
from PIL import Image



class Particle:

    def __init__(self, x=None, y=None, theta=None):
        if x is None:
            self.x = random.randint(0 - cfg.get_px_overlap(), cfg.get_width() + cfg.get_px_overlap())
            self.y = random.randint(0 - cfg.get_px_overlap(), cfg.get_height() + cfg.get_px_overlap())
            self.theta = 2 * math.pi * random.random()
        else:
            self.x = x
            self.y = y
            self.theta = theta
        self.width = cfg.get_part_width()
        self.height = cfg.get_part_height()
        self.length = cfg.get_part_length()
        self.img_width = cfg.get_width()
        self.img_height = cfg.get_height()
        self.effect_range = 10  # ToDO: calculate
        if len(cfg.get_image_path()) == 0:
            self.fromImage = False
        else:
            self.img = Image.open(cfg.get_image_path())
            self.pixels = self.img.convert("L").load()
            self.fromImage = True


    def toMatrix(self):
        matrix = np.zeros((self.img_width, self.img_height))
        for i in range(0, self.img_width):
            for j in range(0, self.img_height):
                matrix[i, j] = self.visualize(i - self.x, j - self.y)  # ToDo: Stick

        return matrix

    def efficient_Matrix(self):
        eff_matrix = np.zeros((2 * self.effect_range, 2 * self.effect_range))
        for i in range(-self.effect_range, self.effect_range):
            for j in range(-self.effect_range, self.effect_range):
                eff_matrix[i + self.effect_range, j + self.effect_range] = \
                    self.visualize(i, j)


        return eff_matrix, self.x, self.y

    def efficient_Matrix_turned(self):
        eff_mat, x, y = self.efficient_Matrix()
        eff_mat_turned, cx, cy = turnMatrix(eff_mat, self.theta)
        # print("x: {}, y:{}, x+cx: {}, y+cy: {}".format(x, y, x+cx, y+cy))
        # return eff_mat_turned, round(x + cx), round(y + cy)  # ToDo Check if x needs to be manipulated first
        return eff_mat_turned, x, y

    def visualize(self, x, y):
        #print(x, y)
        if not self.fromImage:
            return 255 if (abs(x) <= 1 and abs(y) <= 1) else 0
        else:
            cx = self.img.size[0] /2
            cy = self.img.size[1] /2
            try:
                return self.pixels[x + cx, y + cy]
            except IndexError:
                return 0

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
