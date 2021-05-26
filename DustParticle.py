import random

from Distance import Distance
from Particle import Particle
import Configuration as cfg
import numpy as np

class DustParticle(Particle):

    def __init__(self, pos=None):
        if pos is None:
            x = Distance(False, random.randint(0 - cfg.get_px_overlap(), cfg.get_width().px + cfg.get_px_overlap()))
            y = Distance(False, random.randint(0 - cfg.get_px_overlap(), cfg.get_height().px + cfg.get_px_overlap()))
            self.pos = np.array([x, y])
        else:
            self.pos = pos

        super().__init__(self.pos[0], self.pos[1], 0)

        self.img_w = cfg.get_width()
        self.img_h = cfg.get_height()

        self.size = random.random() * 40
        self.arms = random.randint(1,5)

        self.circle_min = 1
        self.cirlce_max = 10

        self.arm_min = 1
        self.arm_max = 10





        self.relpos = []
        knots = []
        self.circles = []



        center = np.array([0, 0])
        for i in range(self.arms):
            a = self.angle()
            vers = self.armLength() * np.array([np.cos(a), np.sin(a)])
            knots.append(center + vers)
            self.relpos.append(center+vers)

        for knot in knots:
            if(random.random() < self.poss()):
                a = self.angle()
                vers = self.armLength() * np.array([np.cos(a), np.sin(a)])
                knots.append(knot + vers)
                self.relpos.append(knot + vers)

        max = 0
        for pos in self.relpos:
            if (np.linalg.norm(pos)) > max:
                max = (np.linalg.norm(pos))

        self.er = max

        max = 0
        for pos in self.relpos:
            if (np.linalg.norm(pos)) > max:
                max = (np.linalg.norm(pos))

        self.effect_range = max


        for pos in self.relpos:
            self.circles.append(self.Circle(pos, self.circleRad()))


    class Circle:
        def __init__(self, pos, rad):
            self.pos = pos
            self.rad = rad


    def poss(self):
        return (1/self.arms) * self.size / len(self.relpos)

    def circleRad(self):
        return random.randint(self.circle_min, self.cirlce_max)

    def armLength(self):
        return random.randint(self.arm_min, self.arm_max)

    def angle(self):
        return 2 * np.pi * random.random()

    def efficient_Matrix_turned(self):
        return self.efficient_Matrix()

    def efficient_Matrix(self):



        eff_matrix = np.zeros((2 * self.er, 2 * self.er))
        for i in range(-1 * self.er, 1 * self.er):
            for j in range(-1 * self.er, 1 * self.er):
                eff_matrix[i + 1 * self.er, j + 1 * self.er] = \
                    self.visualize_pixel(i, j)

        # for atom in self.atoms:
        #    print("Atom pos: {}".format(atom.abspos))
        return eff_matrix, self.x, self.y

    def visualize_pixel(self, x, y):
        #color = random.randint(-50, 50)
        color = random.randint(50, 255)
        ret = 0
        for c in self.circles:
            if np.linalg.norm(c.pos - np.array([x, y])) < c.rad:
                ret += 1

        return ret * color


