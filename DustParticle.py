import random

import matplotlib.pyplot as plt

from Distance import Distance
from Particle import Particle
import Configuration as cfg
import numpy as np

class DustParticle(Particle):

    def __init__(self, pos=None, color=None, size=30, maxArm=30, div=2.4):
        if pos is None:
            x = Distance(False, random.randint(0 - cfg.get_px_overlap(), cfg.get_width().px + cfg.get_px_overlap()))
            y = Distance(False, random.randint(0 - cfg.get_px_overlap(), cfg.get_height().px + cfg.get_px_overlap()))
            self.pos = np.array([x, y])
        else:
            self.pos = pos

        if color is None:
            self.color = random.randint(-25, 50)
        else:
            self.color = color

        super().__init__(self.pos[0], self.pos[1], 0)

        self.img_w = cfg.get_width()
        self.img_h = cfg.get_height()

        #effectRange etwa size/3
        #if size is None:
        #    self.size = random.random() * 40
        #else:
        self.size = size * random.random() / div


        self.arms = random.randint(1,5)

        self.circle_min = 1

        if maxArm is None:
            self.cirlce_max = 10
        else:
            self.cirlce_max = int(0.5*maxArm/div)

        self.arm_min = self.circle_min
        self.arm_max = self.cirlce_max





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





        for pos in self.relpos:
            self.circles.append(self.Circle(pos, self.circleRad()))

        max = 0
        for cir in self.circles:
            if (np.linalg.norm(cir.pos) + cir.rad) > max:
                max = (np.linalg.norm(cir.pos)) + cir.rad

        self.er = int(np.ceil(max))


    class Circle:
        def __init__(self, pos, rad):
            self.pos = pos
            self.rad = rad


    def poss(self):
        return (1/self.arms) * self.size / len(self.relpos)

    def circleRad(self):
        if self.circle_min >= self.cirlce_max:
            return self.circle_min
        try:
            return random.randint(self.circle_min, self.cirlce_max)
        except ValueError:
            print("Error for radnind({}, {})".format(self.circle_min, self.cirlce_max))

    def armLength(self):
        if self.arm_min >= self.arm_max:
            return self.arm_min
        try:
            return random.randint(self.arm_min, self.arm_max)
        except ValueError:
            print("Error for radnind({}, {})".format(self.arm_min, self.arm_max))

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
        #plt.imshow(eff_matrix)
        #plt.show()

        return eff_matrix, self.x, self.y

    def visualize_pixel(self, x, y):
        ret = 0
        for c in self.circles:
            if np.linalg.norm(c.pos - np.array([x, y])) < c.rad:
                ret += 1

        return ret * self.color

    @staticmethod
    def test():
        print("Abhängigkeit von Size:")
        divs = []
        abws = []
        for i in range(2320,2420, 5):
            divs.append(i/1000)

        for div in divs:
            print("DIV: {}".format(div))
            sizes = []
            ef = []
            for size in range(2,100):
                ranges = []
                for i in range(200):
                    d = DustParticle(np.array([200, 200]), size=size, maxArm=size, div=div)
                    ranges.append(d.er)
                sizes.append(size)
                ef.append(np.average(ranges))

            #plt.plot(sizes, ef)
            #plt.title("Dust_Size in Abhh. von Size = maxArm")
            #plt.show()
            dist = 0
            for i in range(len(sizes)):
                dist += np.square(sizes[i] - ef[i])
            abws.append(np.sqrt(dist))

        plt.plot(divs, abws)
        plt.title("Abweichung über divisor")
        plt.show()

        if False:
            print("Abhängigkeit von Armlänge:")

            sizes = []
            ef = []
            for size in range(2,100):
                ranges = []
                for i in range(100):
                    d = DustParticle(np.array([200, 200]), maxArm=size)
                    ranges.append(d.er)
                sizes.append(size)
                ef.append(np.average(ranges))

            plt.plot(sizes, ef)
            plt.title("Dust_Size in Abhh. von Armlänge")
            plt.show()





