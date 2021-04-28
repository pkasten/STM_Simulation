import math, random
import Configuration as cfg
import numpy as np
from Functions import turnMatrix
from PIL import Image
from scipy.special import erf
import matplotlib.pyplot as plt


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
        self.max_height = cfg.get_max_height()
        self.std_deriv = cfg.get_std_deriv()
        self.overlap_threshold = cfg.get_overlap_threshold()
        # self.effect_range = np.square(self.std_deriv) * max(self.length, self.width)
        self.effect_range = max(self.length, self.width)
        self.dragged = False
        self.dragged_dist = 0
        self.dragged_angle = 0
        self.subp1 = None
        self.subp2 = None
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
                matrix[i, j] = self.visualize_pixel(i - self.x, j - self.y)  # ToDo: Stick

        return matrix

    def efficient_Matrix(self):
        eff_matrix = np.zeros((2 * self.effect_range, 2 * self.effect_range))
        for i in range(-self.effect_range, self.effect_range):
            for j in range(-self.effect_range, self.effect_range):
                eff_matrix[i + self.effect_range, j + self.effect_range] = \
                    self.visualize_pixel(i, j)


        return eff_matrix, self.x, self.y

    def efficient_Matrix_turned(self):
        eff_mat, x, y = self.efficient_Matrix()
        eff_mat_turned, cx, cy = turnMatrix(eff_mat, self.theta)
        # print("x: {}, y:{}, x+cx: {}, y+cy: {}".format(x, y, x+cx, y+cy))
        # return eff_mat_turned, round(x + cx), round(y + cy)  # ToDo Check if x needs to be manipulated first
        return eff_mat_turned, x, y

    def drag(self, speed, angle):
        self.dragged = True
        print("Speed: {}".format(speed))
        drag_dist = random.gauss(speed, 0.1 * speed)  # ToDo: soft_code Stddrtiv
        self.dragged_dist = drag_dist
        self.dragged_angle = angle

        rel_height = random.random()
        h1 = rel_height * self.length
        h2 = rel_height * (1 - self.length)
        y_center_p1 = self.y - np.cos(self.theta) * h1/2
        y_center_p2 = self.y + np.cos(self.theta) * h2/2 + self.dragged_dist * np.cos(self.dragged_angle)
        x_center_p1 = self.x - np.sin(self.theta) * h1/2
        x_center_p2 = self.x + np.sin(self.theta) * h1/2 + self.dragged_dist * np.sin(self.dragged_angle)

        p1 = Particle(x_center_p1, y_center_p1, self.theta)
        p2 = Particle(x_center_p2, y_center_p2, self.theta)
        p1.set_length(rel_height * self.length)
        p2.set_length((1 - rel_height) * self.length)

        self.subp1 = p1
        self.subp2 = p2

    #def visualize_dragged(self):


        #self.dragged = True

        #rel_height = random.random()
        #p1 = Particle(self.x, self.y, self.theta)
        #p2 = Particle(self.x + self.dragged_dist * np.sin(self.dragged_angle),
        #              self.y + self.dragged_dist * np.cos(self.dragged_angle), self.theta)
        #p1.set_length(rel_height * self.length)
        #p2.set_length((1 - rel_height) * self.length)

        #self.subp1 = p1
        #self.subp2 = p2
        #return
        # New matrix with indizes combined
        #p1_matrix, p1x, p1y = p1.efficient_Matrix_turned()
        #p2_matrix, p2x, p2y = p2.efficient_Matrix_turned()

        #newmat_w = math.ceil(0.5 * np.shape(p1_matrix)[0] + 0.5 * np.shape(p2_matrix)[0] + math.fabs(p1x - p2x))
        #newmat_h = math.ceil(0.5 * np.shape(p1_matrix)[1] + 0.5 * np.shape(p2_matrix)[1] + math.fabs(p1y - p2y))
        ##print(newmat_h, newmat_w)
        #newmat = np.zeros((newmat_w, newmat_h))
        #newmat2 = np.zeros((newmat_w, newmat_h))
        #for i in range(np.shape(p1_matrix)[0]):
        #    for j in range(np.shape(p1_matrix)[1]):
        #        i_tilt = i - math.ceil(np.shape(p1_matrix)[0]/2) # + p1x
        #        j_tilt = j - math.ceil(np.shape(p1_matrix)[1]/2) # + p1y
        #        print(i, i_tilt, j, j_tilt, newmat_w, newmat_h)
        #        newmat[i_tilt, j_tilt] = p1_matrix[i, j]

        #for i in range(np.shape(p2_matrix)[0]):
        #    for j in range(np.shape(p2_matrix)[1]):
        #        i_tilt = i - math.ceil(np.shape(p2_matrix)[0]/2) + int(np.round(math.fabs(p2x - p1x))) # + p2x
        #        j_tilt = j - math.ceil(np.shape(p2_matrix)[1]/2) + int(np.round(math.fabs(p2y - p1y))) # + p2y
        #        print(i, i_tilt, j, j_tilt, newmat_w, newmat_h)
        #        newmat2[i_tilt, j_tilt] = p2_matrix[i, j]

        #plt.imshow(newmat)
        #plt.show()
        #plt.imshow(newmat2)
        #plt.show()

        #print("newmat:")
        #print(np.max(newmat))
        #self.width = np.shape(newmat)[0]
        #self.height = np.shape(newmat)[1]
        #self.effect_range = max(self.length, self.width)

        #return newmat, self.x, self.y



    def visualize_pixel(self, x, y):
        if not self.fromImage:
            return self._line_gauss(x, y)

        else:
            cx = self.img.size[0] /2
            cy = self.img.size[1] /2
            try:
                return self.pixels[x + cx, y + cy]
            except IndexError:
                return 0

    def _ball(self, x, y):
        return 255 if (abs(x) <= 1 and abs(y) <= 1) else 0

    def _line(self, x, y):
        if -self.width/2 <= x < self.width - self.width/2 and \
            -self.length/2 <= y < self.length - self.length/2:
            if y > 0:
                return self._color(self.height)
            else:
                return self._color(self.height/2)
        else:
            return 0

    def _line_gauss(self, x, y):
        if self.std_deriv == 0:
            return self._line(x, y)

        left_x = -self.width/2
        right_x = self.width - self.width/2
        lower_y = -self.length/2
        upper_y = self.length - self.length/2

        if x < left_x - 3 * np.square(self.std_deriv) or \
            y < lower_y - 3 * np.square(self.std_deriv) or \
            x > right_x + 3 * np.square(self.std_deriv) or \
            y > upper_y + 3 * np.square(self.std_deriv):
            return 0
        elif x < 0 and y < 0:
            return self._color(self.height) * self._gauss(x, left_x, y, lower_y)
        elif x < 0 and y >= 0:
            return self._color(self.height) * self._gauss(x, left_x, y, upper_y)
        elif x >= 0 and y < 0:
            return self._color(self.height) * self._gauss(x, right_x, y, lower_y)
        elif x >= 0 and y >= 0:
            return self._color(self.height) * self._gauss(x, right_x, y, upper_y)
        else:
            print(x, y)
            raise NotImplementedError

    def _gauss(self, x, mu_x, y, mu_y):
        ret = self._gauss1D(x, mu_x) * self._gauss1D(y, mu_y)
        return ret

    def _gauss1D(self, x, mu):
        if x < 0:
            ret = 0.5 * (1 + erf((x - mu) / (np.sqrt(2) * self.std_deriv)))
        else:
            ret = 0.5 * (1 + erf((-x + mu) / (np.sqrt(2) * self.std_deriv)))

        return ret


    def _color(self, height):
        return 255 * height / self.max_height

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def get_theta(self):
        return self.theta

    def set_theta(self, theta):
        self.theta = theta

    def set_length(self, length):
        self.length = length

    def get_dimension(self):
        return max(self.width, self.length)

    def get_distance_to(self, part):
        return np.sqrt(np.square(self.x - part.get_x()) + np.square(self.y - part.get_y()))

    def true_overlap(self, particle):
        dx = particle.get_x() - self.get_x()
        dy = particle.get_y() - self.get_y()
        if(np.sqrt(np.square(dx) + np.square(dy)) > self.get_dimension() + particle.get_dimension()):
            return False
        else:
            thismat, foo, bar = self.efficient_Matrix_turned()
            othmat, foo, bar = particle.efficient_Matrix_turned()
            if np.shape(thismat) != np.shape(othmat):
                small_mat = thismat if np.shape(thismat)[0] < np.shape(othmat)[0] else othmat
                big_mat = thismat if small_mat is othmat else othmat
                if small_mat is not thismat:
                    dx *= -1
                    dy *= -1
                for i in range(np.shape(small_mat)[0]):
                    for j in range(np.shape(small_mat)[1]):
                        try:
                            if small_mat[i, j] > self.overlap_threshold and \
                                    big_mat[i + dx, j + dy] > self.overlap_threshold:
                                return True
                        except IndexError:
                            continue
            else:
                for i in range(np.shape(thismat)[0]):
                    for j in range(np.shape(thismat)[1]):
                        try:
                            if thismat[i, j] > 0.2 and othmat[i + dx, j + dy] > 0.2:
                                return True
                        except IndexError:
                            continue
        return False

    def get_visualization(self):
        ret = []
        if not self.dragged:
            ret.append(self.efficient_Matrix_turned())
            return ret
        else:
            for vis in self.subp1.get_visualization():
                ret.append(vis)
            for vis in self.subp2.get_visualization():
                ret.append(vis)
        return ret


    @staticmethod
    def str_Header():
        return "x, y, theta, width, height, length, dragged, dragged_dist\n"

    def __str__(self):
        args = [self.x, self.y, self.theta, self.width, self.height, self.length, self.dragged, self.dragged_dist]
        return ", ".join(str(arg) for arg in args)
