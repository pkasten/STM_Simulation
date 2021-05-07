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
        # self.std_deriv = cfg.get_std_deriv()
        self.fermi_exp = cfg.get_fermi_exp()
        self.effect_range = max(self.length, self.width)
        if self.fermi_exp != 0:
            self.fermi_range_w = np.log(99) / self.fermi_exp + self.width / 2  # ToDo: Soft Code percentile
            self.fermi_range_h = np.log(99) / self.fermi_exp + self.height / 2
            self.effect_range = int(max(self.length + self.fermi_range_h / 2, self.width + self.fermi_range_w / 2))
        self.overlap_threshold = cfg.get_overlap_threshold()
        # self.effect_range = np.square(self.std_deriv) * max(self.length, self.width)

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
        print("Deprecated 1231453")
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
        drag_dist = random.gauss(speed, 0.1 * speed)  # ToDo: soft_code Stddrtiv
        self.dragged_dist = drag_dist
        self.dragged_angle = angle

        matrix, x, y = self.efficient_Matrix_turned()
        lmat = np.zeros(np.shape(matrix))
        rmat = np.zeros(np.shape(matrix))
        c_x = np.shape(matrix)[0] / 2
        c_y = np.shape(matrix)[1] / 2
        c_x_alt = c_x + (random.random() - 0.5) * self.length * np.sin(self.theta)
        c_y_alt = c_y + (random.random() - 0.5) * self.length * np.cos(self.theta)
        f = lambda x: np.tan(self.dragged_angle) * (x - c_x_alt) + c_y_alt
        for i in range(np.shape(matrix)[0]):
            for j in range(np.shape(matrix)[1]):
                if j < f(i):
                    lmat[i, j] = matrix[i, j]
                    rmat[i, j] = 0
                else:
                    lmat[i, j] = 0
                    rmat[i, j] = matrix[i, j]

        self.subp1 = lmat, x, y
        self.subp2 = rmat, x + self.dragged_dist * np.cos(self.dragged_angle), y + self.dragged_dist * np.sin(
            self.dragged_angle)

    def visualize_pixel(self, x, y):
        if not self.fromImage:
            # return self._line_gauss(x, y)
            return self._line_fermi(x, y)
        else:
            cx = self.img.size[0] / 2
            cy = self.img.size[1] / 2
            try:
                return self.pixels[x + cx, y + cy]
            except IndexError:
                return 0

    def _ball(self, x, y):
        return 255 if (abs(x) <= 1 and abs(y) <= 1) else 0

    def _line(self, x, y):
        if -self.width / 2 <= x < self.width - self.width / 2 and \
                -self.length / 2 <= y < self.length - self.length / 2:
            # if y > 0:
            return self._color(self.height)
            # else:
            #    return self._color(self.height/2)
        else:
            return 0

    # def _line_gauss(self, x, y):
    #    if self.std_deriv == 0:
    #        return self._line(x, y)
    #
    #   left_x = -self.width/2
    #  right_x = self.width - self.width/2
    # lower_y = -self.length/2
    # upper_y = self.length - self.length/2
    #
    #       if x < left_x - 3 * np.square(self.std_deriv) or \
    #          y < lower_y - 3 * np.square(self.std_deriv) or \
    #         x > right_x + 3 * np.square(self.std_deriv) or \
    #        y > upper_y + 3 * np.square(self.std_deriv):
    #       return 0
    #        elif x < 0 and y < 0:
    #           return self._color(self.height) * self._gauss(x, left_x, y, lower_y)
    #      elif x < 0 and y >= 0:
    #         return self._color(self.height) * self._gauss(x, left_x, y, upper_y)
    #    elif x >= 0 and y < 0:
    #       return self._color(self.height) * self._gauss(x, right_x, y, lower_y)
    #  elif x >= 0 and y >= 0:
    #     return self._color(self.height) * self._gauss(x, right_x, y, upper_y)
    # else:#
    #            print(x, y)
    #           raise NotImplementedError

    def _line_fermi(self, x, y):

        if self.fermi_exp == 0:
            return self._line(x, y)

        left_x = -self.width / 2
        right_x = self.width - self.width / 2
        lower_y = -self.length / 2
        upper_y = self.length - self.length / 2

        if x < left_x - self.fermi_range_w or \
                y < lower_y - self.fermi_range_h or \
                x > right_x + self.fermi_range_w or \
                y > upper_y + self.fermi_range_h:

            return 0
        elif x < 0 and y < 0:
            return self._color(self.height) * self._fermi(x, left_x, y, lower_y)
        elif x < 0 and y >= 0:
            return self._color(self.height) * self._fermi(x, left_x, y, upper_y)
        elif x >= 0 and y < 0:
            return self._color(self.height) * self._fermi(x, right_x, y, lower_y)
        elif x >= 0 and y >= 0:
            return self._color(self.height) * self._fermi(x, right_x, y, upper_y)
        else:
            print(x, y)
        raise NotImplementedError

    def _fermi(self, x, mu_x, y, mu_y):
        ret = self._fermi1D(x, mu_x) * self._fermi1D(y, mu_y)
        return ret

    def _fermi1D(self, x, mu):
        if x < 0:
            return 1 / (np.exp(self.fermi_exp * (-x + mu)) + 1)
        else:
            return 1 / (np.exp(self.fermi_exp * (x - mu)) + 1)

    # def _gauss(self, x, mu_x, y, mu_y):
    #    ret = self._gauss1D(x, mu_x) * self._gauss1D(y, mu_y)
    #    return ret

    # def _gauss1D(self, x, mu):
    #    if x < 0:
    #        ret = 0.5 * (1 + erf((x - mu) / (np.sqrt(2) * self.std_deriv)))
    #    else:
    #        ret = 0.5 * (1 + erf((-x + mu) / (np.sqrt(2) * self.std_deriv)))
    #    return ret

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

    def set_height(self, height):
        self.height = height

    def get_dimension(self):
        return max(self.width, self.length)

    def get_distance_to(self, part):
        return np.sqrt(np.square(self.x - part.get_x()) + np.square(self.y - part.get_y()))

    def true_overlap(self, particle):
        if not self.dragged and not particle.dragged:
            #if(self._eval_overlap(particle, self.efficient_Matrix_turned()[0], self.x, self.y)):
            #    print("A")
            return self._eval_overlap(particle, self.efficient_Matrix_turned()[0], self.x, self.y)
        elif self.dragged and not particle.dragged:  # Dragged this
            # print("SubP1: {}, SubP2: {}".format(self.subp1, self.subp2))
            mat, x, y = self.subp1
            ret1 = self._eval_overlap(particle, mat, x, y)
            mat, x, y = self.subp2
            ret2 = self._eval_overlap(particle, mat, x, y)
            # print("Ret1: {}, ret2: {}".format(ret1, ret2))
            #if ret1 or ret2:
            #    print("B")
            return ret1 or ret2
        elif particle.dragged and not self.dragged:
            # print("SubP1: {}, SubP2: {}".format(self.subp1, self.subp2))
            mat, x, y = particle.subp1
            ret1 = self._eval_overlap(self, mat, x, y)
            mat, x, y = particle.subp2
            ret2 = self._eval_overlap(self, mat, x, y)
            # print("Ret1: {}, ret2: {}".format(ret1, ret2))
            #if ret1 or ret2:
             #   print("C")
            return ret1 or ret2
        else:
            mat1, x1, y1 = self.subp1
            mat2, x2, y2 = particle.subp1
            ret1 = self._eval_overlap_matrizes(mat1, x1, y1, mat2, x2, y2)
            mat2, x2, y2 = particle.subp2
            ret2 = self._eval_overlap_matrizes(mat1, x1, y1, mat2, x2, y2)
            mat1, x1, y2 = self.subp2
            mat2, x2, y2 = particle.subp1
            ret3 = self._eval_overlap_matrizes(mat1, x1, y1, mat2, x2, y2)
            mat2, x2, y2 = particle.subp2
            ret4 = self._eval_overlap_matrizes(mat1, x1, y1, mat2, x2, y2)
            #if ret1 or ret2 or ret3 or ret4:
                #print("B")
            return ret1 or ret2 or ret3 or ret4

    def _eval_overlap_matrizes(self, mat1, x1, y1, mat2, x2, y2):
        dx = int(x2 - x1)
        dy = int(y2 - y1)


        if np.sqrt(np.square(dx) + np.square(dy)) > np.sqrt(2) * (
                max(np.shape(mat1)) + max(np.shape(mat2))):
            #print("Out of Range")
            return False
        else:
            thismat = mat1
            othmat = mat2
            # print("Thismat")
            # plt.imshow(thismat)
            # plt.show()

            # print("Othmat")
            # plt.imshow(othmat)
            # plt.show()

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
                                if i+dx < 0 or j+dy < 0:
                                    continue
                                #print("True 1")
                                #print("i: {}, dx: {}, j: {}, dy: {}".format(i, dx, j, dy))
                                #plt.imshow(small_mat)
                                #plt.show()
                                #plt.imshow(big_mat)
                                #plt.show()
                                return True
                        except IndexError as ie:
                            # print(ie)
                            continue
            else:
                for i in range(np.shape(thismat)[0]):
                    for j in range(np.shape(thismat)[1]):
                        try:
                            if thismat[i, j] > self.overlap_threshold and othmat[
                                i + dx, j + dy] > self.overlap_threshold:
                                if i+dx < 0 or j+dy < 0:
                                    continue
                                #print("True2")
                                #print("True 2")
                                #print("i: {}, dx: {}, j: {}, dy: {}".format(i, dx, j, dy))
                                #plt.imshow(thismat)
                                #plt.show()
                                #plt.imshow(othmat)
                                #plt.show()
                                return True
                        except IndexError as ie:
                            # print(ie)
                            continue
        return False

    def _eval_overlap(self, particle, mat, x, y):
        a, b, c = particle.efficient_Matrix_turned()
        #if self._eval_overlap_matrizes(a, b, c, mat, x, y):
            #print("eval_overlap true for {}, {} and {}, {}".format(b, c, x, y))

        return self._eval_overlap_matrizes(a, b, c, mat, x, y)

       # dx = int(particle.get_x() - x)
      #  dy = int(particle.get_y() - y)
        # print("Self.x: {}, dx: {}, dy:{}".format(x, dx, dy))

        #if (np.sqrt(np.square(dx) + np.square(dy)) > self.get_dimension() + particle.get_dimension()):
         #   # print("Out of Range")
          #  return False
        #else:
         #   thismat = mat
          #  othmat, foo, bar = particle.efficient_Matrix_turned()
            # print("Thismat")
            # plt.imshow(thismat)
            # plt.show()

            # print("Othmat")
            # plt.imshow(othmat)
            # plt.show()

        #    if np.shape(thismat) != np.shape(othmat):
         #       small_mat = thismat if np.shape(thismat)[0] < np.shape(othmat)[0] else othmat
          #      big_mat = thismat if small_mat is othmat else othmat
           #     if small_mat is not thismat:
            #        dx *= -1
             #       dy *= -1
              #  for i in range(np.shape(small_mat)[0]):
               #     for j in range(np.shape(small_mat)[1]):
                #        try:
                 #           if small_mat[i, j] > self.overlap_threshold and \
                  #                  big_mat[i + dx, j + dy] > self.overlap_threshold:
                  ##              return True
                  #      except IndexError as ie:
                    #        # print(ie)
                     #       continue
#            else:#
 #               for i in range(np.shape(thismat)[0]):
  #                  for j in range(np.shape(thismat)[1]):
   #                     try:
    #                        if thismat[i, j] > self.overlap_threshold and othmat[
     #                           i + dx, j + dy] > self.overlap_threshold:
      #                          return True
       #                 except IndexError as ie:
        #                    # print(ie)
         #                   continue
        #return False
        # dx = particle.get_x() - x
        # dy = particle.get_y() - y
        # print("Self.x: {}, dx: {}, dy:{}".format(self.x, dx, dy))
        # if (np.sqrt(np.square(dx) + np.square(dy)) > self.get_dimension() + particle.get_dimension()):
        #    print("Out of range")
        #    return False
        # else:
        #    #thismat, foo, bar = self.efficient_Matrix_turned()
        #    thismat = mat
        #    othmat, foo, bar = particle.efficient_Matrix_turned()
        #    if np.shape(thismat) != np.shape(othmat):
        #        small_mat = thismat if np.shape(thismat)[0] < np.shape(othmat)[0] else othmat
        #        big_mat = thismat if small_mat is othmat else othmat
        #        if small_mat is not thismat:
        #            dx *= -1
        #            dy *= -1
        #        for i in range(np.shape(small_mat)[0]):
        #            for j in range(np.shape(small_mat)[1]):
        #                if i > 0 and j > 0 and i + dx > 0 and j + dy > 0:
        #                    try:
        #                        if small_mat[i, j] > particle.overlap_threshold and \
        #                                big_mat[i + dx, j + dy] > particle.overlap_threshold:
        #                            return True
        #                    except IndexError:
        #                        continue
        ##                else:
        #                    continue
        #    else:
        #        for i in range(np.shape(thismat)[0]):
        #            for j in range(np.shape(thismat)[1]):
        #                if i > 0 and j > 0 and i + dx > 0 and j + dy > 0:
        #                    try:
        #                        if thismat[i, j] > self.overlap_threshold and othmat[i + dx, j + dy] > \
        #                                self.overlap_threshold:
        #                            # print("Bigger: thismat[i, j]: {}, i{}, j{}, othmat {}, i+dx{}, j+dy{}".format(thismat[i, j], i, j, othmat[i+dx, j+dy], i+dx, j+dy))
        #                            return True
        #                        else:
        ##                            pass
        #                            # print("Smaller")
        #                    except IndexError:
        #                        continue
        #                else:
        #                    # print("Continuing")
        #                    continue
        # return False

    def get_visualization(self):
        ret = []
        if not self.dragged:
            ret.append(self.efficient_Matrix_turned())
            return ret
        else:
            # print("appending")
            ret.append(self.subp1)
            ret.append(self.subp2)
            # for vis in self.subp1:
            #    ret.append(vis)
            # for vis in self.subp2:
            #    ret.append(vis)
        return ret

    @staticmethod
    def str_Header():
        return "x, y, theta, width, height, length, dragged, dragged_dist\n"

    def __str__(self):
        args = [self.x, self.y, self.theta, self.width, self.height, self.length, self.dragged, self.dragged_dist]
        return ", ".join(str(arg) for arg in args)


class Double_Particle(Particle):
    def __init__(self, x=None, y=None, theta=None):
        super().__init__(x, y, theta)
        self.x *= 2
        self.y *= 2
        self.img_width *= 2
        self.img_height *= 2
