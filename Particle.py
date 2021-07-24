import math, random

from tqdm import tqdm

import Configuration as cfg
import numpy as np
from Functions import turnMatrix
from PIL import Image
from Distance import Distance
from Charge import Charge
from scipy.special import erf
import matplotlib.pyplot as plt


class Particle:



    """
    Class particle abstracts any particle adsorbed to a surface that should be displayable.
    Is extended by molecule, Atom and Dust.
    Key Methods to watch: visualize_pixel,
    """
    def __init__(self, x=None, y=None, theta=None):
        """
        Inititalizes a new particle and obtains basic variables.
        :param x: x-position as a Distance
        :param y: y-position as a Distance
        :param theta: orientation angle
        :return: None
        """

        # Determine position if not provided
        if x is None:
            self.x = Distance(False, random.randint(0 - cfg.get_px_overlap(), cfg.get_width().px + cfg.get_px_overlap()))
            self.y = Distance(False, random.randint(0 - cfg.get_px_overlap(), cfg.get_height().px + cfg.get_px_overlap()))
            self.theta = 2 * math.pi * random.random()
        else:
            self.x = x
            self.y = y
            self.theta = theta

        # set own dimensions
        self.width = cfg.get_part_width()
        self.pos = np.array([self.x, self.y])
        self.height = cfg.get_part_height()
        self.length = cfg.get_part_length()

        # get image properties
        self.img_width = cfg.get_width()
        self.img_height = cfg.get_height()
        self.max_height = cfg.get_max_height()

        # Properties regarding visualization
        self.fermi_exp = cfg.get_fermi_exp()
        self.effect_range = max(self.length, self.width)
        if self.fermi_exp != 0:
            self.fermi_range_w = np.log(99) / self.fermi_exp + self.width.px / 2  # ToDo: Soft Code percentile
            self.fermi_range_h = np.log(99) / self.fermi_exp + self.height.px / 2
            self.effect_range = int(max(self.length.px + self.fermi_range_h / 2, self.width.px + self.fermi_range_w / 2))
        self.overlap_threshold = cfg.get_overlap_threshold()
        # self.effect_range = np.square(self.std_deriv) * max(self.length, self.width)

        # Parameters used for particle dragging
        # Has this particle been dragged already?
        self.dragged = False
        # By how far and at which angle?
        self.dragged_dist = 0
        self.dragged_angle = 0
        # Subp variables contains visualization for each part of a dragged particle
        self.subp1 = None
        self.subp2 = None

        # Chooses whether the particle should be visualized with calculations or should be imported from an image
        if len(cfg.get_image_path()) == 0:
            self.fromImage = False
        else:
            self.img = Image.open(cfg.get_image_path())
            self.pixels = self.img.convert("L").load()
            self.fromImage = True
        #self.charges = []
        #self.calc_charges()



    def set_maxHeight(self, maxH):
        """
        setter method for maximum height parameter.
        :param maxH: new maximum height
        :return:
        """
        self.max_height = maxH

    def efficient_Matrix(self):
        """
        produces the visualization matrix for this particle, without using the orientation.
        Is called by efficient_Matrix_turned which turns it
        :return: visualization matrix, particle position x, y
        """
        eff_matrix = np.zeros((2 * self.effect_range, 2 * self.effect_range))
        for i in tqdm(range(-self.effect_range, self.effect_range)): #ToDo: rem
            for j in range(-self.effect_range, self.effect_range):
                eff_matrix[i + self.effect_range, j + self.effect_range] = \
                    self.visualize_pixel(i, j)

        return eff_matrix, self.x, self.y

    def efficient_Matrix_turned(self):
        """
        Calculates the visualization of this particle as a matrix
        :return: visualization matrix, particle position x, y
        """
        eff_mat, x, y = self.efficient_Matrix()
        eff_mat_turned, cx, cy = turnMatrix(eff_mat, self.theta)
        # print("x: {}, y:{}, x+cx: {}, y+cy: {}".format(x, y, x+cx, y+cy))
        # return eff_mat_turned, round(x + cx), round(y + cy)  # ToDo Check if x needs to be manipulated first
        return eff_mat_turned, x, y

    def get_visualization(self):
        """
        Returns the visualization of this particle.
        Is use by DataFrame to display particles.
        Invokes efficient_matrix_turned, but now accounts for dragged particles
        :return:
        """
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

    def drag(self, speed, angle):
        """
        Implementation of the effect that a particle is dragged by the STM tip
        Sets subp1/2 parameters with their visualizations
        :param speed: dragging speed, determines distance
        :param angle: dragging angle, determines slice border and direction
        :return: None, only sets subp parameters
        """
        self.dragged = True
        drag_dist = random.gauss(speed, 0.1 * speed)  # ToDo: soft_code Stddrtiv
        self.dragged_dist = drag_dist
        self.dragged_angle = angle

        matrix, x, y = self.efficient_Matrix_turned()
        lmat = np.zeros(np.shape(matrix))
        rmat = np.zeros(np.shape(matrix))
        c_x = np.shape(matrix)[0] / 2
        c_y = np.shape(matrix)[1] / 2
        c_x_alt = c_x + (random.random() - 0.5) * self.length.px * np.sin(self.theta)
        c_y_alt = c_y + (random.random() - 0.5) * self.length.px * np.cos(self.theta)
        f = lambda x: np.tan(self.dragged_angle) * (x - c_x_alt) + c_y_alt
        for i in range(np.shape(matrix)[0]):
            for j in range(np.shape(matrix)[1]):
                if j < f(i):
                    lmat[i, j] = matrix[i, j]
                    rmat[i, j] = 0
                else:
                    lmat[i, j] = 0
                    rmat[i, j] = matrix[i, j]

        #print("x: {}, y:{}".format(x, y))
        #print("x2: {}, y2:{}".format(x.px + self.dragged_dist * np.cos(self.dragged_angle),y.px + self.dragged_dist * np.sin(
         #   self.dragged_angle)))


        self.subp1 = lmat, x, y
       #print("X: {}".format(y))
       # print("DD: {}".format(self.dragged_dist))
        try:
            self.subp2 = rmat, Distance(False, x.px + self.dragged_dist.px * np.cos(self.dragged_angle)), Distance(False, y.px + self.dragged_dist.px * np.sin(
                self.dragged_angle))
        except AttributeError:
            self.subp2 = rmat, Distance(False, x.px + self.dragged_dist * np.cos(self.dragged_angle)), Distance(
                False, y.px + self.dragged_dist * np.sin(
                    self.dragged_angle))

    def visualize_pixel(self, x, y):
        """
        Key element of visualization. returns the height for a pixel at postion x, y where
        (0, 0) marks the center of the particle
        :param x: position x
        :param y: position y
        :return: height
        """
        mode = "LineFermi"
        if mode == "Line":
            return self._line(x, y)

        if mode=="LineFermi":
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
        """
        Can be used as visualize pixel to show particles as a ball/circle
        :param x: distance x from center
        :param y: distance y from center
        :return: height
        """
        return 255 if (abs(x) <= 1 and abs(y) <= 1) else 0

    def _line(self, x, y):
        """
        Can be used as visualize pixel to show particles as a sharp line
        :param x: distance x from center
        :param y: distance y from center
        :return: height
        """
        if -self.width.px / 2 <= x < self.width.px - self.width.px / 2 and \
                -self.length.px / 2 <= y < self.length.px - self.length.px / 2:
            # if y > 0:
            return self._color(self.height)
            # else:
            #    return self._color(self.height/2)
        else:
            return 0

    def color(self, h):
        """
        helper method to make _color accessible for other classes
        :param h: height
        :return: color for height h
        """
        return self._color(h)

    def _line_fermi(self, x, y):
        """
        Can be used as visualize pixel to show particles as a line with fermi distribution as border
        :param x: distance x from center
        :param y: distance y from center
        :return: height
        """

        if self.fermi_exp == 0:
            return self._line(x, y)


        left_x = -self.width / 2
        right_x = self.width - self.width / 2
        lower_y = -self.length / 2
        upper_y = self.length - self.length / 2

        top_fak = 1 #ToDO: Remove bzw 1
        lfak = 1
        if x < left_x.px - self.fermi_range_w or \
                y < lower_y.px - self.fermi_range_h or \
                x > right_x.px + self.fermi_range_w or \
                y > upper_y.px + self.fermi_range_h:

            return 0
        elif x < 0 and y < 0:
            return lfak * self._color(self.height) * self._fermi(x, left_x, y, lower_y)
        elif x < 0 and y >= 0:
            return lfak * top_fak * self._color(self.height) * self._fermi(x, left_x, y, upper_y)
        elif x >= 0 and y < 0:
            return self._color(self.height) * self._fermi(x, right_x, y, lower_y)
        elif x >= 0 and y >= 0:
            return top_fak * self._color(self.height) * self._fermi(x, right_x, y, upper_y)
        else:
            print(x, y)
        raise NotImplementedError

    def _fermi(self, x, mu_x, y, mu_y):
        """
        Fermi distribution in 2D
        :param x: position in 1st direction
        :param mu_x: expectation value in 1st direction
        :param y: position in 2nd direction
        :param mu_y: expectation value in 2nd direction
        :return:
        """
        ret = self._fermi1D(x, mu_x) * self._fermi1D(y, mu_y)
        return ret

    def _fermi1D(self, x, mu):
        """
        Implementation of fermis distribution in one dimenstion
        :param x: position
        :param mu: expectation value
        :return:
        """
        if x < 0:
            return 1 / (np.exp(self.fermi_exp * (-x + mu.px)) + 1)
        else:
            return 1 / (np.exp(self.fermi_exp * (x - mu.px)) + 1)

    def _color(self, height):
        """
        Returns height mapped onto a range from 0 to 255
        :param height: current height
        :return: value from 0 to 255, can be used as grayscale visualization
        """

        return 255 * height.px / self.max_height.px

    def get_x(self):
        """
        getter method for x-position
        :return:
        """
        return self.x

    def get_y(self):
        """
        getter method for y-direction
        :return:
        """
        return self.y

    def set_width(self, w):
        """
        setter method for particle width
        :param w: new width
        :return:
        """
        self.width = w

    def set_length(self, l):
        """
        setter method for particle length
        :param l: new length
        :return:
        """
        self.length = l

    def set_x(self, x):
        """
        setter method for x position
        :param x: new x position
        :return:
        """
        self.x = x
     #   self.calc_charges()

    def set_y(self, y):
        """
        setter method for y-position
        :param y: new y position
        :return:
        """
        self.y = y
      #  self.calc_charges()

    def get_theta(self):
        """
        getter method for particle orientation angle
        :return:
        """
        return self.theta

    def set_theta(self, theta):
        """
        setter method for particle orientation
        :param theta: new orientation
        :return:
        """
        self.theta = theta
     #   self.calc_charges()

    def set_height(self, height):
        """
        setter method for particle height
        :param height: new height
        :return:
        """
        self.height = height

    def get_dimension(self):
        """
        returns particle dimension defined as maximum of width and height
        :return: maximum of length and width
        """
        return max(self.width, self.length)

    def get_distance_to(self, part):
        """
        calculates eucliean distance to another particle part
        :param part: other particle
        :return: distance
        """
        return np.sqrt(np.square(self.x - part.get_x()) + np.square(self.y - part.get_y()))

    def true_overlap(self, particle):
        """
        evalues if this particle overlaps with the provided particle
        :param particle: other particle
        :return: True if they do overlap
        """
        if not self.dragged and not particle.dragged:
            #if(self._eval_overlap(particle, self.efficient_Matrix_turned()[0], self.x, self.y)):
            #    print("A")
            #print("Here shall I Be TO")
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

    def overlap_amnt(self, particle):
        """
        Returns a value indicating by how much this particle overlaps with the provided one.
        Was used for Energy optimization to create a slope towards non-overlapping particles
        :param particle: different particle
        :return: value by how much particles do overlap. Just a tendence, no physical meaning
        """

        def eval_overlap_amount(particle, mat, x, y):
            a, b, c = particle.efficient_Matrix_turned()
            return matrix_overlap_amount(a, b, c, mat, x, y)

        def matrix_overlap_amount(mat1, x1, y1, mat2, x2, y2):
            bigmatw = int(np.shape(mat1)[0] / 2 + np.shape(mat2)[0] / 2 + np.abs(x1 - x2))
            bigmath = int(np.shape(mat1)[1] / 2 + np.shape(mat2)[1] / 2 + np.abs(x1 - x2))
            ret = 0
            # bigmat = np.zeros((bigmatw, bigmath))

            origin_x = x1 - np.shape(mat1)[0] / 2 if x1 < x2 else x2 - np.shape(mat2)[0] / 2
            origin_y = y1 - np.shape(mat1)[1] / 2 if y1 < y2 else y2 - np.shape(mat2)[1] / 2

            d_x_1 = x1 - origin_x
            d_y_1 = y1 - origin_y

            d_x_2 = x2 - origin_x
            d_y_2 = y2 - origin_y

            dd_x_1 = np.shape(mat1)[0] / 2
            dd_y_1 = np.shape(mat1)[1] / 2
            dd_x_2 = np.shape(mat2)[0] / 2
            dd_y_2 = np.shape(mat2)[1] / 2

            for i in range(bigmatw):
                for j in range(bigmath):
                    i_t_1 = int(i - d_x_1 + dd_x_1)
                    i_t_2 = int(i - d_x_2 + dd_x_2)
                    j_t_1 = int(j - d_y_1 + dd_y_1)
                    j_t_2 = int(j - d_y_2 + dd_y_2)

                    try:
                        if 0 <= i_t_1 and 0 <= j_t_1:
                            if mat1[i_t_1, j_t_1] > self.overlap_threshold:
                                if 0 <= i_t_2 and 0 <= j_t_2:
                                    if mat2[i_t_2, j_t_2] > self.overlap_threshold:
                                        ret += mat1[i_t_1, j_t_1] + mat2[i_t_2, j_t_2]
                    except IndexError:
                        continue
            return ret

        if not self.dragged and not particle.dragged:
            return eval_overlap_amount(particle, self.efficient_Matrix_turned()[0], self.x, self.y)
        elif self.dragged and not particle.dragged:  # Dragged this
            mat, x, y = self.subp1
            ret1 = eval_overlap_amount(particle, mat, x, y)
            mat, x, y = self.subp2
            ret2 = eval_overlap_amount(particle, mat, x, y)
            return ret1 + ret2
        elif particle.dragged and not self.dragged:
            mat, x, y = particle.subp1
            ret1 = eval_overlap_amount(self, mat, x, y)
            mat, x, y = particle.subp2
            ret2 = eval_overlap_amount(self, mat, x, y)
            return ret1 + ret2
        else:
            mat1, x1, y1 = self.subp1
            mat2, x2, y2 = particle.subp1
            ret1 = matrix_overlap_amount(mat1, x1, y1, mat2, x2, y2)
            mat2, x2, y2 = particle.subp2
            ret2 = matrix_overlap_amount(mat1, x1, y1, mat2, x2, y2)
            mat1, x1, y2 = self.subp2
            mat2, x2, y2 = particle.subp1
            ret3 = matrix_overlap_amount(mat1, x1, y1, mat2, x2, y2)
            mat2, x2, y2 = particle.subp2
            ret4 = matrix_overlap_amount(mat1, x1, y1, mat2, x2, y2)
            #if ret1 or ret2 or ret3 or ret4:
                #print("B")
            return ret1 + ret2 + ret3 + ret4

    def _eval_overlap_matrizes(self, mat1, x1, y1, mat2, x2, y2):
        """
        evaluates if two particles overlap by combining their visualization matrices and checking if both are above
        a specific threshold at the same position
        :param mat1: matrix 1 that should be checked for overlapping
        :param x1: center position of matrix 1
        :param y1: center position of matrix 1
        :param mat2:  matrix 2 that should be checked for overlapping
        :param x2: center position of matrix 2
        :param y2: center position of matrix 2
        :return: True if particles with provieded matrices overlap
        """
        #Testing - ging nicht
        #newmat = np.zeros((int(np.ceil(self.img_width.px)), int(np.ceil(self.img_height.px))))
        #dx = int(np.shape(mat1)[0]/2)
        #dy = int(np.shape(mat1)[1]/2)
        #for i in range(np.shape(mat1)[0]):
        #    for j in range(np.shape(mat2)[1]):
        #        try:
        #            newmat[i + int(np.ceil(x1.px)) - dx, j + int(np.ceil(y1.px)) - dy] = mat1[i, j]
        #        except IndexError:
        #            continue

        #dx = int(np.shape(mat2)[0] / 2)
        #dy = int(np.shape(mat2)[1] / 2)
        #for i in range(np.shape(mat2)[0]):
        #    for j in range(np.shape(mat2)[1]):
        #        try:
        #            if newmat[i + int(np.ceil(x2.px)) - dx, j + int(np.ceil(y2.px)) - dy] > self.overlap_threshold and mat1[i, j] > self.overlap_threshold:
        #                return True
        #            newmat[i + int(np.ceil(x2.px)) - dx, j + int(np.ceil(y2.px)) - dy] = mat1[i, j]
        #        except IndexError:
        #            continue

        #plt.imshow(newmat)
        #plt.show()
        #return False


        #print(np.max(mat1), np.max(mat2))

        #test
        bigmatw = int(np.shape(mat1)[0]/2 + np.shape(mat2)[0]/2 + np.abs(x1.px - x2.px))
        bigmath = int(np.shape(mat1)[1]/2 + np.shape(mat2)[1]/2 + np.abs(x1.px - x2.px))

        #bigmat = np.zeros((bigmatw, bigmath))


        origin_x = x1.px - np.shape(mat1)[0] / 2 if x1 < x2 else x2.px - np.shape(mat2)[0] / 2
        origin_y = y1.px - np.shape(mat1)[1] / 2 if y1 < y2 else y2.px - np.shape(mat2)[1] / 2

        d_x_1 = x1.px - origin_x
        d_y_1 = y1.px - origin_y

        d_x_2 = x2.px - origin_x
        d_y_2 = y2.px - origin_y

        dd_x_1 = np.shape(mat1)[0] / 2
        dd_y_1 = np.shape(mat1)[1] / 2
        dd_x_2 = np.shape(mat2)[0] / 2
        dd_y_2 = np.shape(mat2)[1] / 2


        for i in range(bigmatw):
            for j in range(bigmath):
                i_t_1 = int(i - d_x_1 + dd_x_1)
                i_t_2 = int(i - d_x_2 + dd_x_2)
                j_t_1 = int(j - d_y_1 + dd_y_1)
                j_t_2 = int(j - d_y_2 + dd_y_2)

                try:
                    if 0 <= i_t_1 and 0 <= j_t_1:
                        if mat1[i_t_1, j_t_1] > self.overlap_threshold:
                            if 0 <= i_t_2 and 0 <= j_t_2:
                                if mat2[i_t_2, j_t_2] > self.overlap_threshold:
                                    return True
                except IndexError:
                    continue
        return False

    def _eval_overlap(self, particle, mat, x, y):
        """
        same as _eval_overlap_matrizes, just with provided particle instance and matrix
        :param particle: particle that should be checked for overlapping
        :param mat: matrix of other particle
        :param x: x-position of other particle
        :param y: y-position of other particle
        :return: True if they overlap
        """
        a, b, c = particle.efficient_Matrix_turned()
        #if self._eval_overlap_matrizes(a, b, c, mat, x, y):
            #print("eval_overlap true for {}, {} and {}, {}".format(b, c, x, y))

        return self._eval_overlap_matrizes(a, b, c, mat, x, y)

    @staticmethod
    def str_Header():
        """
        static method. Returns the scheme after which particles are returned as strings
        :return:
        """
        return "type, x, y, theta, width, height, length, dragged, dragged_dist\n"

    def __str__(self):
        """
        string representation of a particle after scheme defined in str_Header
        :return: string representation
        """
        args = ["Particle", self.x, self.y, self.theta, self.width, self.height, self.length, self.dragged, self.dragged_dist]
        return ", ".join(str(arg) for arg in args)


    @DeprecationWarning
    def calc_charges(self):
        """
        Previously used to define charge position
        :return:
        """
        # Dipol
        q_plus = Charge(self.x - self.length * 0.45 * np.sin(self.theta), self.y + self.length * 0.45 * np.cos(self.theta),
                        1)
        q_minus = Charge(self.x + self.length * 0.45 * np.sin(self.theta), self.y - self.length * 0.45 * np.cos(self.theta),
                         -1)
        self.charges = []
        self.charges.append(q_plus)
        self.charges.append(q_minus)


    @DeprecationWarning
    def toMatrix(self):
        """
        DEPRECATED. Previously used instead of efficient_matrix
        Does NOT stop visualization at fermirange
        :return:
        """
        print("Deprecated 1231453")
        matrix = np.zeros((self.img_width, self.img_height))
        for i in range(0, self.img_width):
            for j in range(0, self.img_height):
                matrix[i, j] = self.visualize_pixel(i - self.x, j - self.y)

        return matrix


class Double_Particle(Particle):
    """
    Particle implementation for DoubleTip images. Is just as Partciles, but with doubled image width and height
    """
    def __init__(self, x=None, y=None, theta=None):
        super().__init__(x, y, theta)
        self.x *= 2
        self.y *= 2
        self.img_width *= 2
        self.img_height *= 2
