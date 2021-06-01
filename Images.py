from functools import lru_cache

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import Configuration as cfg
import os, random
from Functions import get_invers_function, measureTime


# import Particle


class MyImage:
    width = cfg.get_width().px
    height = cfg.get_height().px
    # noise = True
    colors = np.zeros((int(np.ceil(width)), int(np.ceil(height))))
    sigma = 5.5
    color_scheme = cfg.get_color_scheme()

    img = ""
    filename_generator = ""

    def __init__(self, matrix=None):
        if matrix is None:
            self.img = self.newImage()
            self.noised = False
        else:
            self.width = np.shape(matrix)[0]
            self.height = np.shape(matrix)[1]
            self.img = self.newImage()
            self.colors = matrix
            self.updateImage()
    @measureTime
    def getWidth(self):
        return self.width

    @measureTime
    def getHeight(self):
        return self.height

    @measureTime
    def setWidth(self, w):
        self.width = w
        self.colors = np.zeros((int(np.ceil(self.width)), int(np.ceil(self.height))))

    @measureTime
    def setHeight(self, h):
        self.height = h
        self.colors = np.zeros((int(np.ceil(self.width)), int(np.ceil(self.height))))

    @measureTime
    def addParticle(self, particle):
        self.colors = self.colors + particle.toMatrix()

    @measureTime
    def addMatrix(self, matrix):
        self.colors = self.colors + matrix

    @DeprecationWarning
    def double_tip(self, strength, rel_dist, angle):  # ToDo: neues Bild damit auch links oben etc was ist
        # ToDo: Use surrounding Pictures
        print("Deprecated 5462")
        vec_x = np.floor(np.sqrt(np.square(self.width) + np.square(self.height)) * rel_dist * np.sin(angle))
        vec_y = np.floor(np.sqrt(np.square(self.width) + np.square(self.height)) * rel_dist * np.cos(angle))
        vec_x = int(vec_x)
        vec_y = int(vec_y)

        newmat = self.colors

        nm_cpy = np.zeros(np.shape(newmat))
        for i in range(np.shape(newmat)[0]):
            for j in range(np.shape(newmat)[1]):
                nm_cpy[i, j] = newmat[i, j]

        for i in range(np.shape(newmat)[0]):
            for j in range(np.shape(newmat)[1]):
                try:
                    newmat[i + vec_x, j + vec_y] += strength * nm_cpy[i, j]
                    # print("Guuu")
                except IndexError as ie:
                    # print(ie)
                    pass

        self.colors = newmat
        self.updateImage()

    @measureTime
    def shift_image(self, f_horiz=None, f_vert=None):

        #testmat = np.ones(np.shape(self.colors))
        #testmat *= 200
        #self.colors = testmat

        style = "Exp"
        factor_x = 1.1
        factor_y = 1.1


        if style == "Linear":
            if f_horiz is None:
                f_h = lambda x: factor_x * x
            else:
                f_h = f_horiz
            if f_vert is None:
                f_v = lambda y: factor_y * y
            else:
                f_v = f_vert

            print("Calcing inv")
            f_h_inv = lambda x: x/factor_x
            f_v_inv = lambda x: x/factor_y
            #f_h_inv = get_invers_function(f_h)
            #f_v_inv = get_invers_function(f_v)
            print("Got inv")
        elif style == "Exp":
            wid, heigth = np.shape(self.colors)
            #gamma_x = np.log(wid * factor_x) / wid
            #gamma_y = np.log(heigth * factor_y) / heigth
            gamma_x = np.log((factor_x - 1)*wid) / wid
            gamma_y = np.log((factor_y - 1)*heigth) / heigth


            if f_horiz is None:
                f_h = lambda x: x + np.exp(gamma_x * x)
            else:
                f_h = f_horiz
            if f_vert is None:
                f_v = lambda y: y + np.exp(gamma_y * y)
            else:
                f_v = f_vert


            f_h_inv = get_invers_function(f_h)
            f_v_inv = get_invers_function(f_v)

        else:
            raise NotImplementedError

        w = int(np.floor(f_h(np.shape(self.colors)[0] - 1)))
        h = int(np.floor(f_v(np.shape(self.colors)[1] - 1)))

        newmat = np.zeros((w, h))
        i_max = np.shape(newmat)[0]
        for i in range(0, np.shape(newmat)[0]):
            print("Shift progress: {:.1f}%".format(100*i/i_max))
            for j in range(0, np.shape(newmat)[1]):
                x_w = f_h_inv(i)
                #y_w = f_v(j)
                x_lw = int(np.floor(x_w))
                x_hi = int(np.ceil(x_w))
                x_mi = x_w % 1.0

                y_w = f_v_inv(j)
                y_lw = int(np.floor(y_w))
                y_hi = int(np.ceil(y_w))
                y_mi = y_w % 1.0
                try:
                    new_wert = x_mi * y_mi * self.colors[x_hi, y_hi]
                    new_wert += x_mi * (1-y_mi) * self.colors[x_hi, y_lw]
                    new_wert += (1-x_mi) * y_mi * self.colors[x_lw, y_hi]
                    new_wert += (1-x_mi) * (1-y_mi) * self.colors[x_lw, y_lw]
                    newmat[i, j] = new_wert
                except IndexError:
                    print("IE Dolle")
                    print(x_lw, x_hi, y_lw, y_hi)


        self.img = self.newImage(*(np.shape(newmat)))
        self.colors = newmat

    def __str__(self):
        return str(self.colors)

    @measureTime
    def updateImage(self):
        pixels = self.img.load()
        for i in range(self.img.size[0]):
            for j in range(self.img.size[1]):
                pixels[i, j] = self.rgb_map(self.colors[i, j])

    @measureTime
    def newImage(self, w=None, h=None):
        if w is not None:
            wid = w
        else:
            wid = self.width
        if h is None:
            hei = self.height
        else:
            hei = h
        img = Image.new('RGB', (int(np.ceil(wid)), int(np.ceil(hei))), 0)
        return img

    # Add Noise to image
    # currently random dirstibuted, possibly change to normal?
    @measureTime
    def noise(self, mu, sigma):
        # if self.noised:
        #    return
        # self.noised = True
        self.colors += np.random.normal(mu, sigma, np.shape(self.colors))

    @measureTime
    def get_matrix(self):
        return self.colors

    @measureTime
    def showImage(self):
        self.img.show()

    @measureTime
    def saveImage(self, filename):
        try:
            self.img.save(filename)
        except FileNotFoundError:
            os.mkdir(cfg.get_image_folder())
            self.img.save(filename)
        return filename

    @measureTime
    @lru_cache
    def rgb_map(self, h):
        mode = self.color_scheme
        if mode == "Gray":
            gsc = int(h)
            return gsc, gsc, gsc

        if mode == "WSXM":
            xs_blue = [0, 85, 160, 220, 255]
            ys_blue = [255, 255, 179, 42, 2]

            xs_green = [0, 17, 47, 106, 182, 232, 255]
            ys_green = [255, 255, 234, 159, 47, 7, 6]

            xs_red = [0, 32, 94, 152, 188, 255]
            ys_red = [255, 136, 57, 17, 6, 6]
        else:
            raise NotImplementedError


        r = None
        g = None
        b = None
        x = h
        for i in range(len(xs_green)-1):
            if xs_green[i] <= x < xs_green[i+1]:
                g = ys_green[i] + ((ys_green[i + 1] - ys_green[i]) / (xs_green[i + 1] - xs_green[i])) * (x - xs_green[i])
                break
        if g is None:
            if x > xs_green[-1]:
                g = ys_green[-1]
            elif x < xs_green[0]:
                g = ys_green[0]
            else:
                raise ValueError

        for i in range(len(xs_blue) - 1):
            #print("{} <= {} < {}".format(xs_blue[i], x, xs_blue[i+1]))
            if xs_blue[i] <= x < xs_blue[i + 1]:
                #print("Yes")
                b = ys_blue[i] + ((ys_blue[i+1] - ys_blue[i])/(xs_blue[i+1] - xs_blue[i])) * (x - xs_blue[i])
                #print("Ret {}".format(b))
                break
        if b is None:
            if x > xs_blue[-1]:
                b = ys_blue[-1]
            elif x < xs_blue[0]:
                b = ys_blue[0]
            else:
                raise ValueError

        for i in range(len(xs_red) - 1):
            if xs_red[i] <= x < xs_red[i + 1]:
                r = ys_red[i] + ((ys_red[i+1] - ys_red[i])/(xs_red[i+1] - xs_red[i])) * (x - xs_red[i])
                break
        if r is None:
            if x > xs_red[-1]:
                r = ys_red[-1]
            elif x < xs_red[0]:
                r = ys_red[0]
            else:
                raise ValueError
        #print("h={:.1f} -> ({}, {}, {})".format(x, 255-int(r), 255-int(g), 255-int(b)))
        return 255-int(r), 255-int(g), 255-int(b)

    @measureTime
    def rgb_map_test(self):
        x = []
        r = []
        g = []
        b = []

        for i in range(255):

            aa, bb, cc = self.rgb_map(i)
            x.append(i)
            r.append(aa)
            g.append(bb)
            b.append(cc)
        plt.plot(x, r, label="r")
        plt.plot(x, g, label="g")
        plt.plot(x, b, label="b")
        plt.show()
