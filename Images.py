import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import Configuration as cfg
import os, random


# import Particle


class MyImage:
    width = cfg.get_width().px
    height = cfg.get_height().px
    # noise = True
    colors = np.zeros((int(np.ceil(width)), int(np.ceil(height))))
    sigma = 5.5

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

    def getWidth(self):
        return self.width

    def getHeight(self):
        return self.height

    def setWidth(self, w):
        self.width = w
        self.colors = np.zeros((int(np.ceil(self.width)), int(np.ceil(self.height))))

    def setHeight(self, h):
        self.height = h
        self.colors = np.zeros((int(np.ceil(self.width)), int(np.ceil(self.height))))

    def addParticle(self, particle):
        self.colors = self.colors + particle.toMatrix()

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

    def shift_image(self, f_horiz=None, f_vert=None):
        style = "Exp"
        print("Prev:")
        plt.imshow(self.colors)
        plt.show()
        if style == "Linear":
            if f_horiz is None:
                f_h = lambda x: 1.1 * x
            else:
                f_h = f_horiz
            if f_vert is None:
                f_v = lambda y: 1.1 * y
            else:
                f_v = f_vert

            f_h_inv = lambda x: x/1.1
            f_v_inv = lambda x: x/1.1
        elif style == "Exp":

            if f_horiz is None:
                f_h = lambda x: 1.1 * x
            else:
                f_h = f_horiz
            if f_vert is None:
                f_v = lambda y: 1.1 * y
            else:
                f_v = f_vert

            f_h_inv = lambda x: x / 1.1
            f_v_inv = lambda x: x / 1.1
        else:
            raise NotImplementedError

        w = int(np.floor(f_h(np.shape(self.colors)[0] - 1)))
        h = int(np.floor(f_v(np.shape(self.colors)[1] - 1)))

        newmat = np.zeros((w, h))

        for i in range(np.shape(newmat)[0]):
            for j in range(np.shape(newmat)[1]):
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



        print("After:")
        plt.imshow(newmat)
        plt.show()

        self.img = self.newImage(*(np.shape(newmat)))
        self.colors = newmat

    def __str__(self):
        return str(self.colors)

    def updateImage(self):
        pixels = self.img.load()
        for i in range(self.img.size[0]):
            for j in range(self.img.size[1]):
                pixels[i, j] = (int(self.colors[i, j]), int(self.colors[i, j]), int(self.colors[i, j]))

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
    def noise(self, mu, sigma):
        # if self.noised:
        #    return
        # self.noised = True
        self.colors += np.random.normal(mu, sigma, np.shape(self.colors))

    def get_matrix(self):
        return self.colors

    def showImage(self):
        self.img.show()

    def saveImage(self, filename):
        try:
            self.img.save(filename)
        except FileNotFoundError:
            os.mkdir(cfg.get_image_folder())
            self.img.save(filename)
        return filename
