from PIL import Image
import numpy as np
import Configuration as cfg
import os, random
# import Particle


class MyImage:
    width = cfg.get_width()
    height = cfg.get_height()
    # noise = True
    colors = np.zeros((int(np.ceil(width.px)), int(np.ceil(height.px))))
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
        self.colors = np.zeros((self.width, self.height))

    def setHeight(self, h):
        self.height = h
        self.colors = np.zeros((self.width, self.height))


    def addParticle(self, particle):
        self.colors = self.colors + particle.toMatrix()

    def addMatrix(self, matrix):
        self.colors = self.colors + matrix

    def double_tip(self, strength, rel_dist, angle): #ToDo: neues Bild damit auch links oben etc was ist
        #ToDo: Use surrounding Pictures
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
                    #print("Guuu")
                except IndexError as ie:
                    #print(ie)
                    pass


        self.colors = newmat
        self.updateImage()




    def __str__(self):
        return str(self.colors)

    def updateImage(self):
        pixels = self.img.load()
        for i in range(self.img.size[0]):
            for j in range(self.img.size[1]):
                pixels[i, j] = (int(self.colors[i, j]), int(self.colors[i, j]), int(self.colors[i, j]))

    def newImage(self):
        img = Image.new('RGB', (int(np.ceil(self.width.px)), int(np.ceil(self.height.px))), 0)
        return img

    # Add Noise to image
    # currently random dirstibuted, possibly change to normal?
    def noise(self, mu, sigma):
        #if self.noised:
        #    return
        #self.noised = True
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
