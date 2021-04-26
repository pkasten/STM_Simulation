from PIL import Image
import numpy as np
import Configuration as cfg
import os
# import Particle


class MyImage:
    width = cfg.get_width()
    height = cfg.get_height()
    noise = True
    colors = np.zeros((width, height))
    sigma = 5.5

    img = ""
    filename_generator = ""

    def __init__(self):
        self.img = self.newImage()

    def getWidth(self):
        return self.width

    def getHeight(self):
        return self.height

    def addParticle(self, particle):
        self.colors = self.colors + particle.toMatrix()

    def addMatrix(self, matrix):
        self.colors = self.colors + matrix

    def __str__(self):
        return str(self.colors)

    def updateImage(self):
        pixels = self.img.load()
        for i in range(self.img.size[0]):
            for j in range(self.img.size[1]):
                pixels[i, j] = (int(self.colors[i, j]), int(self.colors[i, j]), int(self.colors[i, j]))

    def newImage(self):
        img = Image.new('RGB', (self.width, self.height), 0)
        return img

    def showImage(self):
        self.img.show()

    def saveImage(self, filename):
        try:
            self.img.save(filename)
        except FileNotFoundError:
            os.mkdir(cfg.get_image_folder())
            self.img.save(filename)
        return filename
