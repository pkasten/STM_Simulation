import Data
# import Pillow
from PIL import Image
import numpy as np
# import scipy as sp
from Maths import Functions
from Configuration import Configuration

func = Functions.Functions()


class Images:
    width = Configuration.ConfigManager.get_width()
    height = Configuration.ConfigManager.get_height()

    colors = np.zeros((width, height))
    sigma = 5.5

    img = ""

    def __init__(self):
        self.img = self.newImage()

    def getWidth(self):
        return self.width

    def getHeight(self):
        return self.height

    def addPoint(self, coo):
        adding = np.zeros((self.width, self.height))
        adding = 255 * func.normalDistribution(adding, coo, self.sigma)
        print(adding)
        print(np.shape(adding)) #ToDo: Correct
        self.colors = self.colors + adding

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

    def createImage(self, data):
        Configuration.ConfigManager.setup()
        self.newImage()
        while data.hasPoints():
            self.addPoint(data.getPoint())
        self.updateImage()
        self.showImage()
