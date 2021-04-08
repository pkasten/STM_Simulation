import Data
# import Pillow
from PIL import Image
import numpy as np
# import scipy as sp
from Configuration import Files as files
from Maths import Functions
from Configuration import Configuration
from Maths.Functions import measureTime

func = Functions.Functions()


class Images:
    width = Configuration.ConfigManager.get_width()
    height = Configuration.ConfigManager.get_height()

    colors = np.zeros((width, height))
    sigma = 5.5

    img = ""
    filemanager = files.MultiFileManager()
    filename_generator = ""

    def __init__(self, filemanager, filename_generator):
        self.img = self.newImage()
        self.filemanager = filemanager
        self.filename_generator = filename_generator

    @measureTime
    def getWidth(self):
        return self.width

    @measureTime
    def getHeight(self):
        return self.height

    @measureTime
    def addPoint(self, coo):
        adding = np.zeros((self.width, self.height))
        for i in range(self.width):
            for j in range(self.height):
                adding[i, j] = 255 * func.normalDistribution(coo, (i, j), self.sigma)
        self.colors = self.colors + adding

    @measureTime
    def updateImage(self):
        pixels = self.img.load()
        for i in range(self.img.size[0]):
            for j in range(self.img.size[1]):
                pixels[i, j] = (int(self.colors[i, j]), int(self.colors[i, j]), int(self.colors[i, j]))

    @measureTime
    def newImage(self):
        img = Image.new('RGB', (self.width, self.height), 0)
        return img

    @measureTime
    def showImage(self):
        self.img.show()

    @measureTime
    def createImage(self, data):
        Configuration.ConfigManager.setup()
        self.newImage()
        while data.hasPoints():
            self.addPoint(data.getPoint())
        self.updateImage()

    @measureTime
    def saveImage(self):
        fp = self.filename_generator.generate()
        self.img.save(fp)
        return fp
