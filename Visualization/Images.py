import Data
# import Pillow
from PIL import Image
import numpy as np
# import scipy as sp
import os
from Configuration import Files as files
from Maths import Functions
from Configuration import Configuration
from Maths.Functions import measureTime
import math, random

func = Functions.Functions()

aspectRange = 3
aspectRange = 3


class Images:
    width = Configuration.ConfigManager.get_width()
    height = Configuration.ConfigManager.get_height()
    noise = True
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
        dist = aspectRange * self.sigma
        xmin = int(max(0, coo[0] - dist))
        xmax = int(min(self.width, coo[0] + dist))
        ymin = int(max(0, coo[1] - dist))
        ymax = int(min(self.height, coo[1] + dist))
        adding = np.zeros((self.width, self.height))
        for i in range(xmin, xmax):
            for j in range(ymin, ymax):
                adding[i, j] = 255 * func.normalDistribution(coo, (i, j), self.sigma)
        self.colors = self.colors + adding

    @measureTime
    def addPoints(self, coo):  # Not efficient
        adding = np.zeros((self.width, self.height))
        for i in range(self.width):
            for j in range(self.height):
                for pt in coo:
                    adding[i, j] = 255 * func.normalDistribution(pt, (i, j), self.sigma)
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
        coos = []
        #dummyData = data.clone()
        while data.hasPoints():
            # coos.append(data.getPoint())
            self.addPoint(data.getPoint())
        # self.addPoints(coos)
        self.updateImage()

    @measureTime
    def noiseImage(self):
        if self.noise:
            dist = aspectRange * self.sigma
            xmin = 0
            xmax = self.width
            ymin = 0
            ymax = self.height
            adding = np.zeros((self.width, self.height))
            for i in range(xmin, xmax):
                for j in range(ymin, ymax):
                    if(random.random() < 0.2):
                        adding[i, j] += math.floor(random.random() * 50)

            self.colors = self.colors + adding

    @measureTime
    def saveImage(self, index):
        # fp = os.getcwd() + "/bildordner/Image" + str(index) + ".png"
        fp = os.path.join(os.getcwd(), "bildordner", "Image" + str(index) + ".png")
        self.img.save(fp)
        return fp, index
