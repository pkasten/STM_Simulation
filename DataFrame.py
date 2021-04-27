import multiprocessing as mp
import copy, os
import math
from Particle import Particle
from Images import MyImage
#from Maths.Functions import measureTime
#from Configuration.Files import MultiFileManager as fm
import Configuration as cfg
import numpy as np
from Functions import measureTime


class DataFrame:

    def __init__(self, fn_gen):
        self.objects = []
        self.fn_gen = fn_gen
        self.text = ""
        self.img = None

    def getIterator(self):
        return self.objects

    def __len__(self):
        return len(self.objects)

    def addParticle(self, part=None):
        if part is None:
            self.objects.append(Particle())
        else:
            self.objects.append(part)

    def addParticles(self, n):
        for i in range(n):
            self.objects.append(Particle())

    @measureTime
    def createImage(self):
        self.img = MyImage()
        for part in self.objects:
            self.img.addParticle(part)
        self.img.updateImage()
        #img.noise....etc

    @measureTime
    def createImage_efficient(self):
        self.img = MyImage()
        width = cfg.get_width()
        height = cfg.get_height()
        matrix = np.zeros((width, height))

        for part in self.objects:
            eff_mat, x, y = part.efficient_Matrix()
            mat_w = eff_mat.shape[0]

            mat_h = eff_mat.shape[1]
            for i in range(mat_w):
                for j in range(mat_h):
                    new_x = x - math.floor((mat_w/2)) + i
                    new_y = y - math.floor(mat_h/2) + j
                    if not (0 <= new_x < width and 0 <= new_y < height):
                        continue
                    matrix[new_x, new_y] += eff_mat[i, j]

        self.img.addMatrix(matrix)
        self.img.updateImage()

    @measureTime
    def createImage_efficient_with_new_Turn(self):
        self.img = MyImage()
        width = cfg.get_width()
        height = cfg.get_height()
        matrix = np.zeros((width, height))

        for part in self.objects:
            eff_mat, x, y = part.efficient_Matrix_turned()
            mat_w = eff_mat.shape[0]

            mat_h = eff_mat.shape[1]
            for i in range(mat_w):
                for j in range(mat_h):
                    new_x = x - math.floor((mat_w / 2)) + i
                    new_y = y - math.floor(mat_h / 2) + j
                    if not (0 <= new_x < width and 0 <= new_y < height):
                        continue
                    matrix[new_x, new_y] += eff_mat[i, j]

        self.img.addMatrix(matrix)
        self.img.updateImage()

    def createText(self):
        strings = [Particle.str_Header()]
        for part in self.objects:
            strings.append(str(part))
        self.text = "\n".join(strings)

    def save(self):
        if self.img is None:
            self.createImage_efficient()
        if len(self.text) == 0:
            self.createText()
        img_path, dat_path, index = self.fn_gen.generate_Tuple()
        try:
            with open(dat_path, "w") as dat_file:
                dat_file.write(self.text)
        except FileNotFoundError:
            os.mkdir(cfg.get_data_folder())
            with open(dat_path, "w") as dat_file:
                dat_file.write(self.text)
        self.img.saveImage(img_path)


    def hasPoints(self):
        # return not self.points.empty()
        return len(self.objects) > 0

    def __str__(self):
        return str(self.points)

