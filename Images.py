from PIL import Image
import numpy as np
import Configuration as cfg
import os, random
# import Particle


class MyImage:
    width = cfg.get_width()
    height = cfg.get_height()
    # noise = True
    colors = np.zeros((width, height))
    sigma = 5.5


    img = ""
    filename_generator = ""

    def __init__(self):
        self.img = self.newImage()
        self.noised = False

    def getWidth(self):
        return self.width

    def getHeight(self):
        return self.height


    def addParticle(self, particle):
        self.colors = self.colors + particle.toMatrix()

    def addMatrix(self, matrix):
        self.colors = self.colors + matrix

    def double_tip(self, strength, rel_dist, angle, surrounding): #ToDo: neues Bild damit auch links oben etc was ist
        #ToDo: Use surrounding Pictures
        print("Double_Tipping")
        vec_x = np.floor(np.sqrt(np.square(self.width) + np.square(self.height)) * rel_dist * np.sin(angle))
        vec_y = np.floor(np.sqrt(np.square(self.width) + np.square(self.height)) * rel_dist * np.cos(angle))
        vec_x = int(vec_x)
        vec_y = int(vec_y)
        range_A = 0, np.shape(self.colors)[0] - 1, 0, np.shape(self.colors)[1] - 1
        range_B = np.shape(self.colors)[0], 2 * np.shape(self.colors)[0] - 1,  0, np.shape(self.colors)[1] - 1
        range_C = 0, np.shape(self.colors)[0] - 1, 0, np.shape(self.colors)[1], 2 * np.shape(self.colors)[1] - 1
        range_D = np.shape(self.colors)[0], 2 * np.shape(self.colors)[0] - 1,  np.shape(self.colors)[1], 2 * np.shape(self.colors)[1] - 1
        orig = None
        print("D: {}".format(range_D[0]))
        newmat = np.zeros((2* np.shape(self.colors)[0], 2* np.shape(self.colors)[1]))

        def set_Martix_At_Range(newmat, range_X, matrix):
            #nonlocal newmat
            print(range_X)
            for p_local_var in range(range_X[0], range_X[1]):
                for q_local_var in range(range_X[2], range_X[3]):
                    p_tilt = p_local_var - range_X[0]
                    q_tilt = q_local_var - range_X[2]
                    newmat[p_local_var, q_local_var] = matrix[p_tilt, q_tilt]

        if vec_x > 0:
            set_Martix_At_Range(newmat, range_A, surrounding[0])
            set_Martix_At_Range(newmat, range_C, surrounding[1])
            if vec_y > 0:
                set_Martix_At_Range(newmat, range_B, surrounding[2])
                set_Martix_At_Range(newmat, range_D, self.colors)
                orig = range_D
            else:
                set_Martix_At_Range(newmat, range_B, self.colors)
                orig =range_B
                set_Martix_At_Range(newmat, range_D, surrounding[2])
        else:
            set_Martix_At_Range(newmat, range_B, surrounding[0])
            set_Martix_At_Range(newmat, range_D, surrounding[1])
            if vec_y > 0:
                set_Martix_At_Range(newmat, range_A, surrounding[2])
                set_Martix_At_Range(newmat, range_C, self.colors)
                orig = range_C
            else:
                set_Martix_At_Range(newmat, range_A, self.colors)
                orig = range_A
                set_Martix_At_Range(newmat, range_C, surrounding[2])

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

        for p in range(orig[0], orig[1]):
            for q in range(orig[2], orig[3]):
                p_tilt = p - orig[0]
                q_tilt = q - orig[2]
                self.colors[p_tilt, q_tilt] = newmat[p, q]

        self.updateImage()




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

    # Add Noise to image
    # currently random dirstibuted, possibly change to normal?
    def noise(self, mu, sigma):
        if self.noised:
            return
        self.noised = True
        self.colors += np.random.normal(mu, sigma, np.shape(self.colors))



    def showImage(self):
        self.img.show()

    def saveImage(self, filename):
        try:
            self.img.save(filename)
        except FileNotFoundError:
            os.mkdir(cfg.get_image_folder())
            self.img.save(filename)
        return filename
