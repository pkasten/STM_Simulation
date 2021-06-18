from functools import lru_cache

import matplotlib.pyplot as plt
import numpy.fft
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

    # @measureTime
    def getWidth(self):
        return self.width

    # @measureTime
    def getHeight(self):
        return self.height

    # @measureTime
    def setWidth(self, w):
        self.width = w
        self.colors = np.zeros((int(np.ceil(self.width)), int(np.ceil(self.height))))

    # @measureTime
    def setHeight(self, h):
        self.height = h
        self.colors = np.zeros((int(np.ceil(self.width)), int(np.ceil(self.height))))

    # @measureTime
    def addParticle(self, particle):
        self.colors = self.colors + particle.toMatrix()

    # @measureTime
    def addMatrix(self, matrix):
        self.colors = self.colors + matrix

    #@DeprecationWarning
    def double_tip(self, strength, rel_dist, angle):  # ToDo: neues Bild damit auch links oben etc was ist
        # ToDo: Use surrounding Pictures
        vec_x = np.floor(np.sqrt(np.square(self.width) + np.square(self.height)) * rel_dist * np.sin(angle))
        vec_y = np.floor(np.sqrt(np.square(self.width) + np.square(self.height)) * rel_dist * np.cos(angle))
        vec_x = int(vec_x)
        vec_y = int(vec_y)

        #print("Images.double_tip start: ")
        #plt.imshow(self.colors)
        #plt.show()

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
        #print("Double tip end")
        #plt.imshow(newmat)
        #plt.show()
        self.updateImage()

    # @measureTime
    def shift_image(self, f_horiz=None, f_vert=None):

        # testmat = np.ones(np.shape(self.colors))
        # testmat *= 200
        # self.colors = testmat

        style = cfg.get_shift_style()
        factor_x = cfg.get_shift_amount_x()
        factor_y = cfg.get_shift_amount_y()

        if style == "Lin":
            if f_horiz is None:
                f_h = lambda x: factor_x * x
            else:
                f_h = f_horiz
            if f_vert is None:
                f_v = lambda y: factor_y * y
            else:
                f_v = f_vert

            print("Calcing inv")
            f_h_inv = lambda x: x / factor_x
            f_v_inv = lambda x: x / factor_y
            # f_h_inv = get_invers_function(f_h)
            # f_v_inv = get_invers_function(f_v)
            print("Got inv")
        elif style == "Exp":
            wid, heigth = np.shape(self.colors)
            # gamma_x = np.log(wid * factor_x) / wid
            # gamma_y = np.log(heigth * factor_y) / heigth
            gamma_x = np.log((factor_x - 1) * wid) / wid
            gamma_y = np.log((factor_y - 1) * heigth) / heigth

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
            print("Unknown Image Shift Style")
            raise NotImplementedError

        w = int(np.floor(f_h(np.shape(self.colors)[0] - 1)))
        h = int(np.floor(f_v(np.shape(self.colors)[1] - 1)))

        newmat = np.zeros((w, h))
        i_max = np.shape(newmat)[0]
        for i in range(0, np.shape(newmat)[0]):
            #print("Shift progress: {:.1f}%".format(100 * i / i_max))
            for j in range(0, np.shape(newmat)[1]):
                x_w = f_h_inv(i)
                # y_w = f_v(j)
                x_lw = int(np.floor(x_w))
                x_hi = int(np.ceil(x_w))
                x_mi = x_w % 1.0

                y_w = f_v_inv(j)
                y_lw = int(np.floor(y_w))
                y_hi = int(np.ceil(y_w))
                y_mi = y_w % 1.0
                try:
                    new_wert = x_mi * y_mi * self.colors[x_hi, y_hi]
                    new_wert += x_mi * (1 - y_mi) * self.colors[x_hi, y_lw]
                    new_wert += (1 - x_mi) * y_mi * self.colors[x_lw, y_hi]
                    new_wert += (1 - x_mi) * (1 - y_mi) * self.colors[x_lw, y_lw]
                    newmat[i, j] = new_wert
                except IndexError:
                    print("IE Dolle")
                    print(x_lw, x_hi, y_lw, y_hi)

        self.img = self.newImage(*(np.shape(newmat)))
        self.colors = newmat

    def scan_lines(self):
        style = "horizontal"
        if style == "horizontal":
            med_strength = 0.1 * cfg.get_width().px
            variation = med_strength
            poss = 0.5
            y = 0
            while y < np.shape(self.colors)[1]:
                if random.random() < poss:
                    x_med = int(random.randint(0, np.shape(self.colors)[0]))
                    x_len = int(np.random.normal(med_strength, variation) / 2)
                    x_start = max(0, x_med - x_len)
                    x_end = min(x_med + x_len, np.shape(self.colors)[0])
                    #print("xmed = {}, xlen = {}, xstart = {}, xend = {}".format(x_med, x_len, x_start, x_end))
                    if x_start >= x_end:
                     #   print("Start >= End")
                        y += 1
                        continue
                    cols = self.colors[x_start:x_end, y]
                    color = random.choice(cols)
                    for x in range(x_start, x_end):
                        self.colors[x,y] = color
                else:
                    y += 1


    def __str__(self):
        return str(self.colors)

    # @measureTime
    def updateImage(self):
        pixels = self.img.load()
        for i in range(self.img.size[0]):
            for j in range(self.img.size[1]):
                pixels[i, j] = self.rgb_map(self.colors[i, j])

    # @measureTime
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
    # @measureTime
    def noise(self, mu, sigma):
        # if self.noised:
        #    return
        # self.noised = True
        self.colors += np.random.normal(mu, sigma, np.shape(self.colors))

    def noise_function(self):
        noise_mat = np.zeros(np.shape(self.colors))
        w, h = np.shape(noise_mat)
        two_pi = 2 * np.pi
        gen_amplitude = lambda x:1
        shift = 90
        phi = lambda x:two_pi*random.random()
        step = 1



        def f_alpha(n_pts, q_d, alpha, idum):
            xs = []
            nn = n_pts + n_pts
            ha = alpha/2
            q_d = np.sqrt(q_d)

            hfa = [1, nn]
            wfa = [1, nn]
            hfa[1] = 1.0
            wfa[1] = q_d * np.random.normal()

            for i in range(2, n_pts+1):
                hfa.append(hfa[i-1] * (ha + i-2)/i-1)
                wfa.append(q_d * random.random())

            for i in range(n_pts + 1, nn):
                hfa.append(0)
                wfa.append(0)

     #       print("wfa start")
    #        plt.plot(wfa)
   #         plt.show()
  #          print("hfa start")
 #           plt.plot(hfa)
#            plt.show()

            print(len(hfa))

            reth = numpy.fft.rfft(hfa, n_pts) # , 1
            retw = numpy.fft.rfft(wfa, n_pts)

            for i in range(len(reth)):
                hfa[i] = reth[i]

            for i in range(len(retw)):
                wfa[i] = retw[i]

            wfa[1] = wfa[1] * hfa[1]
            wfa[2] = wfa[2] * hfa[2]

     #       print("premodified wfa")
    #        plt.plot(wfa)
   #         plt.show()


            for i in range(3, nn-1, 2):
                wr=wfa[i]
                wi=wfa[i+1]
                wfa[i] = wr * hfa[i] - wi * hfa[i+1]
                wfa[i+1] = wr * hfa[i+1] + wi * hfa[i]

  #          print("Wfa vor iff")
 #           plt.plot(wfa)
#            plt.show()

            retw = np.fft.irfft(wfa, n_pts)

     #       print("retw")
    #        plt.plot(retw)
   #         plt.show()


            for i in range(len(retw)):
                wfa[i] = retw[i]


            for i in range(1, n_pts+1):
                xs.append( wfa[i]/n_pts)


  #          plt.plot(xs)
 #           plt.show()
            return xs



        def f_alpha_2D(n_pts, q_d, alpha):
            vals = np.zeros((n_pts, n_pts))
            nn = n_pts + n_pts
            ha = alpha/2
            q_d = np.sqrt(q_d)

            hfa = np.zeros((nn, nn))
            wfa = np.zeros((nn, nn))

            hfa[1, 1] = 1.0
            wfa[1, 1] = q_d * np.random.normal()
            hfa[0, 1] = 1.0
            wfa[0, 1] = q_d * np.random.normal()
            hfa[1, 0] = 1.0
            wfa[1, 0] = q_d * np.random.normal()
            hfa[0, 0] = 1.0
            wfa[0, 0] = 1

            for i in range(1, n_pts+1):
                hfa[0, i] = q_d * np.random.normal()
                hfa[i, 0] = q_d * np.random.normal()
                wfa[0, i] = 1
                wfa[i, 0] = 1

            for i in range(1, n_pts+1):
                for j in range(1, n_pts+1):
                    #h = hfa[i-1, j] * (ha + i-2)/i-1
                    #h + hfa[i, j-1] * (ha + j-2)/j-1
                    #h + hfa[i-1, j-1] * (ha +
                    h = ha * (2* n_pts - i - j)
                    hfa[i, j] = h
                    wfa[i, j] = (q_d * random.random())

            for i in range(n_pts + 1, nn):
                for j in range(n_pts + 1, nn):
                    hfa[i, j] = 0
                    wfa[i, j] = 0

     #       print("wfa start")
    #        plt.imshow(wfa)
   #         plt.show()
  #          print("hfa start")
 #           plt.imshow(hfa)
#            plt.show()


            reth = numpy.fft.rfft2(hfa, (n_pts, n_pts)) # , 1
            retw = numpy.fft.rfft2(wfa, (n_pts, n_pts))

            #print("ret fft w")
            #plt.imshow(retw)
            #plt.show()
            #print("ret_fft_h")
            #plt.imshow(reth)
            #plt.show()

            for i in range(np.shape(reth)[0]):
                for j in range(np.shape(reth)[1]):
                    hfa[i, j] = reth[i, j]

            for i in range(np.shape(retw)[0]):
                for j in range(np.shape(retw)[1]):
                    wfa[i, j] = retw[i, j]

            wfa[1, 1] = wfa[1, 1] * hfa[1, 1]
            wfa[2, 2] = wfa[2, 2] * hfa[2, 2]
            wfa[2, 1] = wfa[2, 1] * hfa[2, 1]
            wfa[1, 2] = wfa[1, 2] * hfa[1, 2]

  #          print("premodified wfa")
 #           plt.imshow(wfa)
#            plt.show()


            for i in range(3, nn-1, 2):
                for j in range(3, nn-1, 2):
                    wr=wfa[i, j] * 0
                    wi=wfa[i+1, j]  *0
                    wk = wfa[i, j+1] *0
                    wfa[i] = wr * hfa[i, j] - wi * hfa[i+1, j] - wk * hfa[i, j+1]
                    wfa[i+1] = wr * hfa[i+1, j] + wi * hfa[i, j] + wk * hfa[i, j+1]

       #     print("Wfa vor iff")
      #      plt.imshow(wfa)
     #       plt.show()

            retw = np.fft.irfft2(wfa, (n_pts, n_pts))

    #        print("retw")
   #         plt.imshow(retw)
  #          plt.show()

            for i in range(np.shape(retw)[0]):
                for j in range(np.shape(retw)[1]):
                    wfa[i, j] = retw[i, j]


            for i in range(1, n_pts):
                for j in range(1, n_pts):
                    vals[i, j] = ( wfa[i, j]/n_pts)

            print("vals")
            plt.imshow(vals)
            plt.show()
            return vals



        xs = f_alpha(100, 1, 0, 1)
        #plt.plot(range(len(xs)), xs)
        #plt.show()

        print("2d")

        mat = f_alpha_2D(200, 1, 1)
        #print(mat)
        sc = 100 / np.amax(mat)
        shif = 10

        mat *= sc
        for i in range(np.shape(mat)[0]):
            for j in range(np.shape(mat)[1]):
                mat[i, j] += shif
        noise_mat = mat






        def gen_single(width, nu):
            xi = gen_amplitude(nu)
            p = phi(nu)
            print("Single Genned: {:.2f} sin(2pi x {} / {} + {:.2f})".format(xi, nu, width, p))
            return lambda x: xi * np.sin(two_pi * x * nu / width + p)

        def gen_f(width):
            func = lambda x:shift
            fs = []
            for nu in range(1, width, step):
                fs.append(gen_single(width, nu))

            def f(x):
                sum = 0
                for i in range(len(fs)):
                    sum += fs[i](x)

                return sum

            xs = np.linspace(0, width, 100)
            ys = []
            for x in xs:
                ys.append(f(x))
            plt.plot(xs, ys)
            plt.show()

            return f




        #f_x = gen_f(w)



        #for i in range(w):
        #    for j in range(h):
        #        noise_mat[i, j] = f_x(i) + f_x(j) + shift#

        #plt.imshow(noise_mat)
        #plt.show()
        print("Max: {:.2f}".format(np.amax(noise_mat)))
        print("Min: {:.2f}".format(np.amin(noise_mat)))
        print("Avg: {:.2f}".format(np.average(noise_mat)))

        self.colors += noise_mat



    # @measureTime
    def get_matrix(self):
        return self.colors

    # @measureTime
    def showImage(self):
        self.img.show()

    # @measureTime
    def saveImage(self, filename):
        try:
            self.img.save(filename)
        except FileNotFoundError:
            os.mkdir(cfg.get_image_folder())
            self.img.save(filename)
        return filename

    # @measureTime
    @lru_cache
    def rgb_map(self, h):
        mode = self.color_scheme
        mode = mode.lower()
        if mode == "gray":
            gsc = int(h)
            return gsc, gsc, gsc

        if mode == "wsxm":
            xs_blue = [0, 85, 160, 220, 255]
            ys_blue = [255, 255, 179, 42, 2]

            xs_green = [0, 17, 47, 106, 182, 232, 255]
            ys_green = [255, 255, 234, 159, 47, 7, 6]

            xs_red = [0, 32, 94, 152, 188, 255]
            ys_red = [255, 136, 57, 17, 6, 6]
        else:
            print("ERROR: Could not recognize imaging mode {}".format(mode))
            raise NotImplementedError

        r = None
        g = None
        b = None
        x = h
        for i in range(len(xs_green) - 1):
            if xs_green[i] <= x < xs_green[i + 1]:
                g = ys_green[i] + ((ys_green[i + 1] - ys_green[i]) / (xs_green[i + 1] - xs_green[i])) * (
                            x - xs_green[i])
                break
        if g is None:
            if x > xs_green[-1]:
                g = ys_green[-1]
            elif x < xs_green[0]:
                g = ys_green[0]
            else:
                raise ValueError

        for i in range(len(xs_blue) - 1):
            # print("{} <= {} < {}".format(xs_blue[i], x, xs_blue[i+1]))
            if xs_blue[i] <= x < xs_blue[i + 1]:
                # print("Yes")
                b = ys_blue[i] + ((ys_blue[i + 1] - ys_blue[i]) / (xs_blue[i + 1] - xs_blue[i])) * (x - xs_blue[i])
                # print("Ret {}".format(b))
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
                r = ys_red[i] + ((ys_red[i + 1] - ys_red[i]) / (xs_red[i + 1] - xs_red[i])) * (x - xs_red[i])
                break
        if r is None:
            if x > xs_red[-1]:
                r = ys_red[-1]
            elif x < xs_red[0]:
                r = ys_red[0]
            else:
                raise ValueError
        # print("h={:.1f} -> ({}, {}, {})".format(x, 255-int(r), 255-int(g), 255-int(b)))
        return 255 - int(r), 255 - int(g), 255 - int(b)

    # @measureTime
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
