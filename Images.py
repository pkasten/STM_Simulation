import time
from functools import lru_cache

import matplotlib.pyplot as plt
import numpy.fft
from PIL import Image
import numpy as np
from tqdm import tqdm

import Configuration as cfg
import os, random

import Functions
import SXM_info
from Distance import Distance
from Functions import get_invers_function, measureTime
import csv


# import Particle


class MyImage:
    """
    Class to do all the work on Image creation and manipulation
    """

    # Import configuration parameters
    width = cfg.get_width().px
    height = cfg.get_height().px
    # Matrix to store grayscale values of visualization
    colors = np.zeros((int(np.ceil(width)), int(np.ceil(height))))
    sigma = 5.5
    color_scheme = cfg.get_color_scheme()

    img = ""
    filename_generator = ""

    def __init__(self, matrix=None):
        """
        Initializes new Image. Uses matrix as color matrix, Zeros otherwise
        :param matrix: image matrix
        """
        self.use_white_noise = cfg.use_white_noise()
        self.use_line_noise = cfg.use_line_noise()
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
        """
        Returns image width
        :return:
        """
        return self.width

    # @measureTime
    def getHeight(self):
        """
        Returns image height
        :return:
        """
        return self.height

    # @measureTime
    def setWidth(self, w):
        """
        Sets image width
        :param w:
        :return:
        """
        self.width = w
        self.colors = np.zeros((int(np.ceil(self.width)), int(np.ceil(self.height))))

    # @measureTime
    def setHeight(self, h):
        """
        Sets image height
        :param h:
        :return:
        """
        self.height = h
        self.colors = np.zeros((int(np.ceil(self.width)), int(np.ceil(self.height))))

    # @measureTime
    @DeprecationWarning
    def addParticle(self, particle):
        """
        DEPRECATED. Was used to add single particle to visu_matrix
        :param particle:
        :return:
        """
        self.colors = self.colors + particle.toMatrix()

    # @measureTime
    def addMatrix(self, matrix):
        """
        Adds an entire matrix to colors.
        Used for example to add noise to the image
        :param matrix: matrix to be added to colors
        :return:
        """
        self.colors = self.colors + matrix

    def double_tip(self, strength, rel_dist, angle):
        """
        Duplicates the image, shifts it by rel_dist at angle, scale it with strength and add it to colors
        :param strength: Strength of duplicated image
        :param rel_dist: relative shift distance
        :param angle: angle by which shift is done
        :return: None
        """
        vec_x = np.floor(np.sqrt(np.square(self.width) + np.square(self.height)) * rel_dist * np.sin(angle))
        vec_y = np.floor(np.sqrt(np.square(self.width) + np.square(self.height)) * rel_dist * np.cos(angle))
        vec_x = int(vec_x)
        vec_y = int(vec_y)

        # print("Images.double_tip start: ")
        # plt.imshow(self.colors)
        # plt.show()

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
        # print("Double tip end")
        # plt.imshow(newmat)
        # plt.show()
        self.updateImage()

    # @measureTime
    def shift_image(self, f_horiz=None, f_vert=None):
        """
        Used for piezo shift. Shift defined in stettings to the image
        :param f_horiz: Optional, Function for pixel position horizontal
        :param f_vert: Optional, Function for pixel position vertical
        :return:
        """
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
            if factor_x != 1:
                gamma_x = np.log((factor_x - 1) * wid) / wid
            if factor_y != 1:
                gamma_y = np.log((factor_y - 1) * heigth) / heigth

            if f_horiz is None:
                if factor_x == 1:
                    f_h = lambda x: x
                else:
                    f_h = lambda x: x + np.exp(gamma_x * x)
            else:
                f_h = f_horiz
            if f_vert is None:
                if factor_y == 1:
                    f_v = lambda y: y
                else:
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
            # print("Shift progress: {:.1f}%".format(100 * i / i_max))
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
        """
        Add scanlines to the image. can be done horizontally or vertically
        :return:
        """
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
                    # print("xmed = {}, xlen = {}, xstart = {}, xend = {}".format(x_med, x_len, x_start, x_end))
                    if x_start >= x_end:
                        #   print("Start >= End")
                        y += 1
                        continue
                    cols = self.colors[x_start:x_end, y]
                    color = random.choice(cols)
                    for x in range(x_start, x_end):
                        self.colors[x, y] = color
                else:
                    y += 1

    def __str__(self):
        return str(self.colors)

    # @measureTime
    def updateImage(self):
        """
        Updates the managed PIL.Image instance with color matrix
        :return:
        """
        pixels = self.img.load()
        for i in range(self.img.size[0]):
            for j in range(self.img.size[1]):
                pixels[i, j] = self.rgb_map(self.colors[i, j])

    # @measureTime
    def newImage(self, w=None, h=None):
        """
        Creates new PIL.Image instance for this image
        :param w: width
        :param h: height
        :return: Image instance
        """
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

    def noise(self, mu, sigma):
        """
        Adds noise to the image
        :param mu: expectation value (grayscale)
        :param sigma: Standard derivation of white noise
        :return:
        """
        # self.colors += self.noise_spektrum(sigma)
        self.colors += mu * np.ones(np.shape(self.colors))
        if self.use_white_noise:
            self.colors += np.random.normal(0, sigma, np.shape(self.colors))
        if self.use_line_noise:
            self.colors += self.f1_line_noise(0, sigma)

    def _noise_over_time(self, freq, intens):
        print("noise over Time")
        # Normalize
        scale = 1 / np.max(intens)
        temp = []
        for elem in intens:
            temp.append(scale * elem)
        intens = temp
        del temp

        # Test: Plot
        # plt.plot(frequency, intensity)
        # plt.title("Noise Spectrum")
        # plt.show()

        def get_phase_func(freq):
            if freq == 0:
                return lambda x: 0
            max_time = 600
            slotlength = 1 / freq
            slotnumber = int(np.ceil(max_time / slotlength))
            steps = 10
            startphase = lambda: 2 * np.pi * random.random()

           # def _gen_slope(left, right, ber):
           #     m = (right - left) / ber
           #     b = left

           #     def f(x):
            #        return m * x + b

            #    return f

            nextstep = lambda: 2 * np.pi * random.random() - np.pi
            sp = startphase()

            steady_len = steps * slotlength
            slope_len = slotlength

            t = 0
            pairs = [] # (time bis, valueLeft, Slope)
            oldphase = 0
            while t < max_time:
                t += steady_len
                oldphase += nextstep()
                pairs.append((t, oldphase, False))
                t += slope_len
                pairs.append((t, oldphase, True))

            t += steady_len
            pairs.append((t, oldphase, True))
            assert t > max_time

            def func(t):
                t %= max_time
                for i in range(len(pairs)):
                    if t < pairs[i][0]:
                        if pairs[i][2]:
                            dec = slope_len + t - pairs[i][0]
                            m = (pairs[i+1][1] - pairs[i][1])/slope_len
                            return pairs[i][1] + m * dec
                        else:
                            return pairs[i][1]


          #  print("Test Func:")
          #  fac = 100
          #  xs = [x/fac for x in range(2*fac*max_time)]
          #  ys = [func(t) for t in xs]

          #  plt.plot(xs, ys)
          #  plt.title(" Func, maxtime: {}".format(fac * max_time))
           # plt.show()

            return func








            #phases = [lambda x: sp, lambda x: sp]

            #steady = [True, True]
            #for i in range(slotnumber - 1):
            #    steady.append((i % (steps + 1)) != 0)

            #steady.append(True)
            #steady.append(True)

            #phases_appended = [sp, sp]

            #for i in range(2, len(steady)):

                # if i > 5:
                #    print("i={} -> phase[0](0) = {}".format(i, phases[0](0)))
                #    print("i={} -> phase[1](0) = {}".format(i, phases[1](0)))
                #    #print("i={} -> phase[2](0) = {}".format(i, phases[2](0)))
                #    print("i={} -> phase[3](0) = {}".format(i, phases[3](0)))
                #    print("i={} -> phase[4](0) = {}".format(i, phases[4](0)))
                #    print(phases_appended)

                #if steady[i] and steady[i - 1]:
                #    print("A")
                #    phases_appended.append(phases_appended[-1])#

               #     def f(x):
               #         return phases_appended[i]

               #     phases.append(f)
               # elif steady[i] and not steady[i - 1]:
                #    print("B")
                #    old = phases_appended[-1]
                #    new = old + nextstep()
                #    phases_appended.append(new)

                #    def f(x):
                #        return phases_appended[i]

               #     phases.append(f)
               # elif not steady[i] and steady[i - 1]:
               #     print("C")
               #     phases_appended.append(phases_appended[-1])
               #     phases.append("SLOPE")
               #     print("Sloping")
               # else:
               #     print("i = {}, steady[i]={}, steady[i-1]={}".format(i, steady[i], steady[i - 1]))
               #     raise ValueError


        def _to_wave(freq, ampl):
            # 1 Index = 1ms
            phasefkt = get_phase_func(freq)
            #f = lambda t: ampl * np.cos(2 * np.pi * freq * t + phasefkt(t))
            def f(t):
                return ampl * np.cos(2*np.pi * freq * t + phasefkt(t))

            return f


        waves = []
        times = range(10000)
        funcs = []
        for i in range(len(freq)):
            funcs.append(_to_wave(freq[i], intens[i]))

        def noise_ot(t):
            sum = -funcs[0](0)
            for f in funcs:
                sum += f(t)
            return sum

        testing = False
        if testing:
            print("Testing")
            print(intens[0])
            ampl = []
            for i in tqdm(times):
                ampl.append(noise_ot(i / 10000))

            plt.plot(times, ampl)
            plt.title("Sum")
            plt.show()

            # Shift to 0
            shift = np.average(ampl)
            temp = []
            for x in ampl:
                temp.append(x - shift)
            ampl = temp
            del temp

            plt.plot(times, ampl)
            plt.title("Ampl")
            plt.show()


        return noise_ot


    def _not_funcs(self, freq, intens):
        print("noise over Time")
        # Normalize
        scale = 1 / np.max(intens)
        temp = []
        for elem in intens:
            temp.append(scale * elem)
        intens = temp
        del temp


        # Nur wichtige
        no_of_inerest = 30
        interesting = []

        second = lambda elem: elem[1]

        for i in range(no_of_inerest):
            interesting.append((0, 0))

        mini = interesting[0][1]
        for i in range(1, len(freq)):
            curr_int = intens[i]
            if curr_int > mini:
                interesting.append((freq[i], intens[i]))
                interesting.sort(key=second)
                interesting.pop(0)
                mini = interesting[0][1]

        newfreqs = []
        new_intenses = []
        for j in freq:
            if j == 0:
                continue
            newfreqs.append(j)
            new_intenses.append(0)

        keys = [elem[0] for elem in interesting]
        dictionary = {elem[0]: elem[1] for elem in interesting}

        for i in range(len(newfreqs)):
            if newfreqs[i] in keys:
                new_intenses[i] = dictionary[newfreqs[i]]

        freq = newfreqs
        intens = new_intenses

        # Test: Plot
        # plt.plot(frequency, intensity)
        # plt.title("Noise Spectrum")
        # plt.show()

        def get_phase_func(freq):
            if freq == 0:
                return lambda x: 0
            max_time = 600
            slotlength = 1 / freq
            steps = 10
            startphase = lambda: 2 * np.pi * random.random()


            nextstep = lambda: 2 * np.pi * random.random() - np.pi

            steady_len = steps * slotlength
            slope_len = slotlength

            t = 0
            pairs = [] # (time bis, valueLeft, Slope)
            oldphase = 0
            while t < max_time:
                t += steady_len
                oldphase += nextstep()
                pairs.append((t, oldphase, False))
                t += slope_len
                pairs.append((t, oldphase, True))

            t += steady_len
            pairs.append((t, oldphase, True))
            assert t > max_time

            def func(t):
                t %= max_time
                for i in range(len(pairs)):
                    if t < pairs[i][0]:
                        if pairs[i][2]:
                            dec = slope_len + t - pairs[i][0]
                            m = (pairs[i+1][1] - pairs[i][1])/slope_len
                            return pairs[i][1] + m * dec
                        else:
                            return pairs[i][1]

            return func








            #phases = [lambda x: sp, lambda x: sp]

            #steady = [True, True]
            #for i in range(slotnumber - 1):
            #    steady.append((i % (steps + 1)) != 0)

            #steady.append(True)
            #steady.append(True)

            #phases_appended = [sp, sp]

            #for i in range(2, len(steady)):

                # if i > 5:
                #    print("i={} -> phase[0](0) = {}".format(i, phases[0](0)))
                #    print("i={} -> phase[1](0) = {}".format(i, phases[1](0)))
                #    #print("i={} -> phase[2](0) = {}".format(i, phases[2](0)))
                #    print("i={} -> phase[3](0) = {}".format(i, phases[3](0)))
                #    print("i={} -> phase[4](0) = {}".format(i, phases[4](0)))
                #    print(phases_appended)

                #if steady[i] and steady[i - 1]:
                #    print("A")
                #    phases_appended.append(phases_appended[-1])#

               #     def f(x):
               #         return phases_appended[i]

               #     phases.append(f)
               # elif steady[i] and not steady[i - 1]:
                #    print("B")
                #    old = phases_appended[-1]
                #    new = old + nextstep()
                #    phases_appended.append(new)

                #    def f(x):
                #        return phases_appended[i]

               #     phases.append(f)
               # elif not steady[i] and steady[i - 1]:
               #     print("C")
               #     phases_appended.append(phases_appended[-1])
               #     phases.append("SLOPE")
               #     print("Sloping")
               # else:
               #     print("i = {}, steady[i]={}, steady[i-1]={}".format(i, steady[i], steady[i - 1]))
               #     raise ValueError


        def _to_wave(freq, ampl):
            phasefkt = get_phase_func(freq)
            def f(t):
                return ampl * np.cos(2*np.pi * freq * t + phasefkt(t))

            return f


        times = range(10000)
        funcs = []
        for i in range(len(freq)):
            funcs.append(_to_wave(freq[i], intens[i]))


        return funcs


    def noise_spektrum(self, sigma, filename="NoiseSTM.csv", delimiter=";"):
        """
        Returns noise matrix according to provided measurement.
        Still under construction
        :param filename: path to csv file containing intensity over frequency
        :param delimiter: delimiter for CSV file
        :return: noise matrix
        """
        starttime = time.perf_counter()
        scanspeed = Distance(True, SXM_info.get_scanspeed() * 1e10)  # in Angstr/s
        width = cfg.get_width()
        height = cfg.get_height()
        t_line = width / scanspeed  # in s

        def _time_for_pos(x, y):
            return 2 * y * t_line + (x / width.px) * t_line  # 2 Due to repositioning


        timesteps = []
        for x in range(int(np.ceil(width.px))):
            for y in range(int(np.ceil(height.px))):
                timesteps.append(_time_for_pos(x, y))

        frequency = []
        intensity = []
        # Read in
        ct = 0
        all = False
        if all:
            with open(filename, 'r') as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=delimiter)
                for row in csv_reader:
                    ct += 1
                    if not row[0][0].isdigit():
                        continue
                    frequency.append(float(row[0]))
                    intensity.append(float(row[1]))
        else:
            pool = 5
            ct = 0
            pairs = []
            second = lambda elem: elem[1]
            with open(filename, 'r') as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=delimiter)
                for row in csv_reader:
                    if not row[0][0].isdigit():
                        continue
                    ct += 1
                    if ct % pool != 0:
                        pairs.append((float(row[0]), float(row[1])))
                    else:
                        pairs.sort(key=second, reverse=True)
                        pair = pairs.pop(0)
                        frequency.append(pair[0])
                        intensity.append(pair[1])
                        pairs = []


        #nse = self._noise_over_time(frequency, intensity)


        #Testing new Method
        # Nur wichtige
        restrict = False
        if restrict:
            no_of_inerest = 30
            interesting = []

            second = lambda elem: elem[1]

            for i in range(no_of_inerest):
                interesting.append((0, 0))

            mini = interesting[0][1]
            for i in range(1, len(frequency)):
                curr_int = intensity[i]
                if curr_int > mini:
                    interesting.append((frequency[i], intensity[i]))
                    interesting.sort(key=second)
                    interesting.pop(0)
                    mini = interesting[0][1]

            newfreqs = []
            new_intenses = []
            for j in frequency:
                if j == 0:
                    continue
                newfreqs.append(j)
                new_intenses.append(0)

            keys = [elem[0] for elem in interesting]
            dictionary = {elem[0]: elem[1] for elem in interesting}

            for i in range(len(newfreqs)):
                if newfreqs[i] in keys:
                    new_intenses[i] = dictionary[newfreqs[i]]

            frequency = newfreqs
            intensity = new_intenses

        #Def NSE
        def get_phase_func(freq, maxtime=_time_for_pos(width.px, height.px)):
            if freq == 0:
                return lambda x: 0
            max_time = maxtime
            #slotlength = 0.001 *  max(1, np.random.normal(freq, np.sqrt(freq))) / freq
            slotlength = 1/freq
           # print("Old Slotlen: {:.3f}, new: {:.3f}".format(1/freq, slotlength))
            #steps =  max(1, int(np.random.normal(freq, np.sqrt(freq))))
            steps = 10
            startphase = lambda: 2 * np.pi * random.random()


            nextstep = lambda: 2 * np.pi * random.random() - np.pi

            steady_len = steps * slotlength
            slope_len = slotlength

            t = 0
            pairs = [] # (time bis, valueLeft, Slope)
            oldphase = 2 * np.pi * random.random() - np.pi
            #print("Startphase at {:.2f}".format(oldphase))
            dict = {}
            i_steps = 0
            t_dict = timesteps[i_steps]
            app_d_phases = []
            try:
                while t < max_time:
                    t += steady_len
                    #oldphase += nextstep()
                    while t_dict < t:
                        #print("Steady at {:.2f}".format(oldphase))
                        #time.sleep(0.2)
                        dict[t_dict] = oldphase
                        i_steps += 1
                        app_d_phases.append(oldphase)
                        t_dict = timesteps[i_steps]

                    #pairs.append((t, oldphase, False))
                    t += slope_len
                    m = (2 * np.pi * random.random() - np.pi)/ max(slope_len, (timesteps[i_steps + 1] - timesteps[i_steps]))

                    while t_dict < t:
                        #print("DT = {}".format())
                        #print("Slopelen = {}".format(slope_len))
                        #print("m = {}".format(m))
                        oldphase += m * (timesteps[i_steps + 1] - timesteps[i_steps])
                        #print("Slope at {:.2f}".format(oldphase))
                        #time.sleep(2)
                        dict[t_dict] = oldphase
                        i_steps += 1
                        app_d_phases.append(oldphase)
                        t_dict = timesteps[i_steps]
                    #pairs.append((t, oldphase, True))

                t += steady_len
                while t_dict < t:
                    #print("End at {:.2f}".format(oldphase))
                    #time.sleep(0.2)
                    dict[t_dict] = oldphase
                    app_d_phases.append(oldphase)

                    i_steps += 1
                    t_dict = timesteps[i_steps]
                # pairs.append((t, oldphase, True))
                assert t > max_time
            except IndexError:
                pass

            #plt.plot(app_d_phases)
            #plt.title("Appd Phases for f={}".format(freq))
            #plt.show()

            #for key in dict.keys():
            #    print("F={} has key {} with Phase {}".format(freq, key, dict[key]))
            #    time.sleep(0.2)

            keys = [k for k in dict.keys()]
            def func(t):
                try:
                    return dict[t]
                except KeyError:
                    if t not in keys:
                        print("Key {} not Found".format(t))
                        distances = [(t - key) for key in keys]
                        mini = min(distances)
                        for i in range(len(distances)):
                            if distances[i] == mini:
                                print("Closest: {}".format(keys[i]))
                                return dict[keys[i]]
                #t %= max_time
                #for i in range(len(pairs)):
                #    if t < pairs[i][0]:
                #        if pairs[i][2]:
                #            dec = slope_len + t - pairs[i][0]
                #            m = (pairs[i+1][1] - pairs[i][1])/slope_len
                #            return pairs[i][1] + m * dec
                #        else:
                #            return pairs[i][1]

            return func

        # ampl * np.cos(2*np.pi * freq * t + phasefkt(t))

        # ToDo: Include
        phasefkts = []
        for i in range(len(frequency)):
            phasefkts.append(get_phase_func(frequency[i]))

        def nse(t):
            sum = 0
            for i in range(len(frequency)):
                 #sum += intensity[i] * np.cos(2*np.pi * frequency[i] * t + phasefkts[i](t))
                sum += intensity[i] * np.cos(2 * np.pi * frequency[i] * t + phasefkts[i](t))
            return sum

        #Test NSE

        if False:
            ts = []
            xs1 = []
            xs100 = []
            xs500 = []
            ph1 = get_phase_func(1)
            ph100 = get_phase_func(100)
            ph500 = get_phase_func(500)
            for y in range(int(np.ceil(height.px))):
                for x in range(int(np.ceil(width.px))):
                    #print(" ({}, {})".format(x, y))
                    t = _time_for_pos(x, y)
                    ts.append(t)
                    xs1.append(ph1(t))
                    xs100.append(ph100(t))
                    xs500.append(ph500(t))

            plt.plot(ts)
            plt.title("Time")
            plt.show()

            print(xs1)
            print(min(xs1), max(xs1))
            plt.plot(xs1)
            plt.title("Phase for 1Hz")
            plt.show()

            print(xs100)
            print(min(xs100), max(xs100))
            plt.plot(xs100)
            plt.title("Phase for 100Hz")
            plt.show()

            print(xs500)
            print(min(xs500), max(xs500))
            plt.plot(xs500)
            plt.title("Phase for 500Hz")
            plt.show()



        if False:
            print("Start testing")
            xs = range(100 * int(_time_for_pos(width.px, height.px)))
            ys = []
            for i in xs:
                ys.append(nse(i/100))
            #plt.plot(xs, ys)
            #plt.xlabel("100 * time")
            #plt.ylabel("Noise Level weniger Wellen")
            #plt.show()

            # Speedtest plot phasenfkt
            start = time.perf_counter()
            an = int(_time_for_pos(width.px, height.px))
            gen = 100
            times = range(an * gen)
            f = lambda x: np.sin(x) * np.exp(-0.005*x) + x*x - 4*x + 12
            ysf = []
            for t in times:
                ysf.append(f(t/gen))

            plt.plot(times, ysf)
            plt.title("f mit genauigkeit {} nach {:.2f}s".format(gen, time.perf_counter() - start))
            plt.show()
            del ysf

            start = time.perf_counter()
            f = get_phase_func(0.5)
            ysf = []
            for t in times:
                ysf.append(f(t/gen))

            plt.plot(times, ysf)
            plt.title(" Phase f=0.5 mit genauigkeit {} nach {:.2f}s".format(gen, time.perf_counter() - start))
            plt.show()
            del ysf

            start = time.perf_counter()
            f = get_phase_func(500)
            ysf = []
            for t in times:
                ysf.append(f(t/gen))

            plt.plot(times, ysf)
            plt.title(" Phase f=500 mit genauigkeit {} nach {:.2f}s".format(gen, time.perf_counter() - start))
            plt.show()
            del ysf

            start = time.perf_counter()
            f = get_phase_func(500)
            ysf = []
            times = [_time_for_pos(0, i) for i in range(int(width.px))]
            for t in times:
                ysf.append(f(t))

            plt.plot(times, ysf)
            plt.title("1 Line duration: {}".format(time.perf_counter() - start))
            plt.show()
            del ysf

            # End neue Impl







    #Testing
        if False:
            print("testPhase")
            start = time.perf_counter()
            print("gen necessary")
            necessary_pts = []
            w = int(width.px)
            h = int(height.px)
            for i in tqdm(range(w)):
                for j in range(h):
                    necessary_pts.append(_time_for_pos(i, j))

            print("1: {}".format(time.perf_counter() - start))
            start = time.perf_counter()

            print("Gen Funcs")
            funcs = self._not_funcs(frequency, intensity)

            print("2: {}".format(time.perf_counter() - start))
            start = time.perf_counter()

            print("eval Funcs")
            values = []
            for i in range(len(necessary_pts)):
                values.append(0)

            print("3: {}".format(time.perf_counter() - start))
            start = time.perf_counter()

            for f in tqdm(funcs):
                for i in range(len(necessary_pts)):
                    values[i] += f(necessary_pts[i])

            print("4: {}".format(time.perf_counter() - start))
            start = time.perf_counter()

            print("Plot Noise")
            plt.plot(necessary_pts, values)
            plt.title("Testing Noise")
            plt.show()


            print("testPhase-END")



    # end testing

       # print("Start: {}s".format(_time_for_pos(0, 0)))
       # print("End: {}s".format(_time_for_pos(width.px, height.px)))

        w = int(width.px)
        h = int(height.px)
        noisemat = np.zeros((w, h))
        for i in tqdm(range(w)):
            for j in range(h):
                noisemat[i, j] = nse(_time_for_pos(i, j))


        shift = np.average(noisemat)
        nm2 = np.zeros(np.shape(noisemat))
        for i in range(np.shape(noisemat)[0]):
            for j in range(np.shape(noisemat)[1]):
                nm2[i, j] = noisemat[i, j] - shift
        noisemat = nm2
        del nm2

        maxi = np.amax(noisemat)
        mini = np.amin(noisemat)
        diff = maxi - mini
        print("Sigma = {}".format(sigma))
        scale = max(0.5, np.random.normal(1)) * sigma / diff
        temp = np.zeros(np.shape(noisemat))
        for i in range(np.shape(noisemat)[0]):
            for j in range(np.shape(noisemat)[1]):
                temp[i, j] = scale * noisemat[i, j]
        noisemat = temp
        del temp
        maxi = np.amax(noisemat)
        mini = np.amin(noisemat)
        diff = maxi - mini
        print("Diff: {}".format(diff))
        #plt.imshow(Functions.turn_matplotlib(noisemat))
        #plt.title("Image Noise")
        #plt.show()
        print("Noise Spectrum: {:.2f}s".format(time.perf_counter() - starttime))
        return noisemat

    def f1_line_noise(self, mu, sigma):
        """
        Adds 1/f noise to the image
        :param mu: Medium grayscale
        :param sigma: standard derivation
        :return: None
        """
        w, h = np.shape(self.colors)
        noisemat = 0 * np.ones((w, h))

        modify_each_line = True

        def f_alpha(n_pts, q_d, alpha):
            """
            Implementation of 1/f^alpha noise.
            Translated tompython from Kasdin, DOI: 10.1109/5.381848
            :param n_pts: Number of points
            :param q_d: variance
            :param alpha: exponent alpha
            :return: array of noise values
            """
            xs = []
            nn = n_pts + n_pts
            ha = alpha / 2
            q_d = np.sqrt(q_d)

            hfa = [1, nn]
            wfa = [1, nn]
            hfa[1] = 1.0
            wfa[1] = q_d * np.random.normal()

            for i in range(2, n_pts + 1):
                hfa.append(hfa[i - 1] * (ha + i - 2) / i - 1)
                wfa.append(q_d * random.random())

            for i in range(n_pts + 1, nn):
                hfa.append(0)
                wfa.append(0)

            reth = numpy.fft.rfft(hfa, n_pts)  # , 1
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

            for i in range(3, nn - 1, 2):
                wr = wfa[i]
                wi = wfa[i + 1]
                wfa[i] = wr * hfa[i] - wi * hfa[i + 1]
                wfa[i + 1] = wr * hfa[i + 1] + wi * hfa[i]

            #          print("Wfa vor iff")
            #           plt.plot(wfa)
            #            plt.show()

            retw = np.fft.irfft(wfa, n_pts)

            #       print("retw")
            #        plt.plot(retw)
            #         plt.show()

            for i in range(len(retw)):
                wfa[i] = retw[i]

            for i in range(1, n_pts + 1):
                xs.append(wfa[i] / n_pts)

            #          plt.plot(xs)
            #           plt.show()
            return xs

        alph = 2
        xs = f_alpha(h, sigma ** 2, alph)

        ys = []
        for i in range(h):
            ys.append(f_alpha(w, sigma ** 2, 10))

        # print("YS after {:.2f}".format(time.perf_counter() - start))

        for i in range(w):
            for j in range(h):
                if not modify_each_line:
                    noisemat[i, j] += xs[j]
                else:
                    noisemat[i, j] += xs[j] * ys[j][int(round(i / 2))]

        # print("Max: {}".format(np.amax(noisemat)))
        # print("Min: {}".format(np.amin(noisemat)))
        # print("Med: {}".format(np.average(noisemat)))

        scale = 8 * sigma / (np.amax(noisemat) - np.amin(noisemat))
        # ToDo: Scale von scanline aus cfg
        noisemat *= scale
        shift = mu + np.average(noisemat)
        noisemat += shift * np.ones(np.shape(noisemat))

        # print("Max_New: {}".format(np.amax(noisemat)))
        # print("Min_New: {}".format(np.amin(noisemat)))
        # print("Med_New: {}".format(np.average(noisemat)))

        return noisemat

    @DeprecationWarning
    def noise_function(self):
        """
        Deprecated, unsuccessful implemetation of 2D Noise
        :return:
        """

        noise_mat = np.zeros(np.shape(self.colors))
        w, h = np.shape(noise_mat)
        two_pi = 2 * np.pi
        gen_amplitude = lambda x: 1
        shift = 90
        phi = lambda x: two_pi * random.random()
        step = 1

        def f_alpha(n_pts, q_d, alpha, idum):
            xs = []
            nn = n_pts + n_pts
            ha = alpha / 2
            q_d = np.sqrt(q_d)

            hfa = [1, nn]
            wfa = [1, nn]
            hfa[1] = 1.0
            wfa[1] = q_d * np.random.normal()

            for i in range(2, n_pts + 1):
                hfa.append(hfa[i - 1] * (ha + i - 2) / i - 1)
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

            reth = numpy.fft.rfft(hfa, n_pts)  # , 1
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

            for i in range(3, nn - 1, 2):
                wr = wfa[i]
                wi = wfa[i + 1]
                wfa[i] = wr * hfa[i] - wi * hfa[i + 1]
                wfa[i + 1] = wr * hfa[i + 1] + wi * hfa[i]

            #          print("Wfa vor iff")
            #           plt.plot(wfa)
            #            plt.show()

            retw = np.fft.irfft(wfa, n_pts)

            #       print("retw")
            #        plt.plot(retw)
            #         plt.show()

            for i in range(len(retw)):
                wfa[i] = retw[i]

            for i in range(1, n_pts + 1):
                xs.append(wfa[i] / n_pts)

            #          plt.plot(xs)
            #           plt.show()
            return xs

        def f_alpha_2D(n_pts, q_d, alpha):
            vals = np.zeros((n_pts, n_pts))
            nn = n_pts + n_pts
            ha = alpha / 2
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

            for i in range(1, n_pts + 1):
                hfa[0, i] = q_d * np.random.normal()
                hfa[i, 0] = q_d * np.random.normal()
                wfa[0, i] = 1
                wfa[i, 0] = 1

            for i in range(1, n_pts + 1):
                for j in range(1, n_pts + 1):
                    # h = hfa[i-1, j] * (ha + i-2)/i-1
                    # h + hfa[i, j-1] * (ha + j-2)/j-1
                    # h + hfa[i-1, j-1] * (ha +
                    h = ha * (2 * n_pts - i - j)
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

            reth = numpy.fft.rfft2(hfa, (n_pts, n_pts))  # , 1
            retw = numpy.fft.rfft2(wfa, (n_pts, n_pts))

            # print("ret fft w")
            # plt.imshow(retw)
            # plt.show()
            # print("ret_fft_h")
            # plt.imshow(reth)
            # plt.show()

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

            for i in range(3, nn - 1, 2):
                for j in range(3, nn - 1, 2):
                    wr = wfa[i, j] * 0
                    wi = wfa[i + 1, j] * 0
                    wk = wfa[i, j + 1] * 0
                    wfa[i] = wr * hfa[i, j] - wi * hfa[i + 1, j] - wk * hfa[i, j + 1]
                    wfa[i + 1] = wr * hfa[i + 1, j] + wi * hfa[i, j] + wk * hfa[i, j + 1]

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
                    vals[i, j] = (wfa[i, j] / n_pts)

            print("vals")
            plt.imshow(vals)
            plt.show()
            return vals

        xs = f_alpha(100, 1, 0, 1)
        # plt.plot(range(len(xs)), xs)
        # plt.show()

        print("2d")

        mat = f_alpha_2D(200, 1, 1)
        # print(mat)
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
            func = lambda x: shift
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

        # f_x = gen_f(w)

        # for i in range(w):
        #    for j in range(h):
        #        noise_mat[i, j] = f_x(i) + f_x(j) + shift#

        # plt.imshow(noise_mat)
        # plt.show()
        print("Max: {:.2f}".format(np.amax(noise_mat)))
        print("Min: {:.2f}".format(np.amin(noise_mat)))
        print("Avg: {:.2f}".format(np.average(noise_mat)))

        self.colors += noise_mat

    # @measureTime
    def get_matrix(self):
        """
        Getter method for color matrix
        :return:
        """

        return self.colors

    # @measureTime
    def showImage(self):
        """
        Show the image using PIL.Image.show()
        :return:
        """
        self.img.show()

    # @measureTime
    def saveImage(self, filename):
        """
        Save the image under generated filename
        Important: updateImage needs to be called first in order to pass information on to PIL.image instance
        :param filename: filename
        :return:
        """
        try:
            self.img.save(filename)
        except FileNotFoundError:
            os.mkdir(cfg.get_image_folder())
            self.img.save(filename)
        return filename

    # @measureTime
    @lru_cache
    def rgb_map(self, h):
        """
        Maps height values onto rgb values,
        Either as grayscale values or using the WSXM colorscheme with
        WSXM default.lut data
        :param h: height value
        :return: r, g, b values
        """
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
        """
        Method to test the rgb_map
        :return:
        """
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
