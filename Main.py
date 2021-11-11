import numpy as np
import random

from tqdm import tqdm

from Molecule import Molecule
from Particle import Particle
from FilenameGenerator import FilenameGenerator
from DataFrame import DataFrame
from Functions import *
import Configuration as cfg
import math, time
from multiprocessing import Process, Lock
from multiprocessing.managers import BaseManager
import matplotlib.pyplot as plt
import csv
#import skimage

#from skimage.viewer import ImageViewer
import sys
"""
Main class used to interact with the program. Provides basic methods
Important are act(), GenExec and execNthreads()
"""

def test_frame(data_frame):
    """
    Adds Particles at steadily increasing angles line by line to the frame
    Useful to test if angles are defined correctly

    :param data_frame: The frame where particles should be added
    :return: None
    """

    maxn = len(range(50, cfg.get_width(), 100)) * len(range(50, cfg.get_height(), 100))
    th = 0
    dth = 2 * math.pi / maxn
    for y in range(50, cfg.get_height(), 100):
        for x in range(50, cfg.get_width(), 100):
            data_frame.addParticle(Particle(x, y, th))
            th += dth


def test_gaussian_blur(fname):
    """
    Testing method to apply gaussian blur to an image
    :param fname: Filename
    :return: None
    """
    print(fname)
    image = skimage.io.imread(fname=fname)
    viewer = ImageViewer(image)
    viewer.show()
    blurred = skimage.filters.gaussian(image, sigma=(1,1), truncate=3, multichannel=True)
    viewer = ImageViewer(blurred)
    viewer.show()

def measure_speed():
    """
    Generates and saves images for increasing numbers of particles.
    Useful to measure correlation between number of particles and computational time
    Plots time over n

    :return: None
    """

    x = []
    y = []
    for i in range(20):
        start = time.perf_counter()
        dat_frame = DataFrame(FilenameGenerator())
        dat_frame.addParticles(amount=i, overlapping=False)
        dat_frame.get_Image()
        dat_frame.save()
        x.append(i)
        y.append(time.perf_counter() - start)

    plt.plot(x, y)
    plt.show()


def generate(fn_gen):
    """
    Creates an image w/ ordered particles

    :param fn_gen: Filename generator instance
    :return: None
    """
    dat_frame = DataFrame(fn_gen)
    dat_frame.add_Ordered()
    dat_frame.get_Image()
    dat_frame.save()


def multi_test():
    """
    Creates as many Generator instances as specified in Configuration-threds
    :return: None
    """
    BaseManager.register('FilenameGenerator', FilenameGenerator)
    filemanager = BaseManager()
    filemanager.start()
    fn_generator = filemanager.FilenameGenerator()

    gens = []
    for i in range(cfg.get_threads()):
        gens.append(Generator(fn_generator))
    for gen in gens:
        gen.start()


class Generator(Process):
    """
    Multiprocessing process to execute generate()
    """
    def __init__(self, fn_gen):
        super().__init__()
        self.fn_gen = fn_gen

    def run(self):
        generate(self.fn_gen)




def multi_test_fn(t, fgen):
    """
    Runs instances of Generator and kills them after time t
    :param t: time in seconds to wait
    :param fgen: Filename Generator instance
    :return: None
    """
    fn_generator = fgen
    gens = []
    for i in range(cfg.get_threads()):
        gens.append(Generator(fn_generator))
    for gen in gens:
        gen.start()

    time.sleep(t)
    for gen in gens:
        gen.kill()





class Gen1(Process):
    """
    Generator that runs generate multiple times
    """
    def __init__(self, fn, n):
        """

        :param fn: Filename Generator istance
        :param n: number of executions
        """
        super().__init__()
        self.fn = fn
        self.n = n

    def run(self) -> None:
        for i in range(self.n):
            generate(self.fn)


def every_thread_n(n):
    """
    Lets every thread generate n Images
    :param n: Number of images
    :return: None
    """
    BaseManager.register('FilenameGenerator', FilenameGenerator)
    filemanager = BaseManager()
    filemanager.start()
    fn_gen = filemanager.FilenameGenerator()

    ts = []
    for i in range(cfg.get_threads()):
        ts.append(Gen1(fn_gen, n))
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return True


@DeprecationWarning
def test_Length_and_Sigma():
    """
    DEPRECATED.  Used to generate images with different parameters for angle correlation
    :return: None
    """
    BaseManager.register('FilenameGenerator', FilenameGenerator)
    filemanager = BaseManager()
    filemanager.start()
    fn_generator = filemanager.FilenameGenerator()
    l = 0.001
    inc = 0.001
    while l < 3:
        cfg.set_angle_char_len(l)
        testStdderiv(fn_generator, l)
        if inc > 1:
            dinc = 0.2
        elif inc > 0.1:
            dinc = 0.01
        else:
            dinc = 0.002
        inc += dinc
        l += inc


class Gen2(Process):
    """
    Generator used for ordered adding at specific angles
    """
    def __init__(self, fn, a):
        """

        :param fn: Filename Generator instance
        :param a: angle ordered lattice is turned by
        """
        super().__init__()
        self.fn = fn
        self.a = a

    def run(self) -> None:
        dat_frame = DataFrame(self.fn)
        dat_frame.add_Ordered(Molecule, theta=self.a)
        dat_frame.get_Image()
        index = dat_frame.save()
        print("index {} had angle {:.1f}°".format(index, 180 * self.a / 3.14159))


def every_angle():
    """
    Runs Gen2-Instances for angles from 0 to 360 degree
    :return: True to keep parent method alive
    """
    BaseManager.register('FilenameGenerator', FilenameGenerator)
    filemanager = BaseManager()
    filemanager.start()
    fn_gen = filemanager.FilenameGenerator()

    ts = []
    for i in range(cfg.get_threads()):
        a = ((360 / cfg.get_threads()) * i) * np.pi / 180
        ts.append(Gen2(fn_gen, a))
    for t in ts:
        t.start()
        time.sleep(2)
    for t in ts:
        t.join()
    return True


class Gen3(Process):
    """
    Generator used to test Double-Tip method
    """
    def __init__(self, fn, a):
        """

        :param fn: Filename Generator instance
        :param a: angle in which direction the double tipping effect should happen
        """
        super().__init__()
        self.fn = fn
        self.a = a

    def run(self) -> None:
        dat_frame = DataFrame(self.fn)
        dat_frame.add_Ordered(theta=0)
        dat_frame.get_Image(ang=self.a)
        index = dat_frame.save()
        print("index {} had angle {:.1f}°".format(index, 180 * self.a / 3.14159))




@DeprecationWarning
def testStdderiv(fn_gen, l):
    """
    DEPRECATED. Used to test different standard derivations for angle correlation
    :param fn_gen: Filename Generator instance
    :param l: Angle char, length currently used
    :return: None
    """
    fn = fn_gen
    s = 0.005
    inc = 0.01
    while s < 100:
        cfg.set_angle_stdderiv(s)
        print("Sigma = {}, l={} with Index {}".format(s, l, fn.generateIndex()))
        multi_test_fn(8, fn)
        if inc >= 2:
            dinc = 30
        elif inc >= 10:
            dinc = 6
        elif inc > 1:
            dinc = 0.1
        elif inc > 0.1:
            dinc = 0.2
        else:
            dinc = 0.02
        inc += dinc
        s += inc


def test_finv_acc():
    """
    Testing method to measure computational performance of Function.finv()
    Plots Time over accuracy
    :return: None
    """

    times = []
    accs = []
    for i in range(1, 1000, 5):
        start = time.perf_counter()
        f = lambda x: np.exp(x)
        finv = get_invers_function(f, 0, 400, i)

        xs = [i for i in range(400)]
        ys = [finv(i) for i in range(400)]
        times.append(time.perf_counter() - start)
        accs.append(i)

    plt.plot(accs, times)
    plt.title("Computing time over Accuracy")
    plt.show()


def test_finv_len():
    """
    Testing method to measure computational performance of Function.finv()
    Plots Time over number of points
    :return: None
    """

    times = []
    lens = []
    for i in range(1, 100, 5):
        start = time.perf_counter()
        f = lambda x: np.exp(x)
        finv = get_invers_function(f, 0, i)

        xs = [i for i in range(400)]
        ys = [finv(i) for i in range(400)]
        times.append(time.perf_counter() - start)
        lens.append(i)

    plt.plot(lens, times)
    plt.title("Computing time over Points")
    plt.show()


class Gen4(Process):
    """
    Generator for adding a single particle in the middle of the image rotated by given angle
    """
    def __init__(self, fn, a):
        """

        :param fn: Filename Generator Instance
        :param a: Angle
        """
        super().__init__()
        self.fn = fn
        self.a = a

    def run(self) -> None:
        dat_frame = DataFrame(self.fn)
        p = Particle(cfg.get_width() / 2, cfg.get_height() / 2, self.a)
        dat_frame.addParticle(p)
        dat_frame.get_Image()
        index = dat_frame.save()
        print("index {} had angle {:.1f}°".format(index, 180 * self.a / 3.14159))


def every_angle2():
    """
    Invokes Gen4
    :return:
    """
    BaseManager.register('FilenameGenerator', FilenameGenerator)
    filemanager = BaseManager()
    filemanager.start()
    fn_gen = filemanager.FilenameGenerator()

    ts = []
    for i in range(cfg.get_threads()):
        a = ((180 / cfg.get_threads()) * i) * np.pi / 180
        ts.append(Gen4(fn_gen, a))
    for t in ts:
        t.start()
        time.sleep(0.5)
    for t in ts:
        t.join()
    return True

def test_fft():

    def _not_funcs(freq, intens):
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
            #return lambda x:0
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



        def _to_wave(freq, ampl):
            phasefkt = get_phase_func(freq)
            def f(t):
                return ampl * np.cos(2*np.pi * freq * t + phasefkt(t))

            return f


        times = range(10000)
        funcs = []
        for i in range(len(freq)):
            if intens[i] == 0:
                continue
            funcs.append(_to_wave(freq[i], intens[i]))

        def f(t):
            summe = intens[0]
            for f in funcs:
                summe += f(t)
            return summe


        return f



    frequency = []
    intensity = []
    # Read in
    with open("NoiseSTM.csv", 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=";")
        for row in csv_reader:
            if not row[0][0].isdigit():
                continue
            frequency.append(float(row[0]))
            intensity.append(float(row[1]))

    plt.plot(frequency[1:], intensity[1:])
    plt.title("Intens over Freq")
    plt.show()

    l = len(intensity)
    for i in range(l):
        intensity.append(0)

    spectrum = np.real(np.fft.ifft(intensity))[:1000]
    plt.plot(spectrum)
    plt.title("IFFT")
    plt.show()

    ff = np.real(np.fft.fft(intensity))[:1000]
    plt.plot(ff)
    plt.title("FFT")
    plt.show()

    no_t = _not_funcs(frequency, intensity[:l])
    times = range(1000)
    vals = [no_t(t/1000) for t in times]

    plt.plot(times, vals)
    plt.title("Alle Wellen mit Phase zufällig")
    plt.show()

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

    plt.plot(newfreqs, new_intenses)
    plt.title("New Spectrum")
    plt.show()

    no_t = _not_funcs(newfreqs, new_intenses)
    times = range(1000)
    vals = [no_t(t / 1000) for t in times]

    plt.plot(times, vals)
    plt.title("Wenige Wellen Phase zufällig")
    plt.show()

def test_atan():

    def eigen(x, y):
        if x > 0 and y > 0:
            return np.pi / 2 + np.arctan(y / x)
        elif x > 0 and y < 0:
            return np.arctan( - x/y)
        elif x < 0 and y > 0:
            return np.pi + np.arctan(-x/ y)
        elif x < 0 and y < 0:
            return 2* np.pi - np.arctan(x / y)
        elif y == 0 and x < 0:
            return (3/2) * np.pi
        elif y == 0 and x >= 0:
            return np.pi/2
        elif x==0 and y <= 0:
            return 0
        elif x == 0 and y > 0:
            return np.pi
        else:
            raise ValueError

    def mathe(x, y):
        return np.arctan2(-x, y) + np.pi

    xs = [x for x in range(-100, 100)]
    ys = [x for x in range(-100, 100)]
    mat_Eig = np.zeros((len(xs), len(ys)))
    mat_mathe= np.zeros((len(xs), len(ys)))

    #print("Matheig {}, {} = {}".format(50, -50, eigen(50, -50)))

    for x in xs:
        for y in ys:
            mat_Eig[x + 100, y + 100] = eigen(x, y)
            mat_mathe[x + 100, y + 100] = mathe(x, y)

    #mat_Eig = turn_matplotlib(mat_Eig)
    #mat_mathe = turn_matplotlib(mat_mathe)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Horizontally stacked subplots')
    ax1.imshow(mat_Eig)
    ax1.set_title("Eigen")
    ax2.imshow(mat_mathe)
    ax2.set_title("Math")
    plt.show()



def ebene(diff = 120):
    w = int(cfg.get_width().px)
    h = int(cfg.get_height().px)
    maxampl = diff * 2

    a = (random.random() - 0.5) * (maxampl / w)
    b = (random.random() - 0.5) * (maxampl / h)
    midpointx = random.randint(0, w)
    midpointy = random.randint(0, h)
    mat = np.zeros((w, h))
    for i in range(w):
        for j in range(h):
            mat[i, j] = a * (i - midpointx) + (j- midpointy) * b

    #plt.imshow(mat)
    #plt.show()
    return mat

def ebenentest():
    mitt = 30
    xs = [i for i in range(10, 300)]
    ys = []
    for max in tqdm(range(10, 300)):
        temp = []
        for i in range(mitt):
            eb = ebene(diff=max)
            diff = np.amax(eb) - np.amin(eb)
            temp.append(diff)
        ys.append(np.average(temp))

    plt.plot(xs, ys)
    plt.title("Ebenediff über maxampl")
    plt.show()






"""
Add Code here
"""
thrds = cfg.get_threads()
recursions = 200000000


def act(dat):
    """
    Core method. This code will be run by each thread on the given DataFrame

    :param dat: Data frame opertaions should be done on
    :return:
    """
    #pos = np.array([cfg.get_width()/2, cfg.get_height()/2])
    #m = Particle(pos[0], pos[1], theta=0)
    #m = Molecule(pos, theta=0)
    #dat.addParticle(m)
    dat.add_Ordered(ph_grps=5, chirality=-1)
    dat.get_Image()
    dat.save()


class GenExc(Process):
    """
    Core Generator. Code inside act() will be called amnt time by every thread
    """
    def __init__(self, fn, amnt, lck, name="GenExc"):
        """

        :param fn: Filename Generator instance
        :param amnt: Number of time the code should be executed
        """
        super().__init__()
        self.fn = fn
        self.am = amnt
        self.lck = lck
        self.name = name

    def run(self) -> None:
        for i in range(self.am):
            self.lck.acquire()

            new_part_height = random.uniform(0.5, 5)  # Set ranges for variable parameters
            new_img_width_ang_min = 80
            new_img_width_ang_max = 3000
            #new_img_width_ang = random.uniform(new_img_width_ang_min, new_img_width_ang_max)
            #new_img_width_ang = 1 / random.uniform(1 / new_img_width_ang_max, 1 / new_img_width_ang_min)
            new_img_width_ang = np.exp(random.uniform(np.log(new_img_width_ang_min), np.log(new_img_width_ang_max)))
            #print(f"new_img_width_ang: {new_img_width_ang}")
            new_px_ang = 512 / new_img_width_ang
            new_gsc = random.uniform(90, 120)
            new_stdderiv = random.uniform(90, 120)
            new_maxH = random.uniform(0.5, 1 * new_part_height) + new_part_height
            new_fex = random.uniform(0.5, 1)

            cfg.set_part_height(new_part_height)
            cfg.set_px_per_ang(new_px_ang)
            cfg.set_image_dim(new_img_width_ang)
            cfg.set_grayscale_noise(new_gsc)
            cfg.set_noise_stdderiv(new_stdderiv)
            cfg.set_max_height(new_maxH)
            cfg.set_fermi(new_fex)
            self.lck.release()

            dat = DataFrame(self.fn)
            act(dat)

# class ChangeSettings(Process):
#     """
#     Thread that paralelly chages the configuration settings
#     """
#
#     def __init__(self):
#         super().__init__()
#
#     def run(self):
#         while True:
#             time.sleep(14) # Change parameters every 5 minutes
#
#             new_part_height = random.uniform(0.5, 5) # Set ranges for variable parameters
#             new_img_width_ang = random.randint(10, 500) # All possibly wrong, correct
#             new_px_ang = 512/new_img_width_ang
#             new_gsc = random.uniform(30, 200)
#             new_stdderiv = random.uniform(30, 300)
#             new_maxH = random.uniform(0, 2 * new_part_height) + new_part_height
#             new_fex = random.uniform(0.5, 2)
#
#             cfg.set_part_height(new_part_height)
#             cfg.set_image_dim(new_img_width_ang)
#             cfg.set_px_per_ang(new_px_ang)
#             cfg.set_grayscale_noise(new_gsc)
#             cfg.set_noise_stdderiv(new_stdderiv)
#             cfg.set_max_height(new_maxH)
#             cfg.set_fermi(new_fex)
#


def execContinously_vary_params():
    BaseManager.register('FilenameGenerator', FilenameGenerator)
    filemanager = BaseManager()
    filemanager.start()
    fn_gen = filemanager.FilenameGenerator()


    #n = cfg.get_threads()

    n = 14

    # One thread changes settings
    #changes = ChangeSettings()
    #changes.start()

    edit_lock = Lock()


    ts = []


    for i in range(n):
        ts.append(GenExc(fn_gen, 10000000, edit_lock, str(i))) # May images
    for t in ts:
        t.start()
        time.sleep(0.1)
    for t in ts:
        t.join()
    return True



def execNthreads(n, amnt=1):
    """
    Core method. Generates and runs n GenExcs
    :param n: Number of processes to start in parallel
    :param amnt: number of times each process should do the task in act()
    :return:
    """

    BaseManager.register('FilenameGenerator', FilenameGenerator)
    filemanager = BaseManager()
    filemanager.start()
    fn_gen = filemanager.FilenameGenerator()

    ts = []
    for i in range(n):
        ts.append(GenExc(fn_gen, amnt))
    for t in ts:
        t.start()
        time.sleep(0.1)
    for t in ts:
        t.join()
    return True




if __name__ == "__main__":
    clearLog()
    #ebenentest()


    execContinously_vary_params()
    #test_fft()
    #execNthreads(thrds, recursions)
    #test_gaussian_blur("bildordner/BlurImage.png")
    # lo = Lock()
    # start = time.perf_counter()
    # fn = FilenameGenerator()
    #
    # every_angle2()
    # dat = DataFrame(fn)
    # p = Particle(cfg.get_width()/2, cfg.get_height()/2, 0)
    # dat.addParticle(p)
    # dat.get_Image()
    # dat.save()
    # dat = DataFrame(fn)
    # dat.add_Ordered()
    # dat.get_Image()
    # dat.save()
    # every_angle2()
    # mat = 200 * np.ones((200, 200))
    # for i in range(np.shape(mat)[0]):
    #    for j in range(np.shape(mat)[1]):
    #        col = 0
    #        if i > np.shape(mat)[0]/2:
    #            col += 75
    ##        else:
    #            col += 25

    #        if j > np.shape(mat)[1]/2:
    #            col += 100
    #        else:
    #            col += 0
    #        mat[i, j] = col
    # every_thread_n(10)
    # plt.imshow(mat)

    # plt.show()
    # for i in range(0, 360, 10):
    #    mat2, a, b = turn_matrix2(mat, i * 3.14159/180)
    #    plt.imshow(mat2)
    #    plt.show()

    # every_thread_n(1)
    # multi_test(1)
    # m = Molecule()
    # Tests_Gitterpot().test()

    # test_finv_acc()

    # for i in range(1):
    #    dat = DataFrame(fn)
    #    dat.addObjects(Molecule)
    #    #dat.add_Dust(DustParticle(np.array([200, 200]), color=i))
    #    dat.get_Image()
    #    dat.save()

    # start = time.perf_counter()
    # f = lambda x: x*np.sin(x/100) + 20
    # finv = get_invers_function(f)
    # xs = np.linspace(0, 400, 200)
    # ys = [finv(xy) for xy in xs]
    # plt.xkcd()
    # plt.plot(xs, ys)
    # every_angle()
    # p#lt.show()
    # print("Dur: {:.2f}s".format(time.perf_counter() - start))
    # times = []
    # accs = []
    # f#or i in range(1000):
    #    start = time.perf_counter()
    #    f = lambda x:np.exp(x)
    #    finv = get_invers_function(f, 0, 400, i)##

    #    xs = [i for i in range(400)]
    #   ys = [finv(i) for i in range(400)]
    #  times.append(time.perf_counter() - start)
    # accs.append(i)

    # plt.plot(accs, times)
    # plt.title("Computing time over Accuracy")
    # plt.show()
    # every_angle()
    # print(every_thread_n(10))
    # dat = DataFrame(fn)
    # dat.add_Ordered()
    # dat.get_Image()
    # dat.save()

    # dat = DataFrame(fn)
    # p = Particle(Distance(True, 35), Distance(True, 35), 0)
    # dat.addParticle(p)
    # dat.get_Image()
    # dat.save()
    # every_thread_n(5)
    # if every_thread_n(10):
    #    exit()
    #         time.sleep(5)

    # img = MyImage()
    # img.rgb_map_test()
    # for i in range(0, 360, 10):
    #    dat = DataFrame(fn)
    #    dat.add_Ordered(Molecule, theta=np.pi * i / 180)
    #    dat.get_Image()
    #    dat.save()
    ##    print("Dur: {:.2f}s".format(time.perf_counter() - start))
    #   start = time.perf_counter()
    ##    break
    # for i in range(5):
    #    dat = DataFrame(fn)
    #    dat.add_Ordered()
    #    dat.get_Image()
    #    dat.save()

    # DustParticle.test()

    # dat = DataFrame(fn)
    # for n in range(0,8, 3):
    #    dat.addObject(Molecule(np.array([Distance(False, 200), Distance(False, 200 + 100*n)]), n*0.785))
    # dat.addObject(Molecule(np.array([Distance(False, 200), Distance(False, 200 + 100)]), 0))
    # dat.addObject(Particle(Distance(False, 200), Distance(False, 500), 0.9))
    # dat.addObjects(Molecule, amount=1)
    # dat.addObject(Molecule(np.array([Distance(False, 200), Distance(False, 200)]), 1*0.785))
    # dat.addObject(Molecule(np.array([Distance(False, 200), Distance(False, 240)]), 2*0.785))
    # for i in range(30):
    #    print("Iteration I={}".format(i))
    #    dat.addObjects(Molecule, amount=10)

    # dat.objects[0].drag(10, 0)

    # start = time.perf_counter()
    # print(dat.has_overlaps())
    # print("Has Overlaps: " + str(time.perf_counter() - start))
    # start = time.perf_counter()
    # dat.get_Image()
    # print("Get Image: " + str(time.perf_counter() - start))
    # dat.save()

    # filename = "pySPM_Tests/HOPG-gxy1z1-p2020.sxm"
    # filename = "Test_SXM_File.sxm"
    # with open("SXM_ino.txt", "w") as file:
    #   file.write(My_SXM.get_informations(filename))
    # generate(fn)
    # data = np.random.random((256, 256))
    # My_SXM.write_sxm("Test2.sxm", data)
    # My_SXM.write_header("Test4.sxm")
    # My_SXM.write_image("Test4.sxm", data)
    # My_SXM.show_data("Test2.sxm")
    # generate(fn)
    # My_SXM.write_header(os.path.join("pySPM_Tests", "test_header_writer.sxm"))
    # print(My_SXM.get_informations("pySPM_Tests/test_header_writer.sxm"))
    # plt.imshow(My_SXM.get_data_test("pySPM_Tests/HOPG-gxy1z1-p2020.sxm"))
    # plt.show()
    # SXM_info.adjust_to_image(data, "Test3.sxm")
    # My_SXM.write_header("Test3.sxm")
    # My_SXM.write_image("Test3.sxm", data)
    # My_SXM.show_data("Test3.sxm")
    # generate(fn)
    # data = np.random.random((300, 300))
    # My_SXM.write_sxm("Test4.sxm", data)
    # def deg(x):
    #    return x * np.pi / 180

    # for theta in range(44, 45, 1):
    # multi_test(180)
    # dat_frame = DataFrame(fn)
    # for i in range(100):
    #    generate(fn)
    # multi_test(300)
    #    print(theta, fn.generateIndex())
    #   a = Particle(200, 200, deg(0))
    #   b = Particle(220, 220, deg(330))
    #   dat_frame.addParticle(a)
    #   dat_frame.addParticle(b)
    #   print("A overlaps b: {}".format(a.true_overlap(b)))
    # print("B overlaps a: {}".format(b.true_overlap(a)))
    # print("DF has overlaps: {}".format(dat_frame.has_overlaps()))
    #   dat_frame.get_Image()
    #   dat_frame.save()
    # every_thread_one()
    # generate(fn)
    # for i in range(360):
    #    dat = DataFrame(fn)
    #    dat.addParticle(Particle(300, 300, deg(i)))
    #    dat.calc_potential_map()
    #    img = MyImage()
    #    mat = np.zeros((600, 600))
    #    for i in range(600):
    #        for j in range(600):
    #            mat[i, j] = 125 + 125 * dat.potential_map[i, j]
    #    img.addMatrix(mat)
    #    img.updateImage()
    #    img.saveImage(fn.generate_Tuple()[0])

    # plt.imshow(dat.potential_map.transpose())
    # plt.savefig()
    #    dat.get_Image()
    #    dat.save()

    # dat = DataFrame(fn)
    # for i in range(60, cfg.get_width() - 60, 60):
    #    for j in range(60, cfg.get_height() - 60, 60):
    #        dat.add_at_optimum_energy_new(100 + 400 * random.random(), 100 + 400 * random.random(), 6.28 * random.random())
    #        dat.get_Image()
    #        dat.save()
    ##dat.add_ALL_at_optimum_energy_new(10)
    # dat.get_Image()
    # dat.save()

    # dat_frame.addParticle(Particle(100, 100, 0.5))
    # for i in range(5):
    #   a = Particle(200, 200, deg(0))
    #   b = Particle(220, 220, deg(330))
    #   dat_frame.addParticle(a)
    #   dat_frame.addParticle(b)
    #   print("A overlaps b: {}".format(a.true_overlap(b)))
    # print("B overlaps a: {}".format(b.true_overlap(a)))
    # print("DF has overlaps: {}".format(dat_frame.has_overlaps()))
    #   dat_frame.get_Image()
    #   dat_frame.save()
    # every_thread_one()
    # multi_test(1000000)
    # dat_frame.addParticle(Particle(100, 100, 0.5))
    # for i in range(5):
    #    dat_frame.add_at_optimum_energy_new(100 + 400 * random.random(), 100 + 400 * random.random(), 2* np.pi * random.random())
    # dat_frame.add_at_optimum_energy_new(200, 260, deg(180))
    # print(4)
    # dat_frame.add_at_optimum_energy([200, 300, deg(180)])
    # dat_frame.add_at_optimum_energy([250, 240, deg(180)])
    # dat_frame.get_Image()
    # dat_frame.save()
    # generate(fn)
    # for i in range(10):
    #    dat = DataFrame(fn)
    #    dat.addParticles(amount=10)
    #    dat.get_Image()
    #    dat.save()
    # dat_frame.potential_map = dat_frame.calc_potential_map()
    # for i in range(20):
    #    print(i)
    #    dat_frame.add_at_optimum_energy()
    #    dat_frame.potential_map = dat_frame.calc_potential_map()
    #    dat_frame.get_Image()
    #    dat_frame.save(sxm=False, data=False)
    # dat_frame.addParticle(b)
    # dat_frame.get_Image()
    # dat_frame.save()
    # print(dat_frame.has_overlaps())
    # print(a.true_overlap(b))
    # print(1)
    # generate(fn)
    # print(2)
    # generate(fn)
    # print(3)
    # for i in range(10):
    #    generate(fn)
    # multi_test(30)
    # generate(fn)
    # My_SXM.show_data("sxm/Image1.sxm")
    # SXM_info.adjust_to_image(data, "Test4.sxm")
    # My_SXM.write_header("Test4.sxm")
    # My_SXM.write_image("Test4.sxm", data)
    # My_SXM.show_data("Test4.sxm")
    # My_SXM.show_data("Test4.sxm")
    # My_SXM.get_data_test(filename)
    # print(My_SXM.get_informations("Test.sxm"))
    # plt.imshow(My_SXM.get_data("Test.sxm"))
    # plt.show()

    evaluateLog()
