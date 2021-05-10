import os

import numpy as np

import Configuration as conf
import SXM_info
from Images import MyImage
from Particle import Particle
from FilenameGenerator import FilenameGenerator
from DataFrame import DataFrame
from Functions import *
import Configuration as cfg
import math, time
import matplotlib.pyplot as plt
from multiprocessing import Process, Lock, Semaphore
from multiprocessing.managers import BaseManager
from My_SXM import My_SXM


def test_frame(data_frame):
    maxn = len(range(50, cfg.get_width(), 100)) * len(range(50, cfg.get_height(), 100))
    th = 0
    dth = 2 * math.pi / maxn
    # dth = 0.05
    for y in range(50, cfg.get_height(), 100):
        for x in range(50, cfg.get_width(), 100):
            data_frame.addParticle(Particle(x, y, th))
            th += dth

def measure_speed():
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
    dat_frame = DataFrame(fn_gen)
    dat_frame.addParticles()
    dat_frame.get_Image()
    dat_frame.save()

def multi_test(t):

    BaseManager.register('FilenameGenerator', FilenameGenerator)
    filemanager = BaseManager()
    filemanager.start()
    fn_generator = filemanager.FilenameGenerator()

    #fn_gen = FilenameGenerator(lo)
    gens = []
    for i in range(cfg.get_threads()):
        gens.append(Generator(fn_generator))
    for gen in gens:
        gen.start()

    time.sleep(t)
    for gen in gens:
        gen.kill()

def multi_test_fn(t, fgen):

    fn_generator = fgen
    # fn_gen = FilenameGenerator(lo)
    gens = []
    for i in range(cfg.get_threads()):
        gens.append(Generator(fn_generator))
    for gen in gens:
        gen.start()

    time.sleep(t)
    for gen in gens:
        gen.kill()


class Generator(Process):
    def __init__(self, fn_gen):
        super().__init__()
        self.fn_gen = fn_gen



    def run(self):
        #i = 0
        while True:
            #i+= 1
            #print("{} running for {}th time".format(str(self), i))
            generate(self.fn_gen)

def test_Length_and_Sigma():
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


def testStdderiv(fn_gen, l):
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
        #print(s)



if __name__ == "__main__":
    clearLog()
    #lo = Lock()
    start = time.perf_counter()
    fn = FilenameGenerator()
    #filename = "pySPM_Tests/HOPG-gxy1z1-p2020.sxm"
    #filename = "Test_SXM_File.sxm"
    #with open("SXM_ino.txt", "w") as file:
    #   file.write(My_SXM.get_informations(filename))
    #generate(fn)
    #data = np.random.random((256, 256))
    #My_SXM.write_sxm("Test2.sxm", data)
    #My_SXM.write_header("Test4.sxm")
    #My_SXM.write_image("Test4.sxm", data)
    #My_SXM.show_data("Test2.sxm")
    #generate(fn)
    #My_SXM.write_header(os.path.join("pySPM_Tests", "test_header_writer.sxm"))
    #print(My_SXM.get_informations("pySPM_Tests/test_header_writer.sxm"))
    #plt.imshow(My_SXM.get_data_test("pySPM_Tests/HOPG-gxy1z1-p2020.sxm"))
    #plt.show()
    #SXM_info.adjust_to_image(data, "Test3.sxm")
    #My_SXM.write_header("Test3.sxm")
    #My_SXM.write_image("Test3.sxm", data)
    #My_SXM.show_data("Test3.sxm")
    #generate(fn)
    #data = np.random.random((300, 300))
    #My_SXM.write_sxm("Test4.sxm", data)
    def deg(x):
        return x * np.pi / 180

    dat_frame = DataFrame(fn)
    #for i in range(100):
    #    generate(fn)
    #a = Particle(200, 200, deg(110))
    #b = Particle(215, 230, deg(180))
    #dat_frame.addParticle(a)
    #dat_frame.addParticle(b)
    #print("A overlaps b: {}".format(a.true_overlap(b)))
    #print("DF has overlaps: {}".format(dat_frame.has_overlaps()))
    #dat_frame.get_Image()
    #dat_frame.save()

    #dat_frame.addParticle(Particle(100, 100, 0.5))
    dat_frame.calc_potential_map()
    dat_frame.add_at_optimum_energy()
    dat_frame.add_at_optimum_energy([200, 220, deg(180)])
    dat_frame.add_at_optimum_energy([200, 300, deg(180)])
    dat_frame.add_at_optimum_energy([250, 240, deg(180)])
    dat_frame.get_Image()
    dat_frame.save()

    #dat_frame.potential_map = dat_frame.calc_potential_map()
    #for i in range(20):
    #    print(i)
    #    dat_frame.add_at_optimum_energy()
    #    dat_frame.potential_map = dat_frame.calc_potential_map()
    #    dat_frame.get_Image()
    #    dat_frame.save(sxm=False, data=False)
    #dat_frame.addParticle(b)
    #dat_frame.get_Image()
    #dat_frame.save()
    #print(dat_frame.has_overlaps())
    #print(a.true_overlap(b))
    #print(1)
    #generate(fn)
    #print(2)
    #generate(fn)
    #print(3)
    #for i in range(10):
    #    generate(fn)
    #multi_test(30)
    #generate(fn)
    #My_SXM.show_data("sxm/Image1.sxm")
    #SXM_info.adjust_to_image(data, "Test4.sxm")
    #My_SXM.write_header("Test4.sxm")
    #My_SXM.write_image("Test4.sxm", data)
    #My_SXM.show_data("Test4.sxm")
    #My_SXM.show_data("Test4.sxm")
    #My_SXM.get_data_test(filename)
    #print(My_SXM.get_informations("Test.sxm"))
    #plt.imshow(My_SXM.get_data("Test.sxm"))
    #plt.show()

    evaluateLog()




