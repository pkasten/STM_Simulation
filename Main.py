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


class Generator(Process):
    def __init__(self, fn_gen):
        super().__init__()
        self.fn_gen = fn_gen



    def run(self):
        while True:
            generate(self.fn_gen)



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

    #dat_frame = DataFrame(fn)
    #generate(fn)
    #a = Particle(200, 200, 0)
    #b = Particle(230, 200, 0.3)
    #dat_frame.addParticle(a)
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
    multi_test(1800
               )

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




