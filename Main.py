import Configuration as conf
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
    filename = "Test_SXM_File.sxm"
    My_SXM.get_informations(filename)
    #plt.imshow(My_SXM.get_data(filename))
    #plt.show()
   # print(time.perf_counter() - start)
    #dat_frame = DataFrame(fn)
   # print(time.perf_counter() - start)
    #dat_frame.addParticles()
   ## print(time.perf_counter() - start)
    #dat_frame.get_Image()
   # print(time.perf_counter() - start)
    #dat_frame.save()
   # print(time.perf_counter() - start)
    #print(dat_frame.has_overlaps())
    #print(time.perf_counter() - start)
    #multi_test(120)
    #sd = False
    #while not sd:
    #    start = time.perf_counter()
    #    fn = FilenameGenerator()
        #rint(time.perf_counter() - start)
    #   dat_frame = DataFrame(fn)
    #    #rint(time.perf_counter() - start)
    #    dat_frame.addParticles()
    #    #rint(time.perf_counter() - start)
    #    dat_frame.get_Image()
    #    #rint(time.perf_counter() - start)
    #    dat_frame.save()
    #    #rint(time.perf_counter() - start)
    #    sd = dat_frame.has_overlaps()
    #    print(sd)
        #rint(time.perf_counter() - start)


    evaluateLog()




