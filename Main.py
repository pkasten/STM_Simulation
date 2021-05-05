import Configuration as conf
from Images import MyImage
from Particle import Particle
from FilenameGenerator import FilenameGenerator
from DataFrame import DataFrame
from Functions import *
import Configuration as cfg
import math, time
import matplotlib.pyplot as plt


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
    start = time.perf_counter()
    for i in range(90):
        start = time.perf_counter()
        dat_frame = DataFrame(fn_gen)
        dat_frame.addParticles(amount=i)
        dat_frame.get_Image()
        dat_frame.save()
        x.append(i)
        y.append(time.perf_counter() - start)

    plt.plot(x, y)
    plt.show()

if __name__ == "__main__":
    clearLog()
    fn_gen = FilenameGenerator()
    dat_frame = DataFrame(fn_gen)
    # dat_frame.addParticles(conf.get_particles_per_image())
    # dat_frame.createImage_efficient()
    #dat_frame.addParticle(Particle(200, 200, dat_frame._random_angle_range()))
    #test_frame(dat_frame)
    #for i in range(10):
    #dat_frame.addParticles(coverage=0.2, overlapping=False)
    p1 = Particle(200, 200, 0.01)
    p2 = Particle(280, 200, 0.3)
    dat_frame.addParticle(p1)
    dat_frame._drag_particles()
    dat_frame.addParticle(p2)
    print("Has Overlaps: {}".format(dat_frame.has_overlaps()))
    #print(p1.true_overlap(p2))
    #print(p2.true_overlap(p1))
        #dat_frame.addParticles(amount=3)
    #dat_frame._drag_particles()
    #dat_frame.addParticle(Particle(100 ,200, 0))
    #dat_frame.addParticle(Particle(300, 200, 0))
    #dat_frame.addParticle(Particle(200, 100, 0))
    #dat_frame.addParticle(Particle(200, 300, 0))
    dat_frame.get_Image()
    dat_frame.save()


    evaluateLog()




