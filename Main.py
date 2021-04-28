import Configuration as conf
from Images import MyImage
from Particle import Particle
from FilenameGenerator import FilenameGenerator
from DataFrame import DataFrame
from Functions import *
import Configuration as cfg
import math


def test_frame(data_frame):
    maxn = len(range(50, cfg.get_width(), 100)) * len(range(50, cfg.get_height(), 100))
    th = 0
    dth = 2 * math.pi / maxn
    # dth = 0.05
    for y in range(50, cfg.get_height(), 100):
        for x in range(50, cfg.get_width(), 100):
            data_frame.addParticle(Particle(x, y, th))
            th += dth


if __name__ == "__main__":
    clearLog()
    fn_gen = FilenameGenerator()
    dat_frame = DataFrame(fn_gen)
    # dat_frame.addParticles(conf.get_particles_per_image())
    # dat_frame.createImage_efficient()
    dat_frame.addParticle(Particle(200, 200, 3*np.pi/2 + 0.2))
    #test_frame(dat_frame)
    #for i in range(10):
    #dat_frame.addParticles(coverage=0.3, overlapping=False)
    #dat_frame.addParticles(amount=3)
    dat_frame._drag_particles()
    dat_frame.addParticle(Particle(100 ,200, 0))
    dat_frame.addParticle(Particle(300, 200, 0))
    dat_frame.addParticle(Particle(200, 100, 0))
    dat_frame.addParticle(Particle(200, 300, 0))
    dat_frame.get_Image()
    dat_frame.save()
    evaluateLog()


