import Configuration as conf
from Images import MyImage
from Particle import Particle
from FilenameGenerator import FilenameGenerator
from DataFrame import DataFrame
from Functions import *
import Configuration as cfg
import math


def test_frame(data_frame):
    maxn = len(range(25, cfg.get_width(), 50)) * len(range(25, cfg.get_height(), 50))
    th = 0
    #dth = 2 * math.pi / maxn
    dth = 0.05
    for y in range(25, cfg.get_height(), 50):
        for x in range(25, cfg.get_width(), 50):
            data_frame.addParticle(Particle(x, y, th))
            th += dth


if __name__ == "__main__":
    clearLog()
    fn_gen = FilenameGenerator()
    dat_frame = DataFrame(fn_gen)
    # dat_frame.addParticles(conf.get_particles_per_image())
    # dat_frame.createImage_efficient()
    #dat_frame.addParticle(Particle(200, 200, 3*np.pi/2 + 0.2))
    test_frame(dat_frame)
    dat_frame.createImage_efficient_with_new_Turn()
    dat_frame.save()
    evaluateLog()


