import Configuration as conf
from Images import MyImage
from Particle import Particle
from FilenameGenerator import FilenameGenerator
from DataFrame import DataFrame
from Functions import *

if __name__ == "__main__":
    clearLog()
    fn_gen = FilenameGenerator()
    dat_frame = DataFrame(fn_gen)
    dat_frame.addParticles(conf.get_particles_per_image())
    dat_frame.createImage_efficient()
    dat_frame.save()
    evaluateLog()
