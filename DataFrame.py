import multiprocessing as mp
import copy, os
import math, random
from Particle import Particle
from Images import MyImage
#from Maths.Functions import measureTime
#from Configuration.Files import MultiFileManager as fm
import Configuration as cfg
import numpy as np
from Functions import measureTime


class DataFrame:

    def __init__(self, fn_gen):
        self.objects = []
        self.fn_gen = fn_gen
        self.text = ""
        self.img = None
        self.angle_char_len = cfg.get_angle_char_len()
        self.angle_correlation_std = cfg.get_angle_stdderiv()
        self.area = cfg.get_width() * cfg.get_height()
        self.max_dist = np.sqrt(np.square(cfg.get_height()) + np.square(cfg.get_width()))

    def getIterator(self):
        return self.objects

    def __len__(self):
        return len(self.objects)

    def addParticle(self, part=None):
        if part is None:
            self.objects.append(Particle())
        else:
            self.objects.append(part)

    def _overlaps_any(self, part):
        if len(self.objects) == 0:
            return False
        for p in self.objects:
            if part.true_overlap(p):
                return True
        return False

    def _get_thatnot_overlaps(self, maximumtries=1000):
        if len(self.objects) == 0:
            return Particle()
        p = Particle()
        for i in range(maximumtries):
            if self._overlaps_any(p):
                p = Particle()
            else:
                return p
        return p

    def _calc_angle_weight(self, part1, part2):
        drel = part1.get_distance_to(part2) / self.max_dist
        #return np.exp(-self.angle_char_len/drel)
        return self.angle_char_len/drel

    def _calc_angle_for_particle(self, particle):
        if len(self.objects) == 0:
            return np.pi * 2 * random.random()
        amount = 0
        angles = 0
        distances = 0
        for part_it in self.objects:
            weight = self._calc_angle_weight(particle, part_it)
            amount += weight
            distances += part_it.get_distance_to(particle)
            th = part_it.get_theta()
            angles += (th if th < np.pi else -(2 * np.pi - th)) * weight
        med = angles / amount
        std = len(self.objects) * self.angle_correlation_std / amount
        exp_std = np.exp(std) - 1
        print("Particle {} has weighted distance of {}, total distance of {} resulting in sigma={}, expSigma of {}".format(len(self.objects) + 1, amount, distances, std, exp_std))
        return random.gauss(med, exp_std)



    def addParticles(self, amount=None, coverage=None, overlapping=True, maximum_tries=1000):
        #widthout angle correlation
        if self.angle_char_len == 0:
            if not overlapping:
                if amount is not None:
                    for i in range(amount):
                        self.objects.append(self._get_thatnot_overlaps(maximum_tries))
                elif coverage is not None:
                    self.objects.append(self._get_thatnot_overlaps(maximum_tries))
                else:
                    for i in range(cfg.get_particles_per_image()):
                        self.objects.append(self._get_thatnot_overlaps(maximum_tries))
            #w/ angle, w/o overlapping
            else:
                if amount is not None:
                    for i in range(amount):
                        self.objects.append(Particle())
                elif coverage is not None:
                    while self.coverage() < coverage:
                        self.objects.append(Particle())
                else:
                    for i in range(cfg.get_particles_per_image()):
                        self.objects.append(Particle())
        #w/ angle correlation
        else:
            if not overlapping:
                if amount is not None:
                    for i in range(amount):
                        p = self._get_thatnot_overlaps(maximum_tries)
                        p.set_theta(self._calc_angle_for_particle(p))
                        self.objects.append(p)
                elif coverage is not None:
                    while self.coverage() < coverage:
                        p = self._get_thatnot_overlaps(maximum_tries)
                        p.set_theta(self._calc_angle_for_particle(p))
                        self.objects.append(p)
                else:
                    for i in range(cfg.get_particles_per_image()):
                        p = self._get_thatnot_overlaps(maximum_tries)
                        p.set_theta(self._calc_angle_for_particle(p))
                        self.objects.append(p)
            #w/ angle, w/o overlapping
            else:
                if amount is not None:
                    for i in range(amount):
                        p = Particle()
                        p.set_theta(self._calc_angle_for_particle(p))
                        self.objects.append(p)
                elif coverage is not None:
                    while self.coverage() < coverage:
                        p = Particle()
                        p.set_theta(self._calc_angle_for_particle(p))
                        self.objects.append(p)
                else:
                    for i in range(cfg.get_particles_per_image()):
                        p = Particle()
                        p.set_theta(self._calc_angle_for_particle(p))
                        self.objects.append(p)

    @measureTime
    def createImage(self):
        self.img = MyImage()
        for part in self.objects:
            self.img.addParticle(part)
        self.img.updateImage()
        #img.noise....etc

    @measureTime
    def createImage_efficient(self):
        self.img = MyImage()
        width = cfg.get_width()
        height = cfg.get_height()
        matrix = np.zeros((width, height))

        for part in self.objects:
            eff_mat, x, y = part.efficient_Matrix()
            mat_w = eff_mat.shape[0]

            mat_h = eff_mat.shape[1]
            for i in range(mat_w):
                for j in range(mat_h):
                    new_x = x - math.floor((mat_w/2)) + i
                    new_y = y - math.floor(mat_h/2) + j
                    if not (0 <= new_x < width and 0 <= new_y < height):
                        continue
                    matrix[new_x, new_y] += eff_mat[i, j]

        self.img.addMatrix(matrix)
        self.img.updateImage()

    @measureTime
    def createImage_efficient_with_new_Turn(self):
        self.img = MyImage()
        width = cfg.get_width()
        height = cfg.get_height()
        matrix = np.zeros((width, height))

        for part in self.objects:
            eff_mat, x, y = part.efficient_Matrix_turned()
            mat_w = eff_mat.shape[0]

            mat_h = eff_mat.shape[1]
            for i in range(mat_w):
                for j in range(mat_h):
                    new_x = x - math.floor((mat_w / 2)) + i
                    new_y = y - math.floor(mat_h / 2) + j
                    if not (0 <= new_x < width and 0 <= new_y < height):
                        continue
                    matrix[new_x, new_y] += eff_mat[i, j]

        self.img.addMatrix(matrix)
        self.img.updateImage()

    def get_Image(self):
        self.createImage_efficient_with_new_Turn()

    def createText(self):
        strings = [Particle.str_Header()]
        for part in self.objects:
            strings.append(str(part))
        self.text = "\n".join(strings)

    def save(self):
        if self.img is None:
            self.createImage_efficient()
        if len(self.text) == 0:
            self.createText()
        img_path, dat_path, index = self.fn_gen.generate_Tuple()
        try:
            with open(dat_path, "w") as dat_file:
                dat_file.write(self.text)
        except FileNotFoundError:
            os.mkdir(cfg.get_data_folder())
            with open(dat_path, "w") as dat_file:
                dat_file.write(self.text)
        self.img.saveImage(img_path)


    def hasPoints(self):
        # return not self.points.empty()
        return len(self.objects) > 0

    def coverage(self):
        area = cfg.get_width() * cfg.get_height()
        covered = 0
        for part in self.objects:
            covered += np.pi * np.square(part.get_dimension())
        return covered/area

    def has_overlaps(self):
        for part in self.objects:
            for part2 in self.objects:
                if part == part2:
                    continue
                if part.true_overlap(part2):
                    return True
        return False

    def __str__(self):
        return str(self.objects)

