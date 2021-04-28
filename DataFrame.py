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

    # Constructor. Reads Config into local variables
    def __init__(self, fn_gen):
        self.objects = []
        self.fn_gen = fn_gen
        self.text = ""
        self.img = None
        self.angle_char_len = cfg.get_angle_char_len()
        self.angle_correlation_std = cfg.get_angle_stdderiv()
        self.area = cfg.get_width() * cfg.get_height()
        self.max_dist = np.sqrt(np.square(cfg.get_height()) + np.square(cfg.get_width()))
        self.min_angle = cfg.get_angle_range_min()
        self.max_angle = cfg.get_angle_range_max()
        self.use_range = cfg.get_angle_range_usage()
        self.image_noise_mu = cfg.get_image_noise_mu()
        self.image_noise_sigma = cfg.get_image_noise_std_deriv()
        self.use_noise = self.image_noise_sigma != 0
        self.use_dragging = cfg.def_dragging_error
        self.dragging_possibility = cfg.get_dragging_possibility()
        self.dragging_speed = cfg.get_dragging_speed()
        self.raster_angle = cfg.get_raster_angle()

    #returns iterator over Particles
    def getIterator(self):
        return self.objects

    #gets number of particles
    def __len__(self):
        return len(self.objects)

    # adds a given particle or, if not provided a random one
    def addParticle(self, part=None):
        if part is None:
            self.objects.append(Particle())
        else:
            self.objects.append(part)

    #checks wheather part overlaps any existing particle
    def _overlaps_any(self, part):
        if len(self.objects) == 0:
            return False
        for p in self.objects:
            if part.true_overlap(p):
                return True
        return False

    #returns random particle, that does not overlap with any other
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

    # calculates particles weight for importance in surrounding particles
    def _calc_angle_weight(self, part1, part2): # ToDo: Still very sketchy
        drel = part1.get_distance_to(part2) / self.max_dist
        #return np.exp(-self.angle_char_len/drel)
        return self.angle_char_len/drel

    # calculates a random angle for partilce depending on its surrounding with correlation
    def _calc_angle_for_particle(self, particle): # ToDo: Still very sketchy
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
        #print("Particle {} has weighted distance of {}, total distance of {} resulting in sigma={}, expSigma of {}".format(len(self.objects) + 1, amount, distances, std, exp_std))
        return random.gauss(med, exp_std)

    # returns rand
    def _random_angle_range(self):
        if self.min_angle > self.max_angle:
            shiftl = 2 * np.pi - self.min_angle
            ret = random.random() * self.max_angle + shiftl
            ret -= shiftl
            return ret
        else:
            return random.uniform(self.min_angle, self.max_angle)


    def addParticles(self, amount=None, coverage=None, overlapping=True, maximum_tries=1000):
        #widthout angle correlation
        if not self.use_range:
            if self.angle_char_len == 0:
                if not overlapping:
                    if amount is not None:
                        for i in range(amount):
                            p = self._get_thatnot_overlaps(maximum_tries)
                            self.objects.append(p)
                    elif coverage is not None:
                        while self.coverage() < coverage:
                            p = self._get_thatnot_overlaps(maximum_tries)
                            self.objects.append(p)
                    else:
                        for i in range(cfg.get_particles_per_image()):
                            p = self._get_thatnot_overlaps(maximum_tries)
                            self.objects.append(p)
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
        #use angle range
        else:
            if not overlapping:
                if amount is not None:
                    for i in range(amount):
                        p = self._get_thatnot_overlaps(maximum_tries)
                        p.set_theta(self._random_angle_range())
                        self.objects.append(p)
                elif coverage is not None:
                    while self.coverage() < coverage:
                        p = self._get_thatnot_overlaps(maximum_tries)
                        p.set_theta(self._random_angle_range())
                        self.objects.append(p)
                else:
                    for i in range(cfg.get_particles_per_image()):
                        p = self._get_thatnot_overlaps(maximum_tries)
                        p.set_theta(self._random_angle_range())
                        self.objects.append(p)
            #w/ angle, w/o overlapping
            else:
                if amount is not None:
                    for i in range(amount):
                        p = Particle()
                        p.set_theta(self._random_angle_range())
                        self.objects.append(p)
                elif coverage is not None:
                    while self.coverage() < coverage:
                        p = Particle()
                        p.set_theta(self._random_angle_range())
                        self.objects.append(p)
                else:
                    for i in range(cfg.get_particles_per_image()):
                        p = Particle()
                        p.set_theta(self._random_angle_range())
                        self.objects.append(p)

    #deprecated
    @measureTime
    def createImage(self):
        print("Deprecated 3154251534")
        self.img = MyImage()
        for part in self.objects:
            self.img.addParticle(part)


        self.img.updateImage() #Always call last
        #img.noise....etc

    # deprecated
    @measureTime
    def createImage_efficient(self):
        print("Deprecated 46541741")
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
        print("Deprecated 453643")
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
        #self.img.updateImage()

    def create_Image_Visualization(self):
        self.img = MyImage()
        width = cfg.get_width()
        height = cfg.get_height()
        matrix = np.zeros((width, height))

        for part in self.objects:
            for tuple in part.get_visualization():
                eff_mat, x, y = tuple
                mat_w = eff_mat.shape[0]

                #ToDo: possible failure
                x = int(np.round(x))
                y = int(np.round(y))

                mat_h = eff_mat.shape[1]
                for i in range(mat_w):
                    for j in range(mat_h):
                        new_x = x - math.floor((mat_w / 2)) + i
                        new_y = y - math.floor(mat_h / 2) + j
                        if not (0 <= new_x < width and 0 <= new_y < height):
                            continue
                        matrix[new_x, new_y] += eff_mat[i, j]

        self.img.addMatrix(matrix) #Indentd  too far right

    def _drag_particles(self):
        for part in self.objects:
            if random.random() < self.dragging_possibility:
                part.drag(self.dragging_speed, self.raster_angle)

    def get_Image(self):
        #if self.use_dragging: #ToDo: Removed only for testing
        #    self._drag_particles()
        self.create_Image_Visualization()
        if self.use_noise:
            self.img.noise(self.image_noise_mu, self.image_noise_sigma)
        self.img.updateImage()

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
