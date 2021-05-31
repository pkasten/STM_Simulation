import multiprocessing as mp
import copy, os
import math, random
import time

import scipy.optimize

from DustParticle import DustParticle
from Molecule import Molecule, Tests_Gitterpot
from Particle import Particle, Double_Particle
from Images import MyImage
# from Maths.Functions import measureTime
# from Configuration.Files import MultiFileManager as fm
import Configuration as cfg
import numpy as np
import matplotlib.pyplot as plt
from Functions import measureTime
from My_SXM import My_SXM
import scipy.optimize as opt
# from Doubled import Double_Frame
from Charge import Charge
from Distance import Distance
import pickle
from functools import lru_cache


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
        self.max_dist = Distance(True, np.sqrt(np.square(cfg.get_height().ang) + np.square(cfg.get_width().ang)))
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
        self.double_tip_poss = cfg.get_double_tip_possibility()
        self.passed_args_particles = None
        self.passed_args_Obj = None
        self.passed_args_Ordered = None
        self.img_width = cfg.get_width()
        self.img_height = cfg.get_height()
        self.use_crystal_orientations = cfg.get_crystal_orientation_usage()
        self.crystal_directions_num = cfg.get_no_of_orientations()
        self.crystal_directions = cfg.get_crystal_orientations_array()
        if self.use_crystal_orientations:
            self.angle_char_len = 4000
        self.oldPotential = None
        self.add_To_Potential = []
        # self.potential_map = self.calc_potential_map()
        self.overlapping_energy = 1000
        self.overlapping_threshold = cfg.get_overlap_threshold()
        self.part_laenge = cfg.get_part_length()
        self.atomic_step_height = cfg.get_atomic_step_height()
        # AtomStepFermi
        self.fermi_exp = cfg.get_fermi_exp() * cfg.get_nn_dist().px / Distance(True, 0.07).px
        self.fermi_exp = 0.05
        self.fermi_range = np.log(99) / self.fermi_exp + cfg.get_atomic_step_height().px
        # self.fermi_range = 100
        self.dust_amount = cfg.get_dust_amount()
        self.usedust = self.dust_amount != 0
        self.dust_particles = []
        self.max_height = cfg.get_max_height()
        self.use_img_shift = cfg.get_use_img_shift()

    # returns iterator over Particles
    def getIterator(self):
        return self.objects

    # gets number of particles
    def __len__(self):
        return len(self.objects)

    # adds a given particle or, if not provided a random one
    def addParticle(self, part=None):
        if self.passed_args_particles is None:
            self.passed_args_particles = (1, None, True, 1000)
        else:
            if self.passed_args_particles[0] is None:
                self.passed_args_particles = (len(self.objects), self.passed_args_particles[1], self.passed_args_particles[2], self.passed_args_particles[3])
            newargs = self.passed_args_particles[0] + 1, self.passed_args_particles[1], self.passed_args_particles[2], self.passed_args_particles[3]
            self.passed_args_particles = newargs
        if part is None:
            self.objects.append(Particle())

        else:
            self.objects.append(part)
            for c in part.get_charges():
                self.add_To_Potential.append(c)

    # checks wheather part overlaps any existing particle
    def _overlaps_any(self, part):
        # start = time.perf_counter()

        if len(self.objects) == 0:
            # print("Overlaps any took {}".format(time.perf_counter() - start))
            return False
        for p in self.objects:
            if not part.dragged and not p.dragged:
                if math.dist([p.x.px, p.y.px], [part.x.px, part.y.px]) > np.sqrt(2) * max(part.effect_range,
                                                                                          p.effect_range):
                    continue
            if part.true_overlap(p):
                # print("Overlaps any took {}".format(time.perf_counter() - start))
                return True
        # print("Overlaps any took {}".format(time.perf_counter() - start))
        return False

    # returns random particle, that does not overlap with any other
    def _get_thatnot_overlaps(self, maximumtries=1000, calcangle=False):
        if calcangle:
            if len(self.objects) == 0:
                p = Particle()
                p.set_theta(self._calc_angle_for_particle(p))
                return p
            p = Particle()
            p.set_theta(self._calc_angle_for_particle(p))
            for i in range(maximumtries):
                if self._overlaps_any(p):
                    # print("Retry")
                    p = Particle()
                    p.set_theta(self._calc_angle_for_particle(p))
                else:
                    return p
            print("MaxTries Exhausted")
            return p

        if len(self.objects) == 0:
            return Particle()
        p = Particle()
        for i in range(maximumtries):
            if self._overlaps_any(p):
                # print("Retry")
                p = Particle()
            else:
                return p
        # print("MaxTries Exhausted_a")
        return p

    def get_dragged_that_mot_overlaps(self, maximumtries, angle=None, setangle=False):
        def _set_p():
            p = Particle()
            if angle is not None:
                p.set_theta(angle)
            if setangle:
                p.set_theta(self._calc_angle_for_particle(p))
            p.drag(self.dragging_speed, self.raster_angle)
            return p

        if len(self.objects) == 0:
            return _set_p()
        p = _set_p()
        for i in range(maximumtries):
            if self._overlaps_any(p):
                # print("Retry_b")
                p = _set_p()
            else:
                return p
        # print("MaxTries Exhausted_b")
        return p

    # calculates a random angle for partilce depending on its surrounding with correlation
    def _calc_angle_for_particle(self, particle):  # ToDo: Still very sketchy

        if self.use_crystal_orientations:
            print("Deprecated 132813")
            if self._orients_along_crystal(particle):
                # particle.set_height(0.7)
                # print(self.crystal_directions)
                return random.choice(self.crystal_directions)
            else:
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
                return random.gauss(med, exp_std)

        print("WARNING: Not using Crystal Orientation")

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
        # print("Particle {} has weighted distance of {}, total distance of {} resulting in sigma={}, expSigma of {}".format(len(self.objects) + 1, amount, distances, std, exp_std))
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

    def addParticles(self, optimumEnergy=False, amount=None, coverage=None, overlapping=False, maximum_tries=1000):

        if optimumEnergy:
            print("Deprecated 8414813")
            if amount is not None:
                for i in range(amount):
                    self.add_at_optimum_energy_new(self.img_width * random.random(), self.img_height * random.random(),
                                                   2 * np.pi * random.random())
            elif coverage is not None:
                while self.coverage() < coverage:
                    self.add_at_optimum_energy_new(self.img_width * random.random(), self.img_height * random.random(),
                                                   2 * np.pi * random.random())
            else:
                for i in range(cfg.get_particles_per_image()):
                    self.add_at_optimum_energy_new(self.img_width * random.random(), self.img_height * random.random(),
                                                   2 * np.pi * random.random())
            return

        # without angle correlation
        self.passed_args_particles = (optimumEnergy, amount, coverage, overlapping, maximum_tries)
        # print("{}, {}, {}".format(self.use_range, self.angle_char_len, overlapping))
        if not self.use_range:
            if self.angle_char_len == 0:
                if not overlapping:
                    if amount is not None:
                        for i in range(amount):
                            if random.random() < self.dragging_possibility:
                                p = self.get_dragged_that_mot_overlaps(maximum_tries)
                                self.objects.append(p)
                            else:
                                p = self._get_thatnot_overlaps(maximum_tries)
                                self.objects.append(p)
                    elif coverage is not None:
                        while self.coverage() < coverage:
                            if random.random() < self.dragging_possibility:
                                p = self.get_dragged_that_mot_overlaps(maximum_tries)
                                self.objects.append(p)
                            else:
                                p = self._get_thatnot_overlaps(maximum_tries)
                                self.objects.append(p)
                    else:
                        for i in range(cfg.get_particles_per_image()):
                            if random.random() < self.dragging_possibility:
                                p = self.get_dragged_that_mot_overlaps(maximum_tries)
                                self.objects.append(p)
                            else:
                                p = self._get_thatnot_overlaps(maximum_tries)
                                self.objects.append(p)
                # w/ angle, w overlapping
                else:
                    if amount is not None:
                        for i in range(amount):
                            p = Particle()
                            if random.random() < self.dragging_possibility:
                                p.drag(self.dragging_speed, self.raster_angle)
                            self.objects.append(p)
                    elif coverage is not None:
                        while self.coverage() < coverage:
                            p = Particle()
                            if random.random() < self.dragging_possibility:
                                p.drag(self.dragging_speed, self.raster_angle)
                            self.objects.append(p)
                    else:
                        for i in range(cfg.get_particles_per_image()):
                            p = Particle()
                            if random.random() < self.dragging_possibility:
                                p.drag(self.dragging_speed, self.raster_angle)
                            self.objects.append(p)
            # w/ angle correlation
            else:
                if not overlapping:
                    if amount is not None:
                        for i in range(amount):
                            if random.random() < self.dragging_possibility:
                                p = self.get_dragged_that_mot_overlaps(maximum_tries, setangle=True)
                                self.objects.append(p)
                            else:
                                p = self._get_thatnot_overlaps(maximum_tries, calcangle=True)
                                # p.set_theta(self._calc_angle_for_particle(p))
                                self.objects.append(p)
                    elif coverage is not None:
                        while self.coverage() < coverage:
                            if random.random() < self.dragging_possibility:
                                p = self.get_dragged_that_mot_overlaps(maximum_tries, setangle=True)
                                self.objects.append(p)
                            else:
                                p = self._get_thatnot_overlaps(maximum_tries, calcangle=True)
                                # p.set_theta(self._calc_angle_for_particle(p))
                                self.objects.append(p)
                    else:
                        # print("Normal") Normaldurchlauf
                        for i in range(cfg.get_particles_per_image()):
                            if random.random() < self.dragging_possibility:
                                p = self.get_dragged_that_mot_overlaps(maximum_tries, setangle=True)
                                self.objects.append(p)
                            else:
                                p = self._get_thatnot_overlaps(maximum_tries, calcangle=True)
                                # p.set_theta(self._calc_angle_for_particle(p))
                                self.objects.append(p)
                # w/ angle, w overlapping
                else:
                    if amount is not None:
                        for i in range(amount):
                            p = Particle()
                            p.set_theta(self._calc_angle_for_particle(p))
                            if random.random() < self.dragging_possibility:
                                p.drag(self.dragging_speed, self.raster_angle)
                            self.objects.append(p)
                    elif coverage is not None:
                        while self.coverage() < coverage:
                            p = Particle()
                            p.set_theta(self._calc_angle_for_particle(p))
                            if random.random() < self.dragging_possibility:
                                p.drag(self.dragging_speed, self.raster_angle)
                            self.objects.append(p)
                    else:
                        for i in range(cfg.get_particles_per_image()):
                            p = Particle()
                            p.set_theta(self._calc_angle_for_particle(p))
                            if random.random() < self.dragging_possibility:
                                p.drag(self.dragging_speed, self.raster_angle)
                            self.objects.append(p)
        # use angle range
        else:
            if not overlapping:
                if amount is not None:
                    for i in range(amount):
                        if random.random() < self.dragging_possibility:
                            p = self.get_dragged_that_mot_overlaps(maximum_tries, angle=self._random_angle_range())
                            self.objects.append(p)
                        else:
                            p = self._get_thatnot_overlaps(maximum_tries)
                            p.set_theta(self._random_angle_range())
                            self.objects.append(p)
                elif coverage is not None:
                    while self.coverage() < coverage:
                        if random.random() < self.dragging_possibility:
                            p = self.get_dragged_that_mot_overlaps(maximum_tries, angle=self._random_angle_range())
                            self.objects.append(p)
                        else:
                            p = self._get_thatnot_overlaps(maximum_tries)
                            p.set_theta(self._random_angle_range())
                            self.objects.append(p)
                else:
                    for i in range(cfg.get_particles_per_image()):
                        if random.random() < self.dragging_possibility:
                            p = self.get_dragged_that_mot_overlaps(maximum_tries, angle=self._random_angle_range())
                            self.objects.append(p)
                        else:
                            p = self._get_thatnot_overlaps(maximum_tries)
                            p.set_theta(self._random_angle_range())
                            self.objects.append(p)
            # w/ angle, w/ overlapping
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

    def addObject(self, ob):
        self.objects.append(ob)

    def addObjects(self, Object=Molecule, amount=None, coverage=None, overlapping=False, maximum_tries=1000):
        self.passed_args_Obj = Object, amount, coverage, overlapping, maximum_tries

        def get_dragged_that_not_overlaps(maximumtries):
            def _set_p():
                p = Object()
                p.drag(self.dragging_speed, self.raster_angle)
                return p

            if len(self.objects) == 0:
                return _set_p()
            p = _set_p()
            for i in range(maximumtries):
                if self._overlaps_any(p):
                    p = _set_p()
                else:
                    return p
            return p

        def _get_thatnot_overlaps(maximumtries):
            # print("#Obj: {}, has Overlaps: {}".format(len(self.objects), self.has_overlaps()))
            if len(self.objects) == 0:
                return Object()
            p = Object()
            for i in range(maximumtries):
                # print("Added at x={}, y= {}".format(p.pos[0].px, p.pos[1].px))
                if self._overlaps_any(p):
                    # print("Retry")
                    p = Object()
                else:
                    return p
            # print("MaxTries Exhausted_a")
            # print("#Obj: {}, has Overlaps: {}".format(len(self.objects), self.has_overlaps()))
            return p

        #self.passed_args_particles = (amount, coverage, overlapping, maximum_tries)
        # print("{}, {}, {}".format(self.use_range, self.angle_char_len, overlapping))
        if amount is not None:
            for i in range(amount):
                if random.random() < self.dragging_possibility:
                    p = get_dragged_that_not_overlaps(maximum_tries)
                    self.objects.append(p)
                else:
                    p = _get_thatnot_overlaps(maximum_tries)
                    self.objects.append(p)
        elif coverage is not None:
            while self.coverage() < coverage:
                if random.random() < self.dragging_possibility:
                    p = get_dragged_that_not_overlaps(maximum_tries)
                    self.objects.append(p)
                else:
                    p = _get_thatnot_overlaps(maximum_tries)
                    self.objects.append(p)
        else:
            for i in range(cfg.get_particles_per_image()):
                if random.random() < self.dragging_possibility:
                    p = get_dragged_that_not_overlaps(maximum_tries)
                    self.objects.append(p)
                else:
                    p = _get_thatnot_overlaps(maximum_tries)
                    self.objects.append(p)

    def _add_at_pos_dragged(self,Object, pos, theta):
        def _set_p(fak):
            p = Object(pos, theta)
            p.drag(fak * self.dragging_speed.px, self.raster_angle)
            return p

        if len(self.objects) == 0:
            print(1)
            return _set_p(1)
        f = 1.0
        p = _set_p(f)
        for i in range(10):
            if self._overlaps_any(p):
                # print("Retry_b")
                f *= 0.8
                p = _set_p(f)
            else:
                print("-> {:.3f}".format(f))
                return p
        print("MaxTries Exhausted_b")
        print("-> {:.3f}".format(f))
        return p

    def add_Ordered(self, Object=Molecule, theta=None, factor=1.0):
        offset = Distance(False, cfg.get_px_overlap())
        self.passed_args_Ordered = (Object, theta)

        def bog(deg):
            return np.pi * deg / 180

        def add_ordered_NCPh3CN(theta=None):
            if theta is None:
                theta_0 = random.random() * np.pi * 2
            else:
                theta_0 = theta
            # theta_0 = 0 # ToDo Rem
            # theta_0 = bog(-4.5636)
            #print("theta0: {:.1f}°".format(theta_0 / np.pi * 180))
            dist_h = Distance(True, 13.226) * factor
            dist_v = Distance(True, 13.1933) * factor
            gv_a = np.array([dist_h * np.cos(theta_0), dist_h * np.sin(theta_0)])
            gv_b = np.array([-dist_v * np.sin(theta_0), dist_v * np.cos(theta_0)])

            ang_a = theta_0 + bog(19.61545)
            ang_b = theta_0 + bog(59.40909)

            pairs = []

            start = np.array([Distance(True, 0), Distance(True, 0)])
            current = np.array([0, 0])
            a_temp = self.img_width / gv_b[0] if gv_b[0].px != 0 else -np.infty
            b_temp = self.img_height / gv_b[1] if gv_b[1].px != 0 else -np.infty
            j_max = int(np.ceil(max(a_temp, b_temp)))
            c_temp = ((self.img_width + j_max * gv_b[0]) / gv_a[0]) if gv_a[0].px != 0 else j_max
            i_max = int(np.ceil(c_temp))
            # print(i_max, j_max)
            for i in range(-100, 100):
                for j in range(-100, 100):
                    current = start + (gv_a * i) + (gv_b * j)
                    # print(current[0], self.img_width)
                    if self.img_width + offset > current[0] > (-1) * offset and offset + self.img_height > current[
                        1] > (-1) * offset:
                        pairs.append((current, ang_a if (i + j) % 2 == 0 else ang_b))

            for pair in pairs:
                self.objects.append(Object(pos=pair[0], theta=pair[1]))

        def add_ordered_NCPh4CN(theta=None):

            if theta is None:
                theta_0 = random.random() * np.pi * 2
            else:
                theta_0 = theta
            chirality = np.sign(random.random() - 0.5)
            if chirality > 0:
                ang_ud = (theta_0 + bog(217.306)) % (2*np.pi)
                ang_lr = (theta_0 + bog(134.136)) % (2*np.pi)
                ud_lat_ang = (theta_0 + bog(189.595)) % (2*np.pi)
                lr_lat_ang = (theta_0 + bog(100.773)) % (2*np.pi)
                cross_ang = (theta_0 + bog(147.494)) % (2*np.pi)

            else:
                ang_ud = (theta_0 + bog(142.694)) % (2*np.pi)
                ang_lr = (theta_0 + bog(225.864)) % (2*np.pi)
                ud_lat_ang = (theta_0 + bog(170.405)) % (2*np.pi) # Nicht sicher mit unsymmetr Molekülen
                lr_lat_ang = (theta_0 + bog(259.227)) % (2*np.pi)
                cross_ang = (theta_0 + bog(212.506)) % (2*np.pi)


            ud_dist = Distance(True, 24.1689) * factor
            lr_dist = Distance(True, 22.7745) * factor

            crossU_R = Distance(True, 17.1015) * factor

            vec_ud_lr = np.array([crossU_R * np.sin(cross_ang), -crossU_R *np.cos(cross_ang)])
            vec_r = np.array([lr_dist * np.sin(lr_lat_ang), -lr_dist * np.cos(lr_lat_ang)])
            vec_u = np.array([ud_dist * np.sin(ud_lat_ang), -ud_dist * np.cos(ud_lat_ang)])

            #print("LR: {:.1f}°".format(lr_lat_ang/3.14159 * 180))
            #print(vec_r)
            #print("UD: {:.1f}°".format(ud_lat_ang/3.14159 * 180))
            #print(vec_u)

            pairs = []

            i = 5
            j = -10



            start = np.array([Distance(True, 0), Distance(True, 0)])
            for i in range(-100, 100):
                for j in range(-100, 100):
                    current = start + (vec_u * i) + (vec_r * j)
                    if self.img_width + offset > current[0] > (-1) * offset and offset + self.img_height > current[
                        1] > (-1) * offset:
                        pairs.append((current, ang_ud))
                        #print("No {} Appended ({},{})at {}".format(len(pairs), i, j, current))
                    secnd = current + vec_ud_lr
                    if self.img_width + offset > secnd[0] > (-1) * offset and offset + self.img_height > secnd[
                        1] > (-1) * offset:
                        pairs.append((secnd, ang_lr))
                       # print("No {} Appended ({},{})at {}".format(len(pairs), i, j, secnd))


            for pair in pairs:
                #print(pair)
                self.objects.append(Object(pos=pair[0], theta=pair[1]))





        #    if theta is None:
        #        theta_0 = random.random() * np.pi * 2
        #    else:
        #        theta_0 = theta
        #    chirality = np.sign(random.random() - 0.5)#

        #    dist_v = Distance(True, 23.75)
        #    dist_h = Distance(True, 22.23)
        #    chirality = -1

        #    if chirality > 0:
        #        gv_a = np.array([dist_h * np.cos(theta_0), dist_h * np.sin(theta_0)])
        #        gv_b = np.array([-dist_v * np.sin(theta_0), dist_v * np.cos(theta_0)])
        #    else:
        #        gv_a = np.array([dist_h * np.cos(theta_0 + np.pi / 4), dist_h * np.sin(theta_0 + np.pi / 4)])
        #        gv_b = np.array([-dist_v * np.sin(theta_0 + np.pi / 4), dist_v * np.cos(theta_0 + np.pi / 4)])

        #    pairs = []

        #    ang_a = (theta_0 + bog(25)) % (2*np.pi)
        #    ang_b = (theta_0 + bog(123.1)) % (2 * np.pi)

        #    start = np.array([Distance(True, 0), Distance(True, 0)])
        #    current = np.array([0, 0])
        #    a_temp = self.img_width / gv_b[0] if gv_b[0].px != 0 else -np.infty
        #    b_temp = self.img_height / gv_b[1] if gv_b[1].px != 0 else -np.infty
        #    j_max = int(np.ceil(max(a_temp, b_temp)))
        #    c_temp = ((self.img_width + j_max * gv_b[0]) / gv_a[0]) if gv_a[0].px != 0 else j_max
        #    i_max = int(np.ceil(c_temp))
        #    for i in range(-10, max(100, i_max)):
        #        for j in range(-10, max(100, i_max)):
        #            current = start + (gv_a * i) + (gv_b * j)
        #            if self.img_width + offset > current[0] > (-1) * offset and offset + self.img_height > current[
        #                1] > (-1) * offset:
        #                pairs.append((current, ang_a if (i + j) % 2 == 0 else ang_b))

        #    for pair in pairs:
        #        self.objects.append(Object(pos=pair[0], theta=pair[1]))

        def add_ordered_NCPh5CN(theta=None):

            def add_Hexa(center, start_ang, chirality):
                d = Distance(True, 22.457) * factor

                def turnvec(len, ang):
                    return np.array([len * np.sin(ang), -len * np.cos(ang)])

                if chirality > 0:
                    phi1 = (bog(-30.663) + start_ang) % (2*np.pi)
                    phi2 = (bog(29.337) + start_ang) % (2*np.pi)
                    phi3 = (bog(89.337) + start_ang) % (2*np.pi)
                    phi4 = (bog(149.337) + start_ang) % (2*np.pi)
                    phi5 = (bog(209.337) + start_ang) % (2*np.pi)
                    phi6 = (bog(269.337) + start_ang) % (2*np.pi)
                    phis = [phi1, phi2, phi3, phi4, phi5, phi6]

                    theta1 = (bog(41.107) + start_ang) % (2*np.pi)
                    theta2 = (bog(101.107) + start_ang) % (2*np.pi)
                    theta3 = (bog(161.107) + start_ang) % (2*np.pi)
                    theta4 = (bog(221.107) + start_ang) % (2*np.pi)
                    theta5 = (bog(281.107) + start_ang) % (2*np.pi)
                    theta6 = (bog(341.107) + start_ang) % (2*np.pi)
                    thetas = [theta1, theta2, theta3, theta4, theta5, theta6]
                else:
                    phi1 = (bog(30.663) + start_ang) % (2 * np.pi)
                    phi2 = (bog(360-29.337) + start_ang) % (2 * np.pi)
                    phi3 = (bog(360-89.337) + start_ang) % (2 * np.pi)
                    phi4 = (bog(360-149.337) + start_ang) % (2 * np.pi)
                    phi5 = (bog(360-209.337) + start_ang) % (2 * np.pi)
                    phi6 = (bog(360-269.337) + start_ang) % (2 * np.pi)
                    phis = [phi1, phi2, phi3, phi4, phi5, phi6]

                    theta1 = (bog(360-41.107) + start_ang) % (2 * np.pi)
                    theta2 = (bog(360-101.107) + start_ang) % (2 * np.pi)
                    theta3 = (bog(360-161.107) + start_ang) % (2 * np.pi)
                    theta4 = (bog(360-221.107) + start_ang) % (2 * np.pi)
                    theta5 = (bog(360-281.107) + start_ang) % (2 * np.pi)
                    theta6 = (bog(360-341.107) + start_ang) % (2 * np.pi)
                    thetas = [theta1, theta2, theta3, theta4, theta5, theta6]


                pairs = []
                for i in range(6):
                    position = center + turnvec(d, phis[i])
                    if self.img_width + offset > position[0] > (-1) * offset and offset + self.img_height > position[
                        1] > (-1) * offset:
                        pairs.append((position, thetas[i]))


                for pair in pairs:
                    self.objects.append(Object(pos=pair[0], theta=pair[1]))


            if theta is None:
                theta_0 = 2 * np.pi * random.random()
            else:
                theta_0 = theta

            chirality = np.sign(random.random() - 0.5)

            #add_Hexa(np.array([self.img_width/2, self.img_height/2]), theta_0)
            #return

            gv_dist = Distance(True, 55.4938) * factor
            gv_a_w = theta_0 + bog(179.782)
            gv_b_w = theta_0 + bog(119.782)

            gv_a = np.array([gv_dist * np.sin(gv_a_w), -gv_dist * np.cos(gv_a_w)])
            gv_b = np.array([gv_dist * np.sin(gv_b_w), -gv_dist * np.cos(gv_b_w)])

            #print(gv_a, gv_b)

            offset_loc = offset + gv_dist

            start = np.array([Distance(True, 0), Distance(True, 0)])
            current = np.array([0, 0])
            a_temp = self.img_width / gv_b[0] if gv_b[0].px != 0 else -np.infty
            b_temp = self.img_height / gv_b[1] if gv_b[1].px != 0 else -np.infty
            j_max = int(np.ceil(max(a_temp, b_temp)))
            c_temp = ((self.img_width + j_max * gv_b[0]) / gv_a[0]) if gv_a[0].px != 0 else j_max
            i_max = int(np.ceil(c_temp))
            for i in range(-10, max(100, i_max)):
                for j in range(-10, max(100, i_max)):
                    current = start + (gv_a * i) + (gv_b * j) #Sketcy mit 3x Offset
                    if self.img_width + offset_loc > current[0] > (-1) * offset_loc and offset_loc + self.img_height > current[
                        1] > (-1) * offset_loc:
                        add_Hexa(current, theta_0, chirality)

        if type(Object) is not type(Particle):
            raise NotImplementedError
        elif type(Object) is not type(Molecule):
            self.addObjects(Object=Object)
        elif Object.molecule_class != "NCPhCN":
            self.addObjects()
        elif Object.molecule_ph_groups == 3:
            add_ordered_NCPh3CN(theta)
        elif Object.molecule_ph_groups == 4:
            add_ordered_NCPh4CN(theta)
        elif Object.molecule_ph_groups == 5:
            add_ordered_NCPh5CN(theta)
        else:
            raise NotImplementedError

        for part in self.objects:
            if random.random() < self.dragging_possibility:
                self.objects.remove(part)
                self.objects.append(self._add_at_pos_dragged(Molecule, pos=part.pos, theta=part.theta))

    def is_overlapping(self, part):
        for p in self.objects:
            if p.true_overlap(part):
                return self.overlapping_energy
        return 0

    def atomic_step_init(self):
        for obj in self.objects:
            obj.set_maxHeight(cfg.get_max_height() + cfg.get_atomic_step_height())
        # Create Stepborder
        point_a = [random.random() * self.img_width.px, random.random() * self.img_height.px]
        point_b = [random.random() * self.img_width.px, random.random() * self.img_height.px]

        b = (point_a[1] - (point_a[0] / point_b[0]) * point_b[1]) / (1 - point_a[0] / point_b[0])
        m = (point_a[1] - b) / point_b[1]

        # b = 200
        # m = 0.001

        f = lru_cache()(lambda x: m * x + b)

        return f, m, b

    def atomic_step(self, matrix, f, m, b):

        fpoints = []

        mt = False
        if mt:
            start = time.perf_counter()

        # Gitter mit atomic step
        def nearest_ag(gitter, pos):
            mindist = np.inf
            minat = None
            for ag in gitter:
                if np.linalg.norm(ag.pos - pos) < mindist:
                    mindist = np.linalg.norm(ag.pos - pos)
                    minat = ag

            return minat

        def in_range_of_nst(x, y, atoms, radius):
            # return False #ToDo: REm
            for atom in atoms:
                if np.linalg.norm(atom.pos - np.array([x, y])) < radius:
                    return True
            return False

        def fermi(d, mu):
            if d < self.fermi_range:
                return 1 / (np.exp(self.fermi_exp * (d - mu)) + 1)
            else:
                return 0

        def fermi2(d, mu, fex, range):
            if d < range:
                return 1 / (np.exp(fex * (d - mu)) + 1)
            else:
                return 0

        def fermi_ohne_range(d, mu, fex):
            return fermi2(d, mu, fex, np.infty)

        def dist_to_nst(x, y, atoms, radius):

            max = 0
            # return 0 #ToDo: REm
            for atom in atoms:
                if fermi(np.linalg.norm(np.array([x, y]) - atom.pos), radius) > max:
                    max = fermi(np.linalg.norm(np.array([x, y]) - atom.pos), radius)
            return max

        # def dist_to_f(x, y, f):
        #    mind = 1000
        #    for xs in range(0, int(np.ceil(self.img_width.px))):
        #        ys = f(xs)
        #        if not -50 < ys < self.img_height.px + 50:
        #            continue
        #        dist = np.sqrt(np.square(x - xs) + np.square(y - ys))
        #        if dist < mind:
        #            mind = dist#

        #            return mind

        def _calc_fpoints(f):
            lastpt = np.array([0, f(0)])
            inc = 1
            h = 0  # -50
            while h < self.img_width.px:  # +50
                nextpt = np.array([h + inc, f(h + inc)])
                if not (-50 <= h <= self.img_width.px + 50 and -50 <= f(h) <= self.img_height.px + 50):
                    h += 1
                    lastpt = np.array([h, f(h)])
                    # print("Outside: ")
                    # print("f({:.2f}) = {:.2f}".format(h, f(h)))
                    continue
                if np.linalg.norm(lastpt - nextpt) > 2:
                    inc /= 2
                    # print("BigStep: w= {:.2f}, inc={}".format(np.linalg.norm(lastpt - nextpt), inc))
                    # print("f({:.2f}) = {:.2f}".format(h, f(h)))
                    continue
                elif np.linalg.norm(lastpt - nextpt) < 0.5:
                    inc *= 1.5
                    # print("SmallStep:")
                    # print("f({:.2f}) = {:.2f}".format(h, f(h)))
                    continue
                else:
                    h += inc
                    lastpt = np.array([h, f(h)])
                    # print("Appended:")
                    # print("f({:.2f}) = {:.2f}".format(h, f(h)))
                    fpoints.append(lastpt)
                # print("----")

        def dist_to_f(x, y, f):
            if len(fpoints) == 0:
                _calc_fpoints(f)

            mind = 1000
            target = np.array([x, y])
            for lastpt in fpoints:
                if np.linalg.norm(lastpt - target) < mind:
                    mind = np.linalg.norm(lastpt - target)

            return mind

        def find_fermi_range(dh, fex):
            for zetta in range(0, 1000):
                # zetta = 1000 - d
                if dh * fermi_ohne_range(zetta, rad, fex) < 1:
                    return zetta

        show_gitter = False
        use_gitter = True
        show_f = False

        gitter = Tests_Gitterpot.create_larger_gitter()  # Ag-Atom[]
        if (show_gitter):
            matrix += 255 * Tests_Gitterpot.show_gitter(gitter)

        if mt:
            print("STEP1 (Def): {}".format(time.perf_counter() - start))
            start = time.perf_counter()

        # f_er_x_min = np.inf
        # f_er_x_max = - np.inf
        # f_er_y_min = np.inf
        # f_er_y_max = - np.inf

        if use_gitter:
            checking_pairs = []
            if abs(m) > 1:
                #  f_er_x_min = 0
                #   f_er_x_max = self.img_width.px
                for x in range(0, int(np.ceil(self.img_width.px)), 10):
                    yp = f(x)
                    #   if yp < f_er_y_min:
                    #       f_er_y_min = yp
                    #   if yp > f_er_y_max:
                    #       f_er_y_max = yp
                    if not 0 <= yp <= self.img_height.px:
                        continue
                    checking_pairs.append([x, yp])
            else:
                #  f_er_y_min = 0
                #  f_er_y_max = self.img_height.px
                for y in range(0, int(np.ceil(self.img_height.px)), 10):
                    xp = (y - b) / m

                    #     if xp < f_er_x_min:
                    #         f_er_x_min = xp
                    #     if xp > f_er_x_max:
                    #         f_er_x_max = xp

                    if not 0 <= xp <= self.img_width.px:
                        continue

                    checking_pairs.append([xp, y])

            if mt:
                print("STEP2 (Checking Pairs): {}".format(time.perf_counter() - start))
                start = time.perf_counter()

            atoms_near_step = []
            gitter = Tests_Gitterpot.create_larger_gitter()  # Ag-Atom[]
            # print("Positions: {}".format(checking_pairs))

            for pos in checking_pairs:
                nat = nearest_ag(gitter, pos)
                if nat not in atoms_near_step:
                    atoms_near_step.append(nat)

            if mt:
                print("STEP3 (Atoms Near Step): {}".format(time.perf_counter() - start))
                start = time.perf_counter()

            # print("No of Atoms near step: {} - {}".format(len(atoms_near_step), atoms_near_step))

            rad = cfg.get_nn_dist().px

            if len(self.objects) > 0:
                dh = self.objects[0].color(self.atomic_step_height)
            else:
                dh = 255 * self.atomic_step_height / self.max_height
                # dh = 255 * cfg.get_atomic_step_height() / (cfg.get_max_height() + cfg.get_atomic_step_height()))

            # für glatten Übergang
            # dh *= fermi(- self.fermi_range, rad)
            debug_mode = False
            # rem
            # atoms_near_step = []
            fex2 = 0.15 * Distance(True, 2.88).px / cfg.get_nn_dist().px

            fermi_range2 = np.log(99) / fex2 + 50
            fermi_range2 = find_fermi_range(dh, fex2)
            dist_const = rad

            # f_effect_range = f_er_x_min - fermi_range2, f_er_x_max + fermi_range2, f_er_y_min - fermi_range2, f_er_y_max + fermi_range2
            # print(f_effect_range)

            def effh():
                return dist_const / np.sin(0.5 * np.pi - np.arctan(1 / m))

            already_interpolated = False

            def interpolate_lines(givnlines):
                nonlocal already_interpolated
                if already_interpolated:
                    return

                already_interpolated = True
                newlines = []
                for i in range(len(givnlines) - 1):
                    newpta_x = givnlines[i][0][0] * 0.1 + givnlines[i][1][0] * 0.9
                    newpta_y = givnlines[i][0][1] * 0.1 + givnlines[i][1][1] * 0.9
                    newptb_x = givnlines[i + 1][0][0] * 0.9 + givnlines[i + 1][1][0] * 0.1
                    newptb_y = givnlines[i + 1][0][1] * 0.9 + givnlines[i + 1][1][1] * 0.1
                    newlines.append((np.array([newpta_x, newpta_y]), np.array([newptb_x, newptb_y])))
                for kappa in newlines:
                    givnlines.append(kappa)

            def dist_to_line(x, y, lines):

                loc_lines = lines

                interpolate = True
                if interpolate:
                    interpolate_lines(lines)

                zero_threshold = 0.005
                distances = []
                # print(lines)
                # print(loc_lines)
                for line in loc_lines:
                    dy = line[1][1] - line[0][1]
                    dx = line[1][0] - line[0][0]
                    if abs(dx) < zero_threshold:
                        if line[0][1] <= y <= line[1][1]:
                            distances.append(abs(line[0][0] - x))
                        elif y < line[0][1]:
                            distances.append(np.sqrt(np.square(line[0][1] - y) + np.square(line[0][0] - x)))
                        elif y > line[1][1]:
                            distances.append(np.sqrt(np.square(line[1][1] - y) + np.square(line[1][0] - x)))
                        else:
                            raise NotImplementedError
                        continue
                    else:
                        m = dy / dx
                    b = line[0][1] - m * line[0][0]
                    theta = np.arctan(m)
                    m2 = np.tan(theta + np.pi / 2)
                    dy2 = m2
                    dx2 = 1
                    # assert np.sign(m) != np.sign(m)
                    b2 = y - m2 * x
                    if abs(m - m2) < zero_threshold:
                        x_sp = 10000
                    else:
                        x_sp = - (b - b2) / (m - m2)
                    # print("SP inside: {:.2f} <= {:.2f} <= {:.2f}".format(line[0][0], x_sp, line[1][0]))
                    if not (line[0][0] <= x_sp <= line[1][0] or line[0][0] >= x_sp >= line[1][0]):
                        if x_sp < line[0][0] + (line[1][0] - line[0][0]) / 2:
                            x_sp = line[1][0]
                        else:
                            x_sp = line[0][0]
                    y_sp = m * x_sp + b
                    distances.append(np.sqrt(np.square(x - x_sp) + np.square(y - y_sp)))
                return min(distances)

            # print("FR: {:.3f}".format(fermi_range2))

            if mt:
                print("STEP4 (Defs): {}".format(time.perf_counter() - start))
                start = time.perf_counter()

            sign = 1

            mode = "F"  # A
            schieb = 0  # 0

            if mode == "A":
                hmax = np.shape(matrix)[0]
                for h in range(np.shape(matrix)[0]):
                    if h % 5 == 0:
                        print("Progress Atom Step: {:.2f}%".format(100 * h / hmax))

                    y_wert = f(h)
                    for r in range(np.shape(matrix)[1]):

                        d = dist_to_f(h, r, f)

                        innen = y_wert >= sign * r

                        if d - schieb > fermi_range2:
                            if innen:
                                # pass
                                matrix[h, r] += dh
                            continue

                        # in_at = in_range_of_nst(h, r, atoms_near_step, rad)
                        in_at = False
                        if innen:
                            if in_at:
                                opt1 = dh * fermi2(-d - schieb, sign * rad, fex2, fermi_range2) * 0
                                opt2 = dh * dist_to_nst(h, r, atoms_near_step, rad)
                                matrix[h, r] += max(opt1, opt2)
                            else:
                                # pass
                                matrix[h, r] += dh * fermi2(-d - schieb, rad, fex2, fermi_range2)
                        else:
                            if in_at:
                                opt1 = dh * fermi2(d + schieb, sign * rad, fex2, fermi_range2)
                                opt2 = dh * dist_to_nst(h, r, atoms_near_step, rad)
                                matrix[h, r] += max(opt1, opt2)
                            else:
                                matrix[h, r] += dh * fermi2(sign * d + schieb, sign * rad, fex2, fermi_range2)

            if mode == "B":
                hmax = np.shape(matrix)[0]
                for h in range(np.shape(matrix)[0]):
                    if h % 5 == 0:
                        print("Progress Atom Step: {:.2f}%".format(100 * h / hmax))

                    y_wert = f(h)
                    for r in range(np.shape(matrix)[1]):
                        if in_range_of_nst(h, r, atoms_near_step, rad) and y_wert < r:
                            # außen, aber in range of atom
                            matrix[h, r] += dh * dist_to_nst(h, r, atoms_near_step, rad)
                        elif in_range_of_nst(h, r, atoms_near_step, rad) and y_wert >= r:
                            # innen und im atom
                            d = dist_to_f(h, r, f)
                            opt1 = dh * fermi2(-d, rad, fex2, fermi_range2)
                            opt2 = dist_to_nst(h, r, atoms_near_step, rad)
                            matrix[h, r] += max(opt1, opt2)  #

                        elif y_wert < r:
                            # außen, mglw in fermirange
                            d = dist_to_f(h, r, f)
                            if d < fermi_range2:
                                matrix[h, r] += dh * fermi2(d, rad, fex2, fermi_range2)
                        else:
                            d = dist_to_f(h, r, f)
                            if d < fermi_range2:
                                matrix[h, r] += dh * fermi2(-d, rad, fex2, fermi_range2)
                            else:
                                matrix[h, r] += dh
                            # innen

            if mode == "C":
                for h in range(np.shape(matrix)[0]):
                    # if h % 100 == 0:
                    # print("H: {}/{}".format(h, np.shape(matrix)[0]))
                    for r in range(np.shape(matrix)[1]):
                        if in_range_of_nst(h, r, atoms_near_step, rad):
                            add = dh * dist_to_nst(h, r, atoms_near_step, rad)
                            temp = matrix[h, r]
                            matrix[h, r] += add
                            if debug_mode: matrix[h, r] = 0
                            if f(h) > r and add < dh:
                                # matrix[h, r] = temp + max(dh * fermi2(r - f(h) + effh(), effh(), fex2), add)
                                matrix[h, r] = temp + max(dh * fermi2(dist_to_f(h, r, f), effh(), fex2, fermi_range2),
                                                          add)  #

                                if debug_mode: matrix[h, r] = 75
                        else:
                            n = f(h)
                            # if n + fermi_range2 > r:
                            if n + fermi_range2 > r:
                                if abs(n - r) < effh():
                                    matrix[h, r] += dh * fermi2(dist_to_f(h, r, f), effh(), fex2, fermi_range2)
                                    # matrix[h, r] += dh * fermi2(r - n + effh(), effh(), fex2)
                                    if debug_mode: matrix[h, r] = 150
                                else:
                                    if n > r:
                                        matrix[h, r] += dh
                                        if debug_mode: matrix[h, r] = 225  #

            if mode == "D":
                hmax = np.shape(matrix)[0]
                for h in range(np.shape(matrix)[0]):
                    if h % 5 == 0:
                        print("Progress Atom Step: {:.2f}%".format(100 * h / hmax))

                    y_wert = f(h)
                    for r in range(np.shape(matrix)[1]):

                        d = dist_to_f(h, r, f)

                        innen = y_wert >= sign * r

                        if d + schieb > fermi_range2:
                            if innen:
                                # pass
                                matrix[h, r] += dh
                            continue

                        if innen:
                            matrix[h, r] += dh * fermi2(-d - schieb, rad, fex2, fermi_range2)
                        else:
                            matrix[h, r] += dh * fermi2(sign * d + schieb, sign * rad, fex2, fermi_range2)

            if mode == "E":
                x_pt = random.random() * self.img_width.px
                y_pt = random.random() * self.img_height.px
                a = 2 * (0.5 - random.random())
                b = 10 * random.random() - 5
                c = -(a * x_pt ** 2 + b * x_pt - y_pt)

                a = 0.002
                b = 50
                c = 50

                # f = lambda v : a * np.square(v) + b * v + c
                f = lambda v: a * np.square(v - b) + c

                print("F(x) = {:.2f}x^2 + {:.2f}x + {:.2f}".format(a, b, c))
                schieb = 0

                hmax = np.shape(matrix)[0]
                for h in range(np.shape(matrix)[0]):
                    if h % 5 == 0:
                        print("Progress Atom Step: {:.2f}%".format(100 * h / hmax))

                    y_wert = f(h)
                    for r in range(np.shape(matrix)[1]):

                        d = dist_to_f(h, r, f)

                        innen = y_wert >= sign * r

                        if d + schieb > fermi_range2:
                            if innen:
                                # pass
                                matrix[h, r] += dh
                            continue

                        if innen:
                            matrix[h, r] += dh * fermi2(-d - schieb, rad, fex2, fermi_range2)
                        else:
                            matrix[h, r] += dh * fermi2(sign * d + schieb, sign * rad, fex2, fermi_range2)

            if mode == "F":
                points = []
                steps = 15
                updown = random.random() < 0.5
                variance = steps
                sideA = random.random() < 0.5
                show_line = False
                tendence = True

                # Calculate Lines
                if updown:
                    oldstep = random.randint(-variance, variance)
                    oldx = random.random() * self.img_width.px
                    for i in range(0, int(self.img_height.px) + int(self.img_height.px / steps),
                                   int(self.img_height.px / steps)):
                        # points.append(np.array([i, random.random() * self.img_height.px]))
                        # points.append(np.array([i, self.img_height.px/2]))
                        if tendence:
                            newstep = random.randint(-variance, variance) + oldstep
                            new_x = newstep + oldx
                            oldstep = newstep

                        else:
                            new_x = random.randint(-variance, variance) + oldx
                        points.append(np.array([new_x, i]))
                        oldx = new_x

                    lines = []
                    for i in range(len(points) - 1):
                        lines.append((points[i], points[i + 1]))

                    border = []
                    for line in lines:
                        dy = line[1][1] - line[0][1]
                        dx = line[1][0] - line[0][0]

                        for y in range(int(line[0][1]), int(line[1][1])):
                            if 0 <= y < np.shape(matrix)[1]:
                                border.append(np.array([line[0][0] + dx * (y - line[0][1]) / dy, y]))
                    if show_line:
                        for elem in border:
                            if 0 <= int(elem[0]) < self.img_width.px and 0 <= int(elem[1]) < self.img_height.px:
                                matrix[int(elem[0]) - 10, int(elem[1])] = 255

                    if False:
                        dists = np.zeros(np.shape(matrix))
                        hges = np.shape(matrix)[0]
                        for h in range(np.shape(matrix)[0]):
                            print("{}/{}".format(h, hges))
                            for r in range(np.shape(matrix)[1]):
                                dists[h, r] = dist_to_line(h, r, lines)

                        for elem in border:
                            if 0 <= int(elem[0]) < self.img_width.px and 0 <= int(elem[1]) < self.img_height.px:
                                dists[int(elem[0]) - 0, int(elem[1])] = 30

                        xbrakes = []
                        ybrakes = []
                        for line in lines:
                            if line[0][0] not in xbrakes:
                                xbrakes.append(line[0][0])
                            if line[0][1] not in ybrakes:
                                ybrakes.append(line[0][1])

                        hges = np.shape(matrix)[0]
                        for h in range(np.shape(matrix)[0]):
                            print("2nd: {}/{}".format(h, hges))
                            for r in range(np.shape(matrix)[1]):
                                if h in xbrakes or r in ybrakes:
                                    dists[h, r] = 25

                        # for h in range(np.shape(matrix)[0]):
                        #    print("3rd: {}/{}".format(h, hges))
                        #    for r in range(np.shape(matrix)[1]):
                        #        if abs(h - (400 - r)) > 20:
                        #            dists[h, r] = 0

                        plt.imshow(dists)
                        plt.show()

                    leftborder = []
                    rightborder = []
                    for elem in border:
                        leftborder.append(np.array([elem[0] - fermi_range2, elem[1]]))
                        rightborder.append(np.array([elem[0] + fermi_range2, elem[1]]))

                    if False:
                        r = 200
                        lfr = leftborder[r][0]
                        rfr = rightborder[r][0]
                        grenz = border[r][0]
                        xs = range(0, int(self.img_width.px))
                        ys = []
                        for x in xs:
                            if sideA:
                                if x <= lfr:
                                    ys.append(dh)
                                elif lfr < x < grenz:
                                    ys.append(dh * fermi2(-dist_to_line(x, r, lines), 0, fex2, fermi_range2))
                                elif grenz <= x < rfr:
                                    ys.append(dh * fermi2(dist_to_line(x, r, lines), 0, fex2, fermi_range2))
                                elif x >= rfr:
                                    ys.append(0)
                                    pass
                                else:
                                    raise NotImplementedError

                            else:
                                if h <= lfr:
                                    pass
                                elif lfr < h < grenz:
                                    ys.append(dh * fermi2(dist_to_line(x, r, lines), 0, fex2, fermi_range2))
                                elif grenz <= h < rfr:
                                    ys.append(dh * fermi2(-dist_to_line(x, r, lines), 0, fex2, fermi_range2))
                                elif h >= rfr:
                                    ys.append(dh)
                                else:
                                    raise NotImplementedError
                        plt.plot(xs, ys)
                        plt.title("Helligkeitsprofil entlang y = {}".format(r))
                        plt.show()

                    # Display
                    for r in range(np.shape(matrix)[1]):
                        lfr = leftborder[r][0]
                        rfr = rightborder[r][0]
                        grenz = border[r][0]
                        for h in range(np.shape(matrix)[0]):
                            if sideA:
                                if h <= lfr:
                                    matrix[h, r] += dh
                                    # matrix[h, r] = 255
                                elif lfr < h < grenz:
                                    matrix[h, r] += dh * fermi2(-dist_to_line(h, r, lines), 0, fex2, fermi_range2)
                                    # matrix[h, r] = 180
                                elif grenz <= h < rfr:
                                    matrix[h, r] += dh * fermi2(dist_to_line(h, r, lines), 0, fex2, fermi_range2)
                                    # matrix[h, r] = 100
                                elif h >= rfr:
                                    # matrix[h, r] = 50
                                    pass
                                else:
                                    raise NotImplementedError
                            # SideB
                            else:
                                if h <= lfr:
                                    pass
                                    # matrix[h, r] = 50
                                elif lfr < h < grenz:
                                    matrix[h, r] += dh * fermi2(dist_to_line(h, r, lines), 0, fex2, fermi_range2)
                                    # matrix[h, r] = 100
                                elif grenz <= h < rfr:
                                    matrix[h, r] += dh * fermi2((-1) * dist_to_line(h, r, lines), 0, fex2, fermi_range2)
                                    # matrix[h, r] = 150
                                elif h >= rfr:
                                    matrix[h, r] += dh
                                    # matrix[h, r] = 200
                                else:
                                    raise NotImplementedError

                # Not Updown
                else:
                    oldy = random.random() * self.img_height.px
                    oldstep = random.randint(-variance, variance)
                    for i in range(0, int(self.img_width.px) + int(self.img_width.px / steps),
                                   int(self.img_width.px / steps)):
                        # points.append(np.array([i, random.random() * self.img_height.px]))
                        # points.append(np.array([i, self.img_height.px/2]))

                        if tendence:
                            newstep = random.randint(-variance, variance) + oldstep
                            new_y = newstep + oldy
                            oldstep = newstep

                        else:
                            new_y = random.randint(-variance, variance) + oldy
                        points.append(np.array([i, new_y]))
                        oldy = new_y

                    lines = []
                    for i in range(len(points) - 1):
                        lines.append((points[i], points[i + 1]))

                    border = []
                    for line in lines:
                        dy = line[1][1] - line[0][1]
                        dx = line[1][0] - line[0][0]

                        for x in range(int(line[0][0]), int(line[1][0])):
                            if 0 <= x < np.shape(matrix)[0]:
                                border.append(np.array([x, line[0][1] + ((x - line[0][0]) / dx) * dy]))
                    if show_line:
                        for elem in border:
                            if 0 <= int(elem[0]) < self.img_width.px and 0 <= int(elem[1]) < self.img_height.px:
                                matrix[int(elem[0]), int(elem[1])] = 255

                    upperborder = []
                    lowerborder = []
                    for elem in border:
                        upperborder.append(np.array([elem[0], elem[1] - fermi_range2]))
                        lowerborder.append(np.array([elem[0], elem[1] + fermi_range2]))

                    for h in range(np.shape(matrix)[0]):
                        grenz = border[h][1]
                        ufr = upperborder[h][1]
                        lfr = lowerborder[h][1]
                        for r in range(np.shape(matrix)[1]):
                            if sideA:
                                if r <= ufr:
                                    matrix[h, r] += dh
                                    # matrix[h, r] = 255
                                elif ufr < r < grenz:
                                    matrix[h, r] += dh * fermi2(-dist_to_line(h, r, lines), 0, fex2, fermi_range2)
                                    # matrix[h, r] = 180
                                elif grenz <= r < lfr:
                                    matrix[h, r] += dh * fermi2(dist_to_line(h, r, lines), 0, fex2, fermi_range2)
                                    # matrix[h, r] = 100
                                elif r >= lfr:
                                    # matrix[h, r] = 50
                                    pass
                                else:
                                    raise NotImplementedError
                            # SideB
                            else:
                                if r <= ufr:
                                    pass
                                    # matrix[h, r] = 50
                                elif ufr < r < grenz:
                                    matrix[h, r] += dh * fermi2(dist_to_line(h, r, lines), 0, fex2, fermi_range2)
                                    # matrix[h, r] = 100
                                elif grenz <= r < lfr:
                                    matrix[h, r] += dh * fermi2((-1) * dist_to_line(h, r, lines), 0, fex2, fermi_range2)
                                    # matrix[h, r] = 150
                                elif r >= lfr:
                                    matrix[h, r] += dh
                                    # matrix[h, r] = 200
                                else:
                                    raise NotImplementedError

            if mt:
                print("STEP5 (Matrix): {}".format(time.perf_counter() - start))
                start = time.perf_counter()

            # for h in range(np.shape(matrix)[0]):
            #   for r in range(np.shape(matrix)[1]):
            #       for p in fpoints:
            #          if np.linalg.norm(np.array([h, r]) - p) < 2:
            #             matrix[h, r] = 300

            if show_f:
                delta = 1
                for h in range(np.shape(matrix)[0]):
                    for r in range(np.shape(matrix)[1]):
                        if dist_to_f(h, r, f) < delta:
                            matrix[h, r] = 200

                zrs = np.zeros(np.shape(matrix))
                for d in range(int(self.img_width.px)):
                    for e in range(int(self.img_height.px)):
                        zrs[d, e] = dist_to_f(d, e, f)
                plt.imshow(zrs)
                plt.show()

            if False:
                mid = int(self.img_height.px / 2)
                xs = []
                ys = []
                for h in range(np.shape(matrix)[0]):
                    xs.append(h)
                    ys.append(matrix[h, mid])

                plt.plot(xs, ys)
                plt.title("Helligkeitsprofil entlang y = {}".format(mid))
                plt.show()

            # plt.imshow(matrix.transpose())
            # plt.show()

            # for i in range(0, 400, 40):
            #    print("Dist to f: {}-200 : {:.3f}".format(i, dist_to_f(i, 200, f)))#

            # print("Fermi(0, rad) = {}".format(fermi(0, rad)))

    def add_Dust_Part(self, part=None):
        if part is None:
            self.dust_particles.append(DustParticle(size=random.random() * 40))
        else:
            self.dust_particles.append(part)

    def add_Dust(self):

        amnt = int(np.round(np.random.normal(self.dust_amount)))
        for i in range(amnt):
            self.add_Dust_Part()

    def create_Image_Visualization(self):
        self.img = MyImage()
        width = self.img_width
        height = self.img_height
        matrix = np.zeros((int(np.ceil(width.px)), int(np.ceil(height.px))))

        use_atomstep = random.random() < cfg.get_atomic_step_poss()

        # Set Max Height for parts
        if use_atomstep:
            fargs = self.atomic_step_init()
        else:
            fargs = lambda c: 0, 0, 0

        max = len(self.objects)
        ct = 0
        for part in self.objects:
            print("Visu_progress: {:.1f}%".format(100 * ct/max))
            ct += 1
            for tuple in part.get_visualization():
                # print("Tupel:{}".format(tuple))
                eff_mat, x, y = tuple
                mat_w = eff_mat.shape[0]

                # ToDo: possible failure
                x = int(np.round(x.px))
                y = int(np.round(y.px))
                # plt.imshow(eff_mat)
                # plt.show()
                # print(np.max(eff_mat))
                mat_h = eff_mat.shape[1]
                for i in range(mat_w):
                    for j in range(mat_h):
                        new_x = x - math.floor((mat_w / 2)) + i
                        new_y = y - math.floor(mat_h / 2) + j
                        if not (0 <= new_x < width.px and 0 <= new_y < height.px):
                            continue
                        matrix[new_x, new_y] += eff_mat[i, j]
        if self.usedust:
            self.add_Dust()

        if use_atomstep:
            self.atomic_step(matrix, *fargs)

        self.img.addMatrix(matrix)

    def get_Image_Dust(self):
        width = self.img_width
        height = self.img_height
        matrix = np.zeros((int(np.ceil(width.px)), int(np.ceil(height.px))))

        for d in self.dust_particles:
            tuple = d.efficient_Matrix()
            eff_mat, x, y = tuple
            mat_w = eff_mat.shape[0]

            # ToDo: possible failure
            x = int(np.round(x.px))  # no px cause dust
            y = int(np.round(y.px))

            mat_h = eff_mat.shape[1]
            for i in range(mat_w):
                for j in range(mat_h):
                    new_x = x - math.floor((mat_w / 2)) + i
                    new_y = y - math.floor(mat_h / 2) + j
                    if not (0 <= new_x < width.px and 0 <= new_y < height.px):
                        continue
                    matrix[new_x, new_y] += eff_mat[i, j]
        self.img.addMatrix(matrix)

    def get_Image(self):
        if random.random() < self.double_tip_poss:
            print("Double Tipping")
            # ToDo: Step
            strength = 0.3 + 0.5 * random.random()
            rel_dist = 0.1 * random.random()  # ToDO: Let loose
            angle = 2 * np.pi * random.random()
            doubled_frame = Double_Frame(self.fn_gen, strength, rel_dist, angle)
            if self.passed_args_particles is not None:
                doubled_frame.addParticles(*self.passed_args_particles)
            elif self.passed_args_Obj is not None:
                doubled_frame.addObjects(*self.passed_args_Obj)
            elif self.passed_args_Ordered is not None:
                doubled_frame.add_Ordered(*self.passed_args_Ordered)
            else:
                print("Default")
                doubled_frame.addObjects()

            #if self.usedust:
            #    print("Dusting")
            #    doubled_frame.get_Image_Dust()

            self.img = doubled_frame.extract_Smaller()
            self.objects = doubled_frame.get_objects()

            if self.use_noise:
                self.img.noise(self.image_noise_mu, self.image_noise_sigma)

            self.img.updateImage()
            return

        self.create_Image_Visualization()

        if self.usedust:
            self.get_Image_Dust()

        if self.use_noise:
            self.img.noise(self.image_noise_mu, self.image_noise_sigma)

        if self.use_img_shift:
            self.img.shift_image()

        self.img.updateImage()



    def createText(self):
        strings = [Particle.str_Header()]
        for part in self.objects:
            strings.append(str(part))
        self.text = "\n".join(strings)

    def save(self, data=True, image=True, sxm=True):
        if self.img is None:
            # self.createImage_efficient()
            self.get_Image()
        if len(self.text) == 0:
            self.createText()
        img_path, dat_path, sxm_path, index = self.fn_gen.generate_Tuple()
        # print("Saving: {}".format(index))
        print("Saving No {} -> {}".format(index, img_path))
        try:
            with open(dat_path, "w") as dat_file:
                dat_file.write(self.text)
        except FileNotFoundError:
            os.mkdir(cfg.get_data_folder())
            with open(dat_path, "w") as dat_file:
                dat_file.write(self.text)
        self.img.saveImage(img_path)
        My_SXM.write_sxm(sxm_path, self.img.get_matrix())
        # if self.has_overlaps():
        # print("Overlaps detected @ {}".format(index))

    def hasPoints(self):
        # return not self.points.empty()
        return len(self.objects) > 0

    def coverage(self):
        area = self.area
        covered = 0
        for part in self.objects:
            covered += np.pi * np.square(part.get_dimension().px)
        return covered / area.px

    def has_overlaps(self):
        for i in range(len(self.objects)):
            for j in range(i):
                # print("Testing overlap {} - {}".format(i, j))
                if self.objects[i].true_overlap(self.objects[j]):
                    print("Overlap between Part {} @ ({},{}) and Part {} @ ({},{})".format(i, self.objects[i].pos[0].px,
                                                                                           self.objects[i].pos[1].px, j,
                                                                                           self.objects[j].pos[0].px,
                                                                                           self.objects[j].pos[1].px))
                    print("Reverse: {}".format(self.objects[j].true_overlap(self.objects[i])))
                    # print("i: x={}, y={}, dg={}; j: x={}, y={}, dg={}".format(self.objects[i].get_x(), self.objects[i].get_y(), self.objects[i].dragged, self.objects[j].get_x(), self.objects[j].get_y(), self.objects[j].dragged))
                    return True
        return False

    def __str__(self):
        return str(self.objects)

    # Deprecated Stuff....
    # calculates particles weight for importance in surrounding particles
    @DeprecationWarning
    def _calc_angle_weight(self, part1, part2):
        print("Deprecated 5183")
        drel = part1.get_distance_to(part2) / self.max_dist
        # return np.exp(-self.angle_char_len/drel)
        return self.angle_char_len / drel

    @DeprecationWarning
    def _orients_along_crystal(self, particle):
        print("Deprecated 1351361")
        return self._bonding_strength_crystal(particle) < 0.5

    # High if bonds tu particles, low if bonds to crystal
    @DeprecationWarning
    def _bonding_strength_crystal(self, particle):
        print("Deprecated 46463516")
        if len(self.objects) == 0:
            return 0
        for part in self.objects:
            dis = part.get_distance_to(particle)
            if dis < self.max_dist / np.sqrt(2) * 0.2:
                return 1
        return 0

    @DeprecationWarning
    def _drag_particles(self):
        print("Deprecated 6546335416")
        for part in self.objects:
            if random.random() < self.dragging_possibility:
                part.drag(self.dragging_speed, self.raster_angle)

    @DeprecationWarning
    def add_at_optimum_energy_new_theta(self, x_start, y_start, theta_start):
        print("Deprecated 41535154")

        # print(x_start, y_start, theta_start)

        def overlapping_amount(part):
            if len(self.objects) == 0:
                # print("Overlaps any took {}".format(time.perf_counter() - start))
                return 0
            amnt = 0
            for p in self.objects:
                if not part.dragged and not p.dragged:
                    if math.dist([p.x, p.y], [part.x, part.y]) > max(part.effect_range, p.effect_range):
                        continue
                amnt += part.overlap_amnt(p)
            # print("Overlaps any took {}".format(time.perf_counter() - start))
            return amnt

        x_loc = x_start
        y_loc = y_start
        theta_loc = theta_start

        def energyt(theta):
            p = Particle(x_loc, y_loc, theta[0])
            charges = p.get_charges()
            e = 0
            e += overlapping_amount(p)
            # print("Overlapping_amount: {}".format(overlapping_amount(p)))
            for c in charges:
                if c.has_negative_index:
                    e += 1000
                try:
                    e += c.q * self.potential_map[int(c.x), int(c.y)]
                except IndexError:
                    e += self.overlapping_energy
                    continue
            return e

        # for theta in range(360):
        #    print("Theta: {} - E: {}".format(theta, energyt([3.14159 * theta/180])))

        self.potential_map = self.calc_potential_map()
        # plt.imshow(self.potential_map)
        # plt.show()
        vals = opt.fmin(energyt, [theta_loc])
        # print(x_loc, y_loc, vals)
        p = Particle(x_loc, y_loc, vals)

        for i in range(10):
            if self._overlaps_any(p):
                vals = opt.fmin(energyt, [theta_loc])
                p = Particle(x_loc, y_loc, vals[0])
            else:
                break
        self.objects.append(p)
        for c in p.get_charges():
            self.add_To_Potential.append(c)
        self.potential_map = self.calc_potential_map()
        # plt.imshow(self.potential_map)
        # plt.show()

    @DeprecationWarning
    def add_ALL_at_optimum_energy_new(self, n):
        print("Deprecated 1351313")
        # plt.imshow(self.potential_map)
        # plt.show()

        loc_pot_map = np.zeros((self.img_width, self.img_height))
        loc_objs = []
        use_old = False
        to_add = []

        def calc_loc_pot_map():
            if use_old:
                charges = []
                for part in to_add:
                    for c in part.get_charges():
                        charges.append(c)
                for charge in charges:
                    for i in range(self.img_width):
                        for j in range(self.img_height):
                            loc_pot_map[i, j] += charge.calc_Potential(i, j)

            charges = []
            for i in loc_objs:
                for c in i.get_charges():
                    charges.append(c)

            for charge in charges:
                for i in range(self.img_width):
                    for j in range(self.img_height):
                        loc_pot_map[i, j] += charge.calc_Potential(i, j)

        def overlapping_amount(part):
            if len(self.objects) == 0:
                # print("Overlaps any took {}".format(time.perf_counter() - start))
                return 0
            amnt = 0
            for p in loc_objs:
                if not part.dragged and not p.dragged:
                    if math.dist([p.x, p.y], [part.x, part.y]) > max(part.effect_range, p.effect_range):
                        continue
                amnt += part.overlap_amnt(p)
            # print("Overlaps any took {}".format(time.perf_counter() - start))
            return amnt

        initvals = []

        for i in range(n):
            initvals.append(100 + (self.img_width - 200) * random.random())
            initvals.append(100 + (self.img_height - 200) * random.random())
            initvals.append(6.2431 * random.random())

        def energy(args):
            e = 0
            for i in range(0, 3 * n, 3):
                p = Particle(args[i], args[i + 1], args[i + 2])
                e += energyxyt(Particle(args[i], args[i + 1], args[i + 2]))
                loc_objs.append(p)
            # calc_loc_pot_map()
            # for part in loc_objs:
            #    e += energyxyt(part)

            return e

        def energyxyt(part):
            print(part)
            # print("Part: {}".format(part))
            # if part is Particle:
            #    p = part
            # else:
            #    p = part[0]
            es = []
            try:
                for p in part:
                    charges = p.get_charges()
                    e = 0
                    e += overlapping_amount(p)
                    for c in charges:
                        if c.has_negative_index():
                            e += 1000
                        try:
                            # print("Charge {} at ({},{}) adds Energy {}".format(c.q, c.x, c.y, c.q * self.potential_map[int(c.x), int(c.y)]))
                            e += c.q * loc_pot_map[int(c.x), int(c.y)]
                        except IndexError:
                            # print("Index Error")
                            e += self.overlapping_energy
                            continue
                    # print("Energy: {}".format(e))
                    es.append(e)
                return es
            except TypeError:
                p = part
                charges = p.get_charges()
                e = 0
                e += overlapping_amount(p)
                for c in charges:
                    if c.has_negative_index():
                        e += 1000
                    try:
                        # print("Charge {} at ({},{}) adds Energy {}".format(c.q, c.x, c.y, c.q * self.potential_map[int(c.x), int(c.y)]))
                        e += c.q * loc_pot_map[int(c.x), int(c.y)]
                    except IndexError:
                        # print("Index Error")
                        e += self.overlapping_energy
                        continue
                # print("Energy: {}".format(e))
                return e
            except AttributeError:
                print("AE")
                return 1000

        # print(energy(initvals))

        calc_loc_pot_map()
        # plt.imshow(self.potential_map)
        # plt.show()
        vals = opt.fmin(energy, initvals)

        for i in range(0, 3 * n, 3):
            p = Particle(vals[i], vals[i + 1], vals[i + 2])
            for i in range(10):
                if self._overlaps_any(p):
                    print("Overlapped")
                    vals = opt.fmin(energyxyt, [self.img_width * random.random(), self.img_height * random.random(),
                                                2 * np.pi * random.random()])
                    p = Particle(vals[0], vals[1], vals[2])
                else:
                    break
            self.objects.append(p)
            for c in p.get_charges():
                self.add_To_Potential.append(c)
        self.potential_map = self.calc_potential_map()
        plt.imshow(self.potential_map)
        plt.show()

    @DeprecationWarning
    def add_ALL_at_optimum_energy_new(self, n):
        print("Deprecated 1351313")
        # plt.imshow(self.potential_map)
        # plt.show()

        loc_pot_map = np.zeros((self.img_width, self.img_height))
        loc_objs = []
        use_old = False
        to_add = []

        def calc_loc_pot_map():
            if use_old:
                charges = []
                for part in to_add:
                    for c in part.get_charges():
                        charges.append(c)
                for charge in charges:
                    for i in range(self.img_width):
                        for j in range(self.img_height):
                            loc_pot_map[i, j] += charge.calc_Potential(i, j)

            charges = []
            for i in loc_objs:
                for c in i.get_charges():
                    charges.append(c)

            for charge in charges:
                for i in range(self.img_width):
                    for j in range(self.img_height):
                        loc_pot_map[i, j] += charge.calc_Potential(i, j)

        def overlapping_amount(part):
            if len(self.objects) == 0:
                # print("Overlaps any took {}".format(time.perf_counter() - start))
                return 0
            amnt = 0
            for p in loc_objs:
                if not part.dragged and not p.dragged:
                    if math.dist([p.x, p.y], [part.x, part.y]) > max(part.effect_range, p.effect_range):
                        continue
                amnt += part.overlap_amnt(p)
            # print("Overlaps any took {}".format(time.perf_counter() - start))
            return amnt

        initvals = []

        for i in range(n):
            initvals.append(100 + (self.img_width - 200) * random.random())
            initvals.append(100 + (self.img_height - 200) * random.random())
            initvals.append(6.2431 * random.random())

        def energy(args):
            e = 0
            for i in range(0, 3 * n, 3):
                p = Particle(args[i], args[i + 1], args[i + 2])
                e += energyxyt(Particle(args[i], args[i + 1], args[i + 2]))
                loc_objs.append(p)
            # calc_loc_pot_map()
            # for part in loc_objs:
            #    e += energyxyt(part)

            return e

        def energyxyt(part):
            print(part)
            # print("Part: {}".format(part))
            # if part is Particle:
            #    p = part
            # else:
            #    p = part[0]
            es = []
            try:
                for p in part:
                    charges = p.get_charges()
                    e = 0
                    e += overlapping_amount(p)
                    for c in charges:
                        if c.has_negative_index():
                            e += 1000
                        try:
                            # print("Charge {} at ({},{}) adds Energy {}".format(c.q, c.x, c.y, c.q * self.potential_map[int(c.x), int(c.y)]))
                            e += c.q * loc_pot_map[int(c.x), int(c.y)]
                        except IndexError:
                            # print("Index Error")
                            e += self.overlapping_energy
                            continue
                    # print("Energy: {}".format(e))
                    es.append(e)
                return es
            except TypeError:
                p = part
                charges = p.get_charges()
                e = 0
                e += overlapping_amount(p)
                for c in charges:
                    if c.has_negative_index():
                        e += 1000
                    try:
                        # print("Charge {} at ({},{}) adds Energy {}".format(c.q, c.x, c.y, c.q * self.potential_map[int(c.x), int(c.y)]))
                        e += c.q * loc_pot_map[int(c.x), int(c.y)]
                    except IndexError:
                        # print("Index Error")
                        e += self.overlapping_energy
                        continue
                # print("Energy: {}".format(e))
                return e
            except AttributeError:
                print("AE")
                return 1000

        # print(energy(initvals))

        calc_loc_pot_map()
        # plt.imshow(self.potential_map)
        # plt.show()
        vals = opt.fmin(energy, initvals)

        for i in range(0, 3 * n, 3):
            p = Particle(vals[i], vals[i + 1], vals[i + 2])
            for i in range(10):
                if self._overlaps_any(p):
                    print("Overlapped")
                    vals = opt.fmin(energyxyt, [self.img_width * random.random(), self.img_height * random.random(),
                                                2 * np.pi * random.random()])
                    p = Particle(vals[0], vals[1], vals[2])
                else:
                    break
            self.objects.append(p)
            for c in p.get_charges():
                self.add_To_Potential.append(c)
        self.potential_map = self.calc_potential_map()
        plt.imshow(self.potential_map)
        plt.show()

    @DeprecationWarning
    def energy_function(self, x, y, theta):
        # Only for diploe
        print("Deprecated 3451631")
        lenge = self.part_laenge
        x_plus = int(x + 0.5 * lenge * np.sin(theta))
        y_plus = int(y + 0.5 * lenge * np.cos(theta))
        x_minus = int(x - 0.5 * lenge * np.sin(theta))
        y_minus = int(y - 0.5 * lenge * np.cos(theta))  # ToDo: nicht int sonders berechnen
        # if len(x) > 1:
        #    e = []
        #    for i in range(len(x)):
        #        e.append(self.potential_map[x_plus[i], y_plus[i]] - self.potential_map[x_minus[i], y_minus[i]])
        if self._overlaps_any(Particle(x, y, theta)):  # ToDo Readd
            return self.overlapping_energy
        if (x_plus < 0 or x_minus < 0 or y_minus < 0 or y_plus < 0):
            return self.overlapping_energy
        try:
            a1 = self.potential_map[x_plus, y_plus]
            a2 = - self.potential_map[x_minus, y_minus]
            # if np.abs(a1) > self.overlapping_energy / 2 or np.abs(a2) > self.overlapping_energy / 2:
            #    return self.overlapping_energy
            # else:
            # print(a1 - a2)
            # print(a1)
            # print(a2)
            return a1 + a2
        except IndexError:
            return self.overlapping_energy

    @DeprecationWarning
    def opimizable_energy_function(self, x):
        print("Deprecated 165843")
        return self.energy_function(x[0], x[1], x[2])

    @DeprecationWarning
    def add_at_optimum_energy(self, initvals=None):  # ToDo: Include crystal structure
        print("Deprecated 13511453415")
        # SCIPY
        self.potential_map = self.calc_potential_map()
        # initvasl = [self.img_width * random.random(), self.img_height * random.random(), 2 * np.pi * random.random()]
        if initvals is None:
            initvasl = [200, 200, np.pi]
        else:
            initvasl = initvals
        # print(fmin(self.opimizable_energy_function, np.array([200,100,0])))
        vals = opt.fmin(self.opimizable_energy_function, initvasl)
        print(vals)
        p = Particle(vals[0], vals[1], vals[2])
        self.objects.append(p)
        for c in p.get_charges():
            self.add_To_Potential.append(c)
        self.potential_map = self.calc_potential_map()

    @DeprecationWarning
    def add_at_optimum_energy_new(self, x_start, y_start, theta_start):
        print("Deprecatd 456135")

        # plt.imshow(self.potential_map)
        # plt.show()
        def overlapping_amount(part):
            if len(self.objects) == 0:
                # print("Overlaps any took {}".format(time.perf_counter() - start))
                return 0
            amnt = 0
            for p in self.objects:
                if not part.dragged and not p.dragged:
                    if math.dist([p.x, p.y], [part.x, part.y]) > max(part.effect_range, p.effect_range):
                        continue
                amnt += part.overlap_amnt(p)
            # print("Overlaps any took {}".format(time.perf_counter() - start))
            return amnt

        def energyxyt(args3):
            x = args3[0]
            y = args3[1]
            theta = args3[2]
            e = 0
            p = Particle(x, y, theta)
            charges = p.get_charges()

            e += overlapping_amount(p)
            for c in charges:
                if c.has_negative_index():
                    e += 1000
                try:
                    # print("Charge {} at ({},{}) adds Energy {}".format(c.q, c.x, c.y, c.q * self.potential_map[int(c.x), int(c.y)]))
                    e += c.q * self.potential_map[int(c.x), int(c.y)]
                except IndexError:
                    # print("Index Error")
                    e += self.overlapping_energy
                    continue
            # print("Energy: {}".format(e))
            return e

        x_loc = x_start
        y_loc = y_start
        theta_loc = theta_start

        def energyt(theta):
            p = Particle(x_loc, y_loc, theta[0])
            charges = p.get_charges()
            e = 0
            e += overlapping_amount(p)
            for c in charges:
                if c.has_negative_index:
                    e += 1000
                try:
                    e += c.q * self.potential_map[int(c.x), int(c.y)]
                except IndexError:
                    continue
            return e

        def energyxy(xs):
            p = Particle(xs[0], xs[0], theta_loc)
            charges = p.get_charges()
            e = 0
            e += overlapping_amount(p)
            for c in charges:
                if c.has_negative_index:
                    continue
                try:
                    e += c.q * self.potential_map[int(c.x), int(c.y)]
                except IndexError:
                    # print("Index Error")
                    continue

            return e

        self.potential_map = self.calc_potential_map()
        # plt.imshow(self.potential_map)
        # plt.show()
        vals = opt.fmin(energyxyt, [x_loc, y_loc, theta_loc])
        p = Particle(vals[0], vals[1], vals[2])

        for i in range(10):

            if self._overlaps_any(p):
                print("Overlapped")
                vals = opt.fmin(energyxyt, [self.img_width * random.random(), self.img_height * random.random(),
                                            2 * np.pi * random.random()])
                p = Particle(vals[0], vals[1], vals[2])
            else:
                break
        self.objects.append(p)
        for c in p.get_charges():
            self.add_To_Potential.append(c)
        self.potential_map = self.calc_potential_map()
        # plt.imshow(self.potential_map)
        plt.imshow(self.potential_map.transpose())
        f = self.fn_gen.generate_Tuple()[0]
        plt.savefig(f)
        # plt.show()

    @DeprecationWarning
    def calc_pot_Energy_for_particle(self, part, mapchange=False):
        print("Deprecated 4641635")
        if mapchange or self.potential_map is None:
            self.potential_map = self.calc_potential_map()
        charges = part.get_charges()
        # pot_map = self.calc_potential_map()
        e_pot = 0
        for charge in charges:
            try:
                e_pot += charge.q * self.potential_map[
                    int(charge.x), int(charge.y)]  # ToDo: Not round but scale by surrrounding
            except IndexError:
                print("calc_pot_Energy_for_particle for x={}, y={}".format(int(charge.x), int(charge.y)))

        e_pot += self.is_overlapping(part)

        return e_pot

    @DeprecationWarning
    def createImage(self):
        print("Deprecated 3154251534")
        self.img = MyImage()
        for part in self.objects:
            self.img.addParticle(part)

        self.img.updateImage()  # Always call last
        # img.noise....etc

    @DeprecationWarning
    def createImage_efficient(self):
        print("Deprecated 46541741")
        self.img = MyImage()
        width = self.img_width
        height = self.img_height
        matrix = np.zeros((width, height))

        for part in self.objects:
            eff_mat, x, y = part.efficient_Matrix()
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

    @DeprecationWarning
    def createImage_efficient_with_new_Turn(self):
        print("Deprecated 453643")
        self.img = MyImage()
        width = self.img_width
        height = self.img_width
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
        # self.img.updateImage()

    @DeprecationWarning
    def calc_potential_map(self):  # ToDo: Nicht Multithreading safe mit pickle dump
        print("Deprecated 876451646")
        start = time.perf_counter()
        if self.oldPotential is not None:
            pot = self.oldPotential
            for charge in self.add_To_Potential:
                for i in range(self.img_width):
                    for j in range(self.img_height):
                        pot[i, j] += charge.calc_Potential(i, j)

            self.add_To_Potential = []
            self.oldPotential = pot
            print("Short End at {}".format(time.perf_counter() - start))
            # plt.imshow(pot)
            # plt.show()
            return pot

        if os.path.isfile("pot.data"):
            pot = pickle.load(open("pot.data", "rb"))
            self.oldPotential = pot

            # plt.imshow(pot)
            # plt.show()
            if np.shape(pot) == (self.img_width, self.img_height):
                print("Long End at {}".format(time.perf_counter() - start))
                return pot

        def gitter():
            charges = []
            kappa = 1
            for i in range(0, self.img_width, 40):
                # kappa += 1
                for j in range(0, self.img_height, 40):
                    kappa += 1
                    charges.append(Charge(i, j, 0.7))
            pot = np.zeros((self.img_width, self.img_height))

            for part in self.objects:
                for q in part.get_charges():
                    charges.append(q)

            sdf = 0
            for q in charges:
                print(sdf)
                sdf += 1
                for i in range(self.img_width):
                    for j in range(self.img_height):
                        # print(i, j)
                        pot[i, j] += q.calc_Potential(i, j)
            return pot

        def alternating():
            charges = []
            kappa = 1
            for i in range(0, self.img_width, 40):
                kappa += 1
                for j in range(0, self.img_height, 40):
                    kappa += 1
                    charges.append(Charge(i, j, 0.7 * (-1) ** kappa))
            pot = np.zeros((self.img_width, self.img_height))

            for part in self.objects:
                for q in part.get_charges():
                    charges.append(q)

            sdf = 0
            for q in charges:
                print(sdf)
                sdf += 1
                for i in range(self.img_width):
                    for j in range(self.img_height):
                        # print(i, j)
                        pot[i, j] += q.calc_Potential(i, j)
            return pot

        def slope():
            pot = np.zeros((self.img_width, self.img_height))
            for i in range(self.img_width):
                for j in range(self.img_width):
                    pot[i, j] = 100 * i / self.img_width

            return pot

        def ohne():
            return np.zeros((self.img_width, self.img_height))

        pot = ohne()

        print("Long End at {}".format(time.perf_counter() - start))
        with open("pot.data", "wb") as p:
            pickle.dump(pot, p)
        self.oldPotential = pot
        # plt.imshow(pot)
        # plt.show()
        return pot


class Double_Frame(DataFrame):
    def __init__(self, fn_gen, strength, rel_dist, angle):
        super().__init__(fn_gen)
        self.area *= 4
        self.max_dist *= np.sqrt(2)
        self.double_tip_poss = 0
        self.img_width *= 2
        self.img_height *= 2
        self.strength = strength
        self.rel_dist = rel_dist
        self.dt_angle = angle
        self.range = None
        self.shift_x = rel_dist * np.sin(angle)
        self.shift_y = rel_dist * np.cos(angle)
        self.overlap = cfg.get_px_overlap()
        self.dust_particles *=4

        if self.shift_x > 0:
            if self.shift_y > 0:
                self.range = int(int(np.ceil(self.img_width.px)) / 2), int(np.ceil(self.img_width.px)), int(
                    np.ceil((self.img_height.px / 2))), int(np.ceil(self.img_height.px))
            else:
                self.range = int(int(np.ceil(self.img_width.px)) / 2), self.img_width.px, 0, int(
                    np.ceil((self.img_height.px / 2)))
        else:
            if self.shift_y > 0:
                self.range = 0, int(int(np.ceil(self.img_width.px)) / 2), 0, int(
                    np.ceil((self.img_height.px / 2))), int(np.ceil(self.img_height.px))
            else:
                self.range = 0, int(int(np.ceil(self.img_width.px)) / 2), 0, int(np.ceil((self.img_height.px / 2)))

    def addParticle(self, part=None):
        if self.passed_args_particles is None:
            self.passed_args_particles = (1, None, True, 1000)
        else:
            if self.passed_args_particles[0] is None:
                self.passed_args_particles = (len(self.objects), self.passed_args_particles[1], self.passed_args_particles[2], self.passed_args_particles[3])
            self.passed_args_particles[0] += 1
        if part is None:
            self.objects.append(Double_Particle())
        else:
            self.objects.append(part)

    def get_dragged_that_mot_overlaps(self, maximumtries, angle=None, setangle=False):
        # print("DTNO @len {}".format(len(self.objects)))
        def _set_p():
            p = Double_Particle()
            if angle is not None:
                p.set_theta(angle)
            if setangle:
                p.set_theta(self._calc_angle_for_particle(p))
            p.drag(self.dragging_speed, self.raster_angle)
            return p

        if len(self.objects) == 0:
            return _set_p()
        p = _set_p()
        for i in range(maximumtries):
            if self._overlaps_any(p):
                p = _set_p()
            else:
                return p
        return p

    def _get_thatnot_overlaps(self, maximumtries=1000, calcangle=False):
        # print("TNO @len {}".format(len(self.objects)))
        if len(self.objects) == 0:
            return Double_Particle()
        p = Double_Particle()
        for i in range(maximumtries):
            if self._overlaps_any(p):
                p = Double_Particle()
            else:
                return p
        return p

    def addParticles(self,  optimumEnergy=False, amount=None, coverage=None, overlapping=False, maximum_tries=1000):

        self.passed_args_particles = (amount, coverage, overlapping, maximum_tries)

        #print("DF aP Got Args: {}".format(self.passed_args_particles))
        if not self.use_range:
            if self.angle_char_len == 0:
                if not overlapping:
                    if amount is not None:
                        for i in range(4 * amount):
                            if random.random() < self.dragging_possibility:
                                p = self.get_dragged_that_mot_overlaps(maximum_tries)
                                self.objects.append(p)
                            else:
                                p = self._get_thatnot_overlaps(maximum_tries)
                                self.objects.append(p)
                    elif coverage is not None:
                        while self.coverage() < coverage:
                            if random.random() < self.dragging_possibility:
                                p = self.get_dragged_that_mot_overlaps(maximum_tries)
                                self.objects.append(p)
                            else:
                                p = self._get_thatnot_overlaps(maximum_tries)
                                self.objects.append(p)
                    else:
                        for i in range(4 * cfg.get_particles_per_image()):
                            if random.random() < self.dragging_possibility:
                                p = self.get_dragged_that_mot_overlaps(maximum_tries)
                                self.objects.append(p)
                            else:
                                p = self._get_thatnot_overlaps(maximum_tries)
                                self.objects.append(p)
                # w/ angle, w overlapping
                else:
                    if amount is not None:
                        for i in range(4 * amount):
                            p = Double_Particle()
                            if random.random() < self.dragging_possibility:
                                p.drag(self.dragging_speed, self.raster_angle)
                            self.objects.append(p)
                    elif coverage is not None:
                        while self.coverage() < coverage:
                            p = Double_Particle()
                            if random.random() < self.dragging_possibility:
                                p.drag(self.dragging_speed, self.raster_angle)
                            self.objects.append(p)
                    else:
                        for i in range(4 * cfg.get_particles_per_image()):
                            p = Double_Particle()
                            if random.random() < self.dragging_possibility:
                                p.drag(self.dragging_speed, self.raster_angle)
                            self.objects.append(p)
            # w/ angle correlation
            else:
                if not overlapping:
                    if amount is not None:
                        for i in range(4 * amount):
                            if random.random() < self.dragging_possibility:
                                p = self.get_dragged_that_mot_overlaps(maximum_tries, setangle=True)
                                self.objects.append(p)
                            else:
                                p = self._get_thatnot_overlaps(maximum_tries)
                                p.set_theta(self._calc_angle_for_particle(p))
                                self.objects.append(p)
                    elif coverage is not None:
                        while self.coverage() < coverage:
                            if random.random() < self.dragging_possibility:
                                p = self.get_dragged_that_mot_overlaps(maximum_tries, setangle=True)
                                self.objects.append(p)
                            else:
                                p = self._get_thatnot_overlaps(maximum_tries)
                                p.set_theta(self._calc_angle_for_particle(p))
                                self.objects.append(p)
                    else:
                        for i in range(4 * cfg.get_particles_per_image()):
                            if random.random() < self.dragging_possibility:
                                p = self.get_dragged_that_mot_overlaps(maximum_tries, setangle=True)
                                self.objects.append(p)
                            else:
                                p = self._get_thatnot_overlaps(maximum_tries)
                                p.set_theta(self._calc_angle_for_particle(p))
                                self.objects.append(p)
                # w/ angle, w overlapping
                else:
                    if amount is not None:
                        for i in range(4 * amount):
                            p = Double_Particle()
                            p.set_theta(self._calc_angle_for_particle(p))
                            if random.random() < self.dragging_possibility:
                                p.drag(self.dragging_speed, self.raster_angle)
                            self.objects.append(p)
                    elif coverage is not None:
                        while self.coverage() < coverage:
                            p = Double_Particle()
                            p.set_theta(self._calc_angle_for_particle(p))
                            if random.random() < self.dragging_possibility:
                                p.drag(self.dragging_speed, self.raster_angle)
                            self.objects.append(p)
                    else:
                        for i in range(4 * cfg.get_particles_per_image()):
                            p = Double_Particle()
                            p.set_theta(self._calc_angle_for_particle(p))
                            if random.random() < self.dragging_possibility:
                                p.drag(self.dragging_speed, self.raster_angle)
                            self.objects.append(p)
        # use angle range
        else:
            if not overlapping:
                if amount is not None:
                    for i in range(4 * amount):
                        if random.random() < self.dragging_possibility:
                            p = self.get_dragged_that_mot_overlaps(maximum_tries, angle=self._random_angle_range())
                            self.objects.append(p)
                        else:
                            p = self._get_thatnot_overlaps(maximum_tries)
                            p.set_theta(self._random_angle_range())
                            self.objects.append(p)
                elif coverage is not None:
                    while self.coverage() < coverage:
                        if random.random() < self.dragging_possibility:
                            p = self.get_dragged_that_mot_overlaps(maximum_tries, angle=self._random_angle_range())
                            self.objects.append(p)
                        else:
                            p = self._get_thatnot_overlaps(maximum_tries)
                            p.set_theta(self._random_angle_range())
                            self.objects.append(p)
                else:
                    for i in range(4 * cfg.get_particles_per_image()):
                        if random.random() < self.dragging_possibility:
                            p = self.get_dragged_that_mot_overlaps(maximum_tries, angle=self._random_angle_range())
                            self.objects.append(p)
                        else:
                            p = self._get_thatnot_overlaps(maximum_tries)
                            p.set_theta(self._random_angle_range())
                            self.objects.append(p)
            # w/ angle, w/ overlapping
            else:
                if amount is not None:
                    for i in range(4 * amount):
                        p = Double_Particle()
                        p.set_theta(self._random_angle_range())
                        self.objects.append(p)
                elif coverage is not None:
                    while self.coverage() < coverage:
                        p = Double_Particle()
                        p.set_theta(self._random_angle_range())
                        self.objects.append(p)
                else:
                    for i in range(4 * cfg.get_particles_per_image()):
                        p = Double_Particle()
                        p.set_theta(self._random_angle_range())
                        self.objects.append(p)

        # return
        # widthout angle correlation
        # self.passed_args = (amount, coverage, overlapping, maximum_tries)
        # if not self.use_range:
        #   if self.angle_char_len == 0:
        #      if not overlapping:
        #         if amount is not None:
        #            for i in range(4 * amount):
        #               p = self._get_thatnot_overlaps(maximum_tries)
        #              self.objects.append(p)
        #     elif coverage is not None:
        #        while self.coverage() < coverage:
        #           p = self._get_thatnot_overlaps(maximum_tries)
        #          self.objects.append(p)
        # else:

    def _randomPos(self):
        return np.array([self.img_width * random.random(), self.img_height*random.random()])

    def add_Dust_Part(self, part=None):
        if part is None:
            self.dust_particles.append(DustParticle(pos=self._randomPos(), size=random.random() * 40))
        else:
            self.dust_particles.append(part)

    def add_Dust(self):

        amnt = int(np.round(np.random.normal(self.dust_amount)))
        for i in range(amnt):
            self.add_Dust_Part()

    def create_Image_Visualization(self):
        self.img = MyImage()
        self.img.setWidth(int(np.ceil(self.img_width.px)))
        self.img.setHeight(int(np.ceil(self.img_height.px)))
        # print("W: {} - {}".format(self.img_width, self.img.getWidth()))
        # print("H: {} - {}".format(self.img_height, self.img.getHeight()))

        width = int(np.ceil(self.img_width.px))
        height = int(np.ceil(self.img_height.px))
        matrix = np.zeros((width, height))

        use_atomstep = random.random() < cfg.get_atomic_step_poss()



        # Set Max Height for parts
        if use_atomstep:
            for obj in self.objects:
                obj.set_maxHeight(cfg.get_max_height() + cfg.get_atomic_step_height())
            # Create Stepborder
            point_a = [random.randint(self.range[0], self.range[1]), random.randint(self.range[2], self.range[3])]
            point_b = [random.randint(self.range[0], self.range[1]), random.randint(self.range[2], self.range[3])]

            b = (point_a[1] - (point_a[0] / point_b[0]) * point_b[1]) / (1 - point_a[0] / point_b[0])
            m = (point_a[1] - b) / point_b[1]

            f = lambda x: m * x + b

        for part in self.objects:
            for tuple in part.get_visualization():
                # print("Tupel:{}".format(tuple))
                eff_mat, x, y = tuple
                mat_w = eff_mat.shape[0]

                # ToDo: possible failure
                x = int(np.round(x.px))
                y = int(np.round(y.px))
                # plt.imshow(eff_mat)
                # plt.show()
                # print(np.max(eff_mat))
                mat_h = eff_mat.shape[1]
                for i in range(mat_w):
                    for j in range(mat_h):
                        new_x = x - math.floor((mat_w / 2)) + i
                        new_y = y - math.floor(mat_h / 2) + j
                        if not (0 <= new_x < width and 0 <= new_y < height):
                            continue
                        matrix[new_x, new_y] += eff_mat[i, j]

        if self.usedust:
            self.add_Dust()

        if self.usedust:
            print("DD Dusting")
            self.get_Image_Dust()

        if use_atomstep:
            if len(self.objects) > 0:
                dh = self.objects[0].color(self.atomic_step_height)

            else:
                dh = 255 * cfg.get_atomic_step_height() / (cfg.get_max_height() + cfg.get_atomic_step_height())

            # print("Matrix-Shape: {}".format(np.shape(matrix)))

            for h in range(np.shape(matrix)[0]):
                for r in range(np.shape(matrix)[1]):
                    if f(h) > r:
                        matrix[h, r] += dh

        # print("img : {}".format(np.shape(self.img.get_matrix())))
        # print("matrix: {}".format(np.shape(matrix)))
        # print("Matrix")
        # plt.imshow(matrix)
        # plt.show()
        self.img.addMatrix(matrix)  # Indentd  too far right

    def addObjects(self, Object=Molecule, amount=None, coverage=None, overlapping=False, maximum_tries=1000):
        self.passed_args_Obj = Object, amount, coverage, overlapping, maximum_tries

        def random_pos():
            x = random.random() * self.img_width
            y = random.random() * self.img_height
            return np.array([x, y])

        def get_dragged_that_not_overlaps(maximumtries):
            def _set_p():
                p = Object(pos=random_pos())
                p.drag(self.dragging_speed, self.raster_angle)
                return p

            if len(self.objects) == 0:
                return _set_p()
            p = _set_p()
            for i in range(maximumtries):
                if self._overlaps_any(p):
                    p = _set_p()
                else:
                    return p
            return p

        def _get_thatnot_overlaps(maximumtries):
            # print("#Obj: {}, has Overlaps: {}".format(len(self.objects), self.has_overlaps()))
            if len(self.objects) == 0:
                return Object()
            p = Object(pos=random_pos())
            for i in range(maximumtries):
                # print("Added at x={}, y= {}".format(p.pos[0].px, p.pos[1].px))
                if self._overlaps_any(p):
                    # print("Retry")
                    p = Object(pos=random_pos())
                else:
                    return p
            # print("MaxTries Exhausted_a")
            # print("#Obj: {}, has Overlaps: {}".format(len(self.objects), self.has_overlaps()))
            return p

        self.passed_args_particles = (amount, coverage, overlapping, maximum_tries)
        print("DF aObj {}, {}, {}".format(self.use_range, self.angle_char_len, overlapping))
        if amount is not None:
            for i in range(4*amount):
                if random.random() < self.dragging_possibility:
                    p = get_dragged_that_not_overlaps(maximum_tries)
                    self.objects.append(p)
                else:
                    p = _get_thatnot_overlaps(maximum_tries)
                    self.objects.append(p)
        elif coverage is not None:
            while self.coverage() < coverage:
                if random.random() < self.dragging_possibility:
                    p = get_dragged_that_not_overlaps(maximum_tries)
                    self.objects.append(p)
                else:
                    p = _get_thatnot_overlaps(maximum_tries)
                    self.objects.append(p)
        else:
            for i in range(4*cfg.get_particles_per_image()):
                    if random.random() < self.dragging_possibility:
                        p = get_dragged_that_not_overlaps(maximum_tries)
                        self.objects.append(p)
                    else:
                        p = _get_thatnot_overlaps(maximum_tries)
                        self.objects.append(p)

    def extract_Smaller(self):
        self.create_Image_Visualization()
        #print("Start extractSmaller")
        #plt.imshow(self.img.get_matrix())
        #plt.show()

        self.img.double_tip(self.strength, self.rel_dist, self.dt_angle)

        smaller = np.zeros((int(np.ceil(cfg.get_width().px)), int(np.ceil(cfg.get_height().px))))
        bigger = self.img.get_matrix()
        # print(np.shape(smaller), np.shape(bigger))
        # print(self.range)
        for x in range(np.shape(smaller)[0]):
            for y in range(np.shape(smaller)[1]):
                x_tilt = x + self.range[0] - 1
                y_tilt = y + self.range[2] - 1
                smaller[x, y] = bigger[x_tilt, y_tilt]
        #print("Extracted One")
        #plt.imshow(smaller)
        #plt.show()
        # plt.imshow(bigger)
        # plt.show()
        return MyImage(smaller)

    def get_objects(self):
        ret = []
        for part in self.objects:
            if self.range[0] - self.overlap <= part.get_x().px <= self.range[1] + self.overlap and \
                    self.range[2] - self.overlap <= part.get_y().px <= self.range[3] + self.overlap:
                ret.append(part)

        return ret

    def get_Image(self):
        raise NotImplementedError

    def get_Image_Dust_DF(self, mat):
        width = np.shape(mat)[0]
        height = np.shape(mat)[1]
        matrix = mat

        for d in self.dust_particles:
            tuple = d.efficient_Matrix()

            eff_mat, x, y = tuple
            mat_w = eff_mat.shape[0]

            # ToDo: possible failure
            x = int(np.round(x.px))  # no px cause dust
            y = int(np.round(y.px))
            # plt.imshow(eff_mat)
            # plt.show()
            # print(np.max(eff_mat))
            mat_h = eff_mat.shape[1]
            for i in range(mat_w):
                for j in range(mat_h):
                    new_x = x - math.floor((mat_w / 2)) + i
                    new_y = y - math.floor(mat_h / 2) + j
                    if not (0 <= new_x < width and 0 <= new_y < height):
                        continue
                    matrix[new_x, new_y] += eff_mat[i, j]
        #self.img.addMatrix(matrix)

    def createText(self):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError
