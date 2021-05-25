import multiprocessing as mp
import copy, os
import math, random
import time

import scipy.optimize

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
        self.passed_args = None
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

    # returns iterator over Particles
    def getIterator(self):
        return self.objects

    # gets number of particles
    def __len__(self):
        return len(self.objects)

    # adds a given particle or, if not provided a random one
    def addParticle(self, part=None):
        if self.passed_args is None:
            self.passed_args = (1, None, True, 1000)
        else:
            if self.passed_args[0] is None:
                self.passed_args = (len(self.objects), self.passed_args[1], self.passed_args[2], self.passed_args[3])
            newargs = self.passed_args[0] + 1, self.passed_args[1], self.passed_args[2], self.passed_args[3]
            self.passed_args = newargs
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
        self.passed_args = (amount, coverage, overlapping, maximum_tries)
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

        self.passed_args = (amount, coverage, overlapping, maximum_tries)
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

        def dist_to_f(x, y, f):
            mind = 1000
            for xs in range(0, int(np.ceil(self.img_width.px))):
                ys = f(xs)
                if not -50 < ys < self.img_height.px + 50:
                    continue
                dist = np.sqrt(np.square(x - xs) + np.square(y - ys))
                if dist < mind:
                    mind = dist

            return mind

        def find_fermi_range(dh, fex):
            for zetta in range(0, 1000):
                # zetta = 1000 - d
                if dh * fermi_ohne_range(zetta, rad, fex) < 1:
                    return zetta

        show_gitter = False
        use_gitter = True
        show_f = False

        if (show_gitter):
            gitter = Tests_Gitterpot.create_larger_gitter()  # Ag-Atom[]
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
                dh = 255 * cfg.get_atomic_step_height() / (cfg.get_max_height() + cfg.get_atomic_step_height())

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

            # print("FR: {:.3f}".format(fermi_range2))

            if mt:
                print("STEP4 (Defs): {}".format(time.perf_counter() - start))
                start = time.perf_counter()

            sign = 1

            mode = "A"  # A
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

                        if d + schieb > fermi_range2:
                            if innen:
                                # pass
                                matrix[h, r] += dh
                            continue

                        in_at = in_range_of_nst(h, r, atoms_near_step, rad)
                        if innen:
                            if in_at:
                                opt1 = dh * fermi2(-d - schieb, sign * rad, fex2, fermi_range2)
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

            if mt:
                print("STEP5 (Matrix): {}".format(time.perf_counter() - start))
                start = time.perf_counter()

            if show_f:
                delta = 2 * abs(m)
                for h in range(np.shape(matrix)[0]):
                    for r in range(np.shape(matrix)[1]):
                        if abs(f(h) - r) < delta:
                            matrix[h, r] = 200

            # plt.imshow(matrix.transpose())
            plt.show()

            # for i in range(0, 400, 40):
            #    print("Dist to f: {}-200 : {:.3f}".format(i, dist_to_f(i, 200, f)))#

            # print("Fermi(0, rad) = {}".format(fermi(0, rad)))

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
                        if not (0 <= new_x < width.px and 0 <= new_y < height.px):
                            continue
                        matrix[new_x, new_y] += eff_mat[i, j]

        if use_atomstep:
            self.atomic_step(matrix, *fargs)

        self.img.addMatrix(matrix)

    def get_Image(self):
        if random.random() < self.double_tip_poss:
            print("Double Tipping")
            # ToDo: Step
            strength = 0.3 + 0.5 * random.random()
            rel_dist = 0.1 * random.random()  # ToDO: Let loose
            angle = 2 * np.pi * random.random()
            doubled_frame = Double_Frame(self.fn_gen, strength, rel_dist, angle)
            # doubled_frame.addParticles(self.passed_args[0], self.passed_args[1], self.passed_args[2],
            #                          self.passed_args[3])
            doubled_frame.addObjects(*self.passed_args_Obj)

            self.img = doubled_frame.extract_Smaller()
            self.objects = doubled_frame.get_objects()
            if self.use_noise:
                self.img.noise(self.image_noise_mu, self.image_noise_sigma)
            self.img.updateImage()
            return

        self.create_Image_Visualization()

        if self.use_noise:
            self.img.noise(self.image_noise_mu, self.image_noise_sigma)
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
        if self.passed_args is None:
            self.passed_args = (1, None, True, 1000)
        else:
            if self.passed_args[0] is None:
                self.passed_args = (len(self.objects), self.passed_args[1], self.passed_args[2], self.passed_args[3])
            self.passed_args[0] += 1
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

    def _get_thatnot_overlaps(self, maximumtries=1000):
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

    def addParticles(self, amount=None, coverage=None, overlapping=False, maximum_tries=1000):

        self.passed_args = (amount, coverage, overlapping, maximum_tries)
        # print("Got Args: {}".format(self.passed_args))
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

        self.passed_args = (amount, coverage, overlapping, maximum_tries)
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

    def extract_Smaller(self):
        self.create_Image_Visualization()
        # print("Start extractSmaller")
        # plt.imshow(self.img.get_matrix())
        # plt.show()

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
        # plt.imshow(smaller)
        # plt.show()
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

    def createText(self):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError
