import multiprocessing as mp
import copy, os
import math, random
import time
#import numbers

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
import sys
from tqdm import tqdm


class DataFrame:
    """
    Core class of the simulation program. Contains and directs the entire process.
    Contains Particles that should be displayed, generates images and saves them
    """

    # Constructor. Reads Config into local variables
    @measureTime
    def __init__(self, fn_gen):
        """
        Constructor.
        :param fn_gen: Filename Generator instance
        """

        start = time.perf_counter()
        # Initialize important variables
        self.objects = []
        self.fn_gen = fn_gen
        self.text = ""
        self.img = None

        # Get parameters from configuration for possible further use
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

        # passed_args variables are used to store arguments if a adding-function has to be reevaluated (e.g. Double Tip)
        self.passed_args_particles = None
        self.passed_args_Obj = None
        self.passed_args_Ordered = None
        self.passed_args_One = None

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
        self.use_scanlines = cfg.use_scanlines()
        self.use_slope = cfg.get_slope_dist() != Distance(True, 0)
        print("__init__: {:.2f}s".format(time.perf_counter() - start))

        # returns iterator over Particles

    @measureTime
    def getIterator(self):
        """
        Returns iterator over objects
        :return: iterator over self.objs
        """

        return self.objects

    @measureTime
    def __len__(self):
        """
        Returns number of particles inside this frame
        :return: number of particles
        """

        return len(self.objects)

    @measureTime
    def addParticle(self, part=None):
        """
        Adds a single particle, If part is provided, it will be added. Otherwise a new particle is generated and added.
        :param part: optional: particle to add
        :return: None
        """

        self.passed_args_One = Particle()
        if self.passed_args_particles is None:
            self.passed_args_particles = (1, None, True, 1000)
        else:
            if self.passed_args_particles[0] is None:
                self.passed_args_particles = (
                    len(self.objects), self.passed_args_particles[1], self.passed_args_particles[2],
                    self.passed_args_particles[3])
            newargs = self.passed_args_particles[0] + 1, self.passed_args_particles[1], self.passed_args_particles[2], \
                      self.passed_args_particles[3]
            self.passed_args_particles = newargs
        if part is None:
            self.objects.append(Particle())

        else:
            self.objects.append(part)
        #  for c in part.get_charges(): ToDo Add if charges are back
        #     self.add_To_Potential.append(c)

        # checks wheather part overlaps any existing particle

    @measureTime
    def _overlaps_any(self, part):
        """
        Checks if the provided particle overlaps with any existing particle
        :param part: particle to be checked
        :return: True if part overlaps any
        """
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
                return True
        return False

        # returns random particle, that does not overlap with any other

    @measureTime
    def _get_thatnot_overlaps(self, maximumtries=1000, calcangle=False):
        """
        Returns a particle that does not overlap with any existing one.
        :param maximumtries: Tries after which the adding should give up and return None
        :param calcangle: If angle for particle should be calculated
        :return: particle that can be added
        """

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
            return None

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

    def _get_molec_thatnot_overlaps(self, Obj=Molecule, ph=3, maximumtries=1000):
        """
        Same as _get_thatnot_overlaps, now adds Molecules instead of particles
        :param Obj: Class of which things should be added
        :param ph: Number of phenyl groups for added parts
        :param maximumtries: Tries after which the adding should give up and return None
        :return:
        """

        @measureTime
        def _overlaps_M_any(part):
            """
            Assisting function. _overlaps any specified for molecules
            :param part: partcile/molecula that may overlap
            :return: True if it overlaps
            """
            # start = time.perf_counter()

            if len(self.objects) == 0:
                # print("Overlaps any took {}".format(time.perf_counter() - start))
                return False
            for p in self.objects:
                if not part.dragged and not p.dragged:

                    if math.dist([p.x.px, p.y.px], [part.x.px, part.y.px]) > np.sqrt(2) * max(part.effect_range,
                                                                                              p.effect_range):
                        continue
                    else:
                        return True
            # print("Overlaps any took {}".format(time.perf_counter() - start))
            return False

        if len(self.objects) == 0:
            return Obj(molecule_ph_groups=ph)
        p = Obj(molecule_ph_groups=ph)
        for i in range(maximumtries):
            if _overlaps_M_any(p):
                p = Obj(molecule_ph_groups=ph)
            else:
                return p

        return None

    @measureTime
    def get_dragged_that_not_overlaps(self, maximumtries, angle=None, setangle=False):
        """
        Same as _get_thatnot_overlaps, just with dragged particles
        :param maximumtries: Tries after which the adding should give up and return None
        :param angle: angle of particle
        :param setangle: if angle should be set/calculated (deprecated)
        :return:
        """

        @measureTime
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

    @measureTime
    def addParticles(self, optimumEnergy=False, amount=None, coverage=None, overlapping=False, maximum_tries=1000):
        """
        Adds particles to the DataFrame

        :param optimumEnergy: Deprecated. Used to add particles at optimum energy position
        :param amount: Number of particles to add (1st priority)
        :param coverage: if amount is not specified: coverage percentage that should be reached (2nd priority)
        :param overlapping: is particle should be allowed to overlap with each other
        :param maximum_tries: Maximum tries to add not-overlapping particles (see _get_thatnot_overlaps)
        :return: None
        """

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
                                p = self.get_dragged_that_not_overlaps(maximum_tries)
                                self.objects.append(p)
                            else:
                                p = self._get_thatnot_overlaps(maximum_tries)
                                self.objects.append(p)
                    elif coverage is not None:
                        while self.coverage() < coverage:
                            if random.random() < self.dragging_possibility:
                                p = self.get_dragged_that_not_overlaps(maximum_tries)
                                self.objects.append(p)
                            else:
                                p = self._get_thatnot_overlaps(maximum_tries)
                                self.objects.append(p)
                    else:
                        for i in range(cfg.get_particles_per_image()):
                            if random.random() < self.dragging_possibility:
                                p = self.get_dragged_that_not_overlaps(maximum_tries)
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
                                p = self.get_dragged_that_not_overlaps(maximum_tries, setangle=True)
                                self.objects.append(p)
                            else:
                                p = self._get_thatnot_overlaps(maximum_tries, calcangle=True)
                                # p.set_theta(self._calc_angle_for_particle(p))
                                self.objects.append(p)
                    elif coverage is not None:
                        while self.coverage() < coverage:
                            if random.random() < self.dragging_possibility:
                                p = self.get_dragged_that_not_overlaps(maximum_tries, setangle=True)
                                self.objects.append(p)
                            else:
                                p = self._get_thatnot_overlaps(maximum_tries, calcangle=True)
                                # p.set_theta(self._calc_angle_for_particle(p))
                                self.objects.append(p)
                    else:
                        # print("Normal") Normaldurchlauf
                        for i in range(cfg.get_particles_per_image()):
                            if random.random() < self.dragging_possibility:
                                p = self.get_dragged_that_not_overlaps(maximum_tries, setangle=True)
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
                            p = self.get_dragged_that_not_overlaps(maximum_tries, angle=self._random_angle_range())
                            self.objects.append(p)
                        else:
                            p = self._get_thatnot_overlaps(maximum_tries)
                            p.set_theta(self._random_angle_range())
                            self.objects.append(p)
                elif coverage is not None:
                    while self.coverage() < coverage:
                        if random.random() < self.dragging_possibility:
                            p = self.get_dragged_that_not_overlaps(maximum_tries, angle=self._random_angle_range())
                            self.objects.append(p)
                        else:
                            p = self._get_thatnot_overlaps(maximum_tries)
                            p.set_theta(self._random_angle_range())
                            self.objects.append(p)
                else:
                    for i in range(cfg.get_particles_per_image()):
                        if random.random() < self.dragging_possibility:
                            p = self.get_dragged_that_not_overlaps(maximum_tries, angle=self._random_angle_range())
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

    @measureTime
    def addObject(self, ob):
        """
        Adds a passed object to self.objs
        :param ob: passed object to be added
        :return: None
        """
        self.passed_args_One = ob
        self.objects.append(ob)

    @measureTime
    def addObjects(self, Object=Molecule, amount=None, coverage=None, overlapping=False, maximum_tries=1000):
        """
        Same as addParticles, just with Molecules or other specified Object
        :param Object: class of witch instances should be added
        :param amount: Number of particles to add (1st priority)
        :param coverage: if amount is not specified: coverage percentage that should be reached (2nd priority)
        :param overlapping: is particle should be allowed to overlap with each other
        :param maximum_tries: Maximum tries to add not-overlapping particles (see _get_thatnot_overlaps)
        :return:
        """
        self.passed_args_Obj = Object, amount, coverage, overlapping, maximum_tries

        @measureTime
        def get_dragged_that_not_overlaps(maximumtries):
            @measureTime
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

        @measureTime
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

        # self.passed_args_particles = (amount, coverage, overlapping, maximum_tries)
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

    @measureTime
    def add_Ordered(self, Object=Molecule, theta=None, chirality=None, factor=1.0, ph_grps=None, style=None):
        """
        Add particles at positions observed in real STM Data forming regular patterns
        :param Object: Particle Class to be added in an ordered way
        :param theta: Angle by which the resulting lattice is turned. Random if None
        :param chirality: Random if None
        :param factor: Multiplicative factor for increasing the spacing between molecules
        :param ph_grps: number of phenyl groups
        :param style: Specifies the display-style for added particles
        :return: None
        """

        start = time.perf_counter()
        #offset = Distance(False, cfg.get_px_overlap())
        offset = 1.5*Molecule(molecule_ph_groups=5).get_simple_length()
        self.passed_args_Ordered = (Object, theta)
        var_pos = cfg.get_order_pos_var() * Molecule().get_simple_length()
        var_ang = cfg.get_order_ang_var() * 2 * np.pi
        vary_params = cfg.get_use_ordered_variation()


        @measureTime
        def bog(deg):
            """
            Turns degree values into radian
            :param deg: angle in degrees
            :return: angle in radians
            """

            return np.pi * deg / 180

        @measureTime
        def add_ordered_NCPh3CN(theta=None):
            """
            Method that adds Molecules with 3 phenyl groups
            :param theta: Angle by which the resulting lattice is turned. Random if None
            :return: None
            """
            if theta is None:
                theta_0 = random.random() * np.pi * 2
            else:
                theta_0 = theta
            dist_h = Distance(True, 13.226) * factor
            dist_v = Distance(True, 13.1933) * factor
            gv_a = np.array([dist_h * np.cos(theta_0), dist_h * np.sin(theta_0)])
            gv_b = np.array([-dist_v * np.sin(theta_0), dist_v * np.cos(theta_0)])

            ang_a = (theta_0 + bog(19.61545))
            ang_b = (theta_0 + bog(59.40909))

            pairs = []

            start = np.array([Distance(True, 0), Distance(True, 0)])
            current = np.array([0, 0])
            a_temp = self.img_width / gv_b[0] if gv_b[0].px != 0 else -np.infty
            b_temp = self.img_height / gv_b[1] if gv_b[1].px != 0 else -np.infty
            j_max = int(np.ceil(max(a_temp, b_temp)))
            c_temp = ((self.img_width + j_max * gv_b[0]) / gv_a[0]) if gv_a[0].px != 0 else j_max
            i_max = int(np.ceil(c_temp))
            for i in range(-200, 200):
                for j in range(-200, 200):
                    current = start + (gv_a * i) + (gv_b * j)
                    if self.img_width + offset > current[0] > (-1) * offset and offset + self.img_height > current[
                        1] > (-1) * offset:
                        if vary_params:
                            wid = Distance(False, np.random.normal(0, var_pos.px))
                            rang = random.random() * 2 * np.pi
                            pos = current + np.array([wid * np.cos(rang), wid * np.sin(rang)])
                            angdiff = np.random.normal(ang_a if (i + j) % 2 == 0 else ang_b, var_ang)
                            angdiff = angdiff % (2 * np.pi)
                            pairs.append((pos, angdiff))
                        else:
                            ang = ang_a if (i + j) % 2 == 0 else ang_b
                            ang = ang % (np.pi * 2)
                            pairs.append((current, ang))

            for pair in pairs:
                self.objects.append(Object(pos=pair[0], theta=pair[1], molecule_ph_groups=3, style=style))

        @measureTime
        def add_ordered_NCPh4CN(theta=None):
            """
            Method that adds Molecules with 4 phenyl groups
            :param theta: Angle by which the resulting lattice is turned. Random if None
            :return: None
            """

            if theta is None:
                theta_0 = random.random() * np.pi * 2
            else:
                theta_0 = theta
            chirality = np.sign(random.random() - 0.5)
            if chirality > 0:
                ang_ud = (theta_0 + bog(217.306)) % (2 * np.pi)
                ang_lr = (theta_0 + bog(134.136)) % (2 * np.pi)
                ud_lat_ang = (theta_0 + bog(189.595)) % (2 * np.pi)
                lr_lat_ang = (theta_0 + bog(100.773)) % (2 * np.pi)
                cross_ang = (theta_0 + bog(147.494)) % (2 * np.pi)

            else:
                ang_ud = (theta_0 + bog(142.694)) % (2 * np.pi)
                ang_lr = (theta_0 + bog(225.864)) % (2 * np.pi)
                ud_lat_ang = (theta_0 + bog(170.405)) % (2 * np.pi)  # Nicht sicher mit unsymmetr Molekülen
                lr_lat_ang = (theta_0 + bog(259.227)) % (2 * np.pi)
                cross_ang = (theta_0 + bog(212.506)) % (2 * np.pi)

            ud_dist = Distance(True, 24.1689) * factor
            lr_dist = Distance(True, 22.7745) * factor

            crossU_R = Distance(True, 17.1015) * factor

            vec_ud_lr = np.array([crossU_R * np.sin(cross_ang), -crossU_R * np.cos(cross_ang)])
            vec_r = np.array([lr_dist * np.sin(lr_lat_ang), -lr_dist * np.cos(lr_lat_ang)])
            vec_u = np.array([ud_dist * np.sin(ud_lat_ang), -ud_dist * np.cos(ud_lat_ang)])

            # print("LR: {:.1f}°".format(lr_lat_ang/3.14159 * 180))
            # print(vec_r)
            # print("UD: {:.1f}°".format(ud_lat_ang/3.14159 * 180))
            # print(vec_u)

            pairs = []

            i = 5
            j = -10

            start = np.array([Distance(True, 0), Distance(True, 0)])
            for i in range(-100, 100):
                for j in range(-100, 100):
                    current = start + (vec_u * i) + (vec_r * j)
                    if self.img_width + offset > current[0] > (-1) * offset and offset + self.img_height > current[
                        1] > (-1) * offset:
                        if vary_params:
                            wid = Distance(False, np.random.normal(0, var_pos.px))
                            rang = random.random() * 2 * np.pi
                            pos = current + np.array([wid * np.cos(rang), wid * np.sin(rang)])
                            angdiff = np.random.normal(ang_ud, var_ang)
                            angdiff = angdiff % (2 * np.pi)
                            pairs.append((pos, angdiff))
                        else:
                            pairs.append((current, ang_ud))
                        # print("No {} Appended ({},{})at {}".format(len(pairs), i, j, current))
                    secnd = current + vec_ud_lr
                    if self.img_width + offset > secnd[0] > (-1) * offset and offset + self.img_height > secnd[
                        1] > (-1) * offset:
                        if vary_params:
                            wid = Distance(False, np.random.normal(0, var_pos.px))
                            rang = random.random() * 2 * np.pi
                            pos = secnd + np.array([wid * np.cos(rang), wid * np.sin(rang)])
                            angdiff = np.random.normal(ang_lr, var_ang)
                            angdiff = angdiff % (2 * np.pi)
                            pairs.append((pos, angdiff))
                        else:
                            pairs.append((secnd, ang_lr))
                    # print("No {} Appended ({},{})at {}".format(len(pairs), i, j, secnd))

            for pair in pairs:
                # print(pair)
                self.objects.append(Object(pos=pair[0], theta=pair[1], molecule_ph_groups=4, style=style))

        @measureTime
        def add_ordered_NCPh5CN(theta=None, chirality=None):
            """
            Method that adds Molecules with 5 phenyl groups
            :param chirality:
            :param theta: Angle by which the resulting lattice is turned. Random if None
            :return: None
            """

            @measureTime
            def add_Hexa(center, start_ang, chirality):
                """
                Method that adds a ring of six NCPh5CN molecules
                :param center: Position of the hexagons center
                :param start_ang: angle by which wing is turned
                :param chirality: Which chirality the added ring should have
                :return:
                """

                d = Distance(True, 22.457) * factor

                @measureTime
                def turnvec(len, ang):
                    return np.array([len * np.sin(ang), -len * np.cos(ang)])

                if chirality > 0:
                    phi1 = (bog(-30.663) + start_ang) % (2 * np.pi)
                    phi2 = (bog(29.337) + start_ang) % (2 * np.pi)
                    phi3 = (bog(89.337) + start_ang) % (2 * np.pi)
                    phi4 = (bog(149.337) + start_ang) % (2 * np.pi)
                    phi5 = (bog(209.337) + start_ang) % (2 * np.pi)
                    phi6 = (bog(269.337) + start_ang) % (2 * np.pi)
                    phis = [phi1, phi2, phi3, phi4, phi5, phi6]

                    theta1 = (bog(41.107) + start_ang) % (2 * np.pi)
                    theta2 = (bog(101.107) + start_ang) % (2 * np.pi)
                    theta3 = (bog(161.107) + start_ang) % (2 * np.pi)
                    theta4 = (bog(221.107) + start_ang) % (2 * np.pi)
                    theta5 = (bog(281.107) + start_ang) % (2 * np.pi)
                    theta6 = (bog(341.107) + start_ang) % (2 * np.pi)
                    thetas = [theta1, theta2, theta3, theta4, theta5, theta6]
                else:
                    phi1 = (bog(30.663) + start_ang) % (2 * np.pi)
                    phi2 = (bog(360 - 29.337) + start_ang) % (2 * np.pi)
                    phi3 = (bog(360 - 89.337) + start_ang) % (2 * np.pi)
                    phi4 = (bog(360 - 149.337) + start_ang) % (2 * np.pi)
                    phi5 = (bog(360 - 209.337) + start_ang) % (2 * np.pi)
                    phi6 = (bog(360 - 269.337) + start_ang) % (2 * np.pi)
                    phis = [phi1, phi2, phi3, phi4, phi5, phi6]

                    theta1 = (bog(360 - 41.107) + start_ang) % (2 * np.pi)
                    theta2 = (bog(360 - 101.107) + start_ang) % (2 * np.pi)
                    theta3 = (bog(360 - 161.107) + start_ang) % (2 * np.pi)
                    theta4 = (bog(360 - 221.107) + start_ang) % (2 * np.pi)
                    theta5 = (bog(360 - 281.107) + start_ang) % (2 * np.pi)
                    theta6 = (bog(360 - 341.107) + start_ang) % (2 * np.pi)
                    thetas = [theta1, theta2, theta3, theta4, theta5, theta6]

                pairs = []
                for i in range(6):
                    position = center + turnvec(d, phis[i])
                    if self.img_width + offset > position[0] > (-1) * offset and offset + self.img_height > position[
                        1] > (-1) * offset:
                        if vary_params:
                            wid = Distance(False, np.random.normal(0, var_pos.px))
                            rang = random.random() * 2 * np.pi
                            pos = position + np.array([wid * np.cos(rang), wid * np.sin(rang)])
                            angdiff = np.random.normal(thetas[i], var_ang)
                            angdiff = angdiff % (2 * np.pi)
                            pairs.append((pos, angdiff))
                        else:
                            pairs.append((position, thetas[i]))

                for pair in pairs:
                    self.objects.append(Object(pos=pair[0], theta=pair[1], molecule_ph_groups=5, style=style))

            if theta is None:
                theta_0 = 2 * np.pi * random.random()
            else:
                theta_0 = theta
            if chirality is None:
                chirality = np.sign(random.random() - 0.5)

            gv_dist = Distance(True, 55.4938) * factor
            gv_a_w = theta_0 + bog(179.782)
            gv_b_w = theta_0 + bog(119.782)

            gv_a = np.array([gv_dist * np.sin(gv_a_w), -gv_dist * np.cos(gv_a_w)])
            gv_b = np.array([gv_dist * np.sin(gv_b_w), -gv_dist * np.cos(gv_b_w)])

            offset_loc = offset + gv_dist

            start = np.array([Distance(True, 0), Distance(True, 0)])
            current = np.array([0, 0])
            a_temp = self.img_width / gv_b[0] if gv_b[0].px != 0 else -np.infty
            b_temp = self.img_height / gv_b[1] if gv_b[1].px != 0 else -np.infty
            j_max = int(np.ceil(max(a_temp, b_temp)))
            c_temp = ((self.img_width + j_max * gv_b[0]) / gv_a[0]) if gv_a[0].px != 0 else j_max
            i_max = int(np.ceil(c_temp))
            for i in range(-100, max(100, i_max)):
                for j in range(-100, max(100, i_max)):
                    current = start + (gv_a * i) + (gv_b * j)
                    if self.img_width + offset_loc > current[0] > (-1) * offset_loc and offset_loc + self.img_height > \
                            current[
                                1] > (-1) * offset_loc:
                        add_Hexa(current, theta_0, chirality)

        if ph_grps is None:
            random_ph_grps = random.randint(3, 5)
        else:
            random_ph_grps = ph_grps

        if type(Object) is not type(Particle):
            raise NotImplementedError
        elif type(Object) is not type(Molecule):
            self.addObjects(Object=Object)
        elif Object.molecule_class != "NCPhCN":
            self.addObjects()
        # elif Object.molecule_ph_groups == 3:
        #    add_ordered_NCPh3CN(theta)
        # elif Object.molecule_ph_groups == 4:
        #    add_ordered_NCPh4CN(theta)
        # elif Object.molecule_ph_groups == 5:
        #    add_ordered_NCPh5CN(theta)
        elif random_ph_grps == 3:
            add_ordered_NCPh3CN(theta)
        elif random_ph_grps == 4:
            add_ordered_NCPh4CN(theta)
        elif random_ph_grps == 5:
            add_ordered_NCPh5CN(theta, chirality)
        else:
            raise NotImplementedError

        for part in self.objects:
            if random.random() < self.dragging_possibility:
                self.objects.remove(part)
                self.objects.append(self._add_at_pos_dragged(Molecule, pos=part.pos, theta=part.theta))

        print("Add Ordered: {:.2f}s".format(time.perf_counter() - start))

    @measureTime
    def atomic_step_init(self):
        """
        Initiates atomic step calculation by creating special parameters.
        A lot is not used anymore...
        :return: function (unused), slope(unused), y-intercept(unused), lines[],
                    if line is up to down or not, which side is higher
        """
        # for obj in self.objects:
        #    obj.set_maxHeight(cfg.get_max_height() + cfg.get_atomic_step_height())
        # Create Stepborder
        point_a = [random.random() * self.img_width.px, random.random() * self.img_height.px]
        point_b = [random.random() * self.img_width.px, random.random() * self.img_height.px]

        b = (point_a[1] - (point_a[0] / point_b[0]) * point_b[1]) / (1 - point_a[0] / point_b[0])
        m = (point_a[1] - b) / point_b[1]

        # b = 200
        # m = 0.001

        f = lru_cache()(lambda x: m * x + b)

        lns, ud, sA = self.calc_lines()

        return f, m, b, lns, ud, sA

    @measureTime
    def calc_lines(self):
        """
        Creates a line array containing the points along which the atomic step is aligned

        :return: lines-array, if line direction is up-down, which side to use
        """
        points = []

        # No of points at which define the line
        steps = 40

        # If line should be from top to bottom or from left to right
        updown = random.random() < 0.5

        # Variance in direction
        variance = int(160 / steps)

        # Which side of both should be higher
        sideA = random.random() < 0.5

        # If next point should depend on last slope. Makes borders smoother
        tendence = True

        # Calculate Lines
        if updown:
            oldstep = random.randint(-variance, variance)
            oldx = random.random() * self.img_width.px
            for i in range(0, int(self.img_height.px) + int(self.img_height.px / steps),
                           int(self.img_height.px / steps)):

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

            return lines, True, sideA
        else:
            oldy = random.random() * self.img_height.px
            oldstep = random.randint(-variance, variance)
            for i in range(0, int(self.img_width.px) + int(self.img_width.px / steps),
                           int(self.img_width.px / steps)):

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
            return lines, False, sideA

    @measureTime
    def dist_to_line(self, x, y, lines):
        """
        calculates the distance from a point (x,y) to the lines specified in lines
        :param x: x-coordinate
        :param y: y-coordinate
        :param lines: line array
        :return: the distance between point and line
        """

        already_interpolated = False
        maxlen = 50

        @measureTime
        def interpolate_lines(givnlines):
            """
            Interpolates the existing lines to make the tranisitions betwee lines less sharp. Needed to have
            a perpendicular connection between the lines and each point
            :param givnlines: not-interpolated previous lines
            :return: None, appends the new lines directly to the passed array
            """
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

        loc_lines = lines

        # Interpolate if maximum number of lines is not reached yet
        if len(lines) < maxlen:
            interpolate_lines(lines)

        # Threashold from which on a Dx should be treated as 0. Needed to prevent from very high slopes
        zero_threshold = 0.005
        distances = []

        # Calculate the distance to every line segment
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

        # return the minimal distance of all distances to line segments
        return min(distances)

        # @measureTime

    def angle_between(self, part, lines):
        """
        Calculates the angle between the particle orientation and the line segment closest to the particle
        :param part: particle
        :param lines: line segments
        :return: The angle in range from 0 to 90 degree
        """
        loc_lines = lines

        x = part.pos[0].px
        y = part.pos[1].px
        zero_threshold = 0.005
        distances = []
        # print(lines)
        # print(loc_lines)
        for line in loc_lines:
            dy = line[1][1] - line[0][1]
            dx = line[1][0] - line[0][0]
            if abs(dx) < zero_threshold:
                if line[0][1] <= y <= line[1][1]:
                    app = (line, abs(line[0][0] - x))
                   # assert isinstance(app[1], numbers.Number)
                    distances.append(app)
                elif y < line[0][1]:
                    app = (line, np.sqrt(np.square(line[0][1] - y) + np.square(line[0][0])))
                   # assert isinstance(app[1], numbers.Number)
                    distances.append(app)
                elif y > line[1][1]:
                    app = (line, np.sqrt(np.square(line[1][1] - y) + np.square(line[1][0] - x)))
                    #assert isinstance(app[1], numbers.Number)
                    distances.append(app)
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
            app = (line, np.sqrt(np.square(x - x_sp) + np.square(y - y_sp)))
            #assert isinstance(app[1], numbers.Number)
            distances.append(app)

        mindist = np.infty
        closest_pair = None
        for dist in distances:
            try:
                if dist[1] < mindist:
                    closest_pair = dist
                    mindist = dist[1]
            except ValueError:
                print("Error:")
                print("Dist, compl: : {}".format(dist))
                print("Dist: {}".format(dist[1]))
                print("mindist: {}".format(mindist))

        assert closest_pair is not None
        angle_part = part.theta % (2 * np.pi)

        clst_line = closest_pair[0]

        dy = clst_line[1][1] - clst_line[0][1]
        dx = clst_line[1][0] - clst_line[0][0]
        if abs(dx) < zero_threshold:
            theta = 0
        else:
            m = dy / dx
            theta = np.arctan(m)
            theta = theta % (2 * np.pi)

        if theta > np.pi / 2:
            theta = theta - np.pi

        if angle_part > np.pi / 2:
            angle_part = angle_part - np.pi

        delta = np.abs(theta - angle_part)
        delta -= np.pi / 2

        return abs(delta)

    @measureTime
    def atomic_step(self, matrix, f, m, b, lines=None, updown=None, side_A=None):
        """
        Creates a atomic step inside the data
        :param matrix: The previous visualization
        :param f: function, not used anymore
        :param m: slope, not used anymore
        :param b: y-intercept, not used anymore
        :param lines: line segments forming the step
        :param updown: if the step is oriented vertically
        :param side_A: which side should be higher
        :return:
        """

        fpoints = []

        # if mt is True additional timed information wil be provided
        mt = False

        if mt:
            start = time.perf_counter()

            # Gitter mit atomic step

        @measureTime
        def nearest_ag(gitter, pos):
            """
            Returns the nearest Atom from the sample surface from gitter
            :param gitter: array of substrate atoms
            :param pos: position which should be evaluated for nearest atom
            :return: the nearest atom from gitter
            """
            mindist = np.inf
            minat = None
            for ag in gitter:
                if np.linalg.norm(ag.pos - pos) < mindist:
                    mindist = np.linalg.norm(ag.pos - pos)
                    minat = ag

            return minat

        @measureTime
        def in_range_of_nst(x, y, atoms, radius):
            """
            Returns True if position (x, y) is closer to the nearest lattice atom than radius
            :param x: position x
            :param y: position y
            :param atoms: lattice atoms
            :param radius: radius
            :return: True if position (x, y) is closer to the nearest lattice atom than radius
            """
            # return False #ToDo: REm
            for atom in atoms:
                if np.linalg.norm(atom.pos - np.array([x, y])) < radius:
                    return True
            return False

        @measureTime
        def fermi(d, mu):
            """
            calculates fermi distribution at position d with expectation value mu
            Uses standard values for exponent factor gamma and range
            :param d: position
            :param mu: expectation
            :return:
            """
            if d < self.fermi_range:
                return 1 / (np.exp(self.fermi_exp * (d - mu)) + 1)
            else:
                return 0

        @measureTime
        def fermi2(d, mu, fex, range):
            """
            calculates fermi distribution with different gamma fex and different range
            :param d: pos
            :param mu: exp
            :param fex: gamma
            :param range: range
            :return: fermi distr
            """
            if d <= range:
                return 1 / (np.exp(fex * (d - mu)) + 1)
            else:
                return 0

        @measureTime
        def fermi_ohne_range(d, mu, fex):
            """
            Fermi dirtribution with infinite fermi range
            :param d: pos
            :param mu: exp
            :param fex: gamma
            :return: fermi distr
            """
            return fermi2(d, mu, fex, np.infty)

        @measureTime
        def dist_to_nst(x, y, atoms, radius):
            """
            Returns distance to nearest lattice atom
            :param x: pos x
            :param y: pos y
            :param atoms: lattice atoms
            :param radius: radius of these atoms
            :return:
            """
            max = 0
            # return 0 #ToDo: REm
            for atom in atoms:
                if fermi(np.linalg.norm(np.array([x, y]) - atom.pos), radius) > max:
                    max = fermi(np.linalg.norm(np.array([x, y]) - atom.pos), radius)
            return max

        @measureTime
        def _calc_fpoints(f):
            """
            DEPRECATED. Used to calculate line-points from function f
            :param f: function
            :return: None
            """
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

        @measureTime
        def dist_to_f(x, y, f):
            """
            DEPRECATED. Calculates distance from point (x, y) to function f
            :param x: x
            :param y: y
            :param f: function
            :return: distance
            """

            if len(fpoints) == 0:
                _calc_fpoints(f)

            mind = 1000
            target = np.array([x, y])
            for lastpt in fpoints:
                if np.linalg.norm(lastpt - target) < mind:
                    mind = np.linalg.norm(lastpt - target)

            return mind

        @measureTime
        def find_fermi_range(dh, fex):
            """
            Calculates the fermi range from which on the height is smaller than dh
            :param dh: min height
            :param fex: gamma
            :return: range
            """
            for zetta in range(0, 1000):
                # zetta = 1000 - d
                if dh * fermi_ohne_range(zetta, rad, fex) < 1:
                    return zetta

        # Constants what to do
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

        # Array of atoms laying near the step
        atoms_near_step = []
        gitter = Tests_Gitterpot.create_larger_gitter()  # Ag-Atom[]

        for pos in checking_pairs:
            nat = nearest_ag(gitter, pos)
            if nat not in atoms_near_step:
                atoms_near_step.append(nat)

        if mt:
            print("STEP3 (Atoms Near Step): {}".format(time.perf_counter() - start))
            start = time.perf_counter()

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

        @measureTime
        def effh():
            return dist_const / np.sin(0.5 * np.pi - np.arctan(1 / m))

        already_interpolated = False

        @measureTime
        def interpolate_lines(givnlines):
            """
                       Interpolates the existing lines to make the tranisitions betwee lines less sharp. Needed to have
                       a perpendicular connection between the lines and each point
                       :param givnlines: not-interpolated previous lines
                       :return: None, appends the new lines directly to the passed array
            """
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

        @measureTime
        def dist_to_line(x, y, lines):
            """
                    calculates the distance from a point (x,y) to the lines specified in lines
                    :param x: x-coordinate
                    :param y: y-coordinate
                    :param lines: line array
                    :return: the distance between point and line
            """
            loc_lines = lines
            # print("DistToLineM len() {}".format(len(lines)))
            interpolate = True
            if len(lines) < 50:  # ToDo MaxLen
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

        # Mode which implemenetation should be used. Best: G
        mode = "G"  # A

        # Shift of Fermi
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

        # Best
        if mode == "G":
            # Längere Computational dauer
            assert lines is not None
            assert updown is not None
            assert side_A is not None
            sideA = side_A
            show_line = False
            show_Toolbar = False

            if updown:
                border = []
                for line in lines:
                    dy = line[1][1] - line[0][1]
                    dx = line[1][0] - line[0][0]

                    for y in range(int(line[0][1]), int(line[1][1])):
                        if 0 <= y < np.shape(matrix)[1]:
                            border.append(np.array([line[0][0] + dx * (y - line[0][1]) / dy, y]))

                if show_Toolbar:
                    iter = tqdm(range(np.shape(matrix)[1]))
                else:
                    iter = range(np.shape(matrix)[1])

                for r in iter:
                    grenz = border[r][0]
                    for h in range(np.shape(matrix)[0]):
                        if sideA:
                            if h < grenz:
                                matrix[h, r] += dh * fermi2(-dist_to_line(h, r, lines), 0, fex2, fermi_range2)
                            elif grenz <= h:
                                matrix[h, r] += dh * fermi2(dist_to_line(h, r, lines), 0, fex2, fermi_range2)
                            else:
                                raise NotImplementedError
                        # SideB
                        else:
                            if h < grenz:
                                matrix[h, r] += dh * fermi2(dist_to_line(h, r, lines), 0, fex2, fermi_range2)
                            elif grenz <= h:
                                matrix[h, r] += dh * fermi2((-1) * dist_to_line(h, r, lines), 0, fex2, fermi_range2)
                            else:
                                raise NotImplementedError

            else:
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

                if show_Toolbar:
                    iter = tqdm(range(np.shape(matrix)[0]))
                else:
                    iter = range(np.shape(matrix)[0])

                for h in iter:
                    grenz = border[h][1]
                    for r in range(np.shape(matrix)[1]):
                        if sideA:
                            if r < grenz:
                                matrix[h, r] += dh * fermi2(-dist_to_line(h, r, lines), 0, fex2, fermi_range2)
                            elif grenz <= r:
                                matrix[h, r] += dh * fermi2(dist_to_line(h, r, lines), 0, fex2, fermi_range2)
                            else:
                                raise NotImplementedError
                        # SideB
                        else:
                            if r < grenz:
                                matrix[h, r] += dh * fermi2(dist_to_line(h, r, lines), 0, fex2, fermi_range2)
                            elif grenz <= r:
                                matrix[h, r] += dh * fermi2((-1) * dist_to_line(h, r, lines), 0, fex2, fermi_range2)
                            else:
                                raise NotImplementedError
        if mt:
            print("STEP5 (Matrix): {}".format(time.perf_counter() - start))
            start = time.perf_counter()

    @measureTime
    def add_Dust_Part(self, part=None):
        """
        Adds a dust particle to the frame
        :param part: particle to be added. New will be created if None
        :return: None
        """
        if part is None:
            self.dust_particles.append(DustParticle(size=random.random() * 40))
        else:
            self.dust_particles.append(part)

        # @measureTime

    @measureTime
    def add_Dust(self):
        """
        adds specified amount of dust particles
        :return: None
        """
        amnt = int(np.round(np.random.normal(self.dust_amount)))
        for i in range(amnt):
            self.add_Dust_Part()

    @measureTime
    def create_Image_Visualization(self):
        """
        Creates the visualization for the entire situation
        Invoked by get_image
        :return:
        """

        # New image istance for this frame
        self.img = MyImage()

        # create new matrix with specified parameters
        width = self.img_width
        height = self.img_height
        matrix = np.zeros((int(np.ceil(width.px)), int(np.ceil(height.px))))

        # Decides if atomic step should be present
        use_atomstep = random.random() < cfg.get_atomic_step_poss()

        # Set Max Height for parts
        if use_atomstep:
            allargs = self.atomic_step_init()
            fargs = (allargs[0], allargs[1], allargs[2])
            lines = allargs[3]
            updown = allargs[4]
            sideA = allargs[5]
        else:
            fargs = lambda c: 0, 0, 0
            lines = None
            updown = None
            sideA = None

        # Deprecated parameters
        rem_Upper = False
        rem_Lower = False
        rem_Border = True

        # Recalculate positions due to presence of atomic step
        if use_atomstep:
            # self.rem_parts_along_border(lines, matrix, rem_Border, rem_Lower, rem_Upper, sideA, updown)
            self.realign_along_border(lines, matrix, sideA, updown)

        max = len(self.objects)
        ct = 0
        for part in self.objects:
            # print("Visu_progress: {:.1f}%".format(100 * ct / max))
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

        # add dust
        if self.usedust:
            self.add_Dust()

        # modify visu_matrix if atomic step is present
        if use_atomstep:
            self.atomic_step(matrix, *fargs, lines, updown, sideA)

        self.img.addMatrix(matrix)

    @measureTime
    def rem_parts_along_border(self, lines, matrix, rem_Border, rem_Lower, rem_Upper, sideA, updown):
        """
        DEPRECATED. Used to remove particles because of the atomic step
        :param lines: lines defining the step
        :param matrix: visualization matrix
        :param rem_Border: if particles near the step should be removed
        :param rem_Lower: if particles on lower plane should be removed
        :param rem_Upper: if particles on upper plane should be removed
        :param sideA: which side is the upper one
        :param updown: if step is vertically or horizonatlly
        :return: None
        """
        if updown:
            border = []
            for line in lines:
                dy = line[1][1] - line[0][1]
                dx = line[1][0] - line[0][0]

                for y in range(int(line[0][1]), int(line[1][1])):
                    if 0 <= y < np.shape(matrix)[1]:
                        border.append(np.array([line[0][0] + dx * (y - line[0][1]) / dy, y]))
        else:
            border = []
            for line in lines:
                dy = line[1][1] - line[0][1]
                dx = line[1][0] - line[0][0]

                for x in range(int(line[0][0]), int(line[1][0])):
                    if 0 <= x < np.shape(matrix)[0]:
                        border.append(np.array([x, line[0][1] + ((x - line[0][0]) / dx) * dy]))
        ct = 0
        lists_of_indizes_to_rem = []
        for i in range(len(self.objects)):
            part = self.objects[i]
            ct += 1

            posy = 0 if part.pos[1].px < 0 else int(part.pos[1].px)
            posy = min(np.shape(matrix)[1] - 1, posy)
            posx = 0 if part.pos[0].px < 0 else int(part.pos[0].px)
            posx = min(np.shape(matrix)[0] - 1, posx)

            if rem_Border:
                if self.dist_to_line(part.pos[0].px, part.pos[1].px, lines) < 3 * part.get_dimension().px:
                    lists_of_indizes_to_rem.append(i)
                    continue

            if rem_Upper or rem_Lower:
                if updown:
                    if sideA:
                        if posx < border[posy][0]:
                            if rem_Upper:
                                lists_of_indizes_to_rem.append(i)
                                continue
                        else:
                            if rem_Lower:
                                lists_of_indizes_to_rem.append(i)
                                continue
                    else:
                        if posx < border[posy][0]:
                            if rem_Lower:
                                lists_of_indizes_to_rem.append(i)
                                continue
                        else:
                            if rem_Upper:
                                lists_of_indizes_to_rem.append(i)
                                continue
                else:
                    if sideA:
                        if posy < border[posx][1]:
                            if rem_Upper:
                                lists_of_indizes_to_rem.append(i)
                                continue
                        else:
                            if rem_Lower:
                                lists_of_indizes_to_rem.append(i)
                                continue
                    else:
                        if posy < border[posx][1]:
                            if rem_Lower:
                                lists_of_indizes_to_rem.append(i)
                                continue
                        else:
                            if rem_Upper:
                                lists_of_indizes_to_rem.append(i)
                                continue
        for x in lists_of_indizes_to_rem[::-1]:
            del self.objects[x]

        # @measureTime

    @measureTime
    def realign_along_border(self, lines, matrix, sideA, updown):
        """
        New method used instead of just removing the particles at the border
        :param lines: line segments specifying the atomic step
        :param matrix: visu_matrix
        :param sideA: which side is higher
        :param updown: is step is vertical or horizontal
        :return:
        """

        if self.passed_args_Ordered is None:
            return

        def _create_Border():
            """
            calculate particular pixels defining the border
            :return:
            """
            if updown:
                border = []
                for line in lines:
                    dy = line[1][1] - line[0][1]
                    dx = line[1][0] - line[0][0]

                    for y in range(int(line[0][1]), int(line[1][1])):
                        if 0 <= y < np.shape(matrix)[1]:
                            border.append(np.array([line[0][0] + dx * (y - line[0][1]) / dy, y]))
            else:
                border = []
                for line in lines:
                    dy = line[1][1] - line[0][1]
                    dx = line[1][0] - line[0][0]

                    for x in range(int(line[0][0]), int(line[1][0])):
                        if 0 <= x < np.shape(matrix)[0]:
                            border.append(np.array([x, line[0][1] + ((x - line[0][0]) / dx) * dy]))

            return border

        start = time.perf_counter()
        border = _create_Border()
        print("Border creation: {:.2f}ms".format(time.perf_counter() - start))

        def _upper_and_dist(x, y):
            """
            returns if position (x, y) is on the upper side and which distance to the step it has
            :param x: x
            :param y: y
            :return: if its on the upper side, distance
            """

            posy = max(0, y)
            posy = int(np.round(min(np.shape(matrix)[1] - 1, posy)))
            posx = max(0, x)
            posx = int(np.round(min(np.shape(matrix)[0] - 1, posx)))

            dist = self.dist_to_line(posx, posy, lines)

            if updown:
                if sideA:
                    return posx < border[posy][0], dist

                else:
                    return not posx < border[posy][0], dist

            else:
                if sideA:
                    return posy < border[posx][1], dist
                else:
                    return not posy < border[posx][1], dist

        def rempos(part, dist):  # 1 at 3pd, 0 at 7pd
            """
            Definition of possibility to remove a particle depending on its distance to the line
            :param part: particle which might be removed
            :param dist: distance to the line
            :return: possibility to remove the particle
            """
            return (1 + (3 / 7)) - (dist / (7 * part.get_dimension().px))

        # Removes all existing objects
        old_objs = self.objects.copy()
        self.objects = []
        assert len(old_objs) > 0
        ph_groups = old_objs[0].molecule_ph_groups

        # add particles on the upper plane
        self.add_Ordered(*self.passed_args_Ordered, ph_grps=ph_groups)
        upper_obs = self.objects.copy()
        self.objects = []

        # add particles on the lower plane
        self.add_Ordered(*self.passed_args_Ordered, ph_grps=ph_groups)
        lower_obs = self.objects.copy()
        self.objects = []

        # array indices of particles that should be removed on the upper plane
        rem_ind_upper = []
        for i in range(len(upper_obs)):
            part = upper_obs[i]
            rightside, dist = _upper_and_dist(part.pos[0].px, part.pos[1].px)
            if not rightside or rempos(part, dist) > random.random():
                rem_ind_upper.append(i)

        # array indices of particles that should be removed on the lower plane
        rem_ind_lower = []
        for i in range(len(lower_obs)):
            part = lower_obs[i]
            wrongside, dist = _upper_and_dist(part.pos[0].px, part.pos[1].px)
            if wrongside or rempos(part, dist) > random.random():
                rem_ind_lower.append(i)

        # remove the particles with specified indices
        for i in range(len(upper_obs)):
            if i not in rem_ind_upper:
                self.objects.append(upper_obs[i])

        for i in range(len(lower_obs)):
            if i not in rem_ind_lower:
                self.objects.append(lower_obs[i])

        # Add particles at random positions around the steps if they dont overlap
        start = time.perf_counter()
        i = 0
        while i < 100:
            no = self._get_molec_thatnot_overlaps(self.passed_args_Ordered[0], ph=ph_groups, maximumtries=5000)
            if no is None:
                break
            # When particles align across border perpendicular to it
            if self.dist_to_line(no.pos[0].px, no.pos[1].px,
                                 lines) < 0.5 * no.get_dimension().px and self.angle_between(no, lines) > np.pi / 4:
                #print("Prevented from crossing ang")
                continue

            i += 1
            self.objects.append(no)
        #print("Got {} after {:.2f}ms".format(i, time.perf_counter() - start))

    @measureTime
    def get_Image_Dust(self):
        """
        Add dust to the current image self.img
        :return: None
        """
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

        # @measureTime


    @measureTime
    def get_Image(self, ang=None):
        """
        Core method. Performs calculations to create the image and to be able to save it.
        :param ang: Normally None, can specify angle of Double Tip
        :return:
        """
        start = time.perf_counter()

        # Image with double tip
        if random.random() < self.double_tip_poss:
            print("Double Tipping")
            # ToDo: Step
            strength = 0.3 + 0.5 * random.random()
            rel_dist = 0.1 * random.random()  # ToDO: Let loose
            # rel_dist = 0.05
            # strength = 0.8
            if ang is None:
                angle = 2 * np.pi * random.random()
            else:
                angle = ang
            doubled_frame = Double_Frame(self.fn_gen, strength, rel_dist, angle)
            if self.passed_args_particles is not None:
                doubled_frame.addParticles(*self.passed_args_particles)
            elif self.passed_args_Obj is not None:
                doubled_frame.addObjects(*self.passed_args_Obj)
            elif self.passed_args_Ordered is not None:
                doubled_frame.add_Ordered(*self.passed_args_Ordered)
            elif self.passed_args_One is not None:
                doubled_frame.addOne(self.passed_args_One)
            else:
                print("Default")
                doubled_frame.addObjects()

            self.img = doubled_frame.extract_Smaller()
            self.objects = doubled_frame.get_objects()

            if self.use_noise:
                self.img.noise(self.image_noise_mu, self.image_noise_sigma)

            self.img.updateImage()
            print("Get Image: {:.2f}s".format(time.perf_counter() - start))
            return

        # without double tip
        self.create_Image_Visualization()

        # add Dust
        if self.usedust:
            self.get_Image_Dust()


        if self.use_slope:
            self.img.slope()


        # Noise image
        if self.use_noise:
            #self.img.noise_function()
            self.img.noise(self.image_noise_mu, self.image_noise_sigma)

        # add piezo shift
        if self.use_img_shift:
            self.img.shift_image()

        # add scanlines
        if self.use_scanlines:
            self.img.scan_lines()

        # update image matrix
        self.img.updateImage()

        print("get Image: {:.2f}s".format(time.perf_counter() - start))

        # @measureTime

    @measureTime
    def createText(self):
        """
        create string representation
        :return: None
        """
        strings = [Particle.str_Header()]
        for part in self.objects:
            strings.append(str(part))
        self.text = "\n".join(strings)

        # @measureTime

    @measureTime
    def save(self, data=False, image=True, sxm=True):
        """
        Core method. Saves results to disk. Filename is obtained by Filename Generator specified in init()
        :param data: Should results be saved as text describing positions?
        :param image: Should results be saved as an image file?
        :param sxm: Should results be saved as .sxm file?
        :return: Index used for saving
        """
        start = time.perf_counter()
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
        print("Save: {:.2f}s".format(time.perf_counter() - start))
        return index

    @measureTime
    def hasPoints(self):
        """

        :return: True if has objects
        """
        # return not self.points.empty()
        return len(self.objects) > 0

        # @measureTime

    @measureTime
    def coverage(self):
        """
        calculates approximate coverage of frame
        :return: coverage
        """
        area = self.area
        covered = 0
        for part in self.objects:
            covered += np.pi * np.square(part.get_dimension().px)
        return covered / area.px

        # @measureTime

    @measureTime
    def has_overlaps(self):
        """
        Check if current frame has overlapping particles
        :return: True if overlaps were found
        """
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

    @measureTime
    def __str__(self):
        return str(self.objects)

    # Deprecated Stuff....

    @measureTime
    def is_overlapping(self, part):
        """
        DEPRECATED. Returns overlapping energy if it overlaps, 0 otherwise
        :param part: Particle which should be checked for overlapping
        :return:
        """
        for p in self.objects:
            if p.true_overlap(part):
                return self.overlapping_energy
        return 0

    @measureTime
    def _add_at_pos_dragged(self, Object, pos, theta):
        """

        DEPRECATED. Adds the specified Object at a given position dragged at an angle theta
        :param Object: obj to add
        :param pos: position vector where obj should be added
        :param theta: angle at which obj is dragged
        :return:
        """

        @measureTime
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

    @measureTime
    def _calc_angle_for_particle(self, particle):
        """
        DEPREACTED. Calculated the angle for a given particle based on its orientation to the crystal.
        Not used Deprecation Tag because it is used inside some if statement
        :param particle: particle whose angle should be set
        :return:
        """
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

    @measureTime
    def _random_angle_range(self):
        """
        DEPRECATED. Returns a random angle inside the range specified in Configuration
        Not used Deprecation Tag because it is used inside some if statement

        :return: The angle
        """
        if self.min_angle > self.max_angle:
            shiftl = 2 * np.pi - self.min_angle
            ret = random.random() * self.max_angle + shiftl
            ret -= shiftl
            return ret
        else:
            return random.uniform(self.min_angle, self.max_angle)

    @DeprecationWarning
    @measureTime
    def _calc_angle_weight(self, part1, part2):
        """
        DEPRECATED. calculates particles weight for importance in surrounding particles
        :param part1: particle 1
        :param part2: particle 2
        :return: a number indicating how strong both particles are influencing each other
        """
        print("Deprecated 5183")
        drel = part1.get_distance_to(part2) / self.max_dist
        # return np.exp(-self.angle_char_len/drel)
        return self.angle_char_len / drel

    @DeprecationWarning
    @measureTime
    def _orients_along_crystal(self, particle):
        """
        DEPRECATED. Returns whether particle aligns along the crystal or other particles
        :param particle: part
        :return: True if bonds to crystal
        """
        print("Deprecated 1351361")
        return self._bonding_strength_crystal(particle) < 0.5

    #
    @DeprecationWarning
    @measureTime
    def _bonding_strength_crystal(self, particle):
        """
        Indicates bonding strengths to crystal and particles. High if bonds to particles, low if bonds to crystal
        :param particle: part
        :return: 1 if bond to particle, 0 otherwise
        """
        print("Deprecated 46463516")
        if len(self.objects) == 0:
            return 0
        for part in self.objects:
            dis = part.get_distance_to(particle)
            if dis < self.max_dist / np.sqrt(2) * 0.2:
                return 1
        return 0

    @DeprecationWarning
    @measureTime
    def _drag_particles(self):
        """
        DEPRECATED. previously used to drag particles
        :return: None
        """
        print("Deprecated 6546335416")
        for part in self.objects:
            if random.random() < self.dragging_possibility:
                part.drag(self.dragging_speed, self.raster_angle)

    @DeprecationWarning
    @measureTime
    def add_at_optimum_energy_new_theta(self, x_start, y_start, theta_start):
        """
        DEPRECATED. Adds a particle the way its energy is minimized
        :param x_start: init value position x
        :param y_start: init value position y
        :param theta_start: init value theta
        :return: None
        """

        print("Deprecated 41535154")

        @measureTime
        def overlapping_amount(part):
            """
            Inidcates how strong part overlaps with any other particles
            :param part: part
            :return:
            """

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

        @measureTime
        def energyt(theta):
            """
            Calculate energy depending on angle theta
            :param theta: angle
            :return: energy
            """
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
    @measureTime
    def add_ALL_at_optimum_energy_new(self, n):
        """
        DEPRECATED. Adds n particles at their optimal energies
        :param n: number of particles
        :return: None
        """
        print("Deprecated 1351313")
        # plt.imshow(self.potential_map)
        # plt.show()

        loc_pot_map = np.zeros((self.img_width, self.img_height))
        loc_objs = []
        use_old = False
        to_add = []

        @measureTime
        def calc_loc_pot_map():
            """
            Calculates potential array
            :return: None
            """
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

        @measureTime
        def overlapping_amount(part):
            """
                        Inidcates how strong part overlaps with any other particles
                        :param part: part
                        :return:
            """
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

        @measureTime
        def energy(args):
            """
            DEPRECATED. calculates energy depending on x, y, theta
            :param args: x, y, theta
            :return:
            """
            e = 0
            for i in range(0, 3 * n, 3):
                p = Particle(args[i], args[i + 1], args[i + 2])
                e += energyxyt(Particle(args[i], args[i + 1], args[i + 2]))
                loc_objs.append(p)
            # calc_loc_pot_map()
            # for part in loc_objs:
            #    e += energyxyt(part)

            return e

        @measureTime
        def energyxyt(part):
            """
                        DEPRECATED. calculates energy depending on x, y, theta
                        :param part: particle
                        :return:
            """
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
        # plt.show()

    @DeprecationWarning
    @measureTime
    def energy_function(self, x, y, theta):
        """
        DEPRECATED. calculates energy of particle with given params in the potential potentialmap
        :param x: position
        :param y: position
        :param theta: angle
        :return: energy
        """
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
    @measureTime
    def opimizable_energy_function(self, x):
        """
        DEPRECATD. energy_function applicable to SciPy
        :param x:
        :return:
        """
        print("Deprecated 165843")
        return self.energy_function(x[0], x[1], x[2])

    @DeprecationWarning
    @measureTime
    def add_at_optimum_energy(self, initvals=None):
        """
        DEPRECATED. Adds particle at optimum energy (without crystal)
        :param initvals:
        :return:
        """
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
    @measureTime
    def add_at_optimum_energy_new(self, x_start, y_start, theta_start):
        """
        DEPRECATED. Adds particle at optimum energy (without crystal)
        :param initvals:
        :return:
        """

        print("Deprecatd 456135")

        # plt.imshow(self.potential_map)
        # plt.show()
        @measureTime
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

        @measureTime
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

        @measureTime
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

        @measureTime
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
    @measureTime
    def calc_pot_Energy_for_particle(self, part, mapchange=False):
        """
        DEPRECATED. calculates potential energy for particle
        :param part: particle
        :param mapchange: if potential map has changed
        :return:
        """
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
    @measureTime
    def createImage(self):
        """
        DEPRECATED. Fist used to to everything
        :return:
        """
        print("Deprecated 3154251534")
        self.img = MyImage()
        for part in self.objects:
            self.img.addParticle(part)

        self.img.updateImage()  # Always call last
        # img.noise....etc

    @DeprecationWarning
    @measureTime
    def createImage_efficient(self):
        """
        DEPRECATED. Fist used to to everything
        :return:
        """
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
    @measureTime
    def createImage_efficient_with_new_Turn(self):
        """
                DEPRECATED. Fist used to to everything
                :return:
                """
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
    @measureTime
    def calc_potential_map(self):  # ToDo: Nicht Multithreading safe mit pickle dump
        """
        calculate potential map. dumps it to disk
        :return:
        """
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

        @measureTime
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

        @measureTime
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

        @measureTime
        def slope():
            pot = np.zeros((self.img_width, self.img_height))
            for i in range(self.img_width):
                for j in range(self.img_width):
                    pot[i, j] = 100 * i / self.img_width

            return pot

        @measureTime
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
    """
    Class Double Frame extends DataFrame and modifies some methods for the special case of a Double tip.
    Behaves the same but has 4times the size. Also provides methods to extract the double tipped image
    """

    @measureTime
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
        self.dust_particles *= 4

        if self.shift_x >= 0:
            if self.shift_y > 0:
                self.range = int(int(np.ceil(self.img_width.px)) / 2), int(np.ceil(self.img_width.px)), int(
                    np.ceil((self.img_height.px / 2))), int(np.ceil(self.img_height.px))
            else:
                self.range = int(int(np.ceil(self.img_width.px)) / 2), int(np.ceil(self.img_width.px)), 0, int(
                    np.ceil((self.img_height.px / 2)))
        else:
            if self.shift_y > 0:
                self.range = 0, int(int(np.ceil(self.img_width.px)) / 2), int(
                    np.ceil((self.img_height.px / 2))), int(np.ceil(self.img_height.px))
            else:
                self.range = 0, int(int(np.ceil(self.img_width.px)) / 2), 0, int(np.ceil((self.img_height.px / 2)))

    @measureTime
    def addParticle(self, part=None):
        """
        see DataFrame.addParticle
        :return:
        """
        if self.passed_args_particles is None:
            self.passed_args_particles = (1, None, True, 1000)
        else:
            if self.passed_args_particles[0] is None:
                self.passed_args_particles = (
                    len(self.objects), self.passed_args_particles[1], self.passed_args_particles[2],
                    self.passed_args_particles[3])
            self.passed_args_particles[0] += 1
        if part is None:
            self.objects.append(Double_Particle())
        else:
            self.objects.append(part)

    @measureTime
    def get_dragged_that_not_overlaps(self, maximumtries, angle=None, setangle=False):
        """
        see DataFrame.get_dragged_that_not_overlaps
        :return:
        """

        # print("DTNO @len {}".format(len(self.objects)))
        @measureTime
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

    @measureTime
    def _get_thatnot_overlaps(self, maximumtries=1000, calcangle=False):
        """
        see DataFrame._get_thatnot_overlaps
        :return:
        """
        # print("TNO @len {}".format(len(self.objects)))
        if len(self.objects) == 0:
            return Double_Particle()
        p = Double_Particle()
        for i in range(maximumtries):
            if self._overlaps_any(p):
                p = Double_Particle()
            else:
                return p
        return None

    @measureTime
    def addParticles(self, optimumEnergy=False, amount=None, coverage=None, overlapping=False, maximum_tries=1000):
        """
        see DataFrame.addParticles
        :return:
        """
        self.passed_args_particles = (amount, coverage, overlapping, maximum_tries)

        # print("DF aP Got Args: {}".format(self.passed_args_particles))
        if not self.use_range:
            if self.angle_char_len == 0:
                if not overlapping:
                    if amount is not None:
                        for i in range(4 * amount):
                            if random.random() < self.dragging_possibility:
                                p = self.get_dragged_that_not_overlaps(maximum_tries)
                                self.objects.append(p)
                            else:
                                p = self._get_thatnot_overlaps(maximum_tries)
                                self.objects.append(p)
                    elif coverage is not None:
                        while self.coverage() < coverage:
                            if random.random() < self.dragging_possibility:
                                p = self.get_dragged_that_not_overlaps(maximum_tries)
                                self.objects.append(p)
                            else:
                                p = self._get_thatnot_overlaps(maximum_tries)
                                self.objects.append(p)
                    else:
                        for i in range(4 * cfg.get_particles_per_image()):
                            if random.random() < self.dragging_possibility:
                                p = self.get_dragged_that_not_overlaps(maximum_tries)
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
                                p = self.get_dragged_that_not_overlaps(maximum_tries, setangle=True)
                                self.objects.append(p)
                            else:
                                p = self._get_thatnot_overlaps(maximum_tries)
                                p.set_theta(self._calc_angle_for_particle(p))
                                self.objects.append(p)
                    elif coverage is not None:
                        while self.coverage() < coverage:
                            if random.random() < self.dragging_possibility:
                                p = self.get_dragged_that_not_overlaps(maximum_tries, setangle=True)
                                self.objects.append(p)
                            else:
                                p = self._get_thatnot_overlaps(maximum_tries)
                                p.set_theta(self._calc_angle_for_particle(p))
                                self.objects.append(p)
                    else:
                        for i in range(4 * cfg.get_particles_per_image()):
                            if random.random() < self.dragging_possibility:
                                p = self.get_dragged_that_not_overlaps(maximum_tries, setangle=True)
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
                            p = self.get_dragged_that_not_overlaps(maximum_tries, angle=self._random_angle_range())
                            self.objects.append(p)
                        else:
                            p = self._get_thatnot_overlaps(maximum_tries)
                            p.set_theta(self._random_angle_range())
                            self.objects.append(p)
                elif coverage is not None:
                    while self.coverage() < coverage:
                        if random.random() < self.dragging_possibility:
                            p = self.get_dragged_that_not_overlaps(maximum_tries, angle=self._random_angle_range())
                            self.objects.append(p)
                        else:
                            p = self._get_thatnot_overlaps(maximum_tries)
                            p.set_theta(self._random_angle_range())
                            self.objects.append(p)
                else:
                    for i in range(4 * cfg.get_particles_per_image()):
                        if random.random() < self.dragging_possibility:
                            p = self.get_dragged_that_not_overlaps(maximum_tries, angle=self._random_angle_range())
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

    @measureTime
    def _randomPos(self):
        """
        returns a random position vector
        :return: vector
        """
        return np.array([self.img_width * random.random(), self.img_height * random.random()])

    @measureTime
    def add_Dust_Part(self, part=None):
        """
        see DataFrame.add_Dust_Part
        :return:
        """
        if part is None:
            self.dust_particles.append(DustParticle(pos=self._randomPos(), size=random.random() * 40))
        else:
            self.dust_particles.append(part)

    @measureTime
    def add_Dust(self):
        """
        see DataFrame.add_Dust
        :return:
        """
        amnt = int(np.round(np.random.normal(self.dust_amount)))
        for i in range(amnt):
            self.add_Dust_Part()

    @measureTime
    def addOne(self, ob):
        """
        Testing method to add one particle in every quadrand
        :param ob: Object containing position and angle that is added in every quadrand
        :return:
        """
        pos = ob.pos
        ang = ob.theta
        npos1 = pos
        lob = Molecule(npos1, ang)
        self.objects.append(lob)
        npos2 = np.array([pos[0] + self.img_width / 2, pos[1]])
        lob2 = Molecule(npos2, ang)
        self.objects.append(lob2)
        npos3 = np.array([pos[0], pos[1] + self.img_height / 2])
        lob3 = Molecule(npos3, ang)
        self.objects.append(lob3)
        npos4 = np.array([pos[0] + self.img_width / 2, pos[1] + self.img_height / 2])
        lob4 = Molecule(npos4, ang)
        self.objects.append(lob4)

    @measureTime
    def create_Image_Visualization(self):
        """
        see DataFrame.create_Image_Visualization
        :return:
        """
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
            fargs = self.atomic_step_init()
        else:
            fargs = 0, 0, 0

            # for obj in self.objects:
            #    obj.set_maxHeight(cfg.get_max_height() + cfg.get_atomic_step_height())
            # Create Stepborder
            # point_a = [random.randint(self.range[0], self.range[1]), random.randint(self.range[2], self.range[3])]
            # point_b = [random.randint(self.range[0], self.range[1]), random.randint(self.range[2], self.range[3])]

            # b = (point_a[1] - (point_a[0] / point_b[0]) * point_b[1]) / (1 - point_a[0] / point_b[0])
            # m = (point_a[1] - b) / point_b[1]

            # f = lambda x: m * x + b

        #for part in tqdm(self.objects):
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
                #for i in tqdm(range(mat_w)): #ToDo: Rem
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
            self.get_Image_Dust()

        if use_atomstep:
            self.atomic_step(matrix, *fargs)

        # print("img : {}".format(np.shape(self.img.get_matrix())))
        # print("matrix: {}".format(np.shape(matrix)))
        # print("Matrix")
        # plt.imshow(matrix)
        # plt.show()
        self.img.addMatrix(matrix)  # Indentd  too far right

    @measureTime
    def addObjects(self, Object=Molecule, amount=None, coverage=None, overlapping=False, maximum_tries=1000):
        """
        see DataFrame.addObjects
        :return:
        """
        self.passed_args_Obj = Object, amount, coverage, overlapping, maximum_tries

        @measureTime
        def random_pos():
            x = random.random() * self.img_width
            y = random.random() * self.img_height
            return np.array([x, y])

        @measureTime
        def get_dragged_that_not_overlaps(maximumtries):
            @measureTime
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

        @measureTime
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
        # print("DF aObj {}, {}, {}".format(self.use_range, self.angle_char_len, overlapping))
        if amount is not None:
            for i in range(4 * amount):
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
            for i in range(4 * cfg.get_particles_per_image()):
                if random.random() < self.dragging_possibility:
                    p = get_dragged_that_not_overlaps(maximum_tries)
                    self.objects.append(p)
                else:
                    p = _get_thatnot_overlaps(maximum_tries)
                    self.objects.append(p)

    @measureTime
    def extract_Smaller(self):
        """
        returnes a MyImage instance containing the smaller image in its original size
        :return: MyImage instance
        """
        self.create_Image_Visualization()
        # print("Start extractSmaller")
        # plt.imshow(self.img.colors)
        # plt.show()
        # print("extract smaller start")
        # plt.imshow(self.img.colors)
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
        # print("Extracted One")
        # plt.imshow(smaller)
        # plt.show()
        # plt.imshow(bigger)
        # plt.show()

        # print("Extract smaller end")
        # plt.imshow(smaller)
        # plt.show()
        return MyImage(smaller)

    @measureTime
    def get_objects(self):
        """
        returns objects that are inside the extracted frame
        :return: array of objects
        """
        ret = []
        for part in self.objects:
            if self.range[0] - self.overlap <= part.get_x().px <= self.range[1] + self.overlap and \
                    self.range[2] - self.overlap <= part.get_y().px <= self.range[3] + self.overlap:
                ret.append(part)

        return ret

    @measureTime
    def get_Image(self):
        raise NotImplementedError

    @measureTime
    def get_Image_Dust_DF(self, mat):
        """
        see DataFrame.get_Image_Dust
        :return:
        """
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
        # self.img.addMatrix(matrix)

    @measureTime
    def createText(self):
        raise NotImplementedError

    @measureTime
    def save(self):
        raise NotImplementedError
