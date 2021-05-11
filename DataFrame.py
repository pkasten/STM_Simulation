import multiprocessing as mp
import copy, os
import math, random
import time

from Particle import Particle, Double_Particle
from Images import MyImage
#from Maths.Functions import measureTime
#from Configuration.Files import MultiFileManager as fm
import Configuration as cfg
import numpy as np
import matplotlib.pyplot as plt
from Functions import measureTime
from My_SXM import My_SXM
import scipy.optimize as opt
#from Doubled import Double_Frame
from Charge import Charge


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
        self.double_tip_poss = cfg.get_double_tip_possibility()
        self.passed_args = None
        self.img_width = cfg.get_width()
        self.img_height = cfg.get_height()
        self.use_crystal_orientations = cfg.get_crystal_orientation_usage()
        self.crystal_directions_num = cfg.get_no_of_orientations()
        self.crystal_directions = cfg.get_crystal_orientations_array()
        if self.use_crystal_orientations:
            self.angle_char_len = 4000
        self.potential_map = self.calc_potential_map()
        self.overlapping_energy = 1000
        self.overlapping_threshold = cfg.get_overlap_threshold()
        self.part_laenge = cfg.get_part_length()
        self.oldPotential = []
        self.add_To_Potential = []

    #returns iterator over Particles
    def getIterator(self):
        return self.objects

    #gets number of particles
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

    #checks wheather part overlaps any existing particle
    def _overlaps_any(self, part):
        #start = time.perf_counter()

        if len(self.objects) == 0:
            #print("Overlaps any took {}".format(time.perf_counter() - start))
            return False
        for p in self.objects:
            if not part.dragged and not p.dragged:
                if math.dist([p.x, p.y], [part.x, part.y]) > max(part.effect_range, p.effect_range):
                    continue
            if part.true_overlap(p):
                #print("Overlaps any took {}".format(time.perf_counter() - start))
                return True
        #print("Overlaps any took {}".format(time.perf_counter() - start))
        return False

    #returns random particle, that does not overlap with any other
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
                #print("Retry")
                p = Particle()
            else:
                return p
        #print("MaxTries Exhausted_a")
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
                #print("Retry_b")
                p = _set_p()
            else:
                return p
        #print("MaxTries Exhausted_b")
        return p

    # calculates particles weight for importance in surrounding particles
    def _calc_angle_weight(self, part1, part2): # ToDo: Still very sketchy
        drel = part1.get_distance_to(part2) / self.max_dist
        #return np.exp(-self.angle_char_len/drel)
        return self.angle_char_len/drel

    def _orients_along_crystal(self, particle):
        return self._bonding_strength_crystal(particle) < 0.5

    # High if bonds tu particles, low if bonds to crystal
    def _bonding_strength_crystal(self, particle): #ToDo: improve
        if len(self.objects) == 0:
            return 0
        for part in self.objects:
            dis = part.get_distance_to(particle)
            if dis < self.max_dist/np.sqrt(2) * 0.2:
                return 1
        return 0


    # calculates a random angle for partilce depending on its surrounding with correlation
    def _calc_angle_for_particle(self, particle): # ToDo: Still very sketchy

        if self.use_crystal_orientations:
            if self._orients_along_crystal(particle):
                #particle.set_height(0.7)
                #print(self.crystal_directions)
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


    def addParticles(self, amount=None, coverage=None, overlapping=False, maximum_tries=1000):

        for i in range(cfg.get_particles_per_image()):
            print(i)
            self.add_at_optimum_energy_new(self.img_width * random.random(), self.img_height * random.random(), 2*np.pi*random.random())

        #widthout angle correlation
        self.passed_args = (amount, coverage, overlapping, maximum_tries)
        #print("{}, {}, {}".format(self.use_range, self.angle_char_len, overlapping))
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
                #w/ angle, w overlapping
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
            #w/ angle correlation
            else:
                if not overlapping:
                    if amount is not None:
                        for i in range(amount):
                            if random.random() < self.dragging_possibility:
                                p = self.get_dragged_that_mot_overlaps(maximum_tries, setangle=True)
                                self.objects.append(p)
                            else:
                                p = self._get_thatnot_overlaps(maximum_tries, calcangle=True)
                                #p.set_theta(self._calc_angle_for_particle(p))
                                self.objects.append(p)
                    elif coverage is not None:
                        while self.coverage() < coverage:
                            if random.random() < self.dragging_possibility:
                                p = self.get_dragged_that_mot_overlaps(maximum_tries, setangle=True)
                                self.objects.append(p)
                            else:
                                p = self._get_thatnot_overlaps(maximum_tries, calcangle=True)
                                #p.set_theta(self._calc_angle_for_particle(p))
                                self.objects.append(p)
                    else:
                        #print("Normal") Normaldurchlauf
                        for i in range(cfg.get_particles_per_image()):
                            if random.random() < self.dragging_possibility:
                                p = self.get_dragged_that_mot_overlaps(maximum_tries, setangle=True)
                                self.objects.append(p)
                            else:
                                p = self._get_thatnot_overlaps(maximum_tries, calcangle=True)
                                #p.set_theta(self._calc_angle_for_particle(p))
                                self.objects.append(p)
                #w/ angle, w overlapping
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
        #use angle range
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
            #w/ angle, w/ overlapping
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
        width = self.img_width
        height = self.img_height
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
        #self.img.updateImage()

    def calc_potential_map(self):
        start = time.perf_counter()
        try:
            print(self.oldPotential[0,1])
        except TypeError:
            self.oldPotential = None
        if self.oldPotential is not None:
            pot = self.oldPotential
            for charge in self.add_To_Potential:
                for i in range(self.img_width):
                    for j in range(self.img_height):
                        pot[i, j] += charge.calc_Potential(i, j)

            self.add_To_Potential = []
            self.oldPotential = pot
            print("Short End at {}".format(time.perf_counter() - start))
            plt.imshow(pot)
            plt.show()
            return pot

        #if len(self.objects) == 0:
        #    return np.zeros((self.img_width, self.img_height))

        #print("Calcing pot map")
        charges = []

        for i in range(0,self.img_width, 60):
            for j in range(0, self.img_height, 60):
                charges.append(Charge(i, j, 0.5 * (-1)**(i+j)))
        pot = np.zeros((self.img_width, self.img_height))

        #for i in range(self.img_width):
        #    for j in range(self.img_height):
        #        pot[i, j] += ((i - self.img_width/2)**2 + (j-self.img_height/2)**2 )/((self.img_height/2)**2)


        for part in self.objects:
            for q in part.get_charges():
                #print("Anzahl an charges: {}".format(len(part.get_charges())))
                charges.append(q)

        #print("Creating map")
        #print("len(Charges): {}".format(len(charges)))
        #print("len(objy): {}".format(len(self.objects)))
        sdf = 0
        for q in charges:
            print(sdf)
            sdf+=1
            for i in range(self.img_width):
                for j in range(self.img_height):

                    #print(i, j)
                    pot[i, j] += q.calc_Potential(i, j)

        #self.get_Image()

        #mat = self.img.get_matrix()
        #plt.imshow(mat)
        #plt.show()
        #for i in range(self.img_width):
        #    for j in range(self.img_height): #TODO: Check if Overlaps
        #        if mat[i, j] > self.overlapping_threshold:
        #            pot[i, j] = self.overlapping_energy

        #print("Returning map")

        #plt.imshow(self.img.get_matrix())
        #plt.show()
        print("Long End at {}".format(time.perf_counter() - start))
        self.oldPotential = pot
        plt.imshow(pot)
        plt.show()
        return pot

    def calc_pot_Energy_for_particle(self, part, mapchange=False):
        print("Deprecated 4641635")
        if mapchange or self.potential_map is None:
            self.potential_map = self.calc_potential_map()
        charges = part.get_charges()
        #pot_map = self.calc_potential_map()
        e_pot = 0
        for charge in charges:
            try:
                e_pot += charge.q * self.potential_map[int(charge.x), int(charge.y)] #ToDo: Not round but scale by surrrounding
            except IndexError:
                print("calc_pot_Energy_for_particle for x={}, y={}".format(int(charge.x), int(charge.y)))

        e_pot += self.is_overlapping(part)

        return e_pot

    def is_overlapping(self, part):

        for p in self.objects:
            if p.true_overlap(part):
                return self.overlapping_energy
        return 0

    def energy_function(self, x, y, theta):
        #Only for diploe
        lenge = self.part_laenge
        x_plus = int(x + 0.5 * lenge * np.sin(theta))
        y_plus = int(y + 0.5 * lenge * np.cos(theta))
        x_minus = int(x - 0.5 * lenge * np.sin(theta))
        y_minus = int(y - 0.5 * lenge * np.cos(theta)) #ToDo: nicht int sonders berechnen
        #if len(x) > 1:
        #    e = []
        #    for i in range(len(x)):
        #        e.append(self.potential_map[x_plus[i], y_plus[i]] - self.potential_map[x_minus[i], y_minus[i]])
        if self._overlaps_any(Particle(x, y, theta)): #ToDo Readd
            return self.overlapping_energy
        if(x_plus < 0 or x_minus < 0 or y_minus < 0 or y_plus < 0):
            return self.overlapping_energy
        try:
            a1 = self.potential_map[x_plus, y_plus]
            a2 = - self.potential_map[x_minus, y_minus]
            #if np.abs(a1) > self.overlapping_energy / 2 or np.abs(a2) > self.overlapping_energy / 2:
            #    return self.overlapping_energy
            #else:
            #print(a1 - a2)
            #print(a1)
            #print(a2)
            return a1 + a2
        except IndexError:
            return self.overlapping_energy


    def opimizable_energy_function(self, x):
        return self.energy_function(x[0], x[1], x[2])


    def add_at_optimum_energy(self, initvals=None): #ToDo: Include crystal structure

        #if len(self.objects) == 0:
        #    print("Hi")
        #    self.objects.append(Particle(self.img_width/2, self.img_height/2, 0))
        #    self.potential_map = self.calc_potential_map()
        #    return

        #SCIPY
        self.potential_map = self.calc_potential_map()
        #initvasl = [self.img_width * random.random(), self.img_height * random.random(), 2 * np.pi * random.random()]
        if initvals is None:
            initvasl = [200, 200, np.pi]
        else:
            initvasl = initvals
        #print(fmin(self.opimizable_energy_function, np.array([200,100,0])))
        vals = opt.fmin(self.opimizable_energy_function, initvasl)
        print(vals)
        p = Particle(vals[0], vals[1], vals[2])
        self.objects.append(p)
        self.add_To_Potential.append(p)
        self.potential_map = self.calc_potential_map()
        #plt.imshow(self.potential_map)
        #plt.show()

        #plt.imshow(self.potential_map)
        #plt.show()

        #x = 0
        #y = 0
        #theta = 0
        #p = Particle(x, y, theta)
        #dx = 80
        #dtheta = np.pi / 2
        #dy = 80
        #thetas = []
        #energies = []
        #for a in range(4):
        #    thetas.append(a * dtheta)

        #for x in range(0, self.img_width, dx):
        #    for y in range(0, self.img_height, dy):
        #        for theta in thetas:
        #            p.set_x(x)
        #            p.set_y(y)
        #            p.set_theta(theta)
        #            e_pot = self.calc_pot_Energy_for_particle(p, mapchange=False)
        #            #
        #            print(e_pot, x, y, theta)
        #            energies.append((e_pot, x, y, theta))

        #print("Calced Es")
        #minimum_e = 15614561
        #min_args = 0,0,0,0
        #for e in energies:
         #   if e[0] < minimum_e:
         #       min_args = e
         #       minimum_e = e[0]

        #p.set_x(min_args[1])
        #p.set_y(min_args[2])
        #p.set_theta(min_args[3])
        #self.objects.append(p)
        #print("Appended at min: {}".format(min_args))#

        #self.potential_map = self.calc_potential_map()
        #print("Calced Map")

    def add_at_optimum_energy_new(self, x_start, y_start, theta_start):

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
                    #print("Charge {} at ({},{}) adds Energy {}".format(c.q, c.x, c.y, c.q * self.potential_map[int(c.x), int(c.y)]))
                    e += c.q * self.potential_map[int(c.x), int(c.y)]
                except IndexError:
                    #print("Index Error")
                    e += self.overlapping_energy
                    continue
            #print("Energy: {}".format(e))
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
                    #print("Index Error")
                    continue

            return e

        self.potential_map = self.calc_potential_map()
        #plt.imshow(self.potential_map)
        #plt.show()
        vals = opt.fmin(energyxyt, [x_loc, y_loc, theta_loc])
        p = Particle(vals[0], vals[1], vals[2])



        for i in range(10):
            if self._overlaps_any(p):
                vals = opt.fmin(energyxyt, [x_loc, y_loc, theta_loc])
                p = Particle(vals[0], vals[1], vals[2])
            else:
                break
        self.objects.append(p)
        self.add_To_Potential.append(p)
        self.potential_map = self.calc_potential_map()
        #plt.imshow(self.potential_map)
        #plt.show()









    def create_Image_Visualization(self):
        self.img = MyImage()
        width = self.img_width
        height = self.img_height
        matrix = np.zeros((width, height))

        for part in self.objects:
            for tuple in part.get_visualization():
                #print("Tupel:{}".format(tuple))
                eff_mat, x, y = tuple
                mat_w = eff_mat.shape[0]

                #ToDo: possible failure
                x = int(np.round(x))
                y = int(np.round(y))
                #plt.imshow(eff_mat)
                #plt.show()
                #print(np.max(eff_mat))
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
        print("Deprecated 6546335416")
        for part in self.objects:
            if random.random() < self.dragging_possibility:
                part.drag(self.dragging_speed, self.raster_angle)

    def get_Image(self):
        if random.random() < self.double_tip_poss:
            print("Double Tipping")
            strength = 0.3 + 0.5 * random.random()
            #print(strength)
            rel_dist = 0.1 * random.random() #ToDO: Let loose
            angle = 2 * np.pi * random.random()
            #angle = 0
            doubled_frame = Double_Frame(self.fn_gen, strength, rel_dist, angle)
            #print("Created Double Frame")
            doubled_frame.addParticles(self.passed_args[0], self.passed_args[1], self.passed_args[2], self.passed_args[3])
            #print("added Particles")
            #if self.use_dragging:
            #    doubled_frame._drag_particles()
            self.img = doubled_frame.extract_Smaller()
            #print("extracted Smaller")
            self.objects = doubled_frame.get_objects()
            #print("Got objects")
            if self.use_noise:
                self.img.noise(self.image_noise_mu, self.image_noise_sigma)
            self.img.updateImage()
            #print("updated Image")

            return

        #if self.use_dragging: #ToDo: Possibly later
        #    self._drag_particles()
        self.create_Image_Visualization()

        #if random.random() < self.double_tip_poss:
        #    surrounding_frames = []
        #    for i in range(4):
        #        df_a = DataFrame(self.fn_gen)
        #        df_a.addParticles(self.passed_args[0], self.passed_args[1], self.passed_args[2], self.passed_args[3])
        #        df_a.double_tip_poss = 0
        #        df_a.create_Image_Visualization()
        #        surrounding_frames.append(df_a.img.colors)
        #    self.img.double_tip(0.3, 0.1, 2 * np.pi * random.random(), surrounding_frames)
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
            #self.createImage_efficient()
            self.get_Image()
        if len(self.text) == 0:
            self.createText()
        img_path, dat_path, sxm_path, index = self.fn_gen.generate_Tuple()
        #print("Saving: {}".format(index))
        #print("Saving No {}".format(index))
        try:
            with open(dat_path, "w") as dat_file:
                dat_file.write(self.text)
        except FileNotFoundError:
            os.mkdir(cfg.get_data_folder())
            with open(dat_path, "w") as dat_file:
                dat_file.write(self.text)
        self.img.saveImage(img_path)
        My_SXM.write_sxm(sxm_path, self.img.get_matrix())
        #if self.has_overlaps():
            #print("Overlaps detected @ {}".format(index))


    def hasPoints(self):
        # return not self.points.empty()
        return len(self.objects) > 0

    def coverage(self):
        area = self.area
        covered = 0
        for part in self.objects:
            covered += np.pi * np.square(part.get_dimension())
        return covered/area

    def has_overlaps(self):
        for i in range(len(self.objects)):
            for j in range(i):
                #print("Testing overlap {} - {}".format(i, j))
                if self.objects[i].true_overlap(self.objects[j]):
                    #print("Testing overlap {} - {}".format(i, j))
                    #print("i: x={}, y={}, dg={}; j: x={}, y={}, dg={}".format(self.objects[i].get_x(), self.objects[i].get_y(), self.objects[i].dragged, self.objects[j].get_x(), self.objects[j].get_y(), self.objects[j].dragged))
                    return True
        return False

    def __str__(self):
        return str(self.objects)


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
                self.range = int(self.img_width/2), self.img_width, int(self.img_height/2), self.img_height
            else:
                self.range = int(self.img_width/2), self.img_width, 0, int(self.img_height/2)
        else:
            if self.shift_y > 0:
                self.range = 0, int(self.img_width/2), 0, int(self.img_height/2), self.img_height
            else:
                self.range = 0, int(self.img_width/2), 0, int(self.img_height/2)


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
        #print("DTNO @len {}".format(len(self.objects)))
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
        #print("TNO @len {}".format(len(self.objects)))
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
        #print("Got Args: {}".format(self.passed_args))
        if not self.use_range:
            if self.angle_char_len == 0:
                if not overlapping:
                    if amount is not None:
                        for i in range(4*amount):
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
                        for i in range(4*cfg.get_particles_per_image()):
                            if random.random() < self.dragging_possibility:
                                p = self.get_dragged_that_mot_overlaps(maximum_tries)
                                self.objects.append(p)
                            else:
                                p = self._get_thatnot_overlaps(maximum_tries)
                                self.objects.append(p)
                # w/ angle, w overlapping
                else:
                    if amount is not None:
                        for i in range(4*amount):
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
                        for i in range(4*cfg.get_particles_per_image()):
                            p = Double_Particle()
                            if random.random() < self.dragging_possibility:
                                p.drag(self.dragging_speed, self.raster_angle)
                            self.objects.append(p)
            # w/ angle correlation
            else:
                if not overlapping:
                    if amount is not None:
                        for i in range(4*amount):
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
                        for i in range(4*cfg.get_particles_per_image()):
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
                        for i in range(4*amount):
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
                        for i in range(4*cfg.get_particles_per_image()):
                            p = Double_Particle()
                            p.set_theta(self._calc_angle_for_particle(p))
                            if random.random() < self.dragging_possibility:
                                p.drag(self.dragging_speed, self.raster_angle)
                            self.objects.append(p)
        # use angle range
        else:
            if not overlapping:
                if amount is not None:
                    for i in range(4*amount):
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
                    for i in range(4*cfg.get_particles_per_image()):
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
                    for i in range(4*amount):
                        p = Double_Particle()
                        p.set_theta(self._random_angle_range())
                        self.objects.append(p)
                elif coverage is not None:
                    while self.coverage() < coverage:
                        p = Double_Particle()
                        p.set_theta(self._random_angle_range())
                        self.objects.append(p)
                else:
                    for i in range(4*cfg.get_particles_per_image()):
                        p = Double_Particle()
                        p.set_theta(self._random_angle_range())
                        self.objects.append(p)

        #return
        # widthout angle correlation
        #self.passed_args = (amount, coverage, overlapping, maximum_tries)
        #if not self.use_range:
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
        self.img.setWidth(self.img_width)
        self.img.setHeight(self.img_height)
        #print("W: {} - {}".format(self.img_width, self.img.getWidth()))
        #print("H: {} - {}".format(self.img_height, self.img.getHeight()))

        width = self.img_width
        height = self.img_height
        matrix = np.zeros((width, height))

        for part in self.objects:
            for tuple in part.get_visualization():
                #print("Tupel:{}".format(tuple))
                eff_mat, x, y = tuple
                mat_w = eff_mat.shape[0]

                #ToDo: possible failure
                x = int(np.round(x))
                y = int(np.round(y))
                #plt.imshow(eff_mat)
                #plt.show()
                #print(np.max(eff_mat))
                mat_h = eff_mat.shape[1]
                for i in range(mat_w):
                    for j in range(mat_h):
                        new_x = x - math.floor((mat_w / 2)) + i
                        new_y = y - math.floor(mat_h / 2) + j
                        if not (0 <= new_x < width and 0 <= new_y < height):
                            continue
                        matrix[new_x, new_y] += eff_mat[i, j]
        #print("img : {}".format(np.shape(self.img.get_matrix())))
        #print("matrix: {}".format(np.shape(matrix)))
        #print("Matrix")
        #plt.imshow(matrix)
        #plt.show()
        self.img.addMatrix(matrix) #Indentd  too far right

    def extract_Smaller(self):
        self.create_Image_Visualization()
        #print("Start extractSmaller")
        #plt.imshow(self.img.get_matrix())
        #plt.show()

        self.img.double_tip(self.strength, self.rel_dist, self.dt_angle)

        smaller = np.zeros((cfg.get_width(), cfg.get_height()))
        bigger = self.img.get_matrix()
        #print(np.shape(smaller), np.shape(bigger))
        #print(self.range)
        for x in range(np.shape(smaller)[0]):
            for y in range(np.shape(smaller)[1]):
                x_tilt = x + self.range[0] - 1
                y_tilt = y + self.range[2] - 1
                smaller[x, y] = bigger[x_tilt, y_tilt]
        #plt.imshow(smaller)
        #plt.show()
        #plt.imshow(bigger)
        #plt.show()
        return MyImage(smaller)

    def get_objects(self):
        ret = []
        for part in self.objects:
            if self.range[0] - self.overlap <= part.get_x() <= self.range[1] + self.overlap and \
                self.range[2] - self.overlap <= part.get_y() <= self.range[3] + self.overlap:
                ret.append(part)

        return ret

    def get_Image(self):
        raise NotImplementedError

    def createText(self):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError



