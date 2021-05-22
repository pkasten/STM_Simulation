import os

from Particle import Particle
from Atom import Atom, Ag_Atom
import numpy as np
from Distance import Distance
import Configuration as cfg
import pickle
import random, math
import matplotlib.pyplot as plt


class Molecule(Particle):
    molecule_class = "CO2"

    def __init__(self, pos=None, theta=None, lookup_table=None, gitter=None):
        if pos is None:
            x = Distance(False, random.randint(0 - cfg.get_px_overlap(), cfg.get_width().px + cfg.get_px_overlap()))
            y = Distance(False, random.randint(0 - cfg.get_px_overlap(), cfg.get_height().px + cfg.get_px_overlap()))
            self.pos = np.array([x, y])
        else:
            self.pos = pos
        if theta is None:
            self.theta = 2 * math.pi * random.random()
        else:
            self.theta = theta

        super().__init__(self.pos[0], self.pos[1], self.theta)

        self.atoms = []
        self.img_w = cfg.get_width()
        self.img_h = cfg.get_height()
        self.lookup_table = lookup_table

        self.gitter = gitter
        if gitter is not None and lookup_table is None:
            self.lookup_table = self.crate_lookup(gitter)

        # if gitter is None and lookup_table is None:
        # print("WARNING: No Potential lookup")

        if self.molecule_class == "Single":
            self.atoms.append(Atom(np.array([0, 0])))
        elif self.molecule_class == "CO2":
            co_len = Distance(True, 1)
            self.atoms.append(Atom(np.array([0, 0])))
            self.atoms.append(Atom(np.array([-co_len.px, 0])))
            self.atoms.append(Atom(np.array([+co_len.px, 0])))
        elif self.molecule_class == "Star":
            co_len = Distance(True, 0.1)
            self.atoms.append(Atom(np.array([0, 0])))
            self.atoms.append(Atom(np.array([-co_len.px, 0])))
            self.atoms.append(Atom(np.array([+co_len.px, 0])))
            self.atoms.append(Atom(np.array([0, co_len.px])))
            self.atoms.append(Atom(np.array([0, -co_len.px])))

        distant_at = 0
        max_rad = 0
        for atom in self.atoms:
            if np.linalg.norm(atom.relpos) > distant_at:
                distant_at = np.linalg.norm(atom.relpos)
            x = atom.get_effect_range()
            if x > max_rad:
                max_rad = x
        # print("Distant Atom: {} and MaxRad = {}".format(distant_at, max_rad))
        self.effect_range = int(np.ceil(distant_at + max_rad))

        for atom in self.atoms:
            atom.calc_abs_pos(Distance.px_vec(self.pos), self.theta)

    def __str__(self):
        return self.molecule_class

    @staticmethod
    def str():
        return Molecule.molecule_class

    def get_dimension(self):
        if (len(self.atoms) == 0):
            return 0
        return Distance(False, np.sqrt(len(self.atoms) * self.atoms[0].radius.px))

    @DeprecationWarning
    def show(self, x, y):
        ret = 0
        for atom in self.atoms:
            ret += atom.show(x, y)
        return ret

    @DeprecationWarning
    def show_mat(self):
        mat = np.zeros((self.img_w.px, self.img_h.px))
        for i in range(self.img_w.px):
            for j in range(self.img_h.px):
                mat[i, j] = self.show(i, j)

        return mat

    def get_pot(self, gitter):
        pot = 0
        for atom in self.atoms:
            pot += atom.find_pot(gitter)
        return pot

    def crate_lookup(self, gitter):
        self.lookup_table = create_Molecule_lookup_table(gitter, Molecule)

    def gitter_pot(self):
        assert self.lookup_table is not None

        return self.lookup_table.get_nearest_Energy(self.pos, self.theta, self.gitter)[0]

    def efficient_Matrix(self):
        eff_matrix = np.zeros((2 * self.effect_range, 2 * self.effect_range))
        for i in range(-1 * self.effect_range, 1 * self.effect_range):
            for j in range(-1 * self.effect_range, 1 * self.effect_range):
                eff_matrix[i + 1 * self.effect_range, j + 1 * self.effect_range] = \
                    self.visualize_pixel(i, j)

        # for atom in self.atoms:
        #    print("Atom pos: {}".format(atom.abspos))
        return eff_matrix, self.x, self.y

    def efficient_Matrix_turned(self):
        return self.efficient_Matrix()

    def visualize_pixel(self, x, y):
        ret = 0
        for atom in self.atoms:
            vec = atom.abspos - Distance.px_vec(self.pos)
            ret += atom.show_rel(x - vec[0], y - vec[1])
            ret = min(255, ret)
        return ret


class Lookup_Table:

    # def log(self):
    #    log("Lookup_table: ")
    #    for pair in self.pairs:
    #        log("Pair {} :".format(pair.pos))
    #        pair.log()

    def __str__(self):
        s = "Lookup_Table: \n"
        for xc in self.table:
            s += str(xc)
        return s

    # def __str__(self):
    #    s = ""
    #    s += "Lookup_table: \n"
    #    for pair in self.pairs:
    #        s += "Pair {} :\n".format(pair.pos)
    #        s += pair.__str__()
    #    return s

    # class Pair:
    #    def __init__(self, pos):
    #       self.pos = pos
    #        self.ang_dict = {}

    # def log(self):
    #    for angle in self.ang_dict.keys():
    #        log("\t\t  {:.3f} : E= {:.3f}".format(angle, self.ang_dict[angle]))

    #   def __str__(self):
    #      s = ""
    #     for angle in self.ang_dict.keys():
    #        s += "\t\t  {:.3f} : E= {:.3f} \n".format(angle, self.ang_dict[angle])
    #   return s

    # def add(self, angle, energy):
    #   self.ang_dict[angle] = energy

    @staticmethod
    def equalVec(a, b):
        if len(a) != len(b):
            return False
        for i in range(len(a)):
            if a[i] != b[i]:
                return False
        return True

    def __init__(self, table, dist_step, angle_step):
        self.dist_step = dist_step
        self.angle_step = angle_step
        self.pairs = []
        self.table = table
        self.nn_dist = cfg.get_nn_dist()
        self.img_w = cfg.get_width()
        self.img_h = cfg.get_height()

    # def add(self, pos, angle, energy):
    #    exists = False
    #    for p in [p.pos for p in self.pairs]:
    #        if self.equalVec(p, pos):
    #            exists = True
    #            break
    #    if not exists:
    #        self.pairs.append(self.Pair(pos))

    #    for pair in self.pairs:
    #       if self.equalVec(pair.pos, pos):
    #          pair.add(angle, energy)
    #        return
    # print("Error")

    # def get_nearest_Energy(self, pos, angle, gitter):
    #
    #       if np.linalg.norm(pos) > self.nn_dist.px:  # ToDo: Add Lattice Options in settings
    #          pos = pos - Atom(pos).find_nearest_atom(gitter).pos
    #
    #       nearest = None
    #      min_d = np.infty
    #     for p in self.pairs:
    #        d = np.abs(np.linalg.norm(pos - p.pos) / self.dist_step)
    #
    #           if d < min_d:
    #               min_d = d
    #              nearest = p
    ##
    #   min_ang = np.infty
    #      nearest_energy = None
    #     nearest_ang = None
    #    if nearest is None:
    #          return None, None, None
    #       for a in nearest.ang_dict.keys():
    #          d = min(abs(a - angle), abs(2 * np.pi + a - angle))
    #
    #           if d < min_ang:
    #              min_ang = d
    #             nearest_ang = a
    #            nearest_energy = nearest.ang_dict[a]
    #
    #       return nearest_energy, min(abs(nearest_ang - angle), abs(2 * np.pi + nearest_ang - angle)), \
    #             np.linalg.norm(pos - nearest.pos)

    def get_nearest_Energy(self, pos, angle, gitter=None):

        if np.linalg.norm(pos) > self.nn_dist.px:  # ToDo: Add Lattice Options in settings
            pos = pos - Atom(pos).find_nearest_atom(gitter).pos
        x = pos[0]
        y = pos[1]
        ang = angle

        min_x = np.infty
        min_xc = None
        for xj in self.table:
            if abs(xj.x - x) < min_x:
                min_x = abs(xj.x - x)
                min_xc = xj


        min_y = np.infty
        min_yc = None
        for yj in min_xc.y_container:
            if abs(yj.y - y) < min_y:
                min_y = abs(yj.y - y)
                min_yc = yj

        min_a = np.infty
        min_an = None
        for aj in min_yc.ang_Contaienr:
            if min(abs(2*np.pi - aj.ang + ang), abs(ang - aj.ang)) < min_a:
                min_a = min(abs(2*np.pi - aj.ang + ang), abs(ang - aj.ang))
                min_an = aj

        energy = min_an.en

        return energy, min_a, np.sqrt(min_x**2 + min_y**2)

class X_Container:
        def __init__(self, x):
            self.x = x
            self.y_container = []

        def __str__(self):
            s = "X-Container x = {}\n".format(self.x)
            for yc in self.y_container:
                s += str(yc)

            return s

        def add(self, ycont):
            self.y_container.append(ycont)

class Y_Container:
        def __init__(self, y):
            self.y = y
            self.ang_Contaienr = []

        def __str__(self):
            s = "\tY-Container y = {}\n".format(self.y)
            for yc in self.ang_Contaienr:
                s += str(yc)

            return s

        def add(self, a_cont):
            self.ang_Contaienr.append(a_cont)

class Angle:
        def __init__(self, ang, en):
            self.ang = ang
            self.en = en

        def __str__(self):
            s = "\t\tA={:.2f} -> E={:.2f} \n".format(self.ang, self.en)
            return s


def create_Molecule_lookup_table(gitter, Testclass, name=None):

    nn_dist = cfg.get_nn_dist()
    diststeps = int(np.ceil(nn_dist.px/2))
    anglesteps = 180
    steps = diststeps * diststeps * anglesteps
    suff = ""
    if steps > 1000000:
        suff += str(int(np.floor(steps / 1000000))) + "M" + Testclass.molecule_class
    elif steps > 1000:
        suff += str(int(np.floor(steps / 1000))) + "K" + Testclass.molecule_class
    else:
        suff += str(steps) + Testclass.molecule_class

    if name is None:
        name = Testclass.str() + "_lookup" + suff
        print(name)

    if os.path.isfile(name + ".data"):
        table = pickle.load(open(name + ".data", "rb"))
        return table

    angle_step = 360 / anglesteps


    maxdist = (np.sqrt(0.75) - 0.25) * nn_dist.px

    def bog(deg):
        return deg / 180 * np.pi

    # Neu
    cover_range = int(np.ceil(2 * np.sqrt(2) * maxdist))
    dist_step = 2 * cover_range / diststeps
    xs = [i * dist_step - cover_range for i in range(diststeps)]
    ys = [i * dist_step - cover_range for i in range(diststeps)]

    angles = [bog(i * angle_step) for i in range(anglesteps)]

    table = []
    ct = 0
    pzt = 0
    for x in xs:
        ct += 1
        if int(np.ceil(100 * ct / len(xs))) > pzt:
            print("Progress: {}%".format(pzt))
            pzt = int(np.ceil(100 * ct / len(xs)))
        xc = X_Container(x)
        table.append(xc)
        for y in ys:
            yc = Y_Container(y)
            xc.add(yc)
            for theta in angles:
                yc.add(Angle(theta, Testclass(np.array([x, y]), theta).get_pot(gitter)))

    lt = Lookup_Table(table, dist_step, angle_step)

    print(lt)

    with open(name + ".data", "wb") as p:
        pickle.dump(lt, p)

    return lt


def create_gitter():
    ret = []
    nn_dist = cfg.get_nn_dist()
    img_h = cfg.get_height()
    img_w = cfg.get_width()

    gv_a = np.array([1, 0]) * nn_dist.px
    gv_b = np.array([np.cos(np.pi / 1.5), np.sin(np.pi / 1.5)]) * nn_dist.px

    start = np.array([nn_dist.px / 2, nn_dist.px / 2])
    current = np.array([0, 0])
    j_max = int(np.ceil(max(img_w.px / gv_b[0], img_h.px / gv_b[1])))
    i_max = int(np.ceil((img_w.px / gv_a[0]) - j_max * gv_b[0]))
    for i in range(i_max):
        for j in range(j_max):
            current = start + (gv_a * i) + (gv_b * j)
            if current[0] < img_w.px and current[1] < img_h.px and current[0] > 0 and current[1] > 0:
                ret.append(Ag_Atom(current.copy()))
    return ret


def test_Lookup_Table():
    gitter = create_gitter()
    img_h = cfg.get_height()
    img_w = cfg.get_width()
    table = create_Molecule_lookup_table(gitter, Molecule)
    for i in range(3):
        for j in range(3):
            for a in range(4):
                t = a * np.pi / 3
                x = 180 + i * 15
                y = 175 + j * 15

                vec_to_nearest = np.array([x, y]) - Atom(np.array([x, y])).find_nearest_atom(gitter).pos
                e, adist, pdist = table.get_nearest_Energy(vec_to_nearest, t, gitter)
                if e is None:
                    print("None")
                    continue
                print("Energy for {} {}, theta ={:.2f} is {:.2f} with Angerror {:.2f} and PosError {:.2f}".format(
                    x, y, t, e, adist, pdist
                ))

    m1 = Molecule(np.array([random.random() * img_w.px, random.random() * img_h.px]), 2 * np.pi * random.random(),
                  lookup_table=table, gitter=gitter)
    print("Molecule Energy: ")
    print(m1.gitter_pot())
