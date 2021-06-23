import os
import time

from Particle import Particle
from Atom import Atom, Ag_Atom
import numpy as np
from Distance import Distance
import Configuration as cfg
import pickle
import random, math
import matplotlib.pyplot as plt


class Molecule(Particle):
    molecule_class = "NCPhCN"

    # molecule_ph_groups = 3
    # molecule_style = "Complex"

    def __init__(self, pos=None, theta=None, lookup_table=None, gitter=None, molecule_class=None, molecule_ph_groups=0,
                 style=None):
        if molecule_ph_groups > 0:
            self.molecule_ph_groups = molecule_ph_groups
        else:
            self.molecule_ph_groups = random.randint(1, 5)
        if style is not None:
            self.molecule_style = style  # "Complex" "Simple"
        else:
            self.molecule_style = cfg.get_molecule_style()

        if molecule_class is not None:
            self.molecule_class = molecule_class

        # self.molecule_ph_groups = random.randint(1, 5)
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

        # self.ch_len_def = Distance(True, 1.06)
        # self.cn_len_def = Distance(True, 1.47) # Min
        # self.cc_len_def = Distance(True, 1.20)

        # self.ch_len_def = Distance(True, 1.08)
        # self.cn_len_def = Distance(True, 1.78) # Mid
        # self.cc_len_def = Distance(True, 1.37)

        # Add outer molecules Radius to Molecule width and height
        self.add_outer_atomradii = False

        self.ch_len_def = Distance(True, 1.07)
        self.cn_len_def = Distance(True, 1.155)  # Lit
        self.cc_len_def = Distance(True, 1.40)

        self.gitter = gitter
        if gitter is not None and lookup_table is None:
            self.crate_lookup(gitter)

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
            co_len = Distance(True, 1)
            self.atoms.append(Atom(np.array([0, 0])))
            self.atoms.append(Atom(np.array([-co_len.px, 0])))
            self.atoms.append(Atom(np.array([+co_len.px, 0])))
            self.atoms.append(Atom(np.array([0, co_len.px])))
            self.atoms.append(Atom(np.array([0, -co_len.px])))
        elif self.molecule_class == "NCPhCN":
            # ch_len = Distance(True, 1.08)
            ch_len = self.ch_len_def
            cn_len = self.cn_len_def
            cc_len = self.cn_len_def
            # cn_len = Distance(True, 1.80)
            # cc_len = Distance(True, 1.37)
            self.create_NC_PhX_CN(self.molecule_ph_groups, cc_len, ch_len, cn_len)
        else:
            self.atoms.append(Atom(np.array([0, 0])))

        # n of Atoms: {}".format(len(self.atoms)))

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

        self.molecule_style.lower()

        if self.molecule_style == "simple":

            self.simple_length = self.get_simple_length(self.cc_len_def, self.cn_len_def)
            self.simple_width = (2 * self.ch_len_def + 2 * self.cc_len_def) * np.cos(np.pi / 12)
            if self.add_outer_atomradii:
                self.simple_width += 2 * Distance(True, 1.06)
            # print("Simple Length = {}, Simple Width = {}".format(self.simple_length.ang, self.simple_width.ang))
            super().set_width(self.simple_width)
            super().set_length(self.simple_length)

            # print("Molecule x={} has l={:.4f} and w={:.4f}".format(self.molecule_ph_groups, self.simple_length.ang, self.simple_width.ang))

    def __str__(self):
        if self.molecule_class == "NCPhCN":
            return "NCPh{}CN".format(self.molecule_ph_groups)
        return self.molecule_class

    def __repr__(self):
        return "Molecule {} (x={})".format(self.molecule_class, self.molecule_ph_groups)

    def get_C6_Ringdist(self, cc_dist):
        return 2 * cc_dist + Distance(True, 1.52)

    def get_simple_length(self, cc_dist=Distance(True, 1.20), cn_dist=Distance(True, 1.47)):
        n = self.molecule_ph_groups
        if n % 2 == 0:
            amnt = int(n / 2)
            ringdist = self.get_C6_Ringdist(cc_dist)
            c_dist = ringdist / 2 + (amnt - 1) * ringdist + cc_dist + cc_dist
            n_dist = c_dist + cn_dist
            if self.add_outer_atomradii:
                n_dist += Distance(True, 1.06)
            return 2 * n_dist
        else:
            amnt = int((n - 1) / 2)
            ringdist = self.get_C6_Ringdist(cc_dist)
            c_dist = amnt * ringdist + cc_dist + cc_dist
            n_dist = c_dist + cn_dist
            return 2 * n_dist

    def add_C6_Ring(self, center, cc_dist, ch_dist):
        # print(type(center))
        # assert center is type(np.array([0, 0]))
        assert len(center) == 2
        deg60 = np.pi * 60 / 180
        deg120 = np.pi * 120 / 180

        center_px = center
        cc_dist_px = cc_dist.px
        ch_dist_px = ch_dist.px

        posis = []
        # print("Center: {}".format(center_px))
        posis.append(center_px - np.array([cc_dist_px, 0]))
        posis.append(center_px + cc_dist_px * np.array([-np.cos(deg60), np.sin(deg60)]))
        posis.append(center_px + cc_dist_px * np.array([-np.cos(deg120), np.sin(deg120)]))
        posis.append(center_px + cc_dist_px * np.array([1, 0]))
        posis.append(center_px - cc_dist_px * np.array([np.cos(deg60), np.sin(deg60)]))
        posis.append(center_px - cc_dist_px * np.array([np.cos(deg120), np.sin(deg120)]))

        for pos in posis:
            self.atoms.append(Atom(pos, "C"))

        posis = []

        newdist = cc_dist_px + ch_dist_px
        posis.append(center_px + newdist * np.array([-np.cos(deg60), np.sin(deg60)]))
        posis.append(center_px + newdist * np.array([-np.cos(deg120), np.sin(deg120)]))
        posis.append(center_px - newdist * np.array([np.cos(deg60), np.sin(deg60)]))
        posis.append(center_px - newdist * np.array([np.cos(deg120), np.sin(deg120)]))

        for pos in posis:
            self.atoms.append(Atom(pos, "H"))

    def create_NC_PhX_CN(self, n, cc_dist, ch_dist, cn_dist):
        if n % 2 == 0:
            amnt = int(n / 2)
            # print("Amount: {}".format(amnt))
            ringdist = self.get_C6_Ringdist(cc_dist).px
            dist = ringdist / 2
            for i in range(amnt):
                self.add_C6_Ring(np.array([-dist, 0]), cc_dist, ch_dist)
                self.add_C6_Ring(np.array([dist, 0]), cc_dist, ch_dist)
                dist += ringdist
            c_dist = ringdist / 2 + (amnt - 1) * ringdist + cc_dist.px + cc_dist.px
            n_dist = c_dist + cn_dist.px
            self.atoms.append(Atom(np.array([-c_dist, 0]), "C"))
            self.atoms.append(Atom(np.array([c_dist, 0]), "C"))
            self.atoms.append(Atom(np.array([-n_dist, 0]), "N"))
            self.atoms.append(Atom(np.array([n_dist, 0]), "N"))

        else:
            amnt = int((n - 1) / 2)
            ringdist = self.get_C6_Ringdist(cc_dist).px
            dist = ringdist
            self.add_C6_Ring(np.array([0, 0]), cc_dist, ch_dist)
            for i in range(amnt):
                self.add_C6_Ring(np.array([-dist, 0]), cc_dist, ch_dist)
                self.add_C6_Ring(np.array([dist, 0]), cc_dist, ch_dist)
                dist += ringdist
            c_dist = amnt * ringdist + cc_dist.px + cc_dist.px
            n_dist = c_dist + cn_dist.px
            self.atoms.append(Atom(np.array([-c_dist, 0]), "C"))
            self.atoms.append(Atom(np.array([c_dist, 0]), "C"))
            self.atoms.append(Atom(np.array([-n_dist, 0]), "N"))
            self.atoms.append(Atom(np.array([n_dist, 0]), "N"))

    # @staticmethod
    # def str():
    #    return Molecule.molecule_class

    def str(self):
        return self.__str__()

    def color(self, h):
        assert len(self.atoms) > 0
        return self.atoms[0].color(h)

    def set_maxHeight(self, max_h):
        self.max_height = max_h
        for x in self.atoms:
            x.set_maxHeight(max_h)

    def get_dimension(self):
        if (len(self.atoms) == 0):
            return 0
        return Distance(False, np.sqrt(len(self.atoms) * self.atoms[0].radius.px))

    # @DeprecationWarning
    def show(self, x, y):
        ret = 0
        for atom in self.atoms:
            ret += atom.show(x, y)
        return ret

    # @DeprecationWarning
    def show_mat(self):
        print("Deprecated 73289474329")
        mat = np.zeros((int(np.ceil(self.img_w.px)), int(np.ceil(self.img_h.px))))
        for i in range(int(np.ceil(self.img_w.px))):
            for j in range(int(np.ceil(self.img_h.px))):
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
        path = os.path.join(os.getcwd(), "Pickle_Data",
                            "Molec{}Vis{:.2f}_{:.2f}".format(self.molecule_ph_groups, cfg.get_px_per_angstrom(),
                                                             cfg.get_fermi_exp()))
        if os.path.isfile(path):
            with open(path, "rb") as pth:
                return pickle.load(pth), self.x, self.y
        eff_matrix = np.zeros((2 * self.effect_range, 2 * self.effect_range))
        for i in range(-1 * self.effect_range, 1 * self.effect_range):
            for j in range(-1 * self.effect_range, 1 * self.effect_range):
                eff_matrix[i + 1 * self.effect_range, j + 1 * self.effect_range] = \
                    self.visualize_pixel(i, j)

        if not os.path.isfile(path):
            with open(path, "wb") as pth:
                pickle.dump(eff_matrix, pth)

        # for atom in self.atoms:
        #    print("Atom pos: {}".format(atom.abspos))
        return eff_matrix, self.x, self.y

    def efficient_Matrix_turned(self):
        if self.molecule_style == "simple":
            return super().efficient_Matrix_turned()

        return self.efficient_Matrix()


    def visualize_pixel(self, x, y):
        if self.molecule_style == "simple":
            return super().visualize_pixel(x, y)

        mode = "MAX"
        if mode == "ADD":
            ret = 0
            for atom in self.atoms:
                vec = atom.abspos - Distance.px_vec(self.pos)
                ret += atom.show_rel(x - vec[0], y - vec[1])
                ret = min(255, ret)
            return ret
        if mode == "MAX":
            ret = 0
            for atom in self.atoms:
                vec = atom.abspos - Distance.px_vec(self.pos)
                ret = max(ret, atom.show_rel(x - vec[0], y - vec[1]))
                ret = min(255, ret)
            return ret


class Lookup_Table:

    def __str__(self):
        s = "Lookup_Table: \n"
        for xc in self.table:
            s += str(xc)
        return s

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
            if min(abs(2 * np.pi - aj.ang + ang), abs(ang - aj.ang)) < min_a:
                min_a = min(abs(2 * np.pi - aj.ang + ang), abs(ang - aj.ang))
                min_an = aj

        energy = min_an.en

        return energy, min_a, np.sqrt(min_x ** 2 + min_y ** 2)


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
    diststeps = int(np.ceil(nn_dist.px / 2))
    anglesteps = 180
    steps = diststeps * diststeps * anglesteps
    suff = ""
    if steps > 1000000:
        suff += str(int(np.floor(steps / 1000000))) + "M"
    elif steps > 1000:
        suff += str(int(np.floor(steps / 1000))) + "K"
    else:
        suff += str(steps) + Testclass.molecule_class

    if name is None:
        name = str(Testclass()) + "_lookup" + suff
        print("New Name: " + name)

    if os.path.isfile(os.path.join("Pickle_Data", name + ".data")):
        table = pickle.load(open(os.path.join("Pickle_Data", name + ".data"), "rb"))
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

    # print(lt)

    with open(os.path.join("Pickle_Data", name + ".data"), "wb") as p:
        pickle.dump(lt, p)

    return lt


class Tests_Gitterpot:

    @staticmethod
    def test():

        all = False
        print("Test lookup Table:")
        Tests_Gitterpot.test_Lookup_Table()

        print("Show Things")

        gitter = Tests_Gitterpot.create_gitter()
        showmat = 0.5 * Tests_Gitterpot.show_gitter(gitter, True)
        showmat = Tests_Gitterpot.normalize_matrix(showmat)
        if all:
            print("Gitter:")
            plt.imshow(showmat)
            plt.show()

        test = Molecule(np.array([479, 380]), 45 / 180 * np.pi)

        tsm = Tests_Gitterpot.normalize_matrix(test.show_mat())  # ToDo EffMat scaled
        if all:
            print("Molecule:")
            plt.imshow(tsm)
            plt.show()
            # tsm = test.efficient_Matrix_turned()[0]

        nearestats = []
        for atom in test.atoms:
            arr = []
            for elem in atom.find_nearest_atoms(gitter):
                arr.append(elem)
            nearestats.append(arr)

        # print(len(nearestats))
        nearestshw = []
        for arr in nearestats:
            nearestshw.append(Tests_Gitterpot.show_gitter(arr, False))
            # print("Nearestshow show gitter")
            # plt.imshow(Tests_Gitterpot.show_gitter(arr, False))
            # plt.show()

        nearestshow = nearestshw[0]
        for i in range(1, len(nearestshw)):
            nearestshow += nearestshw[i]
        nearestshow = Tests_Gitterpot.normalize_matrix(nearestshow)
        if all:
            print("Sum of Nearestshow")
            plt.imshow(nearestshow)
            plt.show()

        mdp = []
        for arr in nearestats:
            mdp.append(Atom.find_mid_of_nearest(arr))

        # midpoint = mdp[0].show_mat()
        # for i in range(1, len(mdp)):
        #    midpoint += mdp[i].show_mat()

        # midpoint = Tests_Gitterpot.normalize_matrix(midpoint)

        mat = tsm + showmat + nearestshow

        print("Sum of all")
        plt.imshow(mat)
        plt.show()

    @staticmethod
    def normalize_matrix(mat):
        maxi = np.max(mat)
        return mat / maxi

    @staticmethod
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

    @staticmethod
    def create_larger_gitter():
        ret = []
        nn_dist = cfg.get_nn_dist()
        img_h = cfg.get_height()
        img_w = cfg.get_width()

        gv_a = np.array([1, 0]) * nn_dist.px
        gv_b = np.array([np.cos(np.pi / 1.5), np.sin(np.pi / 1.5)]) * nn_dist.px

        start = np.array([0, 0]) - 2 * gv_a - 2 * gv_b
        current = np.array([0, 0])
        j_max = int(np.ceil(max(img_w.px / gv_b[0], img_h.px / gv_b[1]))) + 4
        i_max = int(np.ceil((img_w.px / gv_a[0]) - j_max * gv_b[0])) + 4
        for i in range(i_max):
            for j in range(j_max):
                current = start + (gv_a * i) + (gv_b * j)
                if current[0] < img_w.px and current[1] < img_h.px and current[0] > 0 and current[1] > 0:
                    ret.append(Ag_Atom(current.copy()))
        return ret

    @staticmethod
    def test_Lookup_Table():
        gitter = Tests_Gitterpot.create_gitter()
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

    @staticmethod
    def test_pot_map():
        start = time.perf_counter()
        startstart = start
        gitter = Tests_Gitterpot.create_gitter()

        end = time.perf_counter()
        print("Create Gitter: {:.3f}s".format(end - start))
        start = end

        showmat = Tests_Gitterpot.show_gitter(gitter, True)

        end = time.perf_counter()
        print("ShowGitter: {:.3f}s".format(end - start))
        start = end

        fig, (ax1, ax2) = plt.subplots(2, 1)
        fig.suptitle('Potential und Gitterstruktur')

        ax1.imshow(showmat)
        ax1.set_ylabel("Gitter")

        end = time.perf_counter()
        print("Plot_things: {:.3f}s".format(end - start))
        start = end

        pot = Tests_Gitterpot.create_pot_map(gitter)

        end = time.perf_counter()
        print("Pot_Map: {:.3f}s".format(end - start))

        ax2.imshow(pot)
        ax2.set_ylabel("Potential")

        end = time.perf_counter()
        print("Done: {:.3f}s".format(end - startstart))

        plt.show()

    @staticmethod
    def create_pot_map(gitter):
        img_w = cfg.get_width()
        img_h = cfg.get_height()
        name = "PotMatGitter1"
        if os.path.isfile(os.path.join("Pickle_Data", name + ".data")):
            return pickle.load(open(os.path.join("Pickle_Data", name + ".data"), "rb"))
        ct = 0
        mat = np.zeros((img_w.px, img_h.px))
        for i in range(img_w.px):
            for j in range(img_h.px):
                test = Molecule(np.array([i, j]), 0)
                mat[i, j] = test.get_pot(gitter)

        with open(os.path.join("Pickle_Data", name + ".data"), "wb") as p:
            pickle.dump(mat, p)
        return mat

    @staticmethod
    def show_gitter(gitter, save=True, name="gittermat"):
        img_w = cfg.get_width()
        img_h = cfg.get_height()
        name += str(len(gitter))
        if save:
            if os.path.isfile(os.path.join("Pickle_Data", name + ".data")):
                return pickle.load(open(os.path.join("Pickle_Data", name + ".data"), "rb"))

        showmat = np.zeros((int(np.ceil(img_w.px)), int(np.ceil(img_h.px))))
        ct = 0
        for i in range(int(np.ceil(img_w.px))):
            print("i = {}/{}".format(i, img_w.px))
            for j in range(int(np.ceil(img_h.px))):
                for atom in gitter:
                    showmat[i, j] += atom.show_dot(i, j)
        if save:
            with open(os.path.join("Pickle_Data", name + ".data"), "wb") as p:
                pickle.dump(showmat, p)

        return showmat
