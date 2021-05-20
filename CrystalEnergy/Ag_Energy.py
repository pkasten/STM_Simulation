import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import time

px_durch_ang = 20


class Distance:

    def __init__(self, useAng, arg):
        if (useAng):
            self.px = px_durch_ang * arg
            self.ang = arg
        else:
            self.px = arg
            self.ang = arg / px_durch_ang


nn_dist = Distance(True, 2.88)
img_w = Distance(False, 400)
img_h = Distance(False, 400)


class Ag_Atom:

    def __init__(self, pos):
        self.pos = pos


    def show_mat(self):
        radius = 3
        height = 1
        mat = np.zeros((img_w.px, img_h.px))
        for i in range(img_w.px):
            for j in range(img_h.px):
                if np.sqrt(np.power(i - self.pos[0], 2) + np.power(j - self.pos[1], 2)) < radius:
                    mat[i, j] = height
                else:
                    mat[i, j] = 0

        return mat

    def show(self, x, y):
        radius = 3
        height = 1

        if np.sqrt(np.power(x - self.pos[0], 2) + np.power(y - self.pos[1], 2)) < radius:
            return height
        else:
            return 0

    @staticmethod
    def potential(dist):
        return np.exp(-5*dist/nn_dist.px)

    @staticmethod
    def potential_fromCoords(dist, angle):

        sixty_deg = np.pi /1.5
        distances_to_Atoms = [dist]
        distances_to_Atoms.append(np.sqrt(np.power(nn_dist.px - dist * np.cos(angle), 2)
                                          + np.power(dist * np.sin(angle), 2)))
        distances_to_Atoms.append(np.sqrt(np.power(nn_dist.px - dist * np.cos(sixty_deg-angle), 2)
                                          + np.power(dist * np.sin(sixty_deg-angle), 2)))

        pot = 0
        for d in distances_to_Atoms:
            pot += Ag_Atom.potential(d)

        return pot

class Atom:
    def __init__(self, relpos):
        self.relpos = relpos
        self.abspos = relpos
        self.radius = 3

    def calc_abs_pos(self, moleculepos, moleculetheta):
        turnmat = np.array([[np.cos(moleculetheta), -np.sin(moleculetheta)],
                            [np.sin(moleculetheta), np.cos(moleculetheta)]])
        self.abspos = moleculepos + np.matmul(self.relpos, turnmat)

    def show(self, x, y):
        height = 1
        if np.sqrt(np.power(x - self.abspos[0], 2) + np.power(y - self.abspos[1], 2)) < self.radius:
            return height
        else:
            return 0

    def show_mat(self):
        mat = np.zeros((img_w.px, img_h.px))
        for i in range(img_w.px):
            for j in range(img_h.px):
                mat[i, j] = self.show(i, j)

        return mat

    def find_nearest_atoms(self, gitter):
        threshold = 10000
        nearest = {}
        for atom in gitter:
            if np.linalg.norm(atom.pos - self.abspos) < threshold:
                temp = np.linalg.norm(atom.pos - self.abspos)
                nearest[atom] = temp
                if len(nearest) > 3:
                    maxlen = max(nearest.values())
                    for elem in nearest.keys():
                        if nearest[elem] == maxlen:
                            del nearest[elem]
                            threshold = max(nearest.values())
                            break

        #print("Len(nearest) {} mit {}".format(len(nearest.keys()), nearest.values()))
        return nearest.keys()

    def find_nearest_atoms_dict(self, gitter):
        threshold = 10000
        nearest = {}
        for atom in gitter:
            if np.linalg.norm(atom.pos - self.abspos) < threshold:
                temp = np.linalg.norm(atom.pos - self.abspos)
                nearest[atom] = temp
                if len(nearest) > 3:
                    maxlen = max(nearest.values())
                    for elem in nearest.keys():
                        if nearest[elem] == maxlen:
                            del nearest[elem]
                            threshold = max(nearest.values())
                            break

        #print("Len(nearest) {} mit {}".format(len(nearest.keys()), nearest.values()))
        return nearest

    def find_nearest_atom(self, gitter):
        threshold = 10000
        nearest = None
        for atom in gitter:
            if np.linalg.norm(atom.pos - self.abspos) < threshold:
                nearest = atom
                threshold = np.linalg.norm(atom.pos - self.abspos)

        return nearest

    @staticmethod
    def find_mid_of_nearest(nearest):
        assert len(nearest) == 3

        widths = [atom.pos[0] for atom in nearest]
        heights = [atom.pos[1] for atom in nearest]

        mid_w = np.average(widths)
        mid_h = np.average(heights)

        return Atom(np.array([mid_w, mid_h]))

    @staticmethod
    def angle_between(vec1, vec2):
        assert len(vec1) == 2
        assert len(vec2) == 2

        phi = np.arcsin(abs(np.cross(vec1, vec2)) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
        if (phi > np.pi / 2):
            phi = np.pi - phi
        return phi


    def find_pot_coords(self, gitter):
        nearest = self.find_nearest_atoms_dict(gitter)
        assert len(nearest) == 3
        distances = nearest.values()
        dis = min(distances)
        keys = nearest.keys()
        neighbours = []
        next_at = None
        for key in keys:
            if nearest[key] == dis:
                next_at = key
            else:
                neighbours.append(key)

        vec1 = self.abspos - next_at.pos
        angles = []
        for nb in neighbours:
            vec2 = nb.pos - next_at.pos
            angles.append(self.angle_between(vec1, vec2))

        return dis, min(angles)


    def find_pot(self, gitter):
        d, a = self.find_pot_coords(gitter)
        return Ag_Atom.potential_fromCoords(d, a)




class Molecule:

    def __init__(self, pos, theta):
        self.atoms = []
        self.pos = pos
        self.theta = theta

        molecule_class = "Single"

        if molecule_class == "Single":
            self.atoms.append(Atom(np.array([0, 0])))
        elif molecule_class == "CO2":
            co_len = Distance(True, 1)
            self.atoms.append(Atom(np.array([0, 0])))
            self.atoms.append(Atom(np.array([-co_len.px, 0])))
            self.atoms.append(Atom(np.array([+co_len.px, 0])))

        for atom in self.atoms:
            atom.calc_abs_pos(self.pos, self.theta)

    def show(self, x, y):
        ret = 0
        for atom in self.atoms:
            ret += atom.show(x, y)
        return ret

    def show_mat(self):
        mat = np.zeros((img_w.px, img_h.px))
        for i in range(img_w.px):
            for j in range(img_h.px):
                mat[i, j] = self.show(i, j)

        return mat

    def get_pot(self, gitter):
        pot = 0
        for atom in self.atoms:
            pot += atom.find_pot(gitter)
        return pot


def create_gitter():
    ret = []

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


def show_gitter(gitter, save=False, name="gittermat",):
    if save:
        if os.path.isfile(name + ".data"):
            return pickle.load(open(name + ".data", "rb"))

    showmat = np.zeros((img_w.px, img_h.px))
    ct = 0
    for i in range(img_w.px):
        for j in range(img_h.px):
            for atom in gitter:
                showmat[i, j] += atom.show(i, j)
    if save:
        with open(name + ".data", "wb") as p:
            pickle.dump(showmat, p)

    return showmat


def create_pot_map(gitter):
    name = "PotMatGitter1"
    if os.path.isfile(name + ".data"):
        return pickle.load(open(name + ".data", "rb"))
    ct = 0
    mat = np.zeros((img_w.px, img_h.px))
    for i in range(img_w.px):
        for j in range(img_h.px):
            test = Molecule(np.array([i, j]), 0)
            mat[i, j] = test.get_pot(gitter)

    with open(name + ".data", "wb") as p:
        pickle.dump(mat, p)
    return mat

class Lookup_Table:

    class Couple:
        def __init__(self, pos, angle, energy):
            self.pos = pos
            self.angle = angle
            self.energy = energy


    couples = []

    def __init__(self, dist_step, angle_step):
        self.dist_step = dist_step
        self.angle_step = angle_step

    def add(self, pos, angle, energy):
        self.couples.append(self.Couple(pos, angle, energy))

    def get_nearest_Energy(self, pos, angle):
        nearest = None
        min_d = 1000
        for c in self.couples:
            d = np.square(np.linalg.norm(pos - c.pos)/self.dist_step) \
                + np.square(abs(c.angle - angle)/self.angle_step)

            if d < min_d:
                min_d = d
                nearest = c

        return nearest.energy, min_d, nearest








def create_Molecule_lookup_table(gitter, Testclass, name):

    if os.path.isfile(name + ".data"):
        return pickle.load(open(name + ".data", "rb"))

    anglesteps = 10
    angle_step = 360/anglesteps

    center = Ag_Atom(np.array([img_w.px/2, img_h.px/2]))

    maxdist = (np.sqrt(0.75) - 0.25) * nn_dist.px
    diststeps = 10
    dist_step = maxdist / diststeps

    def bog(deg):
        return deg/180 * np.pi

    angles = [bog(i*angle_step) for i in range(anglesteps)]
    distances = [i * dist_step for i in range(diststeps)]
    molecule_angles = [bog(i*angle_step) for i in range(anglesteps)]

    table = Lookup_Table(dist_step, angle_step)

    vecs = []

    for angle in angles:
        turnmat = np.array([[np.cos(angle), -np.sin(angle)],
                            [np.sin(angle), np.cos(angle)]])
        for dist in distances:
            vec1 = np.array([dist, 0])
            vecs.append(center.pos + np.matmul(vec1, turnmat))

    print(vecs)

    for pos in vecs:
        for theta in molecule_angles:
            table.add(pos, theta, Testclass(pos, theta).get_pot(gitter))

    with open(name + ".data", "wb") as p:
        pickle.dump(table, p)

    return table



def test_pot_map():
    start = time.perf_counter()
    startstart = start
    gitter = create_gitter()

    end = time.perf_counter()
    print("Create Gitter: {:.3f}s".format(end - start))
    start = end

    showmat = show_gitter(gitter, True)

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

    pot = create_pot_map(gitter)

    end = time.perf_counter()
    print("Pot_Map: {:.3f}s".format(end - start))

    ax2.imshow(pot)
    ax2.set_ylabel("Potential")

    end = time.perf_counter()
    print("Done: {:.3f}s".format(end - startstart))

    plt.show()


if __name__ == "__main__":
    gitter = create_gitter()
    table = create_Molecule_lookup_table(gitter, Molecule, "testTable10")

    for i in range(3):
        for j in range(3):
            for a in range(4):
                t = a * np.pi / 3
                x = 180 + i * 15
                y = 175 + j * 15
                e, err, nst = table.get_nearest_Energy(np.array([x, y]), t)
                print("Energy for {} {}, theta ={:.2f} is {:.2f} with error {:.2f} and nearest x={:.2f}, y={:.2f}, a={:.2f}".format(
                    x, y, t, e, err, nst.pos[0], nst.pos[1], nst.angle
                ))
    exit()

    #test_pot_map()
    gitter = create_gitter()
    create_Molecule_lookup_table(gitter, Molecule)
    exit()
    gitter = create_gitter()
    showmat = show_gitter(gitter, True)


    test = Molecule(np.array([179, 171]), 45 / 180 * np.pi)

    tsm = test.show_mat()

    nearestats = []
    for atom in test.atoms:
        for elem in atom.find_nearest_atoms(gitter):
            nearestats.append(elem)

    #print(len(nearestats))
    nearestshow = show_gitter(nearestats, False)



    midpoint = Atom.find_mid_of_nearest(nearestats)
    msm = midpoint.show_mat()


    plt.imshow(nearestshow + showmat + tsm + 2*msm)
    plt.show()

    atti = Atom(np.array([320, 320]))
    verynearest = atti.find_nearest_atom(gitter)
    matti = verynearest.show_mat()
    attimap = atti.show_mat()
    plt.imshow(showmat + attimap + matti)
    plt.show()

    print("Potential of Molecule: {}".format(test.get_pot(gitter)))
