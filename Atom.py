import numpy as np
import Configuration as cfg
from Distance import Distance


class Atom:

    def __init__(self, relpos, type=None):

        radiusmlt = 5 # ToDo_ Realistisch, besser 1?
        self.img_w = cfg.get_width()
        self.img_h = cfg.get_height()
        #print("Relpos: {}".format(relpos))
        if isinstance(relpos[0], Distance):
            #print("Changed")
            self.relpos = Distance.px_vec(relpos)
            #print("Now: Relpos: {}".format(self.relpos))
            self.abspos = self.relpos
        else:
            self.relpos = relpos
            self.abspos = relpos

        #print("Relpos: {} mit {}, {}".format(self.relpos, self.relpos[0], self.relpos[1]))
        self.radius = Distance(True, 0.01)
        if type == "H":
            self.radius = Distance(True, 0.032) * radiusmlt
        elif type == "C":
            self.radius = Distance(True, 0.077) * radiusmlt
        elif type == "N":
            self.radius = Distance(True, 0.070) * radiusmlt

        self.height = cfg.get_part_height()
        self.maxheight = cfg.get_max_height()
        self.fermi_exp = cfg.get_fermi_exp()
        self.fermi_range = np.log(99) / self.fermi_exp + self.height.px / 2
        self.type = type

    def __repr__(self):
        return "{}-Atom bei x={:.1f}, y={:.1f}".format(self.type, self.abspos[0], self.abspos[1])

    def set_maxHeight(self, maxh):
        self.maxheight = maxh

    def calc_abs_pos(self, moleculepos, moleculetheta):
        # As px
        # Um 90° versetzt damit oben
        turnmat = np.array([[np.sin(moleculetheta), -np.cos(moleculetheta)],
                            [np.cos(moleculetheta), np.sin(moleculetheta)]])
        #turnmat = np.array([[0,0],[0,0]])
        self.abspos = moleculepos + np.matmul(self.relpos, turnmat)
        #x = self.relpos*np.sin(moleculetheta)

        #self.abspos = moleculepos + self.relpos

    def show_dot(self, x, y):
        if np.power(x - self.abspos[0], 2) < 25:
            if np.power(y - self.abspos[1], 2) < 25:
                if np.sqrt(np.power(x - self.abspos[0], 2) + np.power(y - self.abspos[1], 2)) < 5:
                    return 255

        return 0

    def show(self, x, y):
        d = np.sqrt(np.power(x - self.abspos[0], 2) + np.power(y - self.abspos[1], 2))
        if d > self.fermi_range:
            return 0

        return self.color(self.height * self._fermi1D(d, self.radius))
        #if np.sqrt(np.power(x - self.abspos[0], 2) + np.power(y - self.abspos[1], 2)) < self.radius:
        #    return self.color(self.height)
        #else:
        #    return 0
    def get_effect_range(self):
        c = 0
        while True:
            c += 1
            if self.color(self.height * self._fermi1D(c, self.radius)) < 1:
                return c


    def color(self, dis):
        return 255 * dis.px/self.maxheight.px

    def _fermi1D(self, x, mu):
        fermi_ret = 1 / (np.exp(self.fermi_exp * (x - mu.px)) + 1)
        #print("Fermi(x={:.2f}m mu={:.2f}) = {:.2f}".format(x, mu.px, fermi_ret))
        return fermi_ret


    def show_rel(self, x, y):

        d = np.sqrt(np.power(x, 2) + np.power(y, 2))
        if d > 30:
            return 0

        return self.color(self.height * self._fermi1D(d, self.radius))

        #if np.sqrt(np.power(x, 2) + np.power(y, 2)) < self.radius.px:
        #    return self.color(self.height)
        #else:
        #    return 0

    def show_mat(self):
        mat = np.zeros((int(np.ceil(self.img_w.px)), int(np.ceil(self.img_h.px))))
        for i in range(int(np.ceil(self.img_w.px))):
            for j in range(int(np.ceil(self.img_h.px))):
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

        # print("Len(nearest) {} mit {}".format(len(nearest.keys()), nearest.values()))
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

        # print("Len(nearest) {} mit {}".format(len(nearest.keys()), nearest.values()))
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


class Ag_Atom:

    def __init__(self, pos):
        self.pos = pos
        self.img_w = cfg.get_width()
        self.img_h = cfg.get_height()
        self.nn_dist = cfg.get_nn_dist()

    def show_mat(self):
        radius = 3
        height = 1
        mat = np.zeros((int(np.ceil(self.img_w.px)), int(np.ceil(self.img_h.px))))
        for i in range(int(np.ceil(self.img_w.px))):
            for j in range(int(np.ceil(self.img_h.px))):
                if np.sqrt(np.power(i - self.pos[0], 2) + np.power(j - self.pos[1], 2)) < radius:
                    mat[i, j] = height
                else:
                    mat[i, j] = 0

        return mat

    def show_dot(self, x, y):
        if np.power(x - self.pos[0], 2) < 25:
            if np.power(y - self.pos[1], 2) < 25:
                if np.sqrt(np.power(x - self.pos[0], 2) + np.power(y - self.pos[1], 2)) < 5:
                    return 255

        return 0

    def show(self, x, y):
        radius = 3
        height = 1

        if np.sqrt(np.power(x - self.pos[0], 2) + np.power(y - self.pos[1], 2)) < radius:
            return height
        else:
            return 0

    @staticmethod
    def potential(dist):
        nn_dist = cfg.get_nn_dist()
        return np.exp(-5 * dist / nn_dist.px)

    @staticmethod
    def potential_fromCoords(dist, angle):

        sixty_deg = np.pi / 1.5
        distances_to_Atoms = [dist]
        nn_dist = cfg.get_nn_dist()
        distances_to_Atoms.append(np.sqrt(np.power(nn_dist.px - dist * np.cos(angle), 2)
                                          + np.power(dist * np.sin(angle), 2)))
        distances_to_Atoms.append(np.sqrt(np.power(nn_dist.px - dist * np.cos(sixty_deg - angle), 2)
                                          + np.power(dist * np.sin(sixty_deg - angle), 2)))

        pot = 0
        for d in distances_to_Atoms:
            pot += Ag_Atom.potential(d)

        return pot
