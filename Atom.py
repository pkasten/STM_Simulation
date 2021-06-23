from functools import lru_cache

import numpy as np
import Configuration as cfg
from Distance import Distance


class Atom:
    """
    Class representing any atom inside Molecules.
    Implements basic methods of Particle. Should possibly extend Particle
    """

    def __init__(self, relpos, type=None):
        """
        Initilaizes a new Atom
        :param relpos: relative position vector from center of the molecule
        :param type: Physical element of this Atom. Implemented are "C" (carbon), "H" (hydrogen) and "N" (nitrogen)
        """

        # save image properties
        self.img_w = cfg.get_width()
        self.img_h = cfg.get_height()

        # Set position values
        if isinstance(relpos[0], Distance):
            #print("Changed")
            self.relpos = Distance.px_vec(relpos)
            #print("Now: Relpos: {}".format(self.relpos))
            self.abspos = self.relpos
        else:
            self.relpos = relpos
            self.abspos = relpos

        # Set radius according to type
        self.radius = Distance(True, 1)
        if type == "H":
            self.radius = Distance(True, 1.06) # Lit
            #self.radius = Distance(True, 0.032) * radiusmlt #
        elif type == "C":
            self.radius = Distance(True, 1.36) # Lit
            #self.radius = Distance(True, 0.077) * radiusmlt #
        elif type == "N":
            self.radius = Distance(True, 1.06) # Lit
            #self.radius = Distance(True, 0.070) * radiusmlt #

        # Some more visualization parameters
        self.height = cfg.get_part_height()
        self.maxheight = cfg.get_max_height()
        self.fermi_exp = cfg.get_fermi_exp()
        #self.fermi_range = np.log(99) / self.fermi_exp + self.height.px / 2
        self.fermi_range = np.log(99) / self.fermi_exp + self.radius.px

        self.type = type

    def __repr__(self):
        """
        String representation of this atom
        :return:
        """
        return "{}-Atom bei x={:.1f}, y={:.1f}".format(self.type, self.abspos[0], self.abspos[1])

    def __str__(self):
        """
        String representation of this atom
        :return:
        """
        return "{}-Atom bei x={:.1f}, y={:.1f}".format(self.type, self.abspos[0], self.abspos[1])

    def set_maxHeight(self, maxh):
        """
        Setter method for maximum height parameter
        :param maxh: new max height
        :return:
        """
        self.maxheight = maxh

    def calc_abs_pos(self, moleculepos, moleculetheta):
        """
        Calculate own absolute position depending on relative position to molecule center self.relpos
        and the molecules position and orientation this atom belongs to
        :param moleculepos: Position of molecule this atom belongs to
        :param moleculetheta: Orientation of molecule this atom belongs to
        :return: None
        """

        turnmat = np.array([[np.sin(moleculetheta), -np.cos(moleculetheta)],
                            [np.cos(moleculetheta), np.sin(moleculetheta)]])
        self.abspos = moleculepos + np.matmul(self.relpos, turnmat)

    def show_dot(self, x, y):
        """
        Method that can be used as visualize pixel to represent the Atom as a dot
        Only used for debug purposes, has fixed radius of 5 px
        :param x: x-position of pixel that should be visualized, (0, 0) represents the atoms position
        :param y: y-position
        :return: 255 if inside the dot, 0 otherwise
        """
        if np.power(x - self.abspos[0], 2) < 25:
            if np.power(y - self.abspos[1], 2) < 25:
                if np.sqrt(np.power(x - self.abspos[0], 2) + np.power(y - self.abspos[1], 2)) < 5:
                    return 255

        return 0

    def show(self, x, y):
        """
        Visualizion methods with absolute positions x, y
        :param x: absolute x-position
        :param y: absolute y-position
        :return: height
        """
        d = np.sqrt(np.power(x - self.abspos[0], 2) + np.power(y - self.abspos[1], 2))
        if d > self.fermi_range:
            return 0

        return self.color(self.height * self._fermi1D(d, self.radius))
        #if np.sqrt(np.power(x - self.abspos[0], 2) + np.power(y - self.abspos[1], 2)) < self.radius:
        #    return self.color(self.height)
        #else:
        #    return 0

    @lru_cache()
    def get_effect_range(self):
        """
        Returns the effect range of visualization, the border after which the height is neglectable
        :return:
        """
        c = 0
        while True:
            c += 1
            if self.color(self.height * self._fermi1D(c, self.radius)) < 1:
                return c

    def color(self, dis):
        """
        transforms specific height dis from 0 to maxheight onto a scale from 0 to 255
        :param dis: height
        :return:
        """
        return 255 * dis.px/self.maxheight.px

    def _fermi1D(self, x, mu):
        """
        Implementation of fermis distribution function
        :param x: position x
        :param mu: expectation value where f(mu) = 0.5
        :return:
        """
        fermi_ret = 1 / (np.exp(self.fermi_exp * (x - mu.px)) + 1)
        #print("Fermi(x={:.2f}m mu={:.2f}) = {:.2f}".format(x, mu.px, fermi_ret))
        return fermi_ret

    @lru_cache
    def show_rel(self, x, y):
        """
        Visualizion methods with relative positions x, y
        :param x: relative x-position
        :param y: relative y-position
        :return: height
        """
        d = np.sqrt(np.power(x, 2) + np.power(y, 2))
        if d > self.fermi_range:
            return 0

        return self.color(self.height * self._fermi1D(d, self.radius))

        #if np.sqrt(np.power(x, 2) + np.power(y, 2)) < self.radius.px:
        #    return self.color(self.height)
        #else:
        #    return 0

    def show_mat(self):
        """
        Visualizes this Atom as a matrix
        :return: the visu-matrix
        """

        mat = np.zeros((int(np.ceil(self.img_w.px)), int(np.ceil(self.img_h.px))))
        for i in range(int(np.ceil(self.img_w.px))):
            for j in range(int(np.ceil(self.img_h.px))):
                mat[i, j] = self.show(i, j)

        return mat

    def find_nearest_atoms(self, gitter):
        """
        DEPRECATED. Used to find the 3 nearest lattice atoms from the array gitter
        :param gitter: lattice atoms
        :return:
        """
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
        """
        DEPRECATED. same as find_nearest_atoms, just with using a dictionary for saving
        :param gitter:
        :return:
        """
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
        """
        DEPRECATED> same as find_nearest_atoms, just returns only the very nearest
        :param gitter: lattice atoms
        :return:
        """
        threshold = 10000
        nearest = None
        for atom in gitter:
            if np.linalg.norm(atom.pos - self.abspos) < threshold:
                nearest = atom
                threshold = np.linalg.norm(atom.pos - self.abspos)

        return nearest

    @staticmethod
    def find_mid_of_nearest(nearest):
        """
        calculates the midpoint of three provided atoms
        :param nearest: three atoms
        :return: A new Atom at the center of this three
        """
        assert len(nearest) == 3

        widths = [atom.pos[0] for atom in nearest]
        heights = [atom.pos[1] for atom in nearest]

        mid_w = np.average(widths)
        mid_h = np.average(heights)

        return Atom(np.array([mid_w, mid_h]))

    @staticmethod
    def angle_between(vec1, vec2):
        """
        Static method to calculate the angle betweeen two vectors
        :param vec1:
        :param vec2:
        :return:
        """
        assert len(vec1) == 2
        assert len(vec2) == 2

        phi = np.arcsin(abs(np.cross(vec1, vec2)) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
        if (phi > np.pi / 2):
            phi = np.pi - phi
        return phi

    def find_pot_coords(self, gitter):
        """
        DEPRECATED. Used to parametrize this atoms position inside the lattice gitter
        :param gitter: lattice
        :return: two parameters defining this's position
        """
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
        """
        calculates potential energy inside the lattice
        :param gitter:
        :return:
        """
        d, a = self.find_pot_coords(gitter)
        return Ag_Atom.potential_fromCoords(d, a)


class Ag_Atom:
    """
    DEPRECATED. Atom representation for atoms inside the lattice structure.
    """
    def __init__(self, pos):
        """
        Initializes new lattice atom at postion vector pos
        :param pos: position vector
        """
        self.pos = pos
        self.img_w = cfg.get_width()
        self.img_h = cfg.get_height()
        self.nn_dist = cfg.get_nn_dist()

    def show_mat(self):
        """
        Visualizes the lattice atom as a matrix
        :return:
        """
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
        """
        see Atom.show_dot()
        """
        if np.power(x - self.pos[0], 2) < 25:
            if np.power(y - self.pos[1], 2) < 25:
                if np.sqrt(np.power(x - self.pos[0], 2) + np.power(y - self.pos[1], 2)) < 5:
                    return 255

        return 0

    def show(self, x, y):
        """
        see Atom.show()
        """
        radius = 3
        height = 1

        if np.sqrt(np.power(x - self.pos[0], 2) + np.power(y - self.pos[1], 2)) < radius:
            return height
        else:
            return 0

    @staticmethod
    def potential(dist):
        """
        Reurns the potential for an Atom in distance dist to this Ag Atom
        :param dist: distance between crystal and adsorbed atom
        :return:
        """
        nn_dist = cfg.get_nn_dist()
        return np.exp(-5 * dist / nn_dist.px)

    @staticmethod
    def potential_fromCoords(dist, angle):
        """
        Returns the lattice potential form parameters dist and angle
        :param dist:
        :param angle:
        :return:
        """

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
