from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np
import time, os, math
import Configuration as cfg

# logfile = os.getcwd() + "/log.txt"
logfile = str(os.path.join(os.getcwd(), "log.txt"))

#If measureTime should be active
MEASURE = False

# statisticsfile = os.getcwd() + "/statistics.txt"
statisticsfile = str(os.path.join(os.getcwd(), "statistics.txt"))


def log(string):
    """
    Logs a provided string to the logfile
    :param string: String to log
    :return: None
    """
    with open(logfile, "a") as file:
        file.write(string)


list_of_known = []


def clearLog():
    """
    Clears the log
    :return: None
    """
    with open(logfile, "w") as f:
        f.write("")


def evaluateLog():
    """
    Evaluates the logfile, genreates new Statsticsfile with total computational times
    :return: None
    """
    known = []
    stat_dict_app = dict()
    stat_dict_totalTime = dict()
    with open(logfile, "r") as lg:
        for line in lg:
            parts = line.split(": ")
            time = parts[1].split(" ")[0]

            if parts[0] not in known:
                known.append(parts[0])
                stat_dict_app[parts[0]] = 0
                stat_dict_totalTime[parts[0]] = 0
            stat_dict_app[parts[0]] += 1
            stat_dict_totalTime[parts[0]] += float(time)
    with open(statisticsfile, "w") as file:
        for key in iter(stat_dict_totalTime.keys()):
            file.write(str(key) + ": {} calls\n".format(stat_dict_app[key]))
            file.write(str(key) + ": {:.3f} ms total\n\n".format(stat_dict_totalTime[key]))

def turnMatrix(mat, theta):
    """
    Turns a visualization matrix around an angle theta
    :param mat: matrix to turn
    :param theta: angle by which matrix should be turned
    :return: new matrix, new center x, new center y
    """
    if (theta % 2 * np.pi) == 0:
        w, h = np.shape(mat)
        cx_old = w / 2
        cy_old = h / 2
        return mat, cx_old, cy_old

    w, h = np.shape(mat)
    b = abs(np.sqrt(np.square(h) / (1 + np.square(np.tan(theta)))))
    a = abs(b * np.tan(theta))

    # mat_loc, th_loc = _prepare_matrix(mat, theta)
    mat_loc = mat
    th_loc = theta

    h_new = math.ceil(w * abs(np.sin(th_loc)) + b)
    w_new = math.ceil(w * abs(np.cos(th_loc)) + a)


    cx = w_new / 2
    cy = h_new / 2
    cx_old = w / 2
    cy_old = h / 2
    center_new = np.array([cx, cy])
    center_old = np.array([cx_old, cy_old])
    new_mat = np.zeros((w_new, h_new))

    for x_abs in range(np.shape(new_mat)[0]):
        for y_abs in range(np.shape(new_mat)[1]):
            x = x_abs - cx
            y = y_abs - cy
            pos = np.array([x, y])
            d = np.linalg.norm(pos)

            #print("hi")
            theta_pos = math.atan2(-x, y) + np.pi

            #if x > 0 and y > 0:
            #    theta_pos = np.pi / 2 + np.arctan(y / x)
            #elif x > 0 and y < 0:
            #    theta_pos = np.arctan(- x / y)
            #elif x < 0 and y > 0:
            #    theta_pos = np.pi + np.arctan(-x / y)
            #elif x < 0 and y < 0:
            #    theta_pos = 2 * np.pi - np.arctan(x / y)
            #elif y == 0 and x < 0:
            #    theta_pos = (3 / 2) * np.pi
            #elif y == 0 and x >= 0:
            #    theta_pos = np.pi / 2
            #elif x == 0 and y <= 0:
            #    theta_pos = 0
            #elif x == 0 and y > 0:
            #    theta_pos = np.pi
            #else:
            #    raise ValueError

            theta_old = (theta_pos - th_loc + 2 * np.pi) % (2 * np.pi)
            assert 0 <= theta_old <= 2 * np.pi
            x_old = d * np.sin(theta_old)
            y_old = - d * np.cos(theta_old)

            pos_old = np.array([x_old, y_old])
            pos_old_abs = pos_old + center_old

            x_old_abs = pos_old_abs[0]
            y_old_abs = pos_old_abs[1]

            x_old_abs_rd = int(np.round(x_old_abs))
            y_old_abs_rd = int(np.round(y_old_abs))

            if not (0 <= x_old_abs_rd <= w and 0 <= y_old_abs_rd <= h):
                continue

            left_x = int(np.floor(pos_old_abs[0]))
            right_x = left_x + 1
            up_y = int(np.floor(pos_old_abs[1]))
            down_y = up_y + 1


            x_sep = pos_old_abs[0] - left_x
            y_sep = pos_old_abs[1] - up_y

            if left_x < 0:
                left_x = 1000000
            if up_y < 0:
                up_y = 10000000

            l_ant = 1 - x_sep
            r_ant = x_sep
            u_ant = 1-y_sep
            d_ant = y_sep

            sumcol = 0
            try:
                sumcol += l_ant * u_ant * mat_loc[left_x, up_y]
                sumcol += l_ant * d_ant * mat_loc[left_x, down_y]
                sumcol += r_ant * u_ant * mat_loc[right_x, up_y]
                sumcol += r_ant * d_ant * mat_loc[right_x, down_y]
            except IndexError:
                sumcol = 0


            sumcol = max(0, sumcol)
            sumcol = min(255, sumcol)

            new_mat[x_abs, y_abs] = sumcol


    return new_mat, (w_new / w) * cx_old, (h_new / h) * cy_old


def approx_invers(f, min=0, max=400):
    """
    DEPRECATED. Approximates the inverse function for provided function f in a given range
    :param f: function to be inverted
    :param min: minimum x of range where f should be inverted
    :param max: maximum x of range where f should be inverted
    :return:
    """
    genauigkeit = 5
    y_steps = 500
    xs_n = []
    ys_n = []
    for i in range(min, max):
        xs_n.append(i)
        ys_n.append(f(i))

    plt.plot(xs_n, ys_n)
    plt.title(f)
    plt.show()

    xs_inv = []
    ys_inv = []

    for l in np.linspace(genauigkeit * int(np.min(ys_n) - 0.5), genauigkeit * int(np.max(ys_n) + 0.5), y_steps):
        i = l / genauigkeit
        testx = []
        testd = []
        new_min = int(f(min) - 2)
        new_max = int(f(max) + 2)

        for k in range(genauigkeit * new_min, genauigkeit * new_max):
            x = k / genauigkeit
            testx.append(x)
            testd.append(abs(f(x) - i))

        dmin = np.min(testd)
        for h in range(len(testx) - 1, 0, -1):
            if testd[h] == dmin:
                xs_inv.append(f(testx[h]))
                ys_inv.append(testx[h])
                break
            # print("Not it")

    print(xs_inv)
    print(ys_inv)


maxInv = max(int(np.ceil(cfg.get_height().px)), int(np.ceil(cfg.get_width().px)))


def get_invers_function(f, min=0, max=maxInv, acc=20):
    """
    Returns the aproximated inverted function of f as a callable expression
    :param f: function to be inverted
    :param min: minimum x of range where f should be inverted
    :param max: maximum x of range where f should be inverted
    :param acc: accuracy of inversion
    :return: inverted function
    """
    genauigkeit = acc
    xs = []
    ys = []
    for i in range(genauigkeit * min, genauigkeit * max):
        x_loc = i / genauigkeit
        y_loc = f(x_loc)
        xs.append(x_loc)
        ys.append(y_loc)

    @lru_cache
    def f_inv(x):
        distances = [abs(y_lc - x) for y_lc in ys]
        dmin = np.min(distances)
        # print("dmin = {:.3f}".format(dmin))
        for i in range(len(distances)):
            if distances[i] == dmin:
                # print("Ret xs[i]  {}".format(xs[i]))
                return xs[i]

        return np.infty

    return f_inv


# def statistics(name, duration):
#
#    if name == "terminatestatistics":
#        print_statistics()
#        return
# print("statistics: " + name)
#    if name not in list_of_known:#
#        list_of_known.append(name)
#        stat_dict_app[name] = 0
#        stat_dict_totalTime[name] = 0
#    #print(stat_dict_app)
#    stat_dict_app[name] += 1
#    stat_dict_totalTime[name] += duration
#    print_statistics()


# def print_statistics():
#    with open(statisticsfile, "w") as file:
#        for key in iter(stat_dict_totalTime.keys()):
#            #print(key)
#            file.write(str(key) + ": {} calls\n".format(stat_dict_app[key]))
#            file.write(str(key) + ": {} ms total\n\n".format(stat_dict_app[key]))


def measureTime(func):
    """
    Measures and logs the time the provided function takes to compute.
    Does nothing if parameter MEASURE is set to False
    :param func: function which should be measured
    :return: None
    """
    if not MEASURE:
        return func

    def measure(*args):
        # if func.__name__ in list_of_known:
        #    return func(*args)
        # else:
        #    list_of_known.append(func.__name__)

        start = time.process_time()
        ret = func(*args)
        duration = time.process_time() - start
        # statistics(func.__name__, duration * 1000)
        log(str(func.__name__) + ": {} ms \n".format(duration * 1000))
        return ret

    return measure


@DeprecationWarning
def turnMatrix_old(mat, theta):  # ToDo: Improve Anti-Aliasing
    """
    DEPRECATED. old method to turn a matrix.
    New version: turnMatrix
    :param mat: matrix to be turned
    :param theta: angle by which matrix should be turned
    :return:
    """
    return turnMatrix(mat, theta)

    shp = np.shape(mat)  # ToDo: Turn Correctly
    matrix = mat
    w = shp[0]
    h = shp[1]
    anti_aliasing = cfg.get_anti_aliasing()

    matrix, theta = _prepare_matrix(matrix, theta)

    b = np.sqrt(np.square(h) / (1 + np.square(np.tan(theta))))
    a = b * np.tan(theta)

    # print(a, b, theta)

    h_new = math.ceil(w * np.sin(theta) + b)
    w_new = math.ceil(w * np.cos(theta) + a)
    cx = w / 2
    cy = h / 2

    limit = 255  # ToDo: SoftCode

    new_mat = np.zeros((w_new, h_new))

    if not anti_aliasing:
        for i in range(0, 2 * w):  # Step 0.5 zum lösen drees Problems von durch rundung nicht getroffenen Feldern
            for j in range(0, 2 * h):
                i_temp = i / 2
                j_temp = j / 2
                if matrix[min(round(i_temp), w - 1), min(round(j_temp), h - 1)] == 0:  # Min für Indexerror
                    continue
                else:
                    # print(i, j)
                    cx = w / 2
                    cy = h / 2
                    i_tilt = i_temp - cx
                    j_tilt = j_temp - cy
                    i_tilt_new = i_tilt * np.cos(theta) - j_tilt * np.sin(theta)
                    j_tilt_new = i_tilt * np.sin(theta) + j_tilt * np.cos(theta)
                    i_new = i_tilt_new + (w_new / w) * cx
                    j_new = j_tilt_new + (h_new / h) * cy
                    try:
                        if i_new % 1 == 0.5 or j_new % 1 == 0.5:
                            new_mat[round(i_new), round(j_new)] += 0.5 * matrix[
                                min(round(i_temp), w - 1), min(round(j_temp), h - 1)]
                            new_mat[round(i_new - 0.1), round(j_new - 0.1)] += 0.5 * matrix[
                                min(round(i_temp - 0.1), w - 1), min(round(j_temp - .01), h - 1)]
                        else:
                            new_mat[round(i_new), round(j_new)] += 0.5 * matrix[
                                min(round(i_temp), w - 1), min(round(j_temp), h - 1)]
                    except IndexError:
                        pass
    else:
        for i in range(w):
            for j in range(h):
                if matrix[i, j] == 0:
                    continue
                else:
                    cx = w / 2
                    cy = h / 2
                    i_tilt = i - cx
                    j_tilt = j - cy
                    i_tilt_new = i_tilt * np.cos(theta) - j_tilt * np.sin(theta)
                    j_tilt_new = i_tilt * np.sin(theta) + j_tilt * np.cos(theta)
                    i_new = i_tilt_new + (w_new / w) * cx
                    j_new = j_tilt_new + (h_new / h) * cy

                    i_new_abs, i_new_chi = np.divmod(i_new, 1)
                    j_new_abs, j_new_chi = np.divmod(j_new, 1)
                    i_new_abs = int(i_new_abs)
                    j_new_abs = int(j_new_abs)
                    a = matrix[i, j]
                    new_mat[i_new_abs, j_new_abs] += a * (1 - i_new_chi) * (1 - j_new_chi)
                    # new_mat[i_new_abs, j_new_abs] = min(limit, new_mat[i_new_abs, j_new_abs])
                    try:
                        new_mat[i_new_abs + 1, j_new_abs] += a * i_new_chi * (1 - j_new_chi)
                        # new_mat[i_new_abs + 1, j_new_abs] = min(limit, new_mat[i_new_abs + 1, j_new_abs])
                    except IndexError:
                        pass
                    try:
                        new_mat[i_new_abs, j_new_abs + 1] += a * (1 - i_new_chi) * j_new_chi
                        # new_mat[i_new_abs, j_new_abs + 1] = min(limit, new_mat[i_new_abs, j_new_abs + 1])
                        new_mat[i_new_abs + 1, j_new_abs + 1] += a * i_new_chi * j_new_chi
                        # new_mat[i_new_abs + 1, j_new_abs + 1] = min(limit, new_mat[i_new_abs + 1, j_new_abs + 1])
                    except IndexError:
                        pass

    return new_mat, (w_new / w) * cx, (h_new / h) * cy

def turn_matplotlib(mat):
    """
    Turns a matrix in a way that matplotlibs pyplot.imshow has the same orientation as the exported image
    :param mat: matrix to be turned
    :return: matrix turned to be visualized by matplotlib
    """
    w, h = np.shape(mat)

    newmat = np.zeros(np.shape(mat))


    for i in range(w):
        for j in range(h):
            newmat[i, j] = mat[h - j - 1, w - i - 1]

    return newmat


@DeprecationWarning
def _prepare_matrix(mat, theta):
    """
    DEPRECATED. Needed to be used before turnMatrix_Old
    :param mat: Matrix
    :param theta: Angle
    :return:
    """
    matrix = mat
    while True:
        if 0 <= theta < np.pi / 2:
            break
        elif np.pi / 2 <= theta < np.pi:
            theta = theta % (np.pi / 2)
            newmat = np.zeros(np.shape(matrix))
            xlen = np.shape(matrix)[0]
            ylen = np.shape(matrix)[1]
            for x in range(xlen):
                for y in range(ylen):
                    newmat[x, y] = matrix[y, xlen - x - 1]
            matrix = newmat
        elif np.pi <= theta < 3 * np.pi / 2:
            newmat = np.zeros(np.shape(matrix))
            xlen = np.shape(matrix)[0]
            ylen = np.shape(matrix)[1]
            for x in range(xlen):
                for y in range(ylen):
                    newmat[x, y] = matrix[x, ylen - y - 1]  # -1 fpr Index Error

            theta = theta % np.pi
            matrix = newmat
        elif 3 * np.pi / 2 <= theta < 2 * np.pi:
            matrix.transpose()
            newmat = np.zeros(np.shape(matrix))
            xlen = np.shape(matrix)[0]
            ylen = np.shape(matrix)[1]
            for x in range(xlen):
                for y in range(ylen):
                    newmat[x, y] = matrix[y, xlen - x - 1]
            matrix = newmat
            newmat = np.zeros(np.shape(matrix))
            for x in range(xlen):
                for y in range(ylen):
                    newmat[x, y] = matrix[xlen - x - 1, ylen - y - 1]  # -1 fpr Index Error

            theta = theta % (3 * np.pi / 2)
            matrix = newmat

        else:
            theta = theta % 2 * np.pi

    return matrix, theta



class Functions:
    """
    DEPRECATED
    """

    @staticmethod
    def dist2D(a, b):
        """
        Caluclate Distance in 2D
        :param a: 2D vector
        :param b: 2D vector
        :return: distacce
        """
        return np.sqrt(np.square(a[0] - b[0]) + np.square(a[1] - b[1]))

    @staticmethod
    @measureTime
    def normalDistribution(center, point, sigma):
        """
        Was used to calculate the 2D normal distribution
        :param center: Expectation value
        :param point: Point to be measured
        :param sigma: Standard Derivation
        :return:
        """
        d = Functions.dist2D(center, point)
        # return np.power(2 * np.pi * np.square(sigma), -0.5) * np.exp(-(np.square(d)) / (2 * np.square(sigma)))
        return np.exp(-(np.square(d)) / (2 * np.square(sigma)))
