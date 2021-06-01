from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np
import time, os, math
import Configuration as cfg

# logfile = os.getcwd() + "/log.txt"
logfile = str(os.path.join(os.getcwd(), "log.txt"))
MEASURE = False

# statisticsfile = os.getcwd() + "/statistics.txt"
statisticsfile = str(os.path.join(os.getcwd(), "statistics.txt"))


def log(string):
    with open(logfile, "a") as file:
        file.write(string)


list_of_known = []


def clearLog():
    with open(logfile, "w") as f:
        f.write("")


def evaluateLog():
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


def turnMatrix(mat, theta):  # ToDo: Improve Anti-Aliasing
    shp = np.shape(mat) #ToDo: Turn Correctly
    matrix = mat
    w = shp[0]
    h = shp[1]
    anti_aliasing = cfg.get_anti_aliasing()  # ToDo: Ask
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
            #matrix = matrix.transpose(0,1)  # ToDo: possible Error
        elif np.pi <= theta < 3 * np.pi / 2:
            newmat = np.zeros(np.shape(matrix))
            xlen = np.shape(matrix)[0]
            ylen = np.shape(matrix)[1]
            # midpoint = ylen/2 + 0.5
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
            # midpoint = ylen/2 + 0.5
            newmat = np.zeros(np.shape(matrix))
            for x in range(xlen):
                for y in range(ylen):
                    newmat[x, y] = matrix[xlen - x - 1, ylen - y - 1]  # -1 fpr Index Error

            theta = theta % (3 * np.pi / 2)
            matrix = newmat

        else:
            theta = theta % 2 * np.pi

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
                i_temp = i/2
                j_temp = j/2
                if matrix[min(round(i_temp), w-1), min(round(j_temp), h-1)] == 0:  # Min für Indexerror
                    continue
                else:
                    #print(i, j)
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
                            new_mat[round(i_new-0.1), round(j_new-0.1)] += 0.5 * matrix[
                                min(round(i_temp-0.1), w - 1), min(round(j_temp-.01), h - 1)]
                        else:
                            new_mat[round(i_new), round(j_new)] += 0.5 * matrix[min(round(i_temp), w-1), min(round(j_temp), h-1)]
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

def approx_invers(f, min=0, max=400):
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

    for l in np.linspace(genauigkeit * int(np.min(ys_n) - 0.5), genauigkeit * int(np.max(ys_n)+0.5), y_steps):
        i = l/genauigkeit
        testx = []
        testd = []
        new_min = int(f(min) - 2)
        new_max = int(f(max) + 2)

        for k in range(genauigkeit * new_min, genauigkeit* new_max):
            x = k / genauigkeit
            testx.append(x)
            testd.append(abs(f(x) - i))

        dmin = np.min(testd)
        for h in range(len(testx) - 1, 0, -1):
            if testd[h] == dmin:
                xs_inv.append(f(testx[h]))
                ys_inv.append(testx[h])
                break
            #print("Not it")

    print(xs_inv)
    print(ys_inv)


maxInv = max( int(np.ceil(cfg.get_height().px)), int(np.ceil(cfg.get_width().px)))
def get_invers_function(f, min=0, max=maxInv, acc=20):
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
        #print("dmin = {:.3f}".format(dmin))
        for i in range(len(distances)):
            if distances[i] == dmin:
                #print("Ret xs[i]  {}".format(xs[i]))
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
        log(str(func.__name__) + ": {:.3f} ms \n".format(duration * 1000))
        return ret

    return measure


class Functions:
    @staticmethod
    def dist2D(a, b):
        return np.sqrt(np.square(a[0] - b[0]) + np.square(a[1] - b[1]))

    @staticmethod
    @measureTime
    def normalDistribution(center, point, sigma):
        d = Functions.dist2D(center, point)
        # return np.power(2 * np.pi * np.square(sigma), -0.5) * np.exp(-(np.square(d)) / (2 * np.square(sigma)))
        return np.exp(-(np.square(d)) / (2 * np.square(sigma)))
