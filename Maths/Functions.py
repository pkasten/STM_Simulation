import numpy as np
import time, os

logfile = os.getcwd() + "/log.txt"
MEASURE = False

statisticsfile = os.getcwd() + "/statistics.txt"


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
