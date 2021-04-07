import multiprocessing as mp
import copy
import time

class DataFrame:

    points = mp.Queue()

    def addPoint(self, xy):
        self.points.put(xy)

    def getPoint(self):
        return self.points.get()

    def hasPoints(self):
        return not self.points.empty()

    def __str__(self):
        return str(self.dump_queue())


    def dump_queue(self):
        result = []
        newqueue = mp.Queue()
        self.points.put('STOP')
        for i in iter(self.points.get, 'STOP'):
            result.append(i)
            if i != 'STOP':
                newqueue.put(i)
        time.sleep(.1)
        self.points = newqueue
        return result

