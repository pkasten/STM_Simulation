import multiprocessing as mp
import copy
import time
from Maths.Functions import measureTime

class DataFrame:

    points = mp.Queue()

    @measureTime
    def addPoint(self, xy):
        self.points.put(xy)

    @measureTime
    def getPoint(self):
        return self.points.get()

    @measureTime
    def hasPoints(self):
        return not self.points.empty()

    @measureTime
    def __str__(self):
        return str(self.dump_queue())

    @measureTime
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

