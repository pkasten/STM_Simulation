import multiprocessing as mp
import copy, os
import time
from Maths.Functions import measureTime
from Configuration.Files import MultiFileManager as fm


class DataFrame:
    # points = mp.Queue()
    points = []
    index = -1
    name = "Data"
    suffix = ".dat"
    filepath = os.getcwd() + "/Daten"

    @measureTime
    def getIterator(self):
        return self.points

    @measureTime
    def __len__(self):
        return len(self.points)

    @measureTime
    def save(self, index):
        self.index = index
        filename = self.name + str(index) + self.suffix
        xs = []
        ys = []
        numerate = []
        i = 0
        # queue2 = mp.Queue()
        content = "Index, X, Y\n"
        sep = ","
        for item in self.points:
            xs.append(item[0])
            ys.append(item[1])
            i += 1
            numerate.append(i)

        # self.points = queue2
        for l in range(len(xs)):
            content += "{},{},{}\n".format(numerate[l], xs[l], ys[l])
        fm.saveTxt(self.filepath, filename, content)

  #  @measureTime
  #  def clone(self):
  #      ret = DataFrame()
   #     for pt in self.points:
   #         ret.points.append(pt)
   #     ret.index = self.index
   #     ret.name = self.name
   #     ret.filepath = self.filepath
   #     ret.suffix = self.suffix
   #     return ret

    @measureTime
    def addPoint(self, xy):
        # self.points.put(xy)
        self.points.append(xy)

    @measureTime
    def getPoint(self):
        # return self.points.get()
        return self.points.pop()

    @measureTime
    def hasPoints(self):
        # return not self.points.empty()
        return len(self.points) > 0

    @measureTime
    def __str__(self):
        # return str(self.dump_queue())
        return str(self.points)

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
