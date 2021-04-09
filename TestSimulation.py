from Data import Simulation as sim
from Data import DataFrame as frame
from Visualization import Images
from Configuration import Files, Configuration
import os, time
from multiprocessing import Process, Lock, Manager, Semaphore
from multiprocessing.managers import BaseManager
from Maths.Functions import measureTime, clearLog, evaluateLog
import matplotlib.pyplot as plt
import math


def compareThreadCount():
    times = []
    # times[0] = 0
    times.append(0)
    imagesperRun = 480
    Configuration.ConfigManager.set_images_pt(25)
    for i in range(1, 6):
        Configuration.ConfigManager.set_threads(i)
        Configuration.ConfigManager.set_images_pt(math.ceil(imagesperRun / i))
        times.append(test())

    x = []
    y = []
    for i in range(1, len(times)):
        x.append(i)
        y.append(times[i])
    plt.plot(x, y)
    plt.title("Comparing computational time over threads")
    plt.xlabel("Threads")
    plt.ylabel("Time in sec")
    plt.savefig(os.getcwd() + "/compareThreadCount.png")


def compareImageSize():
    times = []
    # times[0] = 0
    times.append(0)
    imagesperRun = 10

    Configuration.ConfigManager.set_threads(2)
    Configuration.ConfigManager.set_images_pt(math.ceil(imagesperRun / 2))
    x = []
    for i in range(100, 8000, 100):
        Configuration.ConfigManager.set_width(i)
        Configuration.ConfigManager.set_heigth(i)
        times.append(test(10))
        x.append(i)


def compare_points_per_image():
    times = []
    # times[0] = 0
    times.append(0)
    imagesperRun = 10

    Configuration.ConfigManager.set_threads(2)
    Configuration.ConfigManager.set_images_pt(math.ceil(imagesperRun / 2))
    x = []
    Configuration.ConfigManager.set_width(800)
    Configuration.ConfigManager.set_heigth(800)
    for i in range(10, 100, 10):
        times.append(test(i))
        x.append(i)

    y = []
    for i in range(len(x)):
        y.append(times[i])
    plt.plot(x, y)
    plt.title("Comparing computational time over PointsPerImage")
    plt.xlabel("No. of Points")
    plt.ylabel("time in s")
    plt.savefig(os.getcwd() + "/comparePointsPerImage.png")


def test():
    test(10)


def test(number_of_points):
    # Configuration.MultiConfigManager.set_images_pt(250)

    global nop
    nop = number_of_points
    start = time.perf_counter()
    # Configuration.MultiConfigManager.set_threads(4)
    path = os.getcwd() + "/bildordner"
    moveTo = os.getcwd() + "/bildordner2"
    sem = Semaphore(Files.FileManager.countFiles(path))

    try:
        os.mkdir(path)
    except OSError:
        pass

    BaseManager.register('FilenameGenerator', Files.FilenameGenerator)
    filemanager = BaseManager()
    filemanager.start()
    fn_generator = filemanager.FilenameGenerator(path, ".png")

    # Configuration.MultiConfigManager.set_images_pt(25)

    class DataCreator(Process):
        def run(self):
            for i in range(Configuration.MultiConfigManager.get_images_pt()):
                img = Images.Images(Files.MultiFileManager(), fn_generator)
                data = frame.DataFrame()

                for nono in range(number_of_points):
                    data.addPoint(sim.getPoint(0, img.getWidth(), 0, img.getHeight()))

                index = fn_generator.generateIndex()
                data.save(index)
                #img.noiseImage()
                img.createImage(data)  # ToDo: dont delete, or other method to provide index
                path = img.saveImage(index)[0]
                sem.release()
                print("Image " + path + " saved by: " + str(self.name))

    class Movement(Process):
        mfm = Files.MultiFileManager()
        interrupted = False

        def __init__(self):
            super().__init__()

        def interrupt(self):
            self.interrupted = True

        def run(self):
            # print("Running")
            while not self.interrupted:
                sem.acquire()
                self.mfm.moveFile(path, moveTo)
                print("movedFile. remaining: " + str(self.mfm.countFiles(path)))

    @measureTime
    def genererateTestFiles():

        processes = []
        for i in range(Configuration.MultiConfigManager.get_threads()):
            processes.append(DataCreator())
        for pro in processes:
            pro.start()
        for po in processes:
            po.join()

    @measureTime
    def moveAllTest():
        processes = []
        for i in range(Configuration.MultiConfigManager.get_threads()):
            processes.append(Movement())
            # print("appended")
        for pro in processes:
            pro.start()
            # print("started")
        for po in processes:
            po.join()

    @measureTime
    def generateAndMove(create, move):
        processesC = []
        processesM = []
        for i in range(create):
            processesC.append(DataCreator())
        for j in range(move):
            processesM.append(Movement())
        for i in processesC:
            i.start()
        for j in processesM:
            j.start()
        for i in processesC:
            i.join()
        for j in processesM:
            j.interrupt()
            j.join()

    # print(data)
    # print_statistics()
    genererateTestFiles()
    # moveAllTest()
    # generateAndMove(3,1)
    print("OVR Dauer: {:.3f}".format(time.perf_counter() - start) + " s")
    print("Done")

    return time.perf_counter() - start


if __name__ == "__main__":
    clearLog()
    # test()
    compare_points_per_image()
    evaluateLog()

