from Data import Simulation as sim
from Data import DataFrame as frame
from Visualization import Images
from Configuration import Files, Configuration as cfg
import os, time
from multiprocessing import Process, Semaphore
from multiprocessing.managers import BaseManager
from Maths.Functions import measureTime, clearLog, evaluateLog
import matplotlib.pyplot as plt
import math

runningProcesses = []

def print(st):
    with open("DebugLog.txt", "a") as log:
        log.write(str(st) + "\n")


def compareThreadCount():
    times = []
    # times[0] = 0
    times.append(0)
    imagesperRun = 480
    cfg.set_images_pt(25)
    for i in range(1, 32):
        cfg.set_threads(i)
        cfg.set_images_pt(math.ceil(imagesperRun / i))
        times.append(test(10))

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
    imagesperRun = 16

    cfg.set_threads(os.cpu_count())
    cfg.set_images_pt(math.ceil(imagesperRun / os.cpu_count()))
    x = []
    for i in range(100, 8000, 100):
        cfg.set_width(i)
        cfg.set_heigth(i)
        times.append(test(10))
        x.append(i*i)

    y = []
    for i in range(len(x)):
        y.append(times[i])
    plt.plot(x, y)
    plt.title("Comparing computational time over ImageSize")
    plt.xlabel("ImageSize in px")
    plt.ylabel("time in s")
    plt.savefig(os.path.join(os.getcwd(), "compareImageSize.png"))


def compare_points_per_image():
    times = []
    # times[0] = 0
    times.append(0)
    imagesperRun = 10

    cfg.set_threads(os.cpu_count()/2)
    cfg.set_images_pt(math.ceil(imagesperRun / 2))
    x = []
    cfg.set_width(800)
    cfg.set_heigth(800)
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
    # plt.savefig(os.getcwd() + "/comparePointsPerImage.png")
    plt.savefig(os.path.join(os.getcwd(), "comparePointsPerImage.png"))


def test():
    test(10)


def test(number_of_points):
    # cfg.set_images_pt(250)

    global nop
    nop = number_of_points
    start = time.perf_counter()
    path = os.path.join(os.getcwd(), "bildordner")
    sem = Semaphore(Files.FileManager.countFiles(path))

    try:
        os.mkdir(path)
    except OSError:
        pass

    BaseManager.register('FilenameGenerator', Files.FilenameGenerator)
    filemanager = BaseManager()
    filemanager.start()
    fn_generator = filemanager.FilenameGenerator(path, ".png")


    class DataCreator(Process):

        alive = True
        def stopNow(self):

            self.alive = False


        def run(self):
            for i in range(cfg.get_images_pt()):
                print("prozess Running")
                img = Images.Images(Files.MultiFileManager(), fn_generator)
                data = frame.DataFrame()

                for nono in range(number_of_points):
                    data.addPoint(sim.getPoint(0, img.getWidth(), 0, img.getHeight()))

                index = fn_generator.generateIndex()
                data.save(index)
                # img.noiseImage()
                print("Start imgs.createImg")
                if not self.alive: return
                img.createImageTest(data, str(self))  # ToDo: dont delete, or other method to provide index
                if not self.alive: return
                print("Start Saving")
                path = img.saveImage(index)[0]
                print("Done Saving")
                sem.release()
                print("Image " + path + " saved by: " + str(self.name))


    @measureTime
    def genererateTestFiles():
        global runningProcesses
        processes = []
        for i in range(cfg.get_threads()):
            processes.append(DataCreator())
        for pro in processes:
            print("Starting " + str(pro))
            pro.start()
            runningProcesses.append(pro)

        for po in processes:
            po.join()
            runningProcesses.remove(po)



    # print(data)
    # print_statistics()
    print("Start Generating")
    genererateTestFiles()
    # moveAllTest()
    # generateAndMove(3,1)
    print("OVR Dauer: {:.3f}".format(time.perf_counter() - start) + " s")
    print("Done")

    return time.perf_counter() - start


if __name__ == "__main__":
    clearLog()

    with open("DebugLog.txt", "w") as log:
        log.write(" ")
    #Files.FileManager.clearFolder(os.path.join(os.getcwd(), "bildordner"))
    #Files.FileManager.clearFolder(os.path.join(os.getcwd(), "Daten"))
    if Files.FileManager.countFiles(os.path.join(os.getcwd(), "bildordner2")) > 0:
        Files.FileManager.clearFolder(os.path.join(os.getcwd(), "bildordner2"))
        Files.FileManager.clearFolder(os.path.join(os.path.join(os.getcwd(), "Daten2")))
    Files.FileManager.moveAll(os.path.join(os.getcwd(), "bildordner"), os.path.join(os.getcwd(), "bildordner2"))
    Files.FileManager.moveAll(os.path.join(os.getcwd(), "Daten"), os.path.join(os.getcwd(), "Daten2"))
    # test()
    # compare_points_per_image()
    # compareThreadCount()
    # compare_points_per_image()
    try:
        compareImageSize()
    except KeyboardInterrupt:
        for pr in runningProcesses:
            pr.stopNow()
    evaluateLog()

