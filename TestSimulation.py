from Data import Simulation as sim
from Data import DataFrame as frame
from Visualization import Images
from Configuration import Files, Configuration
import os, time
from multiprocessing import Process, Lock, Manager
from multiprocessing.managers import BaseManager
from Maths.Functions import measureTime, clearLog, evaluateLog


def test():
    Configuration.MultiConfigManager.set_images_pt(250)
    clearLog()
    start = time.perf_counter()

    Configuration.MultiConfigManager.set_threads(4)
    path = os.getcwd() + "/bildordner"
    moveTo = os.getcwd() + "/bildordner2"

    try:
        os.mkdir(path)
    except OSError:
        pass

    BaseManager.register('FilenameGenerator', Files.FilenameGenerator)
    filemanager = BaseManager()
    filemanager.start()
    fn_generator = filemanager.FilenameGenerator(path, ".png")

    class DataCreator(Process):
        def run(self):
            for i in range(Configuration.MultiConfigManager.get_images_pt()):
                img = Images.Images(Files.MultiFileManager(), fn_generator)
                number_of_points = 10
                data = frame.DataFrame()
                for nono in range(number_of_points):
                    data.addPoint(sim.getPoint(0, img.getWidth(), 0, img.getHeight()))

                img.createImage(data)
                path = img.saveImage()
                print("Image " + path + " saved by: " + str(self.name))

    class Movement(Process):
        mfm = Files.MultiFileManager()
        def __init__(self):
            super().__init__()

        def run(self):
            #print("Running")
            while self.mfm.countFiles(path) > 0:
                #print("Inner")
                self.mfm.moveFile(path, moveTo)
                #print("movedFile")

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
            #print("appended")
        for pro in processes:
            pro.start()
            #print("started")
        for po in processes:
            po.join()

    # print(data)
    #print_statistics()
    genererateTestFiles()
    moveAllTest()
    evaluateLog()
    print("OVR Dauer: {:.3f}".format(time.perf_counter() - start) + " s")
    print("Done")




if __name__ == "__main__":
    test()
