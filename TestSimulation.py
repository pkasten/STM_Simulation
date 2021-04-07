from Data import Simulation as sim
from Data import DataFrame as frame
from Visualization import Images
from Configuration import Files, Configuration
import os
from multiprocessing import Process


def test():
    class DataCreator(Process):
        def run(self):
            for i in range(20):
                folder = os.getcwd() + "/testpictures"
                try:
                    os.mkdir(folder)
                except OSError:
                    pass
                Files.FileManager.setDefaultFolder(folder)
                img = Images.Images()
                number_of_points = 10

                data = frame.DataFrame()
                for i in range(number_of_points):
                    data.addPoint(sim.getPoint(0, img.getWidth(), 0, img.getHeight()))

                img.createImage(data)
                path = img.saveImage()
                print("Image " + path + "saved by: " + str(self.name))

    processes = []
    Configuration.MultiConfigManager.set_threads(4)
    for i in range(Configuration.MultiConfigManager.get_threads()):
        processes.append(DataCreator())
    for pro in processes:
        pro.start()
    for po in processes:
        pro.join()

    # print(data)


if __name__ == "__main__":
    test()
