import random
import threading
import time
import Configuration.Files as files
import Configuration.Configuration as config
import os
from multiprocessing import Process

counter = 0
printlock = threading.Lock()
folder = os.getcwd() + "/testFiles"
namescheme = "TestfileNo#.txt"


def testFileManager():
    moves = 1000
    filesNo = 900
    config.ConfigManager.set_threads(4)

    def initTest():
        try:
            os.mkdir("testFiles")
        except FileExistsError as fe:
            None

        for k in range(filesNo):
            filename = str.format("/TestfileNo{}.txt", k)
            path = folder + filename
            with open(path, "w") as f:
                f.write("Testdatei")

    def manageFiles(name, fm):
        rand = random.randrange(1, 6)

        if fm.countFiles(folder) == 0:
            return True
        if rand == 1:
            # print("Thread " + name + " yields First: " + str(fm.firstFile(folder, namescheme)) + "\n")
            fm.firstFile(folder, namescheme)
        elif rand == 2:
            # print("Thread " + name + " yields min: " + str(fm.minIndexFile(folder, namescheme)) + "\n")
            fm.minIndexFile(folder, namescheme)
        elif rand == 3:
            # print("Thread " + name + " yields max: " + str(fm.maxIndexFile(folder, namescheme)) + "\n")
            fm.maxIndexFile(folder, namescheme)
        elif rand == 4:
            fm.removeMax(folder, namescheme)
            # print("Thread " + name + " removes max: " + "\n")
        elif rand == 5:
            fm.removeMin(folder, namescheme)
            # print("Thread " + name + " removes min: " + "\n")
        else:
            print("Unknown")

        return False

    def doTestSingle():
        print("SingleTest")
        start = time.process_time()
        initTest()
        for i in range(moves):
            manageFiles("Single", files.FileManager())
        duration = time.process_time() - start
        print("Single Test: " + str(duration))
        return duration

    def doTestThread():
        cM = files.ThreadsafeFileManager()

        class MyThread(threading.Thread):
            name = "Thread-"
            interrupted = False
            fileManager = files.FileManager()

            def setConfig(self, fileM):
                self.fileManager = fileM

            def interrupt(self):
                self.interrupted = True

            def __init__(self):
                super().__init__()
                global counter
                printlock.acquire()
                # print("Here I am: " + str(counter))
                printlock.release()
                self.name = self.name + str(counter)
                counter += 1

            def run(self):
                while True:
                    if self.interrupted:
                        break
                    if manageFiles(self.name, self.fileManager):
                        break

        print("Second: ThreadLevel")
        start = time.process_time()
        threads = []
        initTest()
        for i in range(config.ConfigManager.get_threads()):
            threads.append(MyThread())
        for t in threads:
            t.setConfig(cM)
            t.start()
        print("Stopping")
        for t in threads:
            t.join()
        end = time.process_time() - start
        print("End: " + str(end))
        return end

    def doTestMulti():

        cM = files.MultiFileManager()

        class MyThread(Process):
            name = "Process-"
            interrupted = False
            fileManager = files.FileManager()

            def setConfig(self, fileM):
                self.fileManager = fileM

            def interrupt(self):
                self.interrupted = True

            def __init__(self):
                super().__init__()
                global counter
                self.name = self.name + str(counter)
                counter += 1

            def run(self):
                while True:
                    if self.interrupted:
                        break
                    if manageFiles(self.name, self.fileManager):
                        break

        print("Multi")
        start = time.process_time()
        threads = []
        initTest()
        for i in range(config.ConfigManager.get_threads()):
            threads.append(MyThread())
        for t in threads:
            t.setConfig(cM)
            t.start()
        print("Stopping")
        for t in threads:
            t.join()
        end = time.process_time() - start
        print("End: " + str(end))
        return end

    def comparison():
        runs = 20
        singles = []
        threadings = []
        multis = []
        for i in range(runs):
            print("Starting Run " + str(i))
            singles.append(doTestSingle())
            #threadings.append(doTestThread())
            multis.append(doTestMulti())

        print("-------------------")
        print("-------------------")
        print("-------------------")
        print("Single average:  " + str(sum(singles) / len(singles)))
        #print("Thread average:  " + str(sum(threadings) / len(threadings)))
        print("Multi average:  " + str(sum(multis) / len(multis)))

    # doTestSingle()
    # doTestThread()
    # doTestMulti()
    comparison()


if __name__ == "__main__":
    testFileManager()
