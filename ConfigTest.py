import random
import threading
import time
import Configuration.Files as files
import Configuration.Configuration as config
import os
from multiprocessing import Process

#counter = 0
#printlock = threading.Lock()
#folder = os.getcwd() + "/testFiles"


def testFileManager():
    moves = 300
    filesNo = 700
    config.ConfigManager.set_threads(4)
    namescheme = "TestfileNo#.txt"

    def initTest():
        try:
            os.mkdir("testFilesA")
        except FileExistsError as fe:
            pass
        try:
            os.mkdir("testFilesB")
        except FileExistsError as fe:
            pass

        folder = os.getcwd() + "/testFilesA"
        for k in range(filesNo):
            filename = str.format("/TestfileNo{}.txt", k)
            path = folder + filename
            with open(path, "w") as f:
                f.write("Testdatei")

    def manageFiles(name, fm, folder):
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
        fm = files.FileManager()
        folderA = os.getcwd() + "/testFilesA"
        folderB = os.getcwd() + "/testFilesB"

        while fm.countFiles(folderA) > 0:
            fm.moveFile(folderA, folderB)

        for i in range(moves):
            manageFiles("Single", files.FileManager(), folderB)
        duration = time.process_time() - start
        print("Single Test: " + str(duration))
        #fm.clearFolder(folderB)
        return duration

    def doTestThread():
        fm = files.ThreadsafeFileManager()
        folderA = os.getcwd() + "/testFilesA"
        folderB = os.getcwd() + "/testFilesB"
        initTest()

        while fm.countFiles(folderA) > 0:
            fm.moveFile(folderA, folderB)

        class MyThread(threading.Thread):
            name = "Thread-"
            interrupted = False
            fileManager = files.FileManager()
            printlock = threading.Lock()
            folderB = os.getcwd() + "/testFilesB"

            def setConfig(self, fileM):
                self.fileManager = fileM

            def interrupt(self):
                self.interrupted = True

            def __init__(self):
                super().__init__()

            def run(self):
                while True:
                    if self.interrupted:
                        break
                    if manageFiles(self.name, self.fileManager, self.folderB):
                        break

        class MovingThread(threading.Thread):
            name = "Thread-"
            interrupted = False
            fileManager = files.ThreadsafeFileManager()
            printlock = threading.Lock()
            folderA = os.getcwd() + "/testFilesA"
            folderB = os.getcwd() + "/testFilesB"

            def setConfig(self, fileM):
                self.fileManager = fileM

            def interrupt(self):
                self.interrupted = True

            def __init__(self):
                super().__init__()

            def run(self):
                while True:
                    if self.interrupted:
                        break
                    if fm.countFiles(folderA) == 0:
                        break
                    if not fm.moveFile(folderA, folderB):
                        break


        print("Second: ThreadLevel")
        start = time.process_time()
        threads = []
        movers = []
        initTest()

        for i in range(config.ConfigManager.get_threads()):
            threads.append(MyThread())
            movers.append(MovingThread())
        for m in movers:
            m.start()
        for m in movers:
            m.join()
        for t in threads:
            t.setConfig(fm)
            t.start()
        print("Stopping")
        while(fm.countFiles(folderA) > 0):
            time.sleep(1)

        for t in threads:
            t.interrupt()
            t.join()
        end = time.process_time() - start
        print("End: " + str(end))
        #fm.clearFolder(folderB)
        return end

    def doTestMulti():

        folderA = os.getcwd() + "/testFilesA"
        folderB = os.getcwd() + "/testFilesB"

        fm = files.MultiFileManager()

        class MyThread(Process):
            name = "Process-"
            interrupted = False
            fileManager = files.FileManager()
            folderB = os.getcwd() + "/testFilesB"

            def setConfig(self, fileM):
                self.fileManager = fileM

            def interrupt(self):
                self.interrupted = True

            def __init__(self):
                super().__init__()


            def run(self):
                while True:
                    if self.interrupted:
                        break
                    if manageFiles(self.name, self.fileManager, self.folderB):
                        break

        class MovingThread(Process):
            name = "TProcess-"
            interrupted = False
            fileManager = files.MultiFileManager()
            #printlock = threading.Lock()
            folderA = os.getcwd() + "/testFilesA"
            folderB = os.getcwd() + "/testFilesB"

            def setConfig(self, fileM):
                self.fileManager = fileM

            def interrupt(self):
                self.interrupted = True

            def __init__(self):
                super().__init__()

            def run(self):
                while True:
                    if self.interrupted:
                        break
                    if fm.countFiles(folderA) == 0:
                        break
                    if not fm.moveFile(folderA, folderB):
                        break

        print("Multi")
        start = time.process_time()
        threads = []
        movers = []
        initTest()
        for i in range(config.ConfigManager.get_threads()):
            threads.append(MyThread())
            movers.append(MovingThread())
        for m in movers:
            m.start()
        for m in movers:
            m.join()
        for t in threads:
            t.setConfig(fm)
            t.start()
        print("Stopping")
        while fm.countFiles(folderA) > 0:
            time.sleep(1)

        for t in threads:
            t.interrupt()
            t.join()
        end = time.process_time() - start
        print("End: " + str(end))
        #fm.clearFolder(folderB)
        return end

    def comparison():
        runs = 20
        singles = []
        threadings = []
        multis = []
        for i in range(runs):
            print("Starting Run " + str(i))
            singles.append(doTestSingle())
            threadings.append(doTestThread())
            multis.append(doTestMulti())

        print("-------------------")
        print("-------------------")
        print("-------------------")
        print("Single average:  " + str(sum(singles) / len(singles)))
        print("Thread average:  " + str(sum(threadings) / len(threadings)))
        print("Multi average:  " + str(sum(multis) / len(multis)))

    #doTestSingle()
    #doTestThread()
    #doTestMulti()
    comparison()


if __name__ == "__main__":
    testFileManager()
