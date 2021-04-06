import random
import threading
import time
import ThreadsafeConfiguration.ThreadsafeFileManager as fm
import os

counter = 0
printlock = threading.Lock()
folder = os.getcwd() + "/testFiles"
namescheme = "TestfileNo#.txt"



def initTest():
    try:
        os.mkdir("testFiles")
    except FileExistsError as fe:
        None

    for k in range(300):
        filename = str.format("/TestfileNo{}.txt", k)
        path = folder + filename
        with open(path, "w") as f:
            f.write("Testdatei")


def manageFiles(name):
    rand = random.randrange(1, 6)

    if fm.countFiles(folder) == 0:
        return True
    if rand == 1:
        print("Thread " + name + " yields First: " + str(fm.firstFile(folder, namescheme)) + "\n")
    elif rand == 2:
        print("Thread " + name + " yields min: " + str(fm.minIndexFile(folder, namescheme)) + "\n")
    elif rand == 3:
        print("Thread " + name + " yields max: " + str(fm.maxIndexFile(folder, namescheme)) + "\n")
    elif rand == 4:
        fm.removeMax(folder, namescheme)
        print("Thread " + name + " removes max: " + "\n")
    elif rand == 5:
        fm.removeMin(folder, namescheme)
        print("Thread " + name + " removes min: " + "\n")
    else:
        print("Unknown")

    return False


class MyThread(threading.Thread):
    name = "Thread-"
    interrupted = False

    def interrupt(self):
        self.interrupted = True

    def __init__(self):
        super().__init__()
        global counter
        printlock.acquire()
        print("Here I am: " + str(counter))
        printlock.release()
        self.name = self.name + str(counter)
        counter += 1

    def run(self):
        while True:
            if self.interrupted:
                break
            if manageFiles(self.name):
                break


threads = []
initTest()
for i in range(7):
    threads.append(MyThread())
for t in threads:
    t.start()
print("Stopping")
for t in threads:
    t.join()
