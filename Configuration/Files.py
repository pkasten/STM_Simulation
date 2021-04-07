import os
import threading
from multiprocessing import Lock as mpLock


class FileManager:

    index = 0
    defaultFolder = os.getcwd()

    @staticmethod
    def setDefaultFolder(path):
        FileManager.defaultFolder = path

    @staticmethod
    def getFileName():
        FileManager.index += 1
        return FileManager.defaultFolder + "/image" + str(FileManager.index) + ".png"

    # Checks wheather the name fits the given scheme
    @staticmethod
    def _fits_scheme(name, namescheme):
        if namescheme is None:
            return True
        suffix = str(namescheme).split('.', maxsplit=1)[1]
        prefix = str(namescheme).split('#', maxsplit=1)[0]
        if not name.startswith(prefix): return False
        if not name.endswith(suffix): return False
        if FileManager.extractNums(name) == "": return False
        return True

    #clears the entire folder. asks for premission
    @staticmethod
    def clearFolder(folder):
        if input("Clear folder " + folder + "?").lower().startswith("n"):
            return
        while FileManager.countFiles(folder) > 0:
            FileManager.removeFirst(folder, None)

    # returns number of files inside folder
    @staticmethod
    def countFiles(folder):
        if folder is None:
            folder = os.getcwd()
        return len(os.listdir(folder))

    # extract all Numbers from given String
    @staticmethod
    def extractNums(string):
        target = ""
        for c in string:
            if c.isdigit():
                target = target + c
        return target

    # returns path to maximum Indexed file. Namescheme Index##### where # means a number
    @staticmethod
    def maxIndexFile(folder, namescheme):
        suffix = str(namescheme).split('.', maxsplit=1)[1]
        prefix = str(namescheme).split('#', maxsplit=1)[0]
        if folder is None:
            folder = os.getcwd()
        files = os.listdir(folder)
        indexes = []
        # besser andere funktion, immer 1.?
        for file in files:
            if not FileManager._fits_scheme(file, namescheme): continue
            index = FileManager.extractNums(file)
            indexes.append(int(index))

        if len(indexes) == 0:
            return

        return prefix + str(max(indexes)) + "." + suffix

    # returns path to minimum Indexed file. Namescheme Index##### where # means a number
    @staticmethod
    def minIndexFile(folder, namescheme):
        suffix = str(namescheme).split('.', maxsplit=1)[1]
        prefix = str(namescheme).split('#', maxsplit=1)[0]
        if folder is None:
            folder = os.getcwd()
        files = os.listdir(folder)
        indexes = []
        # besser andere funktion, immer 1.?
        for file in files:
            if not FileManager._fits_scheme(file, namescheme): continue
            index = FileManager.extractNums(file)
            indexes.append(int(index))

        if len(indexes) == 0:
            return

        return prefix + str(min(indexes)) + "." + suffix

    # returns one File
    @staticmethod
    def firstFile(folder, namescheme):
        if folder is None:
            folder = os.getcwd()
        files = os.listdir(folder)
        for file in files:
            if not FileManager._fits_scheme(file, namescheme): continue
            return file
        return ""

    # removes one File
    @staticmethod
    def removeFirst(folder, namescheme):
        if folder is None:
            folder = os.getcwd()
        os.remove(os.path.join(folder, FileManager.firstFile(folder, namescheme)))

    # removes min File
    @staticmethod
    def removeMin(folder, namescheme):
        if folder is None:
            folder = os.getcwd()
        os.remove(os.path.join(folder, FileManager.minIndexFile(folder, namescheme)))

    # removes max File
    @staticmethod
    def removeMax(folder, namescheme):
        if folder is None:
            folder = os.getcwd()
        os.remove(os.path.join(folder, FileManager.maxIndexFile(folder, namescheme)))

    #moves files from Folder A to Folder B
    @staticmethod
    def moveFile(folderA, folderB):
        file = FileManager.firstFile(folderA, None)
        #print("Moving file " + file + " At Size " + str(FileManager.countFiles(folderA)))
        if file is None:
            return False
        os.rename(os.path.join(folderA, file), os.path.join(folderB, file))
        return True

    # testing functionalities
    def test(self):
        path = os.getcwd() + "/testfiles"
        nameless = input("Enter namescheme")
        folder = input("Enter Folder")
        print("Generating some Testfiles inside " + path)
        suffix = str(nameless).split('.', maxsplit=1)[1]
        prefix = str(nameless).split('#', maxsplit=1)[0]
        try:
            os.mkdir(path)
        except FileExistsError as fe:
            pass

        filenames = []
        for i in range(37):
            filenames.append(str.format("testfiles/{}{}.{}", prefix, i, suffix))

        for file in filenames:
            with open(file, "w") as current:
                current.write("Testdatei")

        if len(folder) < 2:
            folder = path

        print("First File: " + self.firstFile(folder, nameless))
        print("Max indexed File: " + self.maxIndexFile(folder, nameless))
        print("Min indexed Folder: " + self.minIndexFile(folder, nameless))

        print("Removing testfiles")

        for file in os.listdir(path):
            os.remove(os.path.join(path, file))
        os.rmdir(path)

        print("Testing Done")


class ThreadsafeFileManager(FileManager):
    #ToDo: Check if all Methods areimplemented
    edit_lock = threading.Lock()
    filename_lock = threading.Lock()

    @staticmethod
    def getFileName():
        ThreadsafeFileManager.filename_lock.acquire()
        ThreadsafeFileManager.index += 1
        ret = ThreadsafeFileManager.defaultFolder + "/image" + str(ThreadsafeFileManager.index) + ".png"
        ThreadsafeFileManager.filename_lock.release()
        return ret

    # removes one File
    def removeFirst(self, folder, namescheme):
        redo = False
        ff = ThreadsafeFileManager.firstFile(folder, namescheme)
        if ff is None:
            return
        self.edit_lock.acquire()
        try:
            if os.path.exists(folder + "/" + ff):
                os.remove(os.path.join(folder, ff))
            else:
                #print("Not Existant: " + folder + "/" + ff)
                redo = True
        except FileNotFoundError as fnfe:
            pass
        finally:
            self.edit_lock.release()
        if redo:
            self.removeFirst(folder, namescheme)

    def removeFirstNameless(self, folder):
        self.removeFirst(folder, None)


    def clearFolder(self, folder):
        while ThreadsafeFileManager.countFiles(folder) > 0:
            self.removeFirstNameless(folder)

    # moves files from Folder A to Folder B
    def moveFile(self, folderA, folderB):
        #print("Moving TS")
        self.edit_lock.acquire()
        file = ThreadsafeFileManager.firstFile(folderA, None)
        #print("Moving file " + file + " At Size " + str(ThreadsafeFileManager.countFiles(folderA)))
        if file is None:
            print("File is none")
            return False
        try:
            os.rename(folderA + "/" + file, folderB + "/" + file)
        except OSError:
            return False
        finally:
            self.edit_lock.release()


        return True

    # removes min File
    def removeMin(self, folder, namescheme):
        redo = False
        mif = ThreadsafeFileManager.minIndexFile(folder, namescheme)
        if mif is None:
            return
        self.edit_lock.acquire()
        try:
            if os.path.exists(folder + "/" + mif):
                os.remove(os.path.join(folder, mif))
            else:
                #print("Not Existent: " + folder + "/" + mif)
                redo = True
        except FileNotFoundError as fnfe:
            pass
        finally:
            self.edit_lock.release()

        if redo: self.removeMin(folder, namescheme)



    # removes max File
    def removeMax(self, folder, namescheme):
        redo = False
        mif = self.maxIndexFile(folder, namescheme)
        if mif is None:
            return
        self.edit_lock.acquire()
        try:
            if os.path.exists(folder + "/" + mif):
                os.remove(os.path.join(folder, mif))
            else:
                #print("Not Existent: " + folder + "/" + mif)
                #print(self.countFiles(folder))
                redo = True
        except FileNotFoundError as fnfe:
            pass
        finally:
            self.edit_lock.release()
        if redo:
            self.removeMax(folder, namescheme)

    # testing functionalities
    def test(self):
        path = os.getcwd() + "/testfiles"
        nameless = input("Enter namescheme")
        folder = input("Enter Folder")
        print("Generating some Testfiles inside " + path)
        suffix = str(nameless).split('.', maxsplit=1)[1]
        prefix = str(nameless).split('#', maxsplit=1)[0]
        try:
            os.mkdir(path)
        except FileExistsError as fe:
            pass

        filenames = []
        for i in range(37):
            filenames.append(str.format("testfiles/{}{}.{}", prefix, i, suffix))

        for file in filenames:
            with open(file, "w") as current:
                current.write("Testdatei")

        if len(folder) < 2:
            folder = path

        print("First File: " + self.firstFile(folder, nameless))
        print("Max indexed File: " + self.maxIndexFile(folder, nameless))
        print("Min indexed Folder: " + self.minIndexFile(folder, nameless))

        print("Removing testfiles")

        for file in os.listdir(path):
            os.remove(os.path.join(path, file))
        os.rmdir(path)

        print("Testing Done")


class MultiFileManager(ThreadsafeFileManager):
    edit_lock = mpLock()
    filename_lock = mpLock()

    @staticmethod
    def clearFolder(folder):
        while MultiFileManager.countFiles(folder) > 0:
            MultiFileManager.removeFirst(folder, None)

    def moveFile(self, folderA, folderB):
        self.edit_lock.acquire()
        file = MultiFileManager.firstFile(folderA, None)
        #print("Moving file " + file + " At Size " + str(MultiFileManager.countFiles(folderA)))
        if file is None:
            self.edit_lock.release()
            return False
        try:
            os.rename(folderA + "/" + file, folderB + "/" + file)
        except OSError:
            return False
        finally:
            self.edit_lock.release()
        return True
