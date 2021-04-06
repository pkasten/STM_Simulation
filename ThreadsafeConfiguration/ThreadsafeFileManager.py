import os
import threading

mv_from = ""
mv_to = ""
namescheme = ""
edit_lock = threading.Lock()


# Checks wheather the name fits the given scheme
def _fits_scheme(name, namescheme):
    suffix = str(namescheme).split('.', maxsplit=1)[1]
    prefix = str(namescheme).split('#', maxsplit=1)[0]
    if not name.startswith(prefix): return False
    if not name.endswith(suffix): return False
    if extractNums(name) == "": return False
    return True


# returns number of files inside folder
def countFiles(folder):
    if folder is None:
        folder = os.getcwd()
    return len(os.listdir(folder))


# extract all Numbers from given String
def extractNums(string):
    target = ""
    for c in string:
        if c.isdigit():
            target = target + c
    return target


# returns path to maximum Indexed file. Namescheme Index##### where # means a number
def maxIndexFile(folder, namescheme):
    suffix = str(namescheme).split('.', maxsplit=1)[1]
    prefix = str(namescheme).split('#', maxsplit=1)[0]
    if folder is None:
        folder = os.getcwd()
    files = os.listdir(folder)
    indexes = []
    # besser andere funktion, immer 1.?
    for file in files:
        if not _fits_scheme(file, namescheme): continue
        index = extractNums(file)
        indexes.append(int(index))

    if len(indexes) == 0:
        return

    return prefix + str(max(indexes)) + "." + suffix


# returns path to minimum Indexed file. Namescheme Index##### where # means a number
def minIndexFile(folder, namescheme):
    suffix = str(namescheme).split('.', maxsplit=1)[1]
    prefix = str(namescheme).split('#', maxsplit=1)[0]
    if folder is None:
        folder = os.getcwd()
    files = os.listdir(folder)
    indexes = []
    # besser andere funktion, immer 1.?
    for file in files:
        if not _fits_scheme(file, namescheme): continue
        index = extractNums(file)
        indexes.append(int(index))

    if len(indexes) == 0:
        return

    return prefix + str(min(indexes)) + "." + suffix


# returns one File
def firstFile(folder, namescheme):
    if folder is None:
        folder = os.getcwd()
    files = os.listdir(folder)
    for file in files:
        if not _fits_scheme(file, namescheme): continue
        return file
    return ""


# removes one File
def removeFirst(folder, namescheme):
    ff = firstFile(folder, namescheme)
    if ff == "":
        return
    edit_lock.acquire()
    try:
        os.remove(os.path.join(folder, ff))
    except FileNotFoundError as fnfe:
        None
    edit_lock.release()


# removes min File
def removeMin(folder, namescheme):
    mif = minIndexFile(folder, namescheme)
    if mif == "":
        return
    try:
        os.remove(os.path.join(folder, mif))
    except FileNotFoundError as fnfe:
        None
    edit_lock.acquire()
    edit_lock.release()


# removes max File
def removeMax(folder, namescheme):
    mif = maxIndexFile(folder, namescheme)
    if mif == "":
        return
    edit_lock.acquire()
    try:
        os.remove(os.path.join(folder, mif))
    except FileNotFoundError as fnfe:
        None
    edit_lock.release()


# testing functionalities
def test():
    path = os.getcwd() + "/testfiles"
    nameless = input("Enter namescheme")
    folder = input("Enter Folder")
    print("Generating some Testfiles inside " + path)
    suffix = str(nameless).split('.', maxsplit=1)[1]
    prefix = str(nameless).split('#', maxsplit=1)[0]
    try:
        os.mkdir(path)
    except FileExistsError as fe:
        None

    filenames = []
    for i in range(37):
        filenames.append(str.format("testfiles/{}{}.{}", prefix, i, suffix))

    for file in filenames:
        with open(file, "w") as current:
            current.write("Testdatei")

    if len(folder) < 2:
        folder = path

    print("First File: " + firstFile(folder, nameless))
    print("Max indexed File: " + maxIndexFile(folder, nameless))
    print("Min indexed Folder: " + minIndexFile(folder, nameless))

    print("Removing testfiles")

    for file in os.listdir(path):
        os.remove(os.path.join(path, file))
    os.rmdir(path)

    print("Testing Done")


if __name__ == "__main__":
    test()
