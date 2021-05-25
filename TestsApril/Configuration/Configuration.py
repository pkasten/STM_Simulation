import configparser as cp
import os
from Maths.Functions import measureTime

conf = cp.ConfigParser()
settings_folder = os.path.join(os.getcwd(), "Configuration")
settings_file = str(os.path.join(settings_folder, "settings.ini"))
ovr = 'SETTINGS'
def_threads = os.cpu_count()
def_loc = os.getcwd()
def_name = 'file_#.txt'
def_firstStore_images = str(os.path.join(os.getcwd(), "bildordner"))
def_secondStore_images = str(os.path.join(os.getcwd(), "bildordner2"))
def_firstStore_data = str(os.path.join(os.getcwd(), "data"))
def_secondStore_data = str(os.path.join(os.getcwd(), "data2"))
def_width = 400
def_height = 400
def_images_per_thread = 25
initialized = False


# Reset parameters to default values

@measureTime
def _writeDefaults():
    # if not initialized: setup()
    conf[ovr] = {'threads': def_threads,
                 'images_per_thread': def_images_per_thread,
                 'file_location': def_loc,
                 'file_namescheme': def_name,
                 'first_storage_Image': def_firstStore_images,
                 'second_storage_Image': def_secondStore_images,
                 'first_storage_Data': def_firstStore_data,
                 'second_storage_Data': def_secondStore_data,
                 'width': def_width,
                 'height': def_height}
    try:
        with open(settings_file, 'w') as settings:
            conf.write(settings)
    except FileNotFoundError:
        try:
            os.mkdir(settings_folder)
        except FileExistsError:
            with open(settings_file, 'w') as settings:
                settings.write("")
        with open(settings_file, 'w') as settings:
            conf.write(settings)


# Create a new settings-file if not yet existent

@measureTime
def setup():
    global initialized
    initialized = True
    if not os.path.exists(settings_file):
        _writeDefaults()

    conf.read(settings_file)


# return THREADS parameter

@measureTime
def get_threads():
    if not initialized:
        setup()
    conf.read(settings_file)
    return int(conf[ovr]['threads'])


# get Images Per Thread

@measureTime
def get_images_pt():
    if not initialized: setup()
    conf.read(settings_file)
    return int(conf[ovr]['images_per_thread'])

    # set Images Per Thread

@measureTime
def set_images_pt(num):
    if not initialized: setup()
    conf[ovr]['images_per_thread'] = str(num)
    with open(settings_file, "w") as settings:
        conf.write(settings)

    # set THREADS parameter

@measureTime
def set_threads(num):
    if not initialized: setup()
    conf[ovr]['threads'] = str(num)
    with open(settings_file, "w") as settings:
        conf.write(settings)

    # get FILE_LOCATION parameter

@measureTime
def get_root():
    if not initialized: setup()
    conf.read(settings_file)
    return conf[ovr]['file_location']

    # set FILE_LOCATION parameter

@measureTime
def set_root(loc):
    if not initialized: setup()
    if "\\" not in str(loc) and "/" not in str(loc):
        loc = os.path.join(os.getcwd(), loc)
    conf[ovr]['file_location'] = str(loc)
    with open(settings_file, "w") as settings:
        conf.write(settings)

    # get FILE_NAME parameter

@measureTime
def get_fileName():
    if not initialized: setup()
    conf.read(settings_file)
    return conf[ovr]['file_namescheme']

    # set FILE_NAME parameter

@measureTime
def set_fileName(name):
    if not initialized: setup()
    conf[ovr]['file_namescheme'] = str(name)
    with open(settings_file, "w") as settings:
        conf.write(settings)

    # set FIRST_STORAGE parameter

@measureTime
def set_firstStorage_Image(path):
    global settings_file
    if not initialized: setup()
    if "\\" not in str(path) and "/" not in str(path):
        path = os.path.join(os.getcwd(), path)
    conf[ovr]['first_storage_Image'] = str(path)
    with open(settings_file, "w") as settings:
        conf.write(settings)

@measureTime
def set_secondStorage_Image(path):
    if not initialized: setup()
    if "\\" not in str(path) and "/" not in str(path):
        path = os.path.join(os.getcwd(), path)
    conf[ovr]['second_storage_Image'] = str(path)
    with open(settings_file, "w") as settings:
        conf.write(settings)

    # get FIRST_STORAGE parameter

@measureTime
def get_firstStorage_Image():
    if not initialized: setup()
    conf.read(settings_file)
    return conf[ovr]['first_storage_Image']

@measureTime
def get_secondStorage_Image():
    global settings_file
    if not initialized: setup()
    conf.read(settings_file)
    return conf[ovr]['second_storage_Image']

    # set FIRST_STORAGE parameter

@measureTime
def set_firstStorage_Data(path):
    global settings_file
    if not initialized: setup()
    if "\\" not in str(path) and "/" not in str(path):
        path = os.path.join(os.getcwd(), path)
    conf[ovr]['first_storage_Data'] = str(path)
    with open(settings_file, "w") as settings:
        conf.write(settings)

@measureTime
def set_secondStorage_Data(path):
    global settings_file
    if not initialized: setup()
    if "\\" not in str(path) and "/" not in str(path):
        path = os.path.join(os.getcwd(), path)
    conf[ovr]['second_storage_Data'] = str(path)
    with open(settings_file, "w") as settings:
        conf.write(settings)

    # get FIRST_STORAGE parameter

@measureTime
def get_firstStorage_Data():
    global settings_file
    if not initialized: setup()
    conf.read(settings_file)
    return conf[ovr]['first_storage_Data']

@measureTime
def get_secondStorage_Data():
    global settings_file
    if not initialized: setup()
    conf.read(settings_file)
    return conf[ovr]['second_storage_Data']

    # print each line in settings_file

@measureTime
def print_file():
    global settings_file
    if not initialized: setup()
    with open(settings_file, "r") as sf:
        for line in sf:
            print(line, end="")

@measureTime
def get_width():
    global settings_file
    if not initialized: setup()
    conf.read(settings_file)
    return int(conf[ovr]['width'])

@measureTime
def get_height():
    global settings_file
    if not initialized: setup()
    conf.read(settings_file)
    return int(conf[ovr]['height'])

@measureTime
def set_width(w):
    if not initialized: setup()
    conf[ovr]['width'] = str(w)
    with open(settings_file, "w") as settings:
        conf.write(settings)

@measureTime
def set_heigth(h):
    global settings_file
    if not initialized: setup()
    conf[ovr]['height'] = str(h)
    with open(settings_file, "w") as settings:
        conf.write(settings)

    # testing this script


@measureTime
def testCM():
    global settings_file, settings_folder
    settings_folder = os.getcwd()
    settings_file = str(os.path.join(settings_folder, "settings.ini"))

    print("File: ")
    print_file()

    print("Threads: " + str(get_threads()))
    print("Images Pt: " + str(get_images_pt()))
    print("Root: " + str(get_root()))
    print("File Name: " + str(get_fileName()))
    print("Width: " + str(get_width()))
    print("Heigth: " + str(get_height()))
    print("1 Image: " + str(get_firstStorage_Image()))
    print("2 Image: " + str(get_secondStorage_Image()))
    print("1 Data: " + str(get_firstStorage_Data()))
    print("2 Data: " + str(get_secondStorage_Data()))

    print("File: ")
    print_file()

    print("Setting: ")
    set_images_pt(100)
    set_threads(100)
    set_root("Here")
    set_fileName('FN')
    set_heigth(100)
    set_width(100)
    set_firstStorage_Data("1D")
    set_firstStorage_Image("1I")
    set_secondStorage_Data("2D")
    set_secondStorage_Image("2I")

    print("Threads: " + str(get_threads()))
    print("Images Pt: " + str(get_images_pt()))
    print("Root: " + str(get_root()))
    print("File Name: " + str(get_fileName()))
    print("Width: " + str(get_width()))
    print("Heigth: " + str(get_height()))
    print("1 Image: " + str(get_firstStorage_Image()))
    print("2 Image: " + str(get_secondStorage_Image()))
    print("1 Data: " + str(get_firstStorage_Data()))
    print("2 Data: " + str(get_secondStorage_Data()))

    print("Reset")

    _writeDefaults()

    print("Threads: " + str(get_threads()))
    print("Images Pt: " + str(get_images_pt()))
    print("Root: " + str(get_root()))
    print("File Name: " + str(get_fileName()))
    print("Width: " + str(get_width()))
    print("Heigth: " + str(get_height()))
    print("1 Image: " + str(get_firstStorage_Image()))
    print("2 Image: " + str(get_secondStorage_Image()))
    print("1 Data: " + str(get_firstStorage_Data()))
    print("2 Data: " + str(get_secondStorage_Data()))


if __name__ == "__main__":
    testCM()
