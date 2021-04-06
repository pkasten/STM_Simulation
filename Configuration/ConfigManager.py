import configparser as cp
import os

conf = cp.ConfigParser()
settings_file = os.getcwd() + "/Configuration.settings.ini"
ovr = 'SETTINGS'
def_loc = os.getcwd()
def_name = 'file_#.txt'
initialized = False


# Reset parameters to default values
def _writeDefaults():
    if not initialized: setup()
    conf[ovr] = {'threads': 1, 'file_location': def_loc, 'file_namescheme': def_name}
    with open(settings_file, 'w') as settings:
        conf.write(settings)


# Create a new settings-file if not yet existent
def setup():
    if not os.path.exists(settings_file):
        _writeDefaults()

    conf.read(settings_file)


# return THREADS parameter
def get_threads():
    if not initialized: setup()
    conf.read(settings_file)
    return conf[ovr]['threads']


# set THREADS parameter
def set_threads(num):
    if not initialized: setup()
    conf[ovr]['threads'] = str(num)
    with open(settings_file, "w") as settings:
        conf.write(settings)


# get FILE_LOCATION parameter
def get_fileLocation():
    if not initialized: setup()
    conf.read(settings_file)
    return conf[ovr]['file_location']


# set FILE_LOCATION parameter
def set_fileLocation(loc):
    if not initialized: setup()
    conf[ovr]['file_location'] = str(loc)
    with open(settings_file, "w") as settings:
        conf.write(settings)


# get FILE_NAME parameter
def get_fileName():
    if not initialized: setup()
    conf.read(settings_file)
    return conf[ovr]['file_namescheme']


# set FILE_NAME parameter
def set_fileName(name):
    if not initialized: setup()
    conf[ovr]['file_namescheme'] = str(name)
    with open(settings_file, "w") as settings:
        conf.write(settings)


# print each line in settings_file
def print_file():
    if not initialized: setup()
    with open(settings_file, "r") as sf:
        for line in sf:
            print(line)


# testing this script
def test():
    if not initialized: setup()
    print("Testing ConfigManager.py\n")
    print("Setup...")
    setup()
    print("Get Attributes")
    print("Threads: " + get_threads(), end='\n')
    print("File_Location " + get_fileLocation(), end='\n')
    print("NameScheme " + get_fileName(), end='\n')


if __name__ == "__main__":
    test()
