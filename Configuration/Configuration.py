import configparser as cp
import os
from Maths.Functions import measureTime


class ConfigManager:
    conf = cp.ConfigParser()
    settings_file = os.getcwd() + "/Configuration.settings.ini"
    ovr = 'SETTINGS'
    def_loc = os.getcwd()
    def_name = 'file_#.txt'
    def_firstStore = os.getcwd()
    def_secondStore = os.getcwd() + "/second"
    def_width = 200
    def_height = 200
    initialized = False

    # Reset parameters to default values
    @staticmethod
    @measureTime
    def _writeDefaults():
        if not ConfigManager.initialized: ConfigManager.setup()
        ConfigManager.conf[ConfigManager.ovr] = {'threads': 1,
                                                 'file_location': ConfigManager.def_loc,
                                                 'file_namescheme': ConfigManager.def_name,
                                                 'first_storage': ConfigManager.def_firstStore,
                                                 'second_storage': ConfigManager.def_secondStore,
                                                 'width': ConfigManager.def_width,
                                                 'height': ConfigManager.def_height}
        with open(ConfigManager.settings_file, 'w') as settings:
            ConfigManager.conf.write(settings)

    # Create a new settings-file if not yet existent
    @staticmethod
    @measureTime
    def setup():
        ConfigManager.initialized = True
        if not os.path.exists(ConfigManager.settings_file):
            ConfigManager._writeDefaults()

        ConfigManager.conf.read(ConfigManager.settings_file)

    # return THREADS parameter
    @staticmethod
    @measureTime
    def get_threads():
        if not ConfigManager.initialized: ConfigManager.setup()
        ConfigManager.conf.read(ConfigManager.settings_file)
        return int(ConfigManager.conf[ConfigManager.ovr]['threads'])  # ToDo: Castch Exceptions

    # set THREADS parameter
    @staticmethod
    @measureTime
    def set_threads(num):
        if not ConfigManager.initialized: ConfigManager.setup()
        ConfigManager.conf[ConfigManager.ovr]['threads'] = str(num)
        with open(ConfigManager.settings_file, "w") as settings:
            ConfigManager.conf.write(settings)

    # get FILE_LOCATION parameter
    @staticmethod
    @measureTime
    def get_fileLocation():
        if not ConfigManager.initialized: ConfigManager.setup()
        ConfigManager.conf.read(ConfigManager.settings_file)
        return ConfigManager.conf[ConfigManager.ovr]['file_location']

    # set FILE_LOCATION parameter
    @staticmethod
    @measureTime
    def set_fileLocation(loc):
        if not ConfigManager.initialized: ConfigManager.setup()
        ConfigManager.conf[ConfigManager.ovr]['file_location'] = str(loc)
        with open(ConfigManager.settings_file, "w") as settings:
            ConfigManager.conf.write(settings)

    # get FILE_NAME parameter
    @staticmethod
    @measureTime
    def get_fileName():
        if not ConfigManager.initialized: ConfigManager.setup()
        ConfigManager.conf.read(ConfigManager.settings_file)
        return ConfigManager.conf[ConfigManager.ovr]['file_namescheme']

    # set FILE_NAME parameter
    @staticmethod
    @measureTime
    def set_fileName(name):
        if not ConfigManager.initialized: ConfigManager.setup()
        ConfigManager.conf[ConfigManager.ovr]['file_namescheme'] = str(name)
        with open(ConfigManager.settings_file, "w") as settings:
            ConfigManager.conf.write(settings)

    # set FIRST_STORAGE parameter
    @staticmethod
    @measureTime
    def set_firstStorage(path):
        if not ConfigManager.initialized: ConfigManager.setup()
        ConfigManager.conf[ConfigManager.ovr]['first_storage'] = str(path)
        with open(ConfigManager.settings_file, "w") as settings:
            ConfigManager.conf.write(settings)

    # get FIRST_STORAGE parameter
    @staticmethod
    @measureTime
    def get_firstStorage():
        if not ConfigManager.initialized: ConfigManager.setup()
        ConfigManager.conf.read(ConfigManager.settings_file)
        return ConfigManager.conf[ConfigManager.ovr]['first_Storage']

    # set SECOND_STORAGE parameter
    @staticmethod
    @measureTime
    def set_secondStorage(path):
        if not ConfigManager.initialized: ConfigManager.setup()
        ConfigManager.conf[ConfigManager.ovr]['second_storage'] = str(path)
        with open(ConfigManager.settings_file, "w") as settings:
            ConfigManager.conf.write(settings)

    # get SECOND_STORAGE parameter
    @staticmethod
    @measureTime
    def get_secondStorage():
        if not ConfigManager.initialized: ConfigManager.setup()
        ConfigManager.conf.read(ConfigManager.settings_file)
        return ConfigManager.conf[ConfigManager.ovr]['second_Storage']

    # print each line in settings_file
    @staticmethod
    @measureTime
    def print_file():
        if not ConfigManager.initialized: ConfigManager.setup()
        with open(ConfigManager.settings_file, "r") as sf:
            for line in sf:
                print(line)

    @staticmethod
    @measureTime
    def get_width():
        if not ConfigManager.initialized: ConfigManager.setup()
        ConfigManager.conf.read(ConfigManager.settings_file)
        return int(ConfigManager.conf[ConfigManager.ovr]['width'])

    @staticmethod
    @measureTime
    def get_height():
        if not ConfigManager.initialized: ConfigManager.setup()
        ConfigManager.conf.read(ConfigManager.settings_file)
        return int(ConfigManager.conf[ConfigManager.ovr]['height'])

    @staticmethod
    @measureTime
    def set_width(w):
        if not ConfigManager.initialized: ConfigManager.setup()
        ConfigManager.conf[ConfigManager.ovr]['width'] = str(w)
        with open(ConfigManager.settings_file, "w") as settings:
            ConfigManager.conf.write(settings)

    @staticmethod
    @measureTime
    def set_heigth(h):
        if not ConfigManager.initialized: ConfigManager.setup()
        ConfigManager.conf[ConfigManager.ovr]['height'] = str(h)
        with open(ConfigManager.settings_file, "w") as settings:
            ConfigManager.conf.write(settings)

    # testing this script
    @staticmethod
    @measureTime
    def test():
        if not ConfigManager.initialized: ConfigManager.setup()
        print("Testing ConfigManager.py\n")
        print("Setup...")
        ConfigManager.setup()
        print("Get Attributes")
        print("Threads: " + ConfigManager.get_threads(), end='\n')
        print("File_Location " + ConfigManager.get_fileLocation(), end='\n')
        print("NameScheme " + ConfigManager.get_fileName(), end='\n')


class ThreadsafeConfigManager(ConfigManager):
    def __init__(self):
        super(ThreadsafeConfigManager, self).__init__()


class MultiConfigManager(ThreadsafeConfigManager):
    def __init__(self):
        super(MultiConfigManager, self).__init__()
