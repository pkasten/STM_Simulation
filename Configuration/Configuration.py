import configparser as cp
import os


class ConfigManager:
    conf = cp.ConfigParser()
    settings_file = os.getcwd() + "/Configuration.settings.ini"
    ovr = 'SETTINGS'
    def_loc = os.getcwd()
    def_name = 'file_#.txt'
    def_firstStore = os.getcwd()
    def_secondStore = os.getcwd() + "/second"
    initialized = False

    # Reset parameters to default values
    @staticmethod
    def _writeDefaults():
        if not ConfigManager.initialized: ConfigManager.setup()
        ConfigManager.conf[ConfigManager.ovr] = {'threads': 1, 'file_location': ConfigManager.def_loc,
                                                 'file_namescheme': ConfigManager.def_name,
                                                 'first_storage': ConfigManager.def_firstStore,
                                                 'second_storage': ConfigManager.def_secondStore}
        with open(ConfigManager.settings_file, 'w') as settings:
            ConfigManager.conf.write(settings)

    # Create a new settings-file if not yet existent
    @staticmethod
    def setup():
        ConfigManager.initialized = True
        if not os.path.exists(ConfigManager.settings_file):
            ConfigManager._writeDefaults()

        ConfigManager.conf.read(ConfigManager.settings_file)

    # return THREADS parameter
    @staticmethod
    def get_threads():
        if not ConfigManager.initialized: ConfigManager.setup()
        ConfigManager.conf.read(ConfigManager.settings_file)
        return int(ConfigManager.conf[ConfigManager.ovr]['threads'])  # ToDo: Castch Exceptions

    # set THREADS parameter
    @staticmethod
    def set_threads(num):
        if not ConfigManager.initialized: ConfigManager.setup()
        ConfigManager.conf[ConfigManager.ovr]['threads'] = str(num)
        with open(ConfigManager.settings_file, "w") as settings:
            ConfigManager.conf.write(settings)

    # get FILE_LOCATION parameter
    @staticmethod
    def get_fileLocation():
        if not ConfigManager.initialized: ConfigManager.setup()
        ConfigManager.conf.read(ConfigManager.settings_file)
        return ConfigManager.conf[ConfigManager.ovr]['file_location']

    # set FILE_LOCATION parameter
    @staticmethod
    def set_fileLocation(loc):
        if not ConfigManager.initialized: ConfigManager.setup()
        ConfigManager.conf[ConfigManager.ovr]['file_location'] = str(loc)
        with open(ConfigManager.settings_file, "w") as settings:
            ConfigManager.conf.write(settings)

    # get FILE_NAME parameter
    @staticmethod
    def get_fileName():
        if not ConfigManager.initialized: ConfigManager.setup()
        ConfigManager.conf.read(ConfigManager.settings_file)
        return ConfigManager.conf[ConfigManager.ovr]['file_namescheme']

    # set FILE_NAME parameter
    @staticmethod
    def set_fileName(name):
        if not ConfigManager.initialized: ConfigManager.setup()
        ConfigManager.conf[ConfigManager.ovr]['file_namescheme'] = str(name)
        with open(ConfigManager.settings_file, "w") as settings:
            ConfigManager.conf.write(settings)

    # set FIRST_STORAGE parameter
    @staticmethod
    def set_firstStorage(path):
        if not ConfigManager.initialized: ConfigManager.setup()
        ConfigManager.conf[ConfigManager.ovr]['first_storage'] = str(path)
        with open(ConfigManager.settings_file, "w") as settings:
            ConfigManager.conf.write(settings)

    # get FIRST_STORAGE parameter
    @staticmethod
    def get_firstStorage():
        if not ConfigManager.initialized: ConfigManager.setup()
        ConfigManager.conf.read(ConfigManager.settings_file)
        return ConfigManager.conf[ConfigManager.ovr]['first_Storage']

    # set SECOND_STORAGE parameter
    @staticmethod
    def set_secondStorage(path):
        if not ConfigManager.initialized: ConfigManager.setup()
        ConfigManager.conf[ConfigManager.ovr]['second_storage'] = str(path)
        with open(ConfigManager.settings_file, "w") as settings:
            ConfigManager.conf.write(settings)

    # get SECOND_STORAGE parameter
    @staticmethod
    def get_secondStorage():
        if not ConfigManager.initialized: ConfigManager.setup()
        ConfigManager.conf.read(ConfigManager.settings_file)
        return ConfigManager.conf[ConfigManager.ovr]['second_Storage']

    # print each line in settings_file
    @staticmethod
    def print_file():
        if not ConfigManager.initialized: ConfigManager.setup()
        with open(ConfigManager.settings_file, "r") as sf:
            for line in sf:
                print(line)

    # testing this script
    @staticmethod
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
