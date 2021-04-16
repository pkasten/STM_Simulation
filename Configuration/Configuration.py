import configparser as cp
import os
from Maths.Functions import measureTime


class ConfigManager:
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
    @staticmethod
    @measureTime
    def _writeDefaults():
        # if not ConfigManager.initialized: ConfigManager.setup()
        ConfigManager.conf[ConfigManager.ovr] = {'threads': ConfigManager.def_threads,
                                                 'images_per_thread': ConfigManager.def_images_per_thread,
                                                 'file_location': ConfigManager.def_loc,
                                                 'file_namescheme': ConfigManager.def_name,
                                                 'first_storage_Image': ConfigManager.def_firstStore_images,
                                                 'second_storage_Image': ConfigManager.def_secondStore_images,
                                                 'first_storage_Data': ConfigManager.def_firstStore_data,
                                                 'second_storage_Data': ConfigManager.def_secondStore_data,
                                                 'width': ConfigManager.def_width,
                                                 'height': ConfigManager.def_height}
        try:
            with open(ConfigManager.settings_file, 'w') as settings:
                ConfigManager.conf.write(settings)
        except FileNotFoundError:
            try:
                os.mkdir(ConfigManager.settings_folder)
            except FileExistsError:
                with open(ConfigManager.settings_file, 'w') as settings:
                    settings.write("")
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
        if not ConfigManager.initialized:
            ConfigManager.setup()
        ConfigManager.conf.read(ConfigManager.settings_file)
        return int(ConfigManager.conf[ConfigManager.ovr]['threads'])  # ToDo: Catch Exceptions

    # get Images Per Thread
    @staticmethod
    @measureTime
    def get_images_pt():
        if not ConfigManager.initialized: ConfigManager.setup()
        ConfigManager.conf.read(ConfigManager.settings_file)
        return int(ConfigManager.conf[ConfigManager.ovr]['images_per_thread'])  # ToDo: Castch Exceptions

    # set Images Per Thread
    @staticmethod
    @measureTime
    def set_images_pt(num):
        if not ConfigManager.initialized: ConfigManager.setup()
        ConfigManager.conf[ConfigManager.ovr]['images_per_thread'] = str(num)
        with open(ConfigManager.settings_file, "w") as settings:
            ConfigManager.conf.write(settings)

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
    def get_root():
        if not ConfigManager.initialized: ConfigManager.setup()
        ConfigManager.conf.read(ConfigManager.settings_file)
        return ConfigManager.conf[ConfigManager.ovr]['file_location']

    # set FILE_LOCATION parameter
    @staticmethod
    @measureTime
    def set_root(loc):
        if not ConfigManager.initialized: ConfigManager.setup()
        if "\\" not in str(loc) and "/" not in str(loc):
            loc = os.path.join(os.getcwd(), loc)
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
    def set_firstStorage_Image(path):
        if not ConfigManager.initialized: ConfigManager.setup()
        if "\\" not in str(path) and "/" not in str(path):
            path = os.path.join(os.getcwd(), path)
        ConfigManager.conf[ConfigManager.ovr]['first_storage_Image'] = str(path)
        with open(ConfigManager.settings_file, "w") as settings:
            ConfigManager.conf.write(settings)

    @staticmethod
    @measureTime
    def set_secondStorage_Image(path):
        if not ConfigManager.initialized: ConfigManager.setup()
        if "\\" not in str(path) and "/" not in str(path):
            path = os.path.join(os.getcwd(), path)
        ConfigManager.conf[ConfigManager.ovr]['second_storage_Image'] = str(path)
        with open(ConfigManager.settings_file, "w") as settings:
            ConfigManager.conf.write(settings)

    # get FIRST_STORAGE parameter
    @staticmethod
    @measureTime
    def get_firstStorage_Image():
        if not ConfigManager.initialized: ConfigManager.setup()
        ConfigManager.conf.read(ConfigManager.settings_file)
        return ConfigManager.conf[ConfigManager.ovr]['first_storage_Image']

    @staticmethod
    @measureTime
    def get_secondStorage_Image():
        if not ConfigManager.initialized: ConfigManager.setup()
        ConfigManager.conf.read(ConfigManager.settings_file)
        return ConfigManager.conf[ConfigManager.ovr]['second_storage_Image']

    # set FIRST_STORAGE parameter
    @staticmethod
    @measureTime
    def set_firstStorage_Data(path):
        if not ConfigManager.initialized: ConfigManager.setup()
        if "\\" not in str(path) and "/" not in str(path):
            path = os.path.join(os.getcwd(), path)
        ConfigManager.conf[ConfigManager.ovr]['first_storage_Data'] = str(path)
        with open(ConfigManager.settings_file, "w") as settings:
            ConfigManager.conf.write(settings)

    @staticmethod
    @measureTime
    def set_secondStorage_Data(path):
        if not ConfigManager.initialized: ConfigManager.setup()
        if "\\" not in str(path) and "/" not in str(path):
            path = os.path.join(os.getcwd(), path)
        ConfigManager.conf[ConfigManager.ovr]['second_storage_Data'] = str(path)
        with open(ConfigManager.settings_file, "w") as settings:
            ConfigManager.conf.write(settings)

    # get FIRST_STORAGE parameter
    @staticmethod
    @measureTime
    def get_firstStorage_Data():
        if not ConfigManager.initialized: ConfigManager.setup()
        ConfigManager.conf.read(ConfigManager.settings_file)
        return ConfigManager.conf[ConfigManager.ovr]['first_storage_Data']

    @staticmethod
    @measureTime
    def get_secondStorage_Data():
        if not ConfigManager.initialized: ConfigManager.setup()
        ConfigManager.conf.read(ConfigManager.settings_file)
        return ConfigManager.conf[ConfigManager.ovr]['second_storage_Data']

    # print each line in settings_file
    @staticmethod
    @measureTime
    def print_file():
        if not ConfigManager.initialized: ConfigManager.setup()
        with open(ConfigManager.settings_file, "r") as sf:
            for line in sf:
                print(line, end="")


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
        ConfigManager.settings_folder = os.getcwd()
        ConfigManager.settings_file = str(os.path.join(ConfigManager.settings_folder, "settings.ini"))

        print("File: ")
        ConfigManager.print_file()

        print("Threads: " + str(ConfigManager.get_threads()))
        print("Images Pt: " + str(ConfigManager.get_images_pt()))
        print("Root: " + str(ConfigManager.get_root()))
        print("File Name: " + str(ConfigManager.get_fileName()))
        print("Width: " + str(ConfigManager.get_width()))
        print("Heigth: " + str(ConfigManager.get_height()))
        print("1 Image: " + str(ConfigManager.get_firstStorage_Image()))
        print("2 Image: " + str(ConfigManager.get_secondStorage_Image()))
        print("1 Data: " + str(ConfigManager.get_firstStorage_Data()))
        print("2 Data: " + str(ConfigManager.get_secondStorage_Data()))

        print("File: ")
        ConfigManager.print_file()

        print("Setting: ")
        ConfigManager.set_images_pt(100)
        ConfigManager.set_threads(100)
        ConfigManager.set_root("Here")
        ConfigManager.set_fileName('FN')
        ConfigManager.set_heigth(100)
        ConfigManager.set_width(100)
        ConfigManager.set_firstStorage_Data("1D")
        ConfigManager.set_firstStorage_Image("1I")
        ConfigManager.set_secondStorage_Data("2D")
        ConfigManager.set_secondStorage_Image("2I")

        print("Threads: " + str(ConfigManager.get_threads()))
        print("Images Pt: " + str(ConfigManager.get_images_pt()))
        print("Root: " + str(ConfigManager.get_root()))
        print("File Name: " + str(ConfigManager.get_fileName()))
        print("Width: " + str(ConfigManager.get_width()))
        print("Heigth: " + str(ConfigManager.get_height()))
        print("1 Image: " + str(ConfigManager.get_firstStorage_Image()))
        print("2 Image: " + str(ConfigManager.get_secondStorage_Image()))
        print("1 Data: " + str(ConfigManager.get_firstStorage_Data()))
        print("2 Data: " + str(ConfigManager.get_secondStorage_Data()))

        print("Reset")

        ConfigManager._writeDefaults()

        print("Threads: " + str(ConfigManager.get_threads()))
        print("Images Pt: " + str(ConfigManager.get_images_pt()))
        print("Root: " + str(ConfigManager.get_root()))
        print("File Name: " + str(ConfigManager.get_fileName()))
        print("Width: " + str(ConfigManager.get_width()))
        print("Heigth: " + str(ConfigManager.get_height()))
        print("1 Image: " + str(ConfigManager.get_firstStorage_Image()))
        print("2 Image: " + str(ConfigManager.get_secondStorage_Image()))
        print("1 Data: " + str(ConfigManager.get_firstStorage_Data()))
        print("2 Data: " + str(ConfigManager.get_secondStorage_Data()))







if __name__ == "__main__":
    ConfigManager.test()

