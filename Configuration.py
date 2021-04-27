import configparser as cp
import os
from TestsApril.Maths.Functions import measureTime

conf = cp.ConfigParser()
#settings_folder = os.path.join(os.getcwd(), "Configuration")
settings_folder = os.getcwd()
settings_file = str(os.path.join(settings_folder, "settings.ini"))
# ovr = 'SETTINGS'
# def_threads = os.cpu_count()
# def_loc = os.getcwd()
# def_name = 'file_#.txt'
# def_firstStore_images = str(os.path.join(os.getcwd(), "bildordner"))
# def_secondStore_images = str(os.path.join(os.getcwd(), "bildordner2"))
# def_firstStore_data = str(os.path.join(os.getcwd(), "data"))
# def_secondStore_data = str(os.path.join(os.getcwd(), "data2"))
# def_width = 400
# def_height = 400
# def_images_per_thread = 25
initialized = False

cat_pc = 'computer_settings'
def_threads = 'threads', os.cpu_count()
def_images_per_thread = 'images_per_thread', 20
def_image_folder = 'image_folder', str(os.path.join(os.getcwd(), "bildordner"))
def_data_folder = 'data_folder', str(os.path.join(os.getcwd(), "data"))
pc_settings = [def_threads, def_images_per_thread, def_image_folder, def_data_folder]

cat_image_basics = 'image_settings'
def_prefix_image = 'prefix_image', 'Image'
def_suffix_image = 'suffix_image', '.png'
def_prefix_data = 'prefix_data', 'Data'
def_suffix_data = 'suffix_data', '.txt'
def_width = 'width', 400
def_height = 'height', 400
def_particles = 'no_of_particles', 30
def_px_overlap = 'pixels_overlap (in px)', 40
image_basics_settings = [def_prefix_image, def_suffix_image,  def_prefix_data, def_suffix_data, def_width, def_height, def_particles, def_px_overlap]

cat_particle_properties = 'particle_properties'
def_width_part = 'width', 1
def_length_part = 'length', 10
def_height_part = 'height', 1
def_image_path = 'image_path', ""
particle_properties_settings = [def_width_part, def_image_path, def_length_part, def_height_part]


# Reset parameters to default values

@measureTime
def _writeDefaults():

    conf[cat_pc] = {x[0]: x[1] for x in pc_settings}
    conf[cat_image_basics] = {x[0]: x[1] for x in image_basics_settings}
    conf[cat_particle_properties] = {x[0]: x[1] for x in particle_properties_settings}
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
def setupConfigurationManager():
    global initialized
    initialized = True
    if not os.path.exists(settings_file):
        _writeDefaults()

    conf.read(settings_file)


# return THREADS parameter

@measureTime
def get_threads():
    if not initialized:
        setupConfigurationManager()
    conf.read(settings_file)
    return int(conf[cat_pc][def_threads[0]])  # ToDo: Catch Exceptions


# get Images Per Thread

@measureTime
def get_images_pt():
    if not initialized: setupConfigurationManager()
    conf.read(settings_file)
    return int(conf[cat_pc][def_images_per_thread[0]])  # ToDo: Castch Exceptions

    # set Images Per Thread


@measureTime
def set_images_pt(num):
    if not initialized: setupConfigurationManager()
    conf[cat_pc][def_images_per_thread[0]] = str(num)
    with open(settings_file, "w") as settings:
        conf.write(settings)

    # set THREADS parameter


@measureTime
def set_threads(num):
    if not initialized: setupConfigurationManager()
    conf[cat_pc][def_threads[0]] = str(num)
    with open(settings_file, "w") as settings:
        conf.write(settings)

    # get FILE_LOCATION parameter


@measureTime
def set_image_folder(path):
    global settings_file
    if not initialized: setupConfigurationManager()
    if "\\" not in str(path) and "/" not in str(path):
        path = os.path.join(os.getcwd(), path)
    conf[cat_pc][def_image_folder[0]] = str(path)
    with open(settings_file, "w") as settings:
        conf.write(settings)


@measureTime
def get_image_folder():
    if not initialized: setupConfigurationManager()
    conf.read(settings_file)
    return conf[cat_pc][def_image_folder[0]]


@measureTime
def set_data_folder(path):
    global settings_file
    if not initialized: setupConfigurationManager()
    if "\\" not in str(path) and "/" not in str(path):
        path = os.path.join(os.getcwd(), path)
    conf[cat_pc][def_data_folder[0]] = str(path)
    with open(settings_file, "w") as settings:
        conf.write(settings)


@measureTime
def get_data_folder():
    global settings_file
    if not initialized: setupConfigurationManager()
    conf.read(settings_file)
    return conf[cat_pc][def_data_folder[0]]


@measureTime
def print_file():
    global settings_file
    if not initialized: setupConfigurationManager()
    with open(settings_file, "r") as sf:
        for line in sf:
            print(line, end="")


@measureTime
def get_width():
    global settings_file
    if not initialized: setupConfigurationManager()
    conf.read(settings_file)
    return int(conf[cat_image_basics][def_width[0]])


@measureTime
def get_height():
    global settings_file
    if not initialized: setupConfigurationManager()
    conf.read(settings_file)
    return int(conf[cat_image_basics][def_height[0]])


@measureTime
def set_width(w):
    if not initialized: setupConfigurationManager()
    conf[cat_image_basics][def_width[0]] = str(w)
    with open(settings_file, "w") as settings:
        conf.write(settings)


@measureTime
def set_heigth(h):
    global settings_file
    if not initialized: setupConfigurationManager()
    conf[cat_image_basics][def_height[0]] = str(h)
    with open(settings_file, "w") as settings:
        conf.write(settings)

def get_particles_per_image():
    global settings_file
    if not initialized: setupConfigurationManager()
    conf.read(settings_file)
    return int(conf[cat_image_basics][def_particles[0]])

def get_prefix_image():
    global settings_file
    if not initialized: setupConfigurationManager()
    conf.read(settings_file)
    return conf[cat_image_basics][def_prefix_image[0]]

def get_prefix_data():
    global settings_file
    if not initialized: setupConfigurationManager()
    conf.read(settings_file)
    return conf[cat_image_basics][def_prefix_data[0]]

def get_suffix_image():
    global settings_file
    if not initialized: setupConfigurationManager()
    conf.read(settings_file)
    return conf[cat_image_basics][def_suffix_image[0]]

def get_suffix_data():
    global settings_file
    if not initialized: setupConfigurationManager()
    conf.read(settings_file)
    return conf[cat_image_basics][def_suffix_data[0]]

def get_px_overlap():
    global settings_file
    if not initialized: setupConfigurationManager()
    conf.read(settings_file)
    return int(conf[cat_image_basics][def_px_overlap[0]])

def get_part_height():
    global settings_file
    if not initialized: setupConfigurationManager()
    conf.read(settings_file)
    return int(conf[cat_particle_properties][def_height_part[0]])

def get_part_width():
    global settings_file
    if not initialized: setupConfigurationManager()
    conf.read(settings_file)
    return int(conf[cat_particle_properties][def_width_part[0]])

def get_part_length():
    global settings_file
    if not initialized: setupConfigurationManager()
    conf.read(settings_file)
    return int(conf[cat_particle_properties][def_length_part[0]])

def get_image_path():
    global settings_file
    if not initialized: setupConfigurationManager()
    conf.read(settings_file)
    return conf[cat_particle_properties][def_image_path[0]]
