import configparser as cp
import os, numpy as np
from TestsApril.Maths.Functions import measureTime

conf = cp.ConfigParser()
# settings_folder = os.path.join(os.getcwd(), "Configuration")
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
def_particles = 'no_of_particles', int(def_width[1] * def_height[1] / 16000)
def_px_overlap = 'pixels_overlap (in px)', 40
def_anti_aliasing = 'Anti-Aliasing', 1
def_noise_mu = 'Image-noise_Average', 0
def_noise_std_deriv = 'Image-noise-Standard_derivation', 0.1 * def_noise_mu[1]
image_basics_settings = [def_prefix_image, def_suffix_image, def_prefix_data, def_suffix_data, def_width, def_height,
                         def_particles, def_px_overlap, def_anti_aliasing, def_noise_mu, def_noise_std_deriv]

cat_particle_properties = 'particle_properties'
def_width_part = 'width', 3
def_length_part = 'length', 30
def_height_part = 'height', 3
def_image_path = 'image_path', ""
def_part_max_height = 'max_height', 1
def_std_deriv = 'std_derivate_grain_border', def_length_part[1] / 5
def_fermi_exp = '1/kbT', 0.4
def_angle_characteristic_length = 'Angle Characteristic_relative_length', 0.2
def_angle_stdderiv = 'std_derivate_angle_correlation', 1
def_angle_range_min = 'minimum angle (degree)', 0
def_angle_range_max = 'maximum angle (degree)', 0
def_angle_range_usage = 'use angle range?', 0
def_use_crystal_orientation = 'Use Crystal orientation', 1
def_no_of_orientations = 'Number of Crystal Orientations', 2
def_crystal_orientation_1 = 'Crystal Direction 1 (Degrees)', 0
def_crystal_orientation_2 = 'Crystal Direction 2 (Degrees)', 90
def_crystal_orientation_3 = 'Crystal Direction 3 (Degrees)', 0
def_crystal_orientation_4 = 'Crystal Direction 4 (Degrees)', 0
particle_properties_settings = [def_width_part, def_image_path, def_length_part, def_height_part, def_fermi_exp,
                                def_part_max_height, def_angle_range_usage, def_angle_range_min, def_angle_range_max,
                                def_angle_stdderiv, def_angle_characteristic_length, def_use_crystal_orientation,
                                def_no_of_orientations, def_crystal_orientation_1, def_crystal_orientation_2,
                                def_crystal_orientation_3, def_crystal_orientation_4]

cat_special = 'special settings'
def_overlap_threshold = 'overlapping_threshold', 10
def_dragging_error = 'dragging errors', 0
def_dragging_speed = 'dragging speed', 0.1 * def_width[1]
def_dragging_possibility = 'dragging possibility', 0
def_raster_angle = 'raster_angle (degrees)', 0
def_doubletip_possibility= 'possibility of two tips', 0
special_settings = [def_overlap_threshold, def_dragging_error, def_dragging_speed, def_dragging_possibility, def_raster_angle, def_doubletip_possibility]


# Reset parameters to default values

@measureTime
def _writeDefaults():
    conf[cat_pc] = {x[0]: x[1] for x in pc_settings}
    conf[cat_image_basics] = {x[0]: x[1] for x in image_basics_settings}
    conf[cat_particle_properties] = {x[0]: x[1] for x in particle_properties_settings}
    conf[cat_special] = {x[0]: x[1] for x in special_settings}
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


def get_anti_aliasing():
    global settings_file
    if not initialized: setupConfigurationManager()
    conf.read(settings_file)
    return bool(int(conf[cat_image_basics][def_anti_aliasing[0]]))


def get_image_noise_mu():
    global settings_file
    if not initialized: setupConfigurationManager()
    conf.read(settings_file)
    return float(conf[cat_image_basics][def_noise_mu[0]])


def get_image_noise_std_deriv():
    global settings_file
    if not initialized: setupConfigurationManager()
    conf.read(settings_file)
    return float(conf[cat_image_basics][def_noise_std_deriv[0]])


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


def get_max_height():
    global settings_file
    if not initialized: setupConfigurationManager()
    conf.read(settings_file)
    return int(conf[cat_particle_properties][def_part_max_height[0]])


#def get_std_deriv():
#    global settings_file
#    if not initialized: setupConfigurationManager()
#    conf.read(settings_file)
#    return int(float(conf[cat_particle_properties][def_std_deriv[0]]))

def get_fermi_exp():
    global settings_file
    if not initialized: setupConfigurationManager()
    conf.read(settings_file)
    return float(conf[cat_particle_properties][def_fermi_exp[0]])


def get_angle_stdderiv():
    global settings_file
    if not initialized: setupConfigurationManager()
    conf.read(settings_file)
    return float(conf[cat_particle_properties][def_angle_stdderiv[0]])


def get_angle_char_len():
    global settings_file
    if not initialized: setupConfigurationManager()
    conf.read(settings_file)
    return float(conf[cat_particle_properties][def_angle_characteristic_length[0]])


def get_angle_range_min():
    global settings_file
    if not initialized: setupConfigurationManager()
    conf.read(settings_file)
    return np.pi * float(conf[cat_particle_properties][def_angle_range_min[0]]) / 180


def get_angle_range_max():
    global settings_file
    if not initialized: setupConfigurationManager()
    conf.read(settings_file)
    return np.pi * float(conf[cat_particle_properties][def_angle_range_max[0]]) / 180


def get_angle_range_usage():
    global settings_file
    if not initialized: setupConfigurationManager()
    conf.read(settings_file)
    return bool(int((conf[cat_particle_properties][def_angle_range_usage[0]])))

def get_crystal_orientation_usage():
    global settings_file
    if not initialized: setupConfigurationManager()
    conf.read(settings_file)
    return bool(int((conf[cat_particle_properties][def_use_crystal_orientation[0]])))

def get_no_of_orientations():
    global settings_file
    if not initialized: setupConfigurationManager()
    conf.read(settings_file)
    return int((conf[cat_particle_properties][def_no_of_orientations[0]]))

def get_crystal_orientation_1():
    global settings_file
    if not initialized: setupConfigurationManager()
    conf.read(settings_file)
    return np.pi * float(conf[cat_particle_properties][def_crystal_orientation_1[0]]) / 180

def get_crystal_orientation_2():
    global settings_file
    if not initialized: setupConfigurationManager()
    conf.read(settings_file)
    return np.pi * float(conf[cat_particle_properties][def_crystal_orientation_2[0]]) / 180

def get_crystal_orientation_3():
    global settings_file
    if not initialized: setupConfigurationManager()
    conf.read(settings_file)
    return np.pi * float(conf[cat_particle_properties][def_crystal_orientation_3[0]]) / 180

def get_crystal_orientation_4():
    global settings_file
    if not initialized: setupConfigurationManager()
    conf.read(settings_file)
    return np.pi * float(conf[cat_particle_properties][def_crystal_orientation_4[0]]) / 180

def get_crystal_orientations_array():
    if get_no_of_orientations() == 1:
        return [get_crystal_orientation_1()]
    elif get_no_of_orientations() == 2:
        return [get_crystal_orientation_1(), get_crystal_orientation_2()]
    elif get_no_of_orientations() == 3:
        return [get_crystal_orientation_1(), get_crystal_orientation_2(), get_crystal_orientation_3()]
    elif get_no_of_orientations() == 4:
        return [get_crystal_orientation_1(), get_crystal_orientation_2(), get_crystal_orientation_3(), get_crystal_orientation_4()]
    else:
        raise NotImplementedError

def get_overlap_threshold():
    global settings_file
    if not initialized: setupConfigurationManager()
    conf.read(settings_file)
    return int(conf[cat_special][def_overlap_threshold[0]])


def get_dragging_error():
    global settings_file
    if not initialized: setupConfigurationManager()
    conf.read(settings_file)
    return bool(int(conf[cat_special][def_dragging_error[0]]))

def get_raster_angle():
    global settings_file
    if not initialized: setupConfigurationManager()
    conf.read(settings_file)
    return np.pi * float(conf[cat_special][def_raster_angle[0]]) / 180

def get_dragging_speed():
    global settings_file
    if not initialized: setupConfigurationManager()
    conf.read(settings_file)
    return float(conf[cat_special][def_dragging_speed[0]])


def get_dragging_possibility():
    global settings_file
    if not initialized: setupConfigurationManager()
    conf.read(settings_file)
    return float(conf[cat_special][def_dragging_possibility[0]])

def get_double_tip_possibility():
    global settings_file
    if not initialized: setupConfigurationManager()
    conf.read(settings_file)
    return float(conf[cat_special][def_doubletip_possibility[0]])
