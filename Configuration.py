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
def_image_folder = 'Image Folder', str(os.path.join(os.getcwd(), "bildordner"))
def_data_folder = 'Data Folder', str(os.path.join(os.getcwd(), "data"))
def_sxm_folder = 'SXM Folder', str(os.path.join(os.getcwd(), "sxm"))
pc_settings = [def_threads, def_images_per_thread, def_image_folder, def_data_folder, def_sxm_folder]

val_threads = None
val_images_per_thread = None
val_image_folder = None
val_data_folder = None
val_sxm_folder = None


cat_image_basics = 'image_settings'
def_prefix_image = 'prefix_image', 'Image'
def_suffix_image = 'suffix_image', '.png'
def_prefix_data = 'prefix_data', 'Data'
def_suffix_data = 'suffix_data', '.txt'
def_prefix_sxm = 'prefix_sxm', 'Image'
def_suffix_sxm = 'suffix_sxm', '.sxm'
def_width = 'width', 400
def_height = 'height', 400
def_particles = 'no_of_particles', int(def_width[1] * def_height[1] / 16000)
def_px_overlap = 'pixels_overlap (in px)', 40
def_anti_aliasing = 'Anti-Aliasing', 1
def_noise_mu = 'Image-noise_Average', 0
def_noise_std_deriv = 'Image-noise-Standard_derivation', 0.1 * def_noise_mu[1]
image_basics_settings = [def_prefix_image, def_suffix_image, def_prefix_data, def_suffix_data, def_prefix_sxm,
                         def_suffix_sxm, def_width, def_height,
                         def_particles, def_px_overlap, def_anti_aliasing, def_noise_mu, def_noise_std_deriv]


val_prefix_image = None
val_suffix_image = None
val_prefix_data = None
val_suffix_data = None
val_prefix_sxm = None
val_suffix_sxm = None
val_width = None
val_height = None
val_particles = None
val_px_overlap = None
val_anti_aliasing = None
val_noise_mu = None
val_noise_std_deriv = None


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

val_width_part = None
val_length_part = None
val_height_part = None
val_image_path = None
val_part_max_height = None
val_std_deriv = None
val_fermi_exp = None
val_angle_characteristic_length =None
val_angle_stdderiv = None
val_angle_range_min = None
val_angle_range_max = None
val_angle_range_usage = None
val_use_crystal_orientation = None
val_no_of_orientations = None
val_crystal_orientation_1 = None
val_crystal_orientation_2 = None
val_crystal_orientation_3 = None
val_crystal_orientation_4 = None

cat_special = 'special settings'
def_overlap_threshold = 'overlapping_threshold', 10
def_dragging_error = 'dragging errors', 0
def_dragging_speed = 'dragging speed', 0.1 * def_width[1]
def_dragging_possibility = 'dragging possibility', 0
def_raster_angle = 'raster_angle (degrees)', 0
def_doubletip_possibility= 'possibility of two tips', 0
special_settings = [def_overlap_threshold, def_dragging_error, def_dragging_speed, def_dragging_possibility, def_raster_angle, def_doubletip_possibility]

val_overlap_threshold = None
val_dragging_error = None
val_dragging_speed = None
val_dragging_possibility = None
val_raster_angle = None
val_doubletip_possibility = None

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

    update_params()


# Create a new settings-file if not yet existent

@measureTime
def setupConfigurationManager():
    global initialized
    initialized = True
    if not os.path.exists(settings_file):
        _writeDefaults()

    conf.read(settings_file)
    update_params()

def update_params():
    conf.read(settings_file)
    global val_threads, val_images_per_thread, val_image_folder, val_data_folder, val_width, val_height, val_particles
    global val_prefix_image, val_prefix_data, val_suffix_image, val_suffix_data, val_px_overlap, val_anti_aliasing
    global val_noise_mu, val_noise_std_deriv, val_height_part, val_width_part, val_length_part, val_image_path
    global val_part_max_height, val_fermi_exp, val_angle_stdderiv, val_angle_characteristic_length, val_angle_range_min
    global val_angle_range_max, val_angle_range_usage, val_use_crystal_orientation, val_no_of_orientations, val_crystal_orientation_1
    global val_crystal_orientation_2, val_crystal_orientation_3, val_crystal_orientation_4, val_overlap_threshold
    global val_dragging_error, val_raster_angle, val_dragging_speed, val_dragging_possibility, val_doubletip_possibility
    global val_prefix_sxm, val_suffix_sxm, val_sxm_folder
    val_threads = int(conf[cat_pc][def_threads[0]])
    val_images_per_thread = int(conf[cat_pc][def_images_per_thread[0]])
    val_image_folder = conf[cat_pc][def_image_folder[0]]
    val_data_folder = conf[cat_pc][def_data_folder[0]]
    val_sxm_folder = conf[cat_pc][def_sxm_folder[0]]
    val_width = int(conf[cat_image_basics][def_width[0]])
    val_height = int(conf[cat_image_basics][def_height[0]])
    val_particles = int(conf[cat_image_basics][def_particles[0]])
    val_prefix_image = conf[cat_image_basics][def_prefix_image[0]]
    val_prefix_data = conf[cat_image_basics][def_prefix_data[0]]
    val_prefix_sxm = conf[cat_image_basics][def_prefix_sxm[0]]
    val_suffix_image = conf[cat_image_basics][def_suffix_image[0]]
    val_suffix_data = conf[cat_image_basics][def_suffix_data[0]]
    val_suffix_sxm = conf[cat_image_basics][def_suffix_sxm[0]]
    val_px_overlap = int(conf[cat_image_basics][def_px_overlap[0]])
    val_anti_aliasing = bool(int(conf[cat_image_basics][def_anti_aliasing[0]]))
    val_noise_mu = float(conf[cat_image_basics][def_noise_mu[0]])
    val_noise_std_deriv = float(conf[cat_image_basics][def_noise_std_deriv[0]])
    val_height_part = int(conf[cat_particle_properties][def_height_part[0]])
    val_width_part = int(conf[cat_particle_properties][def_width_part[0]])
    val_length_part = int(conf[cat_particle_properties][def_length_part[0]])
    val_image_path = conf[cat_particle_properties][def_image_path[0]]
    val_part_max_height = int(conf[cat_particle_properties][def_part_max_height[0]])
    val_fermi_exp = float(conf[cat_particle_properties][def_fermi_exp[0]])
    val_angle_stdderiv = float(conf[cat_particle_properties][def_angle_stdderiv[0]])
    val_angle_characteristic_length = float(conf[cat_particle_properties][def_angle_characteristic_length[0]])
    val_angle_range_min = np.pi * float(conf[cat_particle_properties][def_angle_range_min[0]]) / 180
    val_angle_range_max = np.pi * float(conf[cat_particle_properties][def_angle_range_max[0]]) / 180
    val_angle_range_usage = bool(int((conf[cat_particle_properties][def_angle_range_usage[0]])))
    val_use_crystal_orientation = bool(int((conf[cat_particle_properties][def_use_crystal_orientation[0]])))
    val_no_of_orientations = int((conf[cat_particle_properties][def_no_of_orientations[0]]))
    val_crystal_orientation_1 = np.pi * float(conf[cat_particle_properties][def_crystal_orientation_1[0]]) / 180
    val_crystal_orientation_2 = np.pi * float(conf[cat_particle_properties][def_crystal_orientation_2[0]]) / 180
    val_crystal_orientation_3 = np.pi * float(conf[cat_particle_properties][def_crystal_orientation_3[0]]) / 180
    val_crystal_orientation_4 = np.pi * float(conf[cat_particle_properties][def_crystal_orientation_4[0]]) / 180
    val_overlap_threshold = int(conf[cat_special][def_overlap_threshold[0]])
    val_dragging_error = bool(int(conf[cat_special][def_dragging_error[0]]))
    val_raster_angle = np.pi * float(conf[cat_special][def_raster_angle[0]]) / 180
    val_dragging_speed = float(conf[cat_special][def_dragging_speed[0]])
    val_dragging_possibility = float(conf[cat_special][def_dragging_possibility[0]])
    val_doubletip_possibility = float(conf[cat_special][def_doubletip_possibility[0]])


# return THREADS parameter

@measureTime
def get_threads():
    if not initialized: setupConfigurationManager()
    return val_threads


# get Images Per Thread

@measureTime
def get_images_pt():
    if not initialized: setupConfigurationManager()
    return val_images_per_thread
    #conf.read(settings_file)
    #return int(conf[cat_pc][def_images_per_thread[0]])  # ToDo: Castch Exceptions

    # set Images Per Thread


@measureTime
def set_images_pt(num):
    if not initialized: setupConfigurationManager()
    conf[cat_pc][def_images_per_thread[0]] = str(num)
    with open(settings_file, "w") as settings:
        conf.write(settings)
    update_params()

    # set THREADS parameter


@measureTime
def set_threads(num):
    if not initialized: setupConfigurationManager()
    conf[cat_pc][def_threads[0]] = str(num)
    with open(settings_file, "w") as settings:
        conf.write(settings)
    update_params()

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
    update_params()


@measureTime
def get_image_folder():
    if not initialized: setupConfigurationManager()
    return val_image_folder
    #conf.read(settings_file)
    #return conf[cat_pc][def_image_folder[0]]


@measureTime
def set_data_folder(path):
    global settings_file
    if not initialized: setupConfigurationManager()
    if "\\" not in str(path) and "/" not in str(path):
        path = os.path.join(os.getcwd(), path)
    conf[cat_pc][def_data_folder[0]] = str(path)
    with open(settings_file, "w") as settings:
        conf.write(settings)
    update_params()


@measureTime
def get_data_folder():
    global settings_file
    if not initialized: setupConfigurationManager()
    return val_data_folder

@measureTime
def get_sxm_folder():
    global settings_file
    if not initialized: setupConfigurationManager()
    return val_sxm_folder



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
    return val_width


@measureTime
def get_height():
    global settings_file
    if not initialized: setupConfigurationManager()
    return val_height


@measureTime
def set_width(w):
    if not initialized: setupConfigurationManager()
    conf[cat_image_basics][def_width[0]] = str(w)
    with open(settings_file, "w") as settings:
        conf.write(settings)
    update_params()


@measureTime
def set_heigth(h):
    global settings_file
    if not initialized: setupConfigurationManager()
    conf[cat_image_basics][def_height[0]] = str(h)
    with open(settings_file, "w") as settings:
        conf.write(settings)
    update_params()


def get_particles_per_image():
    global settings_file
    if not initialized: setupConfigurationManager()
    return val_particles


def get_prefix_image():
    global settings_file
    if not initialized: setupConfigurationManager()
    return val_prefix_image


def get_prefix_data():
    global settings_file
    if not initialized: setupConfigurationManager()
    return val_prefix_data

def get_prefix_sxm():
    global settings_file
    if not initialized: setupConfigurationManager()
    return val_prefix_sxm


def get_suffix_image():
    global settings_file
    if not initialized: setupConfigurationManager()
    return val_suffix_image


def get_suffix_data():
    global settings_file
    if not initialized: setupConfigurationManager()
    return val_suffix_data

def get_suffix_sxm():
    global settings_file
    if not initialized: setupConfigurationManager()
    return val_suffix_sxm


def get_px_overlap():
    global settings_file
    if not initialized: setupConfigurationManager()
    return val_px_overlap


def get_anti_aliasing():
    global settings_file
    if not initialized: setupConfigurationManager()
    return val_anti_aliasing


def get_image_noise_mu():
    global settings_file
    if not initialized: setupConfigurationManager()
    return val_noise_mu


def get_image_noise_std_deriv():
    global settings_file
    if not initialized: setupConfigurationManager()
    return val_noise_std_deriv


def get_part_height():
    global settings_file
    if not initialized: setupConfigurationManager()
    return val_height_part


def get_part_width():
    global settings_file
    if not initialized: setupConfigurationManager()
    return val_width_part


def get_part_length():
    global settings_file
    if not initialized: setupConfigurationManager()
    return val_length_part


def get_image_path():
    global settings_file
    if not initialized: setupConfigurationManager()
    return val_image_path


def get_max_height():
    global settings_file
    if not initialized: setupConfigurationManager()
    return val_part_max_height


#def get_std_deriv():
#    global settings_file
#    if not initialized: setupConfigurationManager()
#    conf.read(settings_file)
#    return int(float(conf[cat_particle_properties][def_std_deriv[0]]))

def get_fermi_exp():
    global settings_file
    if not initialized: setupConfigurationManager()
    return val_fermi_exp


def get_angle_stdderiv():
    global settings_file
    if not initialized: setupConfigurationManager()
    return val_angle_stdderiv
def set_angle_stdderiv(sigmea):
    if not initialized: setupConfigurationManager()
    conf[cat_particle_properties][def_angle_stdderiv[0]] = str(sigmea)
    with open(settings_file, "w") as settings:
        conf.write(settings)
    update_params()


def get_angle_char_len():
    global settings_file
    if not initialized: setupConfigurationManager()
    return val_angle_characteristic_length

def set_angle_char_len(l):
    if not initialized: setupConfigurationManager()
    conf[cat_particle_properties][def_angle_characteristic_length[0]] = str(l)
    with open(settings_file, "w") as settings:
        conf.write(settings)
    update_params()


def get_angle_range_min():
    global settings_file
    if not initialized: setupConfigurationManager()
    return val_angle_range_min


def get_angle_range_max():
    global settings_file
    if not initialized: setupConfigurationManager()
    return val_angle_range_max


def get_angle_range_usage():
    global settings_file
    if not initialized: setupConfigurationManager()
    return val_angle_range_usage

def get_crystal_orientation_usage():
    global settings_file
    if not initialized: setupConfigurationManager()
    return val_use_crystal_orientation

def get_no_of_orientations():
    global settings_file
    if not initialized: setupConfigurationManager()
    return val_no_of_orientations

def get_crystal_orientation_1():
    global settings_file
    if not initialized: setupConfigurationManager()
    return val_crystal_orientation_1

def get_crystal_orientation_2():
    global settings_file
    if not initialized: setupConfigurationManager()
    return val_crystal_orientation_2

def get_crystal_orientation_3():
    global settings_file
    if not initialized: setupConfigurationManager()
    return val_crystal_orientation_3

def get_crystal_orientation_4():
    global settings_file
    if not initialized: setupConfigurationManager()
    return val_crystal_orientation_4

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
    return val_overlap_threshold


def get_dragging_error():
    global settings_file
    if not initialized: setupConfigurationManager()
    return val_dragging_error

def get_raster_angle():
    global settings_file
    if not initialized: setupConfigurationManager()
    return val_raster_angle

def get_dragging_speed():
    global settings_file
    if not initialized: setupConfigurationManager()
    return val_dragging_speed

def get_dragging_possibility():
    global settings_file
    if not initialized: setupConfigurationManager()
    return val_dragging_possibility

def get_double_tip_possibility():
    global settings_file
    if not initialized: setupConfigurationManager()
    return val_doubletip_possibility
