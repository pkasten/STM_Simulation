import configparser as cp
import os, numpy as np
from TestsApril.Maths.Functions import measureTime
from Distance import Distance


"""
Configuration manager. Loads setting from file settings_file into the program
using a configparser and provides getter methods for each setting
"""

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
def_px_per_angstrom = 'Pixel per Angstrom', 2
def_prefix_image = 'Prefix for Image file', 'Image'
def_suffix_image = 'Suffix for Image file', '.png'
def_prefix_data = 'Prefix for data file', 'Data'
def_suffix_data = 'Suffix for data file', '.txt'
def_prefix_sxm = 'Prefix for sxm file', 'Image'
def_suffix_sxm = 'Suffix for sxm file', '.sxm'
def_color_scheme = 'Color scheme (Gray, WSXM)', 'WSXM'
def_width = 'Image width (Ang)', 100
def_height = 'Image height (Ang)', 100
def_particles = 'Default number of particles per Image', 20
def_px_overlap = 'Pixel overlap where Particles can be (in px)', 40
def_anti_aliasing = 'Use Anti Aliasing (1 True, 0 False)', 1
def_use_white_noise = "Use white noise", 0
def_use_line_noise = 'Use line noise/ 1/f noise', 1
def_noise_mu = 'Grayscale value of mean noise', 40
def_noise_std_deriv = 'Image noise standard derivation', 0.7 * def_noise_mu[1]
image_basics_settings = [def_prefix_image, def_suffix_image, def_prefix_data, def_suffix_data, def_prefix_sxm,
                         def_suffix_sxm, def_color_scheme, def_px_per_angstrom, def_width, def_height,
                         def_particles, def_px_overlap, def_anti_aliasing,def_use_white_noise,
                         def_use_line_noise, def_noise_mu, def_noise_std_deriv]


val_prefix_image = None
val_px_per_angstrom = None
val_suffix_image = None
val_prefix_data = None
val_suffix_data = None
val_prefix_sxm = None
val_suffix_sxm = None
val_color_scheme = None
val_width = None
val_height = None
val_particles = None
val_px_overlap = None
val_anti_aliasing = None
val_use_white_noise = None
val_use_line_noise = None
val_noise_mu = None
val_noise_std_deriv = None


cat_particle_properties = 'particle_properties'
def_width_part = 'Particle width (Ang)', 0.4
def_length_part = 'Particle length (Ang)', 20
def_height_part = 'Particle height (Ang)', 5
def_image_path = 'Picture how particles should look like, leave empty for generated image', ""
def_molecule_style = 'Visualization style for molecules (simple, complex)', "simple"
def_part_max_height = 'Maximum Height (Ang)', 7.3
def_std_deriv = 'Standard Derivation of Grain Border (Deprecated)', def_length_part[1] / 5
def_fermi_exp = 'Exponent 1/kbT in fermi distribution', 0.4
def_angle_characteristic_length = 'Angle Characteristic_relative_length (Deprecated)', 0
def_angle_stdderiv = 'std_derivate_angle_correlation (Deprecated)', 0
def_angle_range_min = 'minimum angle of particles (degree)', 0
def_angle_range_max = 'maximum angle of particles (degree)', 0
def_angle_range_usage = 'use angle range?', 0
def_use_crystal_orientation = 'Use Crystal orientation?', 0
def_no_of_orientations = 'Number of Crystal Orientations', 0
def_crystal_orientation_1 = 'Crystal Direction 1 (Degrees)', 0
def_crystal_orientation_2 = 'Crystal Direction 2 (Degrees)', 0
def_crystal_orientation_3 = 'Crystal Direction 3 (Degrees)', 0
def_crystal_orientation_4 = 'Crystal Direction 4 (Degrees)', 0
def_use_ordered_variation = 'Use Variation in ordered position?', 1
def_order_pos_var = 'Variation in Particle Position when ordered (percent of length)', 0.05
def_order_ang_var = 'Variation in Particle Angle when ordered (percent of 2pi)', 0.05

particle_properties_settings = [def_width_part, def_image_path, def_molecule_style, def_length_part, def_height_part, def_fermi_exp,
                                def_part_max_height, def_angle_range_usage, def_angle_range_min, def_angle_range_max,
                                def_angle_stdderiv, def_angle_characteristic_length, def_use_crystal_orientation,
                                def_no_of_orientations, def_crystal_orientation_1, def_crystal_orientation_2,
                                def_crystal_orientation_3, def_crystal_orientation_4, def_use_ordered_variation, def_order_pos_var, def_order_ang_var]

val_width_part = None
val_length_part = None
val_height_part = None
val_image_path = None
val_molecule_style = None
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
val_use_ordered_variation = None
val_order_pos_var = None
val_order_ang_var = None


cat_special = 'special settings'
def_overlap_threshold = 'Height threshold when particles are overlapping', 10
def_dragging_error = 'Use dragging errors', 0
def_dragging_speed = 'dragging speed (in Ang)',  0.5 * def_width[1]
def_dragging_possibility = 'dragging possibility', 0
def_raster_angle = 'raster_angle (degrees)', 0
def_doubletip_possibility= 'possibility of two tips', 0
def_atomic_step_height = 'Height of atomic step (Ang)', 2.3
def_atomic_step_poss = 'Possibility of atomic step', 0
def_dust_amount = 'Medium no of dust particles', 15
def_use_imgshift = 'Use Image shift', 0
def_shift_amount_x = 'Stretching factor of image Shift (x-Direction)', 1.05
def_shift_amount_y = 'Stretching factor of image Shift (y-Direction)', 1.05
def_shift_style = 'Image Shifting style (Lin/Exp)', "Exp"
def_use_scanlines = 'Use scanlines', 1
special_settings = [def_overlap_threshold, def_dragging_error, def_dragging_speed, def_dragging_possibility, def_raster_angle, def_doubletip_possibility, def_atomic_step_height, def_atomic_step_poss, def_dust_amount, def_use_imgshift, def_shift_style, def_shift_amount_x, def_shift_amount_y, def_use_scanlines]

val_overlap_threshold = None
val_dragging_error = None
val_dragging_speed = None
val_dragging_possibility = None
val_raster_angle = None
val_doubletip_possibility = None
val_atomc_step_height = None
val_atomic_step_poss = None
val_dust_amount = None
val_use_img_shift = None
val_shift_style = None
val_shift_amount_x = None
val_shift_amount_y = None
val_use_scanlines = None

cat_lattice = 'lattice'
def_nn_dist = 'Distance between nearest neighbours (Ang)', 2.88
lattice_settings = [def_nn_dist]

val_nn_dist = None

# Reset parameters to default values

@measureTime
def _writeDefaults():
    conf[cat_pc] = {x[0]: x[1] for x in pc_settings}
    conf[cat_image_basics] = {x[0]: x[1] for x in image_basics_settings}
    conf[cat_particle_properties] = {x[0]: x[1] for x in particle_properties_settings}
    conf[cat_lattice] = {x[0] : x[1] for x in lattice_settings}
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
        print("No settings file found. Created file in {}".format(settings_file))
        exit(0)

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
    global val_prefix_sxm, val_suffix_sxm, val_sxm_folder, val_px_per_angstrom, val_nn_dist, val_atomc_step_height, val_atomic_step_poss
    global val_dust_amount, val_use_img_shift, val_color_scheme, val_order_ang_var, val_order_pos_var, val_shift_amount_x, val_shift_amount_y, val_shift_style
    global val_use_ordered_variation, val_use_white_noise, val_use_line_noise, val_use_scanlines, val_molecule_style
    val_threads = int(conf[cat_pc][def_threads[0]])
    val_images_per_thread = int(conf[cat_pc][def_images_per_thread[0]])
    val_image_folder = conf[cat_pc][def_image_folder[0]]
    val_data_folder = conf[cat_pc][def_data_folder[0]]
    val_sxm_folder = conf[cat_pc][def_sxm_folder[0]]
    val_px_per_angstrom = float(conf[cat_image_basics][def_px_per_angstrom[0]])
    val_width = Distance(True, float(conf[cat_image_basics][def_width[0]]))
    val_height = Distance(True, float(conf[cat_image_basics][def_height[0]]))
    val_particles = int(conf[cat_image_basics][def_particles[0]])
    val_prefix_image = conf[cat_image_basics][def_prefix_image[0]]
    val_prefix_data = conf[cat_image_basics][def_prefix_data[0]]
    val_prefix_sxm = conf[cat_image_basics][def_prefix_sxm[0]]
    val_suffix_image = conf[cat_image_basics][def_suffix_image[0]]
    val_suffix_data = conf[cat_image_basics][def_suffix_data[0]]
    val_suffix_sxm = conf[cat_image_basics][def_suffix_sxm[0]]
    val_color_scheme = conf[cat_image_basics][def_color_scheme[0]]
    val_molecule_style = conf[cat_particle_properties][def_molecule_style[0]]
    val_px_overlap = int(conf[cat_image_basics][def_px_overlap[0]])
    val_anti_aliasing = bool(int(conf[cat_image_basics][def_anti_aliasing[0]]))
    val_use_white_noise = bool(int(conf[cat_image_basics][def_use_white_noise[0]]))
    val_use_line_noise = bool(int(conf[cat_image_basics][def_use_line_noise[0]]))
    val_noise_mu = float(conf[cat_image_basics][def_noise_mu[0]])
    val_noise_std_deriv = float(conf[cat_image_basics][def_noise_std_deriv[0]])
    val_height_part = Distance(True, float(conf[cat_particle_properties][def_height_part[0]]))
    val_width_part = Distance(True, float(conf[cat_particle_properties][def_width_part[0]]))
    val_length_part = Distance(True, float(conf[cat_particle_properties][def_length_part[0]]))
    val_image_path = conf[cat_particle_properties][def_image_path[0]]
    val_part_max_height = Distance(True, float(conf[cat_particle_properties][def_part_max_height[0]]))
    val_fermi_exp = float(conf[cat_particle_properties][def_fermi_exp[0]]) / val_px_per_angstrom
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
    val_dragging_speed = Distance(True, float(conf[cat_special][def_dragging_speed[0]]))
    val_dragging_possibility = float(conf[cat_special][def_dragging_possibility[0]])
    val_atomc_step_height = Distance(True, float(conf[cat_special][def_atomic_step_height[0]]))
    val_atomic_step_poss = float(conf[cat_special][def_atomic_step_poss[0]])
    val_doubletip_possibility = float(conf[cat_special][def_doubletip_possibility[0]])
    val_use_img_shift = bool(int(conf[cat_special][def_use_imgshift[0]]))
    val_use_scanlines = bool(int(conf[cat_special][def_use_scanlines[0]]))
    val_shift_amount_x = float(conf[cat_special][def_shift_amount_x[0]])
    val_shift_amount_y = float(conf[cat_special][def_shift_amount_y[0]])
    val_shift_style = conf[cat_special][def_shift_style[0]]
    val_nn_dist = Distance(True, float(conf[cat_lattice][def_nn_dist[0]]))
    val_dust_amount = float(conf[cat_special][def_dust_amount[0]])
    val_use_ordered_variation = bool(int(conf[cat_particle_properties][def_use_ordered_variation[0]]))
    val_order_pos_var = float(conf[cat_particle_properties][def_order_pos_var[0]]) / 100
    val_order_ang_var = float(conf[cat_particle_properties][def_order_ang_var[0]]) / 100

# return THREADS parameter

@measureTime
def get_threads():
    """
    Getter method for threads parameter. Initializes ConfigManager if not done yet.
    :return: Number of threads
    """
    if not initialized: setupConfigurationManager()
    return val_threads


# get Images Per Thread

@measureTime
def get_images_pt():
    if not initialized: setupConfigurationManager()
    return val_images_per_thread
    #conf.read(settings_file)
    #return int(conf[cat_pc][def_images_per_thread[0]])

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

def get_px_per_angstrom():
    global settings_file
    if not initialized: setupConfigurationManager()
    return val_px_per_angstrom

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

def use_white_noise():
    global settings_file
    if not initialized: setupConfigurationManager()
    return val_use_white_noise

def use_line_noise():
    global settings_file
    if not initialized: setupConfigurationManager()
    return val_use_line_noise


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
        if get_no_of_orientations() == 0:
            return []
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

def get_atomic_step_height():
    global settings_file
    if not initialized: setupConfigurationManager()
    return val_atomc_step_height

def get_atomic_step_poss():
    global settings_file
    if not initialized: setupConfigurationManager()
    return val_atomic_step_poss

def get_nn_dist():
    global settings_file
    if not initialized: setupConfigurationManager()
    return val_nn_dist

def get_dust_amount():
    global settings_file
    if not initialized: setupConfigurationManager()
    return val_dust_amount

def get_use_img_shift():
    if not initialized: setupConfigurationManager()
    return val_use_img_shift

def get_color_scheme():
    if not initialized: setupConfigurationManager()
    return val_color_scheme

def get_molecule_style():
    if not initialized: setupConfigurationManager()
    return val_molecule_style

def get_order_ang_var():
    if not initialized: setupConfigurationManager()
    return val_order_ang_var

def get_order_pos_var():
    if not initialized: setupConfigurationManager()
    return val_order_pos_var

def get_shift_amount_x():
    if not initialized: setupConfigurationManager()
    return val_shift_amount_x

def get_shift_amount_y():
    if not initialized: setupConfigurationManager()
    return val_shift_amount_x

def get_shift_style():
    if not initialized: setupConfigurationManager()
    return val_shift_style

def get_use_ordered_variation():
    if not initialized: setupConfigurationManager()
    return val_use_ordered_variation

def use_scanlines():
    if not initialized: setupConfigurationManager()
    return val_use_scanlines
