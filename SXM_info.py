import configparser as cp
import os, numpy as np, datetime

conf = cp.ConfigParser()
settings_folder = os.getcwd()
settings_file = str(os.path.join(settings_folder, "spm_info.ini"))
initialized = False

cat = 'Nanonis SXM Settings'

def_NANONIS_VERSION = 'NANONIS_VERSION', [['2']]
def_SCANIT_TYPE = 'SCANIT_TYPE', [['FLOAT', 'MSBFIRST']]
def_REC_DATE = 'REC_DATE', [['11.09.2020']]
def_REC_TIME = 'REC_TIME', [['18:09:55']]
def_REC_TEMP = 'REC_TEMP', [['290.0000000000']]
def_ACQ_TIME = "ACQ_TIME", [['6.6']]
def_SCAN_PIXELS = "SCAN_PIXELS", [['256', '256']]
def_SCAN_FILE = "SCAN_FILE", [['D:\\STM', 'DATA\\2020-07-09\\HOPG-gxy1z1-p2020.sxm']]
def_SCAN_TIME = "SCAN_TIME", [['1.280E-2', '1.280E-2']]
def_SCAN_RANGE = "SCAN_RANGE", [['1.806093E-9', '1.806093E-9']]
def_SCAN_OFFSET = "SCAN_OFFSET", [['-9.711871E-8', '5.935157E-8']]
def_SCAN_ANGLE = "SCAN_ANGLE", [['0.000E+0']]
def_SCAN_DIR = "SCAN_DIR", [['down']]
def_BIAS = "BIAS", [['9.270E-2']]
def_Z_CONTROLLER = "Z-CONTROLLER", [['Name', 'on', 'Setpoint', 'P-gain', 'I-gain', 'T-const'],
                                    ['log', 'Current', '1', '8.810E-10', 'A', '1.397E-19', 'm', '1.367E-6', 'm/s',
                                     '1.021E-13', 's']]
def_COMMENT = "COMMENT", []
def_Bias_Bias_V = "Bias>Bias (V)", [['92.7036E-3']]
def_Bias_Calibration_V_V = "Bias>Calibration (V/V)", [['1E+0']]
def_Bias_Offset_V = "Bias>Offset (V)", [['0E+0']]
def_Current_Current_A = "Current>Current (A)", [['885.631E-12']]
def_Current_Calibration_A_V = "Current>Calibration (A/V)", [['1E-9']]
def_Current_Offset_A = "Current>Offset (A)", [['0E+0']]
def_Current_Gain = "Current>Gain", [['Not', 'switchable']]
def_Piezo_Configuration_Active_Calib = "Piezo Configuration>Active Calib.", [['Default']]
def_Piezo_Configuration_Calib_X_m_V = "Piezo Configuration>Calib. X (m/V)", [['11.68E-9']]
def_Piezo_Configuration_Calib_Y_m_V = "Piezo Configuration>Calib. Y (m/V)", [['11.68E-9']]
def_Piezo_Configuration_Calib_Z_m_V = "Piezo Configuration>Calib. Z (m/V)", [['1.562E-9']]
def_Piezo_Configuration_HV_Gain_X = "Piezo Configuration>HV Gain X", [['1']]
def_Piezo_Configuration_HV_Gain_Y = "Piezo Configuration>HV Gain Y", [['1']]
def_Piezo_Configuration_HV_Gain_Z = "Piezo Configuration>HV Gain Z", [['1']]
def_Piezo_Configuration_Tilt_X_deg = "Piezo Configuration>Tilt X (deg)", [['0']]
def_Piezo_Configuration_Tilt_Y_deg = "Piezo Configuration>Tilt Y (deg)", [['0']]
def_Piezo_Configuration_Curvature_radius_X_m = "Piezo Configuration>Curvature radius X (m)", [['Inf']]
def_Piezo_Configuration_Curvature_radius_Y_m = "Piezo Configuration>Curvature radius Y (m)", [['Inf']]
def_Piezo_Configuration_2nd_order_corr_X_V_m_2 = "Piezo Configuration>2nd order corr X (V/m^2)", [['0E+0']]
def_Piezo_Configuration_2nd_order_corr_Y_V_m_2 = "Piezo Configuration>2nd order corr Y (V/m^2)", [['0E+0']]
def_Piezo_Configuration_Drift_X_m_s = "Piezo Configuration>Drift X (m/s)", [['0E+0']]
def_Piezo_Configuration_Drift_Y_m_s = "Piezo Configuration>Drift Y (m/s)", [['0E+0']]
def_Piezo_Configuration_Drift_Z_m_s = "Piezo Configuration>Drift Z (m/s)", [['0E+0']]
def_Piezo_Configuration_Drift_correction_status_on_off = "Piezo Configuration>Drift correction status (on/off)", [
    ['FALSE']]
def_Z_Controller_Z_m = "Z-Controller>Z (m)", [['-14.8895E-9']]
def_Z_Controller_Controller_name = "Z-Controller>Controller name", [['log', 'Current']]
def_Z_Controller_Controller_status = "Z-Controller>Controller status:", [['ON']]
def_Z_Controller_Setpoint = "Z-Controller>Setpoint", [['880.956E-12']]
def_Z_Controller_Setpoint_unit = "Z-Controller>Setpoint unit", [['A']]
def_Z_Controller_P_gain = "Z-Controller>P gain", [['139.658E-21']]
def_Z_Controller_I_gain = "Z-Controller>I gain", [['1.36739E-6']]
def_Z_Controller_Time_const_s = "Z-Controller>Time const (s)", [['102.135E-15']]
def_Z_Controller_TipLift_m = "Z-Controller>TipLift (m)", [['0E+0']]
def_Z_Controller_Switch_off_delay_s = "Z-Controller>Switch off delay (s)", [['0E+0']]
def_Scan_Scanfield = "Scan>Scanfield", [['-97.1187E-9;59.3516E-9;1.80609E-9;1.80609E-9;0E+0']]
def_Scan_series_name = "Scan>series name", [['HOPG-gxy1z1-p2']]
def_Scan_channels = "Scan>channels", [['Z', '(m)']]
def_Scan_pixels_line = "Scan>pixels/line", [['256']]
def_Scan_lines = "Scan>lines", [['256']]
def_Scan_speed_forw_m_s = "Scan>speed forw. (m/s)", [['141.101E-9']]
def_Scan_speed_backw_m_s = "Scan>speed backw. (m/s)", [['141.101E-9']]
def_DATA_INFO = "DATA_INFO", [['Channel', 'Name', 'Unit', 'Direction', 'Calibration', 'Offset'],
                              ['14', 'Z', 'm', 'both', '1.562E-9', '0.000E+0']]
def_SCANIT_END = "SCANIT_END", []

settings = [
    def_NANONIS_VERSION, def_SCANIT_TYPE, def_REC_DATE, def_REC_TIME, def_REC_TEMP, def_ACQ_TIME, def_SCAN_PIXELS,
    def_SCAN_FILE, def_SCAN_TIME, def_SCAN_RANGE, def_SCAN_OFFSET, def_SCAN_ANGLE, def_SCAN_DIR, def_BIAS,
    def_Z_CONTROLLER, def_COMMENT, def_Bias_Bias_V, def_Bias_Calibration_V_V, def_Bias_Offset_V, def_Current_Current_A,
    def_Current_Calibration_A_V, def_Current_Offset_A, def_Current_Gain, def_Piezo_Configuration_Active_Calib,
    def_Piezo_Configuration_Calib_X_m_V, def_Piezo_Configuration_Calib_Y_m_V, def_Piezo_Configuration_Calib_Z_m_V,
    def_Piezo_Configuration_HV_Gain_X, def_Piezo_Configuration_HV_Gain_Y, def_Piezo_Configuration_HV_Gain_Z,
    def_Piezo_Configuration_Tilt_X_deg, def_Piezo_Configuration_Tilt_Y_deg,
    def_Piezo_Configuration_Curvature_radius_X_m,
    def_Piezo_Configuration_Curvature_radius_Y_m, def_Piezo_Configuration_2nd_order_corr_X_V_m_2,
    def_Piezo_Configuration_2nd_order_corr_Y_V_m_2, def_Piezo_Configuration_Drift_X_m_s,
    def_Piezo_Configuration_Drift_Y_m_s, def_Piezo_Configuration_Drift_Z_m_s,
    def_Piezo_Configuration_Drift_correction_status_on_off, def_Z_Controller_Z_m, def_Z_Controller_Controller_name,
    def_Z_Controller_Controller_status, def_Z_Controller_Setpoint, def_Z_Controller_Setpoint_unit,
    def_Z_Controller_P_gain, def_Z_Controller_I_gain, def_Z_Controller_Time_const_s, def_Z_Controller_TipLift_m,
    def_Z_Controller_Switch_off_delay_s, def_Scan_Scanfield, def_Scan_series_name, def_Scan_channels,
    def_Scan_pixels_line, def_Scan_lines, def_Scan_speed_forw_m_s, def_Scan_speed_backw_m_s, def_DATA_INFO,
    def_SCANIT_END]


def _writeDefaults():
    global settings, settings_file
    conf[cat] = {x[0]: x[1] for x in settings}
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

def rewrite_file():
    global settings, settings_file
    conf[cat] = {x[0]: x[1] for x in settings}
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

def update_params():
    conf.read(settings_file)
    global settings
    for elem in settings:
        elem[1] = conf[cat][elem[0]]

def get_header_arr():
    return settings

def get_header_dict():
    return {elem[0]: elem[1] for elem in settings}

def updateTime():
    global def_REC_TIME
    global def_REC_DATE
    new_date = def_REC_DATE[0], [[datetime.date.today().strftime("%d.%m.%y")]]
    new_time = def_REC_TIME[0], [[print(datetime.datetime.now().strftime("%H:%M:%S"))]]
    def_REC_TIME = new_time
    def_REC_DATE = new_date


def adjust_to_image(data):
    global def_SCAN_PIXELS
    width = np.shape(data)[0]
    height = np.shape(data)[1]
    new_sp = def_SCAN_PIXELS[0], [[str(width), str(height)]]
    def_SCAN_PIXELS = new_sp
    rewrite_file()
    update_params()






