"""
Configuration File containing global default values
"""
import math

from data.physionet_dataset import PHYSIONET
from util.dot_dict import DotDict

PLOT_TO_PDF = True
VERBOSE = False
# Calculate the difference in accuracy between (new) Validation Dataset and known Training Dataset
# if True the differences are stored in the results.txt
TEST_OVERFITTING = True
# Preloads all subjects data for n_classes classification Training in memory
# for -benchmark: decrease --subjects_cs (see main.py) to decrease memory usage when benchmarking
DATA_PRELOAD = True
global_config = DotDict(FREQ_FILTER_HIGHPASS=None,
                        FREQ_FILTER_LOWPASS=None,
                        USE_NOTCH_FILTER=False)

# Training Settings
EPOCHS = 100
SPLITS = 5
VALIDATION_SUBJECTS = 0
N_CLASSES = [2, 3, 4]
# Learning Rate Settings
LR = DotDict(
    start=0.01,
    milestones=[20, 40],
    gamma=0.1
)

BATCH_SIZE = 16

# Benchmark Settings
SUBJECTS_CS = 10
GPU_WARMUPS = 20

eegnet_config = DotDict(pool_size=4)

# Time Interval per EEG Trial (T=0: start of MI Cue)
eeg_config = DotDict(EEG_TMIN=PHYSIONET.EEG_TMIN,
                     EEG_TMAX=PHYSIONET.EEG_TMAX,
                     TRIAL_SLICES=1,
                     SAMPLERATE=PHYSIONET.SAMPLERATE,
                     SAMPLES=(PHYSIONET.EEG_TMAX - PHYSIONET.EEG_TMIN) * PHYSIONET.SAMPLERATE)


def set_eeg_trials_slices(slices):
    # eeg_config.EEG_TMIN = 0
    # eeg_config.EEG_TMAX = 4
    eeg_config.TRIALS_SLICES = slices
    eeg_config.SAMPLES = math.floor(((eeg_config.EEG_TMAX - eeg_config.EEG_TMIN) * eeg_config.SAMPLERATE) / slices)


def set_eeg_times(tmin, tmax):
    eeg_config.EEG_TMIN = tmin
    eeg_config.EEG_TMAX = tmax
    eeg_config.SAMPLES = (tmax - tmin) * eeg_config.SAMPLERATE


def reset_eeg_times():
    eeg_config.EEG_TMIN = PHYSIONET.EEG_TMIN
    eeg_config.EEG_TMAX = PHYSIONET.EEG_TMAX
    eeg_config.SAMPLERATE = PHYSIONET.SAMPLERATE
    eeg_config.SAMPLES = (PHYSIONET.EEG_TMAX - PHYSIONET.EEG_TMIN) * PHYSIONET.SAMPLERATE


def set_poolsize(size):
    eegnet_config.pool_size = size


results_folder = './results'
training_results_folder = f"/training"
benchmark_results_folder = f"/benchmark"
live_sim_results_folder = f"/live_sim"
training_ss_results_folder = f"/training_ss"

trained_model_name = "trained_model.pt"
trained_ss_model_name = "trained_ss_model.pt"
chs_names_txt = "ch_names.txt"
# Folder where MNE downloads Physionet Dataset to
# on initial Run MNE needs to download the Dataset
datasets_folder = './data/datasets/'

# Selections of Channels for reduced amount of needed EEG Channels
# Visualization:
# https://upload.wikimedia.org/wikipedia/commons/thumb/f/fb/EEG_10-10_system_with_additional_information.svg/640px-EEG_10-10_system_with_additional_information.svg.png?1615815815740

# Source:
# https://www.researchgate.net/publication/324826717_Motor_Imagery_EEG_Signal_Processing_and_Classification_using_Machine_Learning_Approach
MOTORIMG_CHANNELS_21 = [
    'Fc5', 'Fc3', 'Fc1', 'Fcz', 'Fc2', 'Fc4', 'Fc6',
    'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
    'Cp5', 'Cp3', 'Cp1', 'Cpz', 'Cp2', 'Cp4', 'Cp6',
]
MOTORIMG_CHANNELS_18 = [
    'Fc5', 'Fc3', 'Fc1', 'Fc2', 'Fc4', 'Fc6',
    'C5', 'C3', 'C1', 'C2', 'C4', 'C6',
    'Cp5', 'Cp3', 'Cp1', 'Cp2', 'Cp4', 'Cp6',
]
MOTORIMG_CHANNELS_16_openbci = [
    'F3', 'Fz', 'F4',
    'Fc5', 'Fc1', 'Fc2', 'Fc6',
    'T7', 'C3', 'Cz', 'C4', 'T8',
    'Cp5', 'Cp1', 'Cp2', 'Cp6',
]
MOTORIMG_CHANNELS_16 = [
    'Fc5', 'Fc3', 'Fc1', 'Fc2', 'Fc4', 'Fc6',
    'C3', 'C1', 'C2', 'C4',
    'Cp5', 'Cp3', 'Cp1', 'Cp2', 'Cp4', 'Cp6',
]
MOTORIMG_CHANNELS_16_2 = [
    'Fc5', 'Fc3', 'Fc1', 'Fc2', 'Fc4', 'Fc6',
    'C5', 'C3', 'C1', 'C2', 'C4', 'C6',
    'Cp3', 'Cp1', 'Cp2', 'Cp4',
]
MOTORIMG_CHANNELS_16_bs = [
    'Fp1', 'Fpz', 'Fp2',
    'Af3', 'Af4',
    'F7', 'F8',
    'T10',
    'P7', 'Po7', 'O1', 'Oz', 'O2', 'Po4', 'P8',
    'Iz'
]
MOTORIMG_CHANNELS_14 = [
    'Fc3', 'Fc1', 'Fcz', 'Fc2', 'Fc4',
    'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
    'Cp3', 'Cp4',
]
MOTORIMG_CHANNELS_14_2 = [
    'Fc3', 'Fc1', 'Fcz', 'Fc2', 'Fc4',
    'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
    'Cp1', 'Cp2',
]
MOTORIMG_CHANNELS_14_3 = [
    'Fc5', 'Fc3', 'Fc1', 'Fc2', 'Fc4', 'Fc6',
    'C3', 'C1', 'C2', 'C4',
    'Cp3', 'Cp1', 'Cp2', 'Cp4',
]
MOTORIMG_CHANNELS_14_4 = [
    'Fc5', 'Fc3', 'Fc1', 'Fc2', 'Fc4', 'Fc6',
    'C5', 'C3', 'C1', 'C2', 'C4', 'C6',
    'Cp1', 'Cp2'
]
# Source:
# https://www.researchgate.net/publication/43407016_Exploring_Large_Virtual_Environments_by_Thoughts_Using_a_Brain-Computer_Interface_Based_on_Motor_Imagery_and_High-Level_Commands/figures?lo=1
MOTORIMG_CHANNELS_12 = [
    'Fc3', 'Fcz', 'Fc4',
    'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
    'Cp3', 'Cpz', 'Cp4',
]
MOTORIMG_CHANNELS_8 = [
    'Fc5', 'Fc1', 'Fc2', 'Fc6',

    'Cp5', 'Cp1', 'Cp2', 'Cp6',
]
MOTORIMG_CHANNELS_8_2 = [
    'Fc3', 'Fc4',
    'C5', 'C1', 'C2', 'C6',
    'Cp3', 'Cp4'
]
MOTORIMG_CHANNELS_8_3 = [
    'Fc1', 'Fc2',
    'C5', 'C3', 'C4', 'C6',
    'Cp1', 'Cp2'
]
MOTORIMG_CHANNELS_8_4 = [
    'Fc3', 'Fc4',
    'C3', 'C1', 'C2', 'C4',
    'Cp3', 'Cp4'
]
MOTORIMG_CHANNELS_8_5 = [
    'Fc5', 'Fc6',
    'C3', 'C1', 'C2', 'C4',
    'Cp5', 'Cp6'
]
MOTORIMG_CHANNELS_8_6 = [
    'Fc5', 'Fc6',
    'C5', 'C3', 'C1', 'C2', 'C4', 'C6'
]
# source:
# https://www.mdpi.com/1424-8220/17/7/1557/htm
MOTORIMG_CHANNELS_5 = [
    "C3", "Cz", "C4", "Cp3", "Cp4"
]

MOTORIMG_CHANNELS = {
    '5': MOTORIMG_CHANNELS_5, '8': MOTORIMG_CHANNELS_8,
    '8_2': MOTORIMG_CHANNELS_8_2, '8_3': MOTORIMG_CHANNELS_8_3,
    '8_4': MOTORIMG_CHANNELS_8_4, '8_5': MOTORIMG_CHANNELS_8_5,
    '8_6': MOTORIMG_CHANNELS_8_6, '12': MOTORIMG_CHANNELS_12,
    '14': MOTORIMG_CHANNELS_14, '14_2': MOTORIMG_CHANNELS_14_2,
    '14_3': MOTORIMG_CHANNELS_14_3, '14_4': MOTORIMG_CHANNELS_14_4,
    '16': MOTORIMG_CHANNELS_16, '16_2': MOTORIMG_CHANNELS_16_2,
    '16_openbci': MOTORIMG_CHANNELS_16_openbci,
    '16_bs': MOTORIMG_CHANNELS_16_bs,
    '18': MOTORIMG_CHANNELS_18, '21': MOTORIMG_CHANNELS_21
}
