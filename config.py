"""
Configuration File containing global default values
"""

from util.dot_dict import DotDict

PLOT_TO_PDF = False
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
datasets_folder = './datasets/'

BASELINE_CORRECTION = False

# Learning Rate Settings
LR = DotDict(
    start=0.01,
    milestones=[20, 50],
    gamma=0.1
)
# Time Interval per EEG Trial (T=0: start of MI Cue)
DEFAULT_EEG_TMIN = 0
DEFAULT_EEG_TMAX = 3
DEFAULT_SAMPLERATE = 160
eeg_config = DotDict(EEG_TMIN=DEFAULT_EEG_TMIN,
                     EEG_TMAX=DEFAULT_EEG_TMAX,
                     SAMPLERATE=DEFAULT_SAMPLERATE,
                     SAMPLES=(DEFAULT_EEG_TMAX - DEFAULT_EEG_TMIN) * DEFAULT_SAMPLERATE)

# Amount of picked Trials in 1 Run of 1 Subject
TRIALS_PER_SUBJECT_RUN = 21

# Training Settings
EPOCHS = 100
SPLITS = 5
VALIDATION_SUBJECTS = 0
N_CLASSES = [2, 3, 4]

BATCH_SIZE = 16

eegnet_config = DotDict(pool_size=4)

# Benchmark Settings
SUBJECTS_CS = 10
GPU_WARMUPS = 20

# Available 64 EEG Channels from Physionet Dataset
# raw.info['ch_names']
MNE_CHANNELS = [
    'Fc5', 'Fc3', 'Fc1', 'Fcz', 'Fc2', 'Fc4', 'Fc6',
    'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
    'Cp5', 'Cp3', 'Cp1', 'Cpz', 'Cp2', 'Cp4', 'Cp6',
    'Fp1', 'Fpz', 'Fp2',
    'Af7', 'Af3', 'Afz', 'Af4', 'Af8',
    'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8',
    'Ft7', 'Ft8',
    'T7', 'T8', 'T9', 'T10',
    'Tp7', 'Tp8',
    'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8',
    'Po7', 'Po3', 'Poz', 'Po4', 'Po8',
    'O1', 'Oz', 'O2',
    'Iz']
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
    '18': MOTORIMG_CHANNELS_18, '21': MOTORIMG_CHANNELS_21}

