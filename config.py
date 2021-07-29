"""
Configuration File containing global default values
"""
import math
import os
import sys

import matplotlib.pyplot as plt
from data.datasets.phys.phys_dataset import PHYS
from util.dot_dict import DotDict

PLOT_TO_PDF = False
VERBOSE = False
SHOW_PLOTS = False

# Turn interactive plotting off
if SHOW_PLOTS is False:
    plt.ioff()
else:
    plt.ion()
# Calculate the difference in accuracy between Testing Dataset and Training Dataset
# if True the differences are stored in the results.txt
TEST_OVERFITTING = True
# Preloads all subjects data for n_classes classification Training in memory
# for -benchmark: decrease --subjects_cs (see main.py) to decrease memory usage when benchmarking
global_config = DotDict(FREQ_FILTER_HIGHPASS=None,
                        FREQ_FILTER_LOWPASS=None,
                        USE_NOTCH_FILTER=False)

# Training Settings
EPOCHS = 100
SPLITS = PHYS.cv_folds
VALIDATION_SUBJECTS = 0
N_CLASSES = [2]

# Learning Rate Settings
LR = DotDict(
    start=0.01,
    milestones=[20, 50],
    gamma=0.1
)

BATCH_SIZE = 16

# Benchmark Settings
SUBJECTS_CS = 10
GPU_WARMUPS = 20
# Jetson Nano cant handle bigger Batch Sizes when device='cpu'
JETSON_CPU_MAX_BS = 15

eegnet_config = DotDict(pool_size=4)

# Global System Sample Rate
# after preloading any Dataset the EEG Data gets resampled
# to this Samplerate (see training_cv() in machine_learning/modes.py)
SYSTEM_SAMPLE_RATE = 250

# Time Interval per EEG Trial (T=0: start of MI Cue)
# Trials Slicing (divide every Trial in equally long Slices)
eeg_config = DotDict(TMIN=PHYS.CONFIG.TMIN,
                     TMAX=PHYS.CONFIG.TMAX,
                     CUE_OFFSET=PHYS.CONFIG.CUE_OFFSET,
                     TRIALS_SLICES=1,
                     SAMPLERATE=PHYS.CONFIG.SAMPLERATE,
                     SAMPLES=int((PHYS.CONFIG.TMAX - PHYS.CONFIG.TMIN) * PHYS.CONFIG.SAMPLERATE),
                     # FOR LSMR21:
                     # 0 = disallow Trials with Artifacts, 1 = use all Trials
                     ARTIFACTS=1,
                     # 0 = use all Trials
                     # 1 = use only Trials with forcedresult = 1
                     # 2 = use only Trials with results = 1
                     TRIAL_CATEGORY=0)


def set_eeg_config(cfg: DotDict):
    eeg_config.TMIN = cfg.TMIN + cfg.CUE_OFFSET
    eeg_config.TMAX = cfg.TMAX + cfg.CUE_OFFSET
    eeg_config.TRIALS_SLICES = 1
    eeg_config.CUE_OFFSET = cfg.CUE_OFFSET
    eeg_config.SAMPLERATE = cfg.SAMPLERATE
    eeg_config.SAMPLES = int((cfg.TMAX - cfg.TMIN) * cfg.SAMPLERATE)


def set_eeg_artifacts_trial_category(artifacts: int = eeg_config.ARTIFACTS,
                                     trial_category: int = eeg_config.TRIAL_CATEGORY):
    eeg_config.ARTIFACTS = artifacts
    eeg_config.TRIAL_CATEGORY = trial_category


def set_eeg_samplerate(sr):
    eeg_config.SAMPLERATE = sr
    eeg_config.SAMPLES = int((eeg_config.TMAX - eeg_config.TMIN) * eeg_config.SAMPLERATE)


def set_eeg_trials_slices(slices):
    # eeg_config.TMIN = 0
    # eeg_config.TMAX = 4
    eeg_config.TRIALS_SLICES = slices
    eeg_config.SAMPLES = math.floor(((eeg_config.TMAX - eeg_config.TMIN) * eeg_config.SAMPLERATE) / slices)


def set_eeg_times(tmin, tmax, cue_offset):
    eeg_config.TMIN = tmin + cue_offset
    eeg_config.TMAX = tmax + cue_offset
    eeg_config.CUE_OFFSET = cue_offset
    eeg_config.SAMPLES = int((tmax - tmin) * eeg_config.SAMPLERATE)


def set_eeg_cue_offset(cue_offset):
    eeg_config.TMIN -= eeg_config.CUE_OFFSET
    eeg_config.TMIN += cue_offset
    eeg_config.TMAX -= eeg_config.CUE_OFFSET
    eeg_config.TMAX += cue_offset
    eeg_config.CUE_OFFSET = cue_offset
    eeg_config.SAMPLES = math.floor(
        ((eeg_config.TMAX - eeg_config.TMIN) * eeg_config.SAMPLERATE) / eeg_config.TRIALS_SLICES)


def reset_eeg_times():
    eeg_config.TMIN = PHYS.CONFIG.TMIN
    eeg_config.TMAX = PHYS.CONFIG.TMAX
    eeg_config.SAMPLERATE = PHYS.CONFIG.SAMPLERATE
    eeg_config.SAMPLES = int((PHYS.CONFIG.TMAX - PHYS.CONFIG.TMIN) * PHYS.CONFIG.SAMPLERATE)


def set_poolsize(size):
    eegnet_config.pool_size = size


def set_bandpassfilter(fmin=None, fmax=None, notch=False):
    global_config.FREQ_FILTER_HIGHPASS = fmin
    global_config.FREQ_FILTER_LOWPASS = fmax
    global_config.USE_NOTCH_FILTER = notch


# Project's root path
ROOT = os.path.dirname(os.path.abspath(__file__))

sys.path.append(ROOT)
to_path = lambda x: os.path.join(ROOT, x)

results_folder = to_path('results')
training_results_folder = '/training'
benchmark_results_folder = '/benchmark'
live_sim_results_folder = '/live_sim'
training_ss_results_folder = '/training_ss'

trained_model_name = "trained_model.pt"
trained_ss_model_name = "trained_ss_model.pt"
chs_names_txt = "ch_names.txt"
# Folder where MNE downloads Physionet Dataset to
# on initial Run MNE needs to download the Dataset

datasets_folder = '/opt/datasets'

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
# Inspired by:
# https://www.sciencedirect.com/science/article/pii/S0925231215002295
MOTORIMG_CHANNELS_16_csp = [
    'Af3', 'Afz',
    'F3', 'Fz',
    'Fc3', 'Fc2',
    'C5', 'C3', 'C1', 'Cz',
    'Cp3', 'Cp1',
    'P3', 'P1', 'Pz',
    'Poz'
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
    '16_csp': MOTORIMG_CHANNELS_16_csp,
    '16_bs': MOTORIMG_CHANNELS_16_bs,
    '18': MOTORIMG_CHANNELS_18, '21': MOTORIMG_CHANNELS_21
}

# Neural Response Frequency bands (FMIN,FMAX)-Tuples
#  all: no Bandpass
#   F1: 0-8Hz
#   F2: 8-16Hz
#   F1: 16-28Hz
FBS = [(None, None), (None, 8), (8, 16), (16, 28)]
FBS_NAMES = ['all', 'f1', 'f2', 'f3']
