"""
Configuration File containing global default values
"""
import math
from dataclasses import dataclass
from typing import List

import matplotlib.pyplot as plt
import torch.types

from util.dot_dict import DotDict
from util.misc import calc_n_samples

PLOT_TO_PDF = False
VERBOSE = False
SHOW_PLOTS = False
# if True EEG Data is always resampled to CONFIG.SYSTEM_SAMPLE_RATE
RESAMPLE = True

# Turn interactive plotting off
if SHOW_PLOTS is False:
    plt.ioff()
else:
    plt.ion()
# Calculate the difference in accuracy between Testing Dataset and Training Dataset
# if True the differences are stored in the results.txt
TEST_OVERFITTING = True


class MIConfig(object):
    """
    Object containing all relevant Machine Learning Training Parameters
    """
    # Training Settings
    EPOCHS: int = 100
    SPLITS: int = 5
    VALIDATION_SUBJECTS: int = 0
    N_CLASSES: List[int] = [2]

    # Learning Rate Settings
    LR: DotDict = DotDict(
        start=0.01,
        milestones=[20, 50],
        gamma=0.1
    )

    BATCH_SIZE: int = 16

    # Benchmark Settings
    SUBJECTS_CS: int = 10
    GPU_WARMUPS: int = 20
    # Jetson Nano cant handle bigger Batch Sizes when device='cpu'
    JETSON_CPU_MAX_BS: int = 15

    def set_lr_milestones(self, milestones: List[int]):
        self.LR.milestones = milestones


@dataclass
class FilterConfig(object):
    """
    Object containing all relevant Filter Parameters to be used
    """
    FREQ_FILTER_HIGHPASS: float = None
    FREQ_FILTER_LOWPASS: float = None
    USE_NOTCH_FILTER: bool = False
    NORMALIZE: bool = False

    def set_filters(self, fmin=None, fmax=None, notch=False):
        if fmin == 0:
            fmin = None
        self.FREQ_FILTER_HIGHPASS = fmin
        self.FREQ_FILTER_LOWPASS = fmax
        self.USE_NOTCH_FILTER = notch

    def __repr__(self):
        return f"""
Bandpass Filter: [{self.FREQ_FILTER_HIGHPASS};{self.FREQ_FILTER_LOWPASS}]Hz
Notch Filter (60Hz): {self.USE_NOTCH_FILTER}
"""


@dataclass
class ANNConfig(object):
    """
    Object containing all additional configuration Parameters for the used Artificial Neural Network
    """
    # Pool Size of EEGNet
    POOL_SIZE: int = 4
    DROPOUT: float = 0.4
    KERNLENGTH: int = 80

    def set_model_params(self, dropout: float = None, kernlength: int = None, pool_size: int = None):
        if dropout is not None:
            self.DROPOUT = dropout
        if kernlength is not None:
            self.KERNLENGTH = kernlength
        if pool_size is not None:
            self.POOL_SIZE = pool_size

    def __repr__(self):
        return f"""
Model: EEGNet
Pool Size: {self.POOL_SIZE}
Dropout: {self.DROPOUT}
Kernel Length: {self.KERNLENGTH}
"""


@dataclass
class EEGConfig(object):
    """
    Object containing all configuration Variables for loading the selected EEG Dataset
    Values are copied from the selected Dataset's DSConstants class ({Dataset Name}_dataset.py)
    (see /util/cmd_parser.py 'check_common_arguments()' -> 'CONFIG.EEG.set_config(dataset.CONSTANTS.CONFIG)')
    """
    # Time Interval per EEG Trial (T=0: start of MI Cue)
    # Trials Slicing (divide every Trial in equally long Slices)
    TMIN: float = -1
    TMAX: float = -1
    CUE_OFFSET: float = None
    TRIALS_SLICES: int = 1
    SAMPLERATE: float = -1
    SAMPLES: int = -1
    # FOR LSMR21:
    # 0 = disallow Trials with Artifacts, 1 = use all Trials
    ARTIFACTS: int = 1
    # 0 = use all Trials
    # 1 = use only Trials with forcedresult = 1
    # 2 = use only Trials with results = 1
    TRIAL_CATEGORY: int = 0

    # ONLY USED FOR 'PHYS' DATASET
    REST_TRIALS_FROM_BASELINE_RUN: bool = True
    REST_TRIALS_LESS: int = 0

    def __repr__(self):
        return f"""
EEG Epoch interval: [{self.TMIN - self.CUE_OFFSET};{self.TMAX - self.CUE_OFFSET}]s
Cue Offset: {self.CUE_OFFSET}
Included Trials with Artifacts: {'Yes' if self.ARTIFACTS == 1 else 'No'}
Trial Category: {self.TRIAL_CATEGORY}
Trials Slices: {self.TRIALS_SLICES}
"""

    def set_cue_offset(self, cue_offset):
        if self.CUE_OFFSET is not None:
            self.TMIN -= self.CUE_OFFSET
            self.TMAX -= self.CUE_OFFSET
        self.CUE_OFFSET = cue_offset
        self.TMIN += self.CUE_OFFSET
        self.TMAX += self.CUE_OFFSET
        self.SAMPLES = math.floor(
            ((self.TMAX - self.TMIN) * self.SAMPLERATE) / self.TRIALS_SLICES)

    def set_trials_slices(self, slices: int):
        # eeg_config.TMIN = 0
        # eeg_config.TMAX = 4
        self.TRIALS_SLICES = slices
        self.SAMPLES = math.floor(((self.TMAX - self.TMIN) * self.SAMPLERATE) / slices)

    def set_times(self, tmin=None, tmax=None, cue_offset=None):
        if cue_offset is not None:
            self.CUE_OFFSET = cue_offset
        if tmin is not None:
            self.TMIN = tmin + self.CUE_OFFSET
        if tmax is not None:
            self.TMAX = tmax + self.CUE_OFFSET
        self.SAMPLES = calc_n_samples(self.TMIN, self.TMAX, self.SAMPLERATE)

    def set_samples(self, samples):
        self.SAMPLES = samples

    def set_samplerate(self, sr):
        self.SAMPLERATE = sr
        self.SAMPLES = calc_n_samples(self.TMIN, self.TMAX, self.SAMPLERATE)

    def _set_config_(self, cfg):
        # If CUE_OFFSET is manually set with set_cue_offset() before main.py (e.g. in batch_training 'init' methods)
        # do not overwrite manually set CUE_OFFSET
        if self.CUE_OFFSET is None:
            self.CUE_OFFSET = cfg.CUE_OFFSET
        self.TMIN = cfg.TMIN + self.CUE_OFFSET
        self.TMAX = cfg.TMAX + self.CUE_OFFSET
        self.TRIALS_SLICES = 1
        self.SAMPLERATE = cfg.SAMPLERATE
        self.SAMPLES = calc_n_samples(cfg.TMIN, cfg.TMAX, cfg.SAMPLERATE)
        if RESAMPLE:
            self.set_samplerate(CONFIG.SYSTEM_SAMPLE_RATE)

    def set_artifacts_trial_category(self, artifacts: int = None, trial_category: int = None):
        if artifacts is not None:
            self.ARTIFACTS = artifacts
        if trial_category is not None:
            self.TRIAL_CATEGORY = trial_category


class Config(object):
    """
    Singleton Object for all relevant global Config Variables
    """
    # Global System Sample Rate
    # after preloading any Dataset the EEG Data gets resampled
    # to this Samplerate (see training_cv() in machine_learning/modes.py)
    SYSTEM_SAMPLE_RATE: int = 250

    DEVICE: torch.types.Device = torch.device("cpu")

    EEG: EEGConfig = EEGConfig()
    NET: ANNConfig = ANNConfig()
    FILTER: FilterConfig = FilterConfig()
    MI: MIConfig = MIConfig()

    def reset(self):
        self.EEG = EEGConfig()
        self.NET = ANNConfig()
        self.FILTER = FilterConfig()
        self.MI = MIConfig()

    def set_eeg_config(self, cfg):
        self.EEG._set_config_(cfg)
        self.NET.KERNLENGTH = self.EEG.SAMPLERATE // 2

    def __repr__(self):
        return f"""
System Sample Rate: {self.SYSTEM_SAMPLE_RATE}
## EEG Config:{self.EEG}
## Filter Config:{self.FILTER} 
## Net Model Config:{self.NET}"""


CONFIG = Config()

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
