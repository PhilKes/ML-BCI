"""
Configuration File containing global default values
"""

from util.dot_dict import DotDict

PLOTS = True
VERBOSE = False
# Calculate the difference in accuracy between (new) Validation Dataset and known Training Dataset
# if True the differences are stored in the results.txt
TEST_OVERFITTING = True
# Preloads all subjects data for n_classes classification Training in memory
# for -benchmark: decrease --subjects_cs (see main.py) to decrease memory usage when benchmarking
DATA_PRELOAD = True
global_config = DotDict(FREQ_FILTER_HIGHPASS=60, FREQ_FILTER_LOWPASS=2, USE_NOTCH_FILTER=True)

results_folder = './results'
training_results_folder = f"/training"
benchmark_results_folder = f"/benchmark"

trained_model_name = "trained_model.pt"
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
EEG_TMIN = 0
EEG_TMAX = 3
SAMPLERATE = 160
SAMPLES = (EEG_TMAX - EEG_TMIN) * SAMPLERATE

# Amount of picked Trials in 1 Run of 1 Subject
TRIALS_PER_SUBJECT_RUN = 21

# Training Settings
EPOCHS = 100
SPLITS = 5
VALIDATION_SUBJECTS = 10
N_CLASSES = [2, 3, 4]

BATCH_SIZE = 16

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
MOTORIMG_CHANNELS_18 = [
    'Fc5', 'Fc3', 'Fc1', 'Fc2', 'Fc4', 'Fc6',
    'C5', 'C3', 'C1', 'C2', 'C4', 'C6',
    'Cp5', 'Cp3', 'Cp1', 'Cp2', 'Cp4', 'Cp6',
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

MOTORIMG_CHANNELS = {'8': MOTORIMG_CHANNELS_8, '8_2': MOTORIMG_CHANNELS_8_2,
                     '8_3': MOTORIMG_CHANNELS_8_3, '8_4': MOTORIMG_CHANNELS_8_4,
                     '8_5': MOTORIMG_CHANNELS_8_5, '8_6': MOTORIMG_CHANNELS_8_6,
                     '5': MOTORIMG_CHANNELS_5, '18': MOTORIMG_CHANNELS_18}

# # Calculated with python_test.py get_mean_std():
# channel_means = [-4.4257e-06, -3.6615e-06, -3.5425e-06, -3.1105e-06, -1.9982e-06,
#                  -3.3686e-06, -4.0484e-06, -3.2589e-06, -1.2037e-06, -3.1303e-06,
#                  -1.7123e-06, -1.3769e-06, -3.8620e-06, -3.8488e-06, -3.8019e-06,
#                  -3.4305e-06, -2.2203e-06, -3.4104e-06, -2.4583e-06, -2.7768e-06,
#                  -2.1735e-06, -3.1017e-05, -2.8601e-05, -3.1928e-05, -2.5437e-05,
#                  -1.9414e-05, -1.3425e-05, -1.7509e-05, -2.3842e-05, -9.7920e-06,
#                  -1.0631e-05, -7.9275e-06, -7.3165e-06, -6.1011e-06, -6.9056e-06,
#                  -7.8441e-06, -7.9372e-06, -8.7261e-06, -2.7639e-06, -2.2479e-06,
#                  -1.4207e-07, -8.4886e-08, -2.4083e-06, -9.0723e-07, -8.6527e-08,
#                  -8.9375e-07, -2.7776e-07, -1.2364e-07, -3.0605e-06, -2.8032e-06,
#                  -3.5766e-06, -3.1065e-06, -4.2012e-06, -3.6984e-06, -4.8093e-06,
#                  -3.0346e-06, -3.5540e-06, -4.0859e-06, -2.8972e-06, -5.5720e-06,
#                  -4.2342e-06, -2.3905e-06, -3.9677e-06, -3.4984e-06]
# channel_stds = [6.7977e-05, 6.7956e-05, 6.7134e-05, 6.3398e-05, 6.3251e-05, 6.6822e-05,
#                 6.4781e-05, 6.4687e-05, 6.1571e-05, 5.9905e-05, 5.6210e-05, 5.7690e-05,
#                 6.0474e-05, 5.8444e-05, 5.7063e-05, 5.8189e-05, 5.7179e-05, 5.7279e-05,
#                 5.3735e-05, 5.5777e-05, 5.5726e-05, 1.2370e-04, 1.2028e-04, 1.2554e-04,
#                 1.1895e-04, 9.8800e-05, 8.8457e-05, 9.7427e-05, 1.1697e-04, 8.5738e-05,
#                 8.4617e-05, 7.5524e-05, 7.5537e-05, 7.1688e-05, 7.6756e-05, 7.6226e-05,
#                 8.2670e-05, 8.3560e-05, 6.6245e-05, 6.4996e-05, 6.0052e-05, 5.6108e-05,
#                 5.8593e-05, 6.8983e-05, 5.8227e-05, 6.1241e-05, 6.2516e-05, 6.0535e-05,
#                 5.8508e-05, 6.1415e-05, 5.3972e-05, 5.4090e-05, 5.4028e-05, 5.5977e-05,
#                 5.6267e-05, 6.1128e-05, 6.0736e-05, 6.0434e-05, 5.3701e-05, 5.7876e-05,
#                 6.0989e-05, 6.0698e-05, 5.9087e-05, 5.0683e-05]
# # Calculated using torch.mean on whole Dataset
# torch_mean = -0.0258
# torch_std = 1.0560
#
# means = -6.1795e-06
# stds = 7.1783e-05
# TRANSFORM = torchvision.transforms.Normalize(channel_means, channel_stds)
# TRANSFORM = lambda x: x * 1e4
