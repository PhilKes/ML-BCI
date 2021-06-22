"""
File containing all relevant Properties of the
Physionet Motorimagery Dataset
https://physionet.org/content/eegmmidb/1.0.0/
"""

from mne.datasets import eegbci

from util.dot_dict import DotDict

PHYS_name = 'Physionet MI dataset'
PHYS_short_name = 'PHYS'
# Available 64 EEG Channels from Physionet Dataset
# raw.info['ch_names']
PHYS_CHANNELS = [
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

PHYS_cv_folds = 5
# Time Interval per EEG Trial (T=0: start of MI Cue)
PHYS_CONFIG = DotDict(
    TMIN=0.0,
    TMAX=2.0,
    SAMPLERATE=160,
    # if True: load Run 0 for Rest Trials, ignore in other Runs
    # not relevant for 2 class Classification
    REST_TRIALS_FROM_BASELINE_RUN=True,
    REST_TRIALS_LESS=0,
    CUE_OFFSET=0.0
)


# How many less Rest Trials than for other classes (per Subject)
def set_rests_config(from_bl_run=None, less_rests=None):
    if from_bl_run is not None:
        set_rest_from_bl_run(from_bl_run)
    if less_rests is not None:
        set_rest_trials_less(less_rests)


def set_rest_from_bl_run(val):
    PHYS_CONFIG.REST_TRIALS_FROM_BASELINE_RUN = val


def set_rest_trials_less(val):
    PHYS_CONFIG.REST_TRIALS_LESS = val


# Amount of picked Trials in 1 Run of 1 Subject
TRIALS_PER_SUBJECT_RUN = 21

# Some Subjects are excluded due to differing numbers of Trials in the recordings
excluded_subjects = [88, 92, 100, 104]
PHYS_ALL_SUBJECTS = [i for i in range(1, 110) if i not in excluded_subjects]

runs_rest = [1]  # Baseline, eyes open
runs_t1 = [3, 7, 11]  # Task 1 (open and close left or right fist)
runs_t2 = [4, 8, 12]  # Task 2 (imagine opening and closing left or right fist)
runs_t3 = [5, 9, 13]  # Task 3 (open and close both fists or both feet)
runs_t4 = [6, 10, 14]  # Task 4 (imagine opening and closing both fists or both feet)

runs = [runs_rest, runs_t1, runs_t2, runs_t3, runs_t4]

# Mapping of n_class Classification to correct Tasks
# e.g. 3class Classification uses Task 0 (Baseline Rest Run) + Task 2 (Left/right Fist)
n_classes_tasks = {1: [0], 2: [2], 3: [0, 2], 4: [0, 2, 4]}

# Sample run for n_class live_sim mode
n_classes_live_run = {2: runs_t2[-1], 3: runs_t2[-1], 4: runs_t2[-1]}

# Maximum available trials
trials_for_classes_per_subject_avail = {2: 42, 3: 84, 4: 153}

# All total trials per class per n_class-Classification
classes_trials = {
    "2class": {
        0: 445,  # Left
        1: 437,  # Right
    },
    "3class": {
        0: 882,  # Rest
        1: 445,  # Left
        2: 437,  # Right
    },
    "4class": {
        0: 1748,  # Rest
        1: 479,  # Left
        2: 466,  # Right
        3: 394,  # Both Fists
    },
}

class_labels = {
    2: ['L', 'R'],
    3: ['Rest', 'L', 'R'],
    4: ['Rest', 'L', 'R', 'D'],
}

mne_dataset = eegbci
