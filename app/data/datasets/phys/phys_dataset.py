"""
File containing all relevant Properties of the
Physionet Motorimagery Dataset
https://physionet.org/content/eegmmidb/1.0.0/
"""

from mne.datasets import eegbci



# Default Constants of the PHYS Dataset
# DO NOT MODIFY FROM ANYWHERE ELSE IN THE CODE
from app.config import EEGConfig
from app.data.datasets.DSConstants import DSConstants


class PHYSConstants(DSConstants):
    def __init__(self):
        super().__init__()

        self.name = 'Physionet MI dataset'
        self.short_name: str = 'PHYS'
        # Available 64 EEG Channels from Physionet Dataset
        # raw.info['ch_names']
        self.CHANNELS = [
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

        self.cv_folds = 5
        # Time Interval per EEG Trial (T=0: start of MI Cue)
        # These are Constants, DO NOT MODIFY!
        self.CONFIG: EEGConfig = EEGConfig(
            TMIN=0.0,
            TMAX=2.0,
            SAMPLERATE=160,
            # if True: load Run 0 for Rest Trials, ignore in other Runs
            # not relevant for 2 class Classification
            REST_TRIALS_FROM_BASELINE_RUN=True,
            REST_TRIALS_LESS=0,
            CUE_OFFSET=0.0
        )
        self.TRIAL_TMAX = 6 - self.CONFIG.CUE_OFFSET
        self.TRIAL_TMIN = -2 - self.CONFIG.CUE_OFFSET
        self.REST_PHASES = [(0.0, 2.0), (6.0, 8.0)]

        # Every Run contains 7 Trials per Class
        self.TRIALS_PER_CLASS_PER_RUN = 7
        # Amount of picked Trials in 1 Run of 1 Subject
        self.TRIALS_PER_SUBJECT_RUN = 3 * self.TRIALS_PER_CLASS_PER_RUN

        # Some Subjects are excluded due to differing numbers of Trials in the recordings
        self.ALL_SUBJECTS = [i for i in range(1, 20) if i not in [88, 92, 100, 104]]

        runs_rest = [1]  # Baseline, eyes open
        runs_t1 = [3, 7, 11]  # Task 1 (open and close left or right fist)
        runs_t2 = [4, 8, 12]  # Task 2 (imagine opening and closing left or right fist)
        runs_t3 = [5, 9, 13]  # Task 3 (open and close both fists or both feet)
        runs_t4 = [6, 10, 14]  # Task 4 (imagine opening and closing both fists or both feet)

        self.runs = [runs_rest, runs_t1, runs_t2, runs_t3, runs_t4]

        # Mapping of n_class Classification to correct Tasks
        # e.g. 3class Classification uses Task 0 (Baseline Rest Run) + Task 2 (Left/right Fist)
        # Keep following in Sync with PHYSDataLoader.load_n_classes_tasks() 'exclude_bothfists'
        # additional changes in phys_data_loading
        # WITHOUT 'rest' Trials for 3/4-class:
        # Sync with data_utils.py get_trials_size() Method
        self.n_classes_tasks = {1: [0], 2: [2], 3: [2, 4], 4: [2, 4]}
        # WITH 'rest' Trials for 3/4-class:
        # self.n_classes_tasks = {1: [0], 2: [2], 3: [0, 2], 4: [0, 2, 4]}

        # Sample run for n_class live_sim mode
        self.n_classes_live_run = {2: runs_t2[-1], 3: runs_t2[-1], 4: runs_t2[-1]}

        # Maximum available trials
        self.trials_for_classes_per_subject_avail = {2: 42, 3: 84, 4: 153}

        # All total trials per class per n_class-Classification
        self.classes_trials = {
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

        self.class_labels = {
            2: ['L', 'R'],
            3: ['L', 'R', 'D'],
            4: ['Rest', 'L', 'R', 'U', 'D'],
        }

        self.mne_dataset = eegbci

    # How many less Rest Trials than for other classes (per Subject)
    def set_rests_config(self, from_bl_run=None, less_rests=None):
        if from_bl_run is not None:
            self.set_rest_from_bl_run(from_bl_run)
        if less_rests is not None:
            self.set_rest_trials_less(less_rests)

    def set_rest_from_bl_run(self, val):
        self.CONFIG.REST_TRIALS_FROM_BASELINE_RUN = val

    def set_rest_trials_less(self, val):
        self.CONFIG.REST_TRIALS_LESS = val


PHYS = PHYSConstants()
