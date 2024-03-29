"""
File containing all relevant Properties of the
LSMR-21 dataset:
https://figshare.com/articles/dataset/Human_EEG_Dataset_for_Brain-Computer_Interface_and_Meditation/13123148
"""
from typing import List

from config import EEGConfig
from data.datasets.DSConstants import DSConstants


# Default Constants of the LSMR21 Dataset
# DO NOT MODIFY FROM ANYWHERE ELSE IN THE CODE
class LSMR21Constants(DSConstants):
    def __init__(self):
        super().__init__()

        self.name = 'Human EEG Dataset for Brain-Computer Interface and Meditation'
        self.short_name = 'LSMR21'

        # No excluded subjects
        self.ALL_SUBJECTS = [i for i in range(62)]  # 62 Subjects
        self.runs = [i for i in range(11)]
        # Maximum number of Trials per class in 1 Subject Run
        self.trials_per_class_per_sr = 80
        self.cv_folds = 5

        # # No excluded subjects
        # ALL_SUBJECTS = [i for i in range(1,62) if i not in []]  # 62 Subjects
        #
        # runs = [i for i in range(1,12)]
        #
        # trials_per_subject = len(runs) * 450
        # cv_folds = 5

        # Available 62 EEG Channels
        self.CHANNELS = ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7',
                         'FC5',
                         'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4',
                         'C6', 'T8',
                         'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ',
                         'P2', 'P4',
                         'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2']

        # Time Interval per EEG Trial (T=4: start of MI Cue)
        # Time: 2s blank Screen + 2s Target Presentation + 0-6s Feedback Control + 1s additional = max. 11s
        # TMAX = 7s (0-6s Feedback Control (MI Cue) + 1s additional)
        # These are Constants, DO NOT MODIFY!
        # TO load LSMR from original Matlab files:
        # * change SAMPLERATE to 1000 Hz
        # * change parameter from_matlab to True in LSMR21DataLoader.load_subject_run
        # * change Return type of LSMR21DataLoader.load_subject_run to LSMRSubjectRun
        self.CONFIG: EEGConfig = EEGConfig(
            TMIN=0.0,
            TMAX=2.0,
            # SAMPLERATE=1000,
            # TODO KEEP IN SYNC WITH NUMPY DOWNSAMPLING
            SAMPLERATE=250,
            CUE_OFFSET=2.0
        )
        self.TRIAL_TMAX = 11.0 - self.CONFIG.CUE_OFFSET
        self.TRIAL_TMIN = -self.CONFIG.CUE_OFFSET
        self.ORIGINAL_SAMPLERATE = 1000
        # Mapping of n_class Classification to correct Tasks
        # 1: Left/Right, 2: Up/Down, 3: 2D (Left/Right/Up/Down)
        # TODO:
        #  2 class: only Task 1 Trials? (Task 3 also contains Left/Right Trials)
        #  3 class: only Tasks 1,2 Trials? (Task 3 also contains Left/Right/Up Trials)
        #  4 class: only 3 Trials? (Tasks 1+2 also contain Left/Right/Up/Down combined Trials)
        self.n_classes_tasks = {2: [1], 3: [1, 2], 4: [1, 2]}

        # n_classes_targets = {2: [1, 2], 3: [1, 2, 3], 4: [1, 2, 3, 4]}

    @classmethod
    def set_runs(cls, runs: List[int]):
        cls.runs = runs


LSMR21 = LSMR21Constants()
