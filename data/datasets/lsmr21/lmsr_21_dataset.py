"""
File containing all relevant Properties of the
LSMR-21 dataset
"""

from util.dot_dict import DotDict


class LSMR21:
    name = 'Human EEG Dataset for Brain-Computer Interface and Meditation'
    short_name = 'LSMR21'

    # No excluded subjects
    ALL_SUBJECTS = [1, 2, 7, 26]  # 62 Subjects
    runs = [1, 11]
    trials_per_subject = 115
    cv_folds = 2

    # # No excluded subjects
    # ALL_SUBJECTS = [i for i in range(1,62) if i not in []]  # 62 Subjects
    #
    # runs = [i for i in range(1,12)]
    #
    # trials_per_subject = len(runs) * 450
    # cv_folds = 5

    # Available 62 EEG Channels
    CHANNELS = ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5',
                'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8',
                'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4',
                'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2']
    # Time Interval per EEG Trial (T=4: start of MI Cue)
    # Time: 2s blank Screen + 2s Target Presentation + 0-6s Feedback Control + 1s additional = max. 11s
    # TMAX = 7s (0-6s Feedback Control (MI Cue) + 1s additional)
    CONFIG = DotDict(
        TMIN=0.0,
        TMAX=2.0,
        SAMPLERATE=1000,
        # TODO KEEP IN SYNC WITH NUMPY DOWNSAMPLING
        #SAMPLERATE=250,
        CUE_OFFSET=4.0
    )

    # Mapping of n_class Classification to correct Tasks
    # 1: Left/Right, 2: Up/Down, 3: 2D (Left/Right/Up/Down)
    n_classes_tasks = {2: [1], 3: [1, 2], 4: [1, 2, 3]}

    n_classes_targets = {2: [1, 2], 3: [1, 2, 3], 4: [1, 2, 3, 4]}
