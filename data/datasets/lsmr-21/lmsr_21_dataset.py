"""
File containing all relevant Properties of the
LSMR-21 dataset
"""

from util.dot_dict import DotDict


class LSMR_21:
    name = 'Human EEG Dataset for Brain-Computer Interface and Meditation'
    short_name = 'LSMR-21'

    # No excluded subjects
    ALL_SUBJECTS = [i for i in range(1, 62) if i not in []]  # 62 Subjects

    # Available 62 EEG Channels
    CHANNELS = ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5',
                'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8',
                'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4',
                'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2']
    cv_folds = 5
    # Time Interval per EEG Trial (T=4: start of MI Cue)
    CONFIG = DotDict(
        TMIN=0.0,
        TMAX=10.0,
        SAMPLERATE=1000,
        CUE_OFFSET=4.0
    )
