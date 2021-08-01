"""
File containing all relevant Properties of the
BCI competition IV 2a dataset

History:
  2021-05-10: Getting started - ms
"""
from config import EEGConfig
from util.dot_dict import DotDict


class BCIC:
    name = 'BCI competition IV 2A MI dataset'
    short_name = 'BCIC'

    # No excluded subjects
    ALL_SUBJECTS = [i for i in range(1, 10) if i not in []]  # 9 subjects

    # Available 22 EEG Channels from BCIC Dataset
    CHANNELS = [
        'c01', 'c02', 'c03', 'c04', 'c05', 'c06', 'c07',
        'c08', 'c09', 'c10', 'c11', 'c12', 'c13', 'c14',
        'c15', 'c16', 'c17', 'c18', 'c19', 'c20', 'c21',
        'c22']
    cv_folds = 9
    # Time Interval per EEG Trial (T=2: start of MI Cue)
    CONFIG: EEGConfig = EEGConfig(
        TMIN=0.0,
        TMAX=2.0,
        SAMPLERATE=250,
        CUE_OFFSET=2.0
    )
