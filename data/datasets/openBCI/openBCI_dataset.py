"""
File containing all relevant Properties of the
BCI competition IV 2a dataset

History:
  2021-05-10: Getting started - ms
"""

from util.dot_dict import DotDict


class OpenBCI:
    name = 'OpenBCI Dataset'
    short_name = 'openBCI'

    # No excluded subjects
    ALL_SUBJECTS = [1]#[i for i in range(1, 10) if i not in []]  # 9 subjects

    # Available 22 EEG Channels from BCIC Dataset
    # TODO Channelnamen nach 10-10 system
    CHANNELS = [
        'fp1', 'fp2', 'c3', 'c4', 'p7', 'p8', 'o1',
        'o2', 'f7', 'f8', 'f3', 'f4', 't7', 't8',
        'p3', 'p4']

    # Todo filter
    cv_folds = 2
    # Time Interval per EEG Trial (T=2: start of MI Cue)
    CONFIG = DotDict(
        TMIN=0.0,
        TMAX=2.0,
        SAMPLERATE=125,
        CUE_OFFSET=0
    )
