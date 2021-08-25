"""
File containing all relevant Properties of the
BCI competition IV 2a dataset

History:
  2021-05-10: Getting started - ms
"""
from config import EEGConfig
from data.datasets.DSConstants import DSConstants
from util.dot_dict import DotDict


# Default Constants of the OpenBCI Dataset
# DO NOT MODIFY FROM ANYWHERE ELSE IN THE CODE
class OpenBCIConstants(DSConstants):

    def __init__(self):
        super().__init__()
        self.name = 'OpenBCI Dataset'
        self.short_name = 'OBCI'

        self.trials_per_subject = 50

        # No excluded subjects
        self.ALL_SUBJECTS = [i for i in range(1, 3) if i not in []]

        # Available 16 EEG Channels from OpenBCI Dataset
        self.CHANNELS = ['Fp1', 'Fp2', 'C3', 'C4', 'P7', 'P8', 'O1', 'O2', 'F7', 'F8', 'F3', 'F4', 'T7', 'T8', 'P3',
                         'P4']

        self.cv_folds = 2
        # Time Interval per EEG Trial (T=2: start of MI Cue)
        self.CONFIG: EEGConfig = EEGConfig(
            TMIN=0.0,
            TMAX=4.0,
            SAMPLERATE=125,
            CUE_OFFSET=2
        )


OpenBCI = OpenBCIConstants()
