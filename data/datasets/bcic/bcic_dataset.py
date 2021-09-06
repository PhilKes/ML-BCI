"""
File containing all relevant Properties of the
BCI competition IV 2a dataset
http://www.bbci.de/competition/iv/desc_2a.pdf

History:
  2021-05-10: Getting started - ms
"""
from config import EEGConfig
from data.datasets.DSConstants import DSConstants


# Default Constants of the BCIC Dataset
# DO NOT MODIFY FROM ANYWHERE ELSE IN THE CODE
class BCICConstants(DSConstants):

    def __init__(self):
        super().__init__()

        self.name = 'BCI competition IV 2A MI dataset'
        self.short_name = 'BCIC'

        # No excluded subjects
        self.ALL_SUBJECTS = [i for i in range(1, 10) if i not in []]  # 9 subjects

        # Available 22 EEG Channels from BCIC Dataset
        self.CHANNELS = [
            'c01', 'c02', 'c03', 'c04', 'c05', 'c06', 'c07',
            'c08', 'c09', 'c10', 'c11', 'c12', 'c13', 'c14',
            'c15', 'c16', 'c17', 'c18', 'c19', 'c20', 'c21',
            'c22']
        self.cv_folds = 9
        # Default Time Interval per EEG Trial (T=2: start of MI Cue because Trial start with 2 Seconds Rest)
        # These are Constants, DO NOT MODIFY!
        self.CONFIG: EEGConfig = EEGConfig(
            TMIN=0.0,
            TMAX=2.0,
            SAMPLERATE=250,
            CUE_OFFSET=2.0
        )
        self.TRIAL_TMAX = 7.5 - self.CONFIG.CUE_OFFSET
        self.TRIAL_TMIN = -self.CONFIG.CUE_OFFSET
        # Time Intervals in each Trial when the Subject rests (in seconds)
        self.REST_PHASES = [(0.0, 2.0), (6.0, 7.5)]
        self.event_type_to_label = {
            769: 0,
            770: 1,
            771: 2,
            772: 3,
            783: -1
        }


BCIC = BCICConstants()
