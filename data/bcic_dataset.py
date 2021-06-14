"""
File containing all relevant Properties of the
BCI competition IV 2a dataset

History:
  2021-05-10: Getting started - ms
"""

from mne.datasets import eegbci

from util.dot_dict import DotDict

BCIC_excluded_subjects = []                         # No excluded subjects
BCIC_ALL_SUBJECTS = [i for i in range(1, 10)]       # 9 subjects

# Available 22 EEG Channels from Physionet Dataset
BCIC_CHANNELS = [
    'c01', 'c02', 'c03', 'c04', 'c05', 'c06', 'c07',
    'c08', 'c09', 'c10', 'c11', 'c12', 'c13', 'c14',
    'c15', 'c16', 'c17', 'c18', 'c19', 'c20', 'c21',
    'c22']

# Time Interval per EEG Trial (T=0: start of MI Cue)
BCIC_CONFIG = DotDict(
    TMIN=4.0,
    TMAX=6.0,
    SAMPLERATE=250,
)