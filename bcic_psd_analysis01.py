"""
File: bcic_psd_analysis01.py

Description:
  Calculate and plot mean power spectral density of class specific EEG signals
  given with BCIC data set.

  Optional filter parameters can be adjusted by parameters:
    FREQ_FILTER_HIGHPASS and FREQ_FILTER_LOWPASS which are both defined in config.py,
    butter_bandpass_filt(..., order=X) in File bcic_iv2a_dataset.py, where X denotes
    the order of the applied filter.

Author: Manfred Strahnen

History:
  2021-05-25: Getting started - ms (Manfred Strahnen
"""
import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy import signal
from config import eeg_config
from data.bcic_dataset import BCIC_CHANNELS
from data.data_utils import butter_bandpass_filt
from config import global_config, eeg_config
from data.bcic_iv2a_dataset   import BCIC_IV2a_dataset, plot_psds
from data.bcic_dataset import BCIC_CONFIG

"""
Subroutine: calc_psds()
  Loads BCI data set with the specified parameters and calculates the PSDs,
  which afterwards are stored in files.
"""
def calc_psds():
    # Setting of needed parameters
    subjects = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    n_classes = 4
    training = 1
    eeg_config.TMIN = BCIC_CONFIG.TMIN
    eeg_config.TMAX = BCIC_CONFIG.TMAX
    eeg_config.TRIAL_SLICES = 1
    eeg_config.SAMPLERATE = BCIC_CONFIG.SAMPLERATE
    eeg_config.SAMPLES = (int) ((BCIC_CONFIG.TMAX - BCIC_CONFIG.TMIN) * BCIC_CONFIG.SAMPLERATE)
    print("  - Subjects: ", subjects)
    print("  - n_classes: ", n_classes)

    print('  - Load BCIC data set')
    equal_trials=True
    ch_names = BCIC_CHANNELS

    ds_r = BCIC_IV2a_dataset(subjects=subjects, n_classes=n_classes, ch_names=ch_names)
    ds_r.load_subjects_data(training)
    ds_r.print_stats()

    print("  - Calculate PSDs")
    ds_r.calc_psds()
    print("  - PSDs are save in files!")


print("PSD analysis of BCIC data set started with following parameters:")
# calc_psds() only has to be called if one of the parameters defined therein has been changed !!!
#calc_psds()

print(' Read PSDs from files and plot them ')
plot_psds()
print("The End")