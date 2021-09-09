"""
File: bcic_psd_analysis.py

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
from config import CONFIG
from data.datasets.bcic.bcic_data_loading import BCICDataLoader
from data.datasets.bcic.bcic_dataset import BCIC
from data.datasets.bcic.bcic_iv2a_dataset import BCIC_IV2a_dataset, plot_psds
from paths import results_folder

"""
Subroutine: calc_psds()
  Loads BCI data set with the specified parameters and calculates the PSDs,
  which afterwards are stored in files.
"""


def calc_psds():
    # Setting of needed parameters
    subjects = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    n_class = 4
    training = 1
    CONFIG.EEG.TMIN = BCIC.CONFIG.TMIN
    CONFIG.EEG.TMAX = BCIC.CONFIG.TMAX
    CONFIG.EEG.TRIALS_SLICES = 1
    CONFIG.EEG.SAMPLERATE = BCIC.CONFIG.SAMPLERATE
    CONFIG.EEG.SAMPLES = (int)((BCIC.CONFIG.TMAX - BCIC.CONFIG.TMIN) * BCIC.CONFIG.SAMPLERATE)
    print("  - Subjects: ", subjects)
    print("  - n_class: ", n_class)

    print('  - Load BCIC data set')
    equal_trials = True
    ch_names = BCIC.CHANNELS
    # TODO refactor
    data, labels = BCICDataLoader.load_subjects_data(subjects=subjects, n_class=n_class, ch_names=ch_names)
    BCICDataLoader.print_stats(labels)

    print("  - Calculate PSDs")
    BCIC_IV2a_dataset.calc_psds(n_class, subjects, data, labels, path=f"{results_folder}/psds_BCIC")
    print("  - PSDs are save in files!")


########################################################################################
if __name__ == '__main__':
    print("PSD analysis of BCIC data set started with following parameters:")
    # calc_psds() only has to be called if one of the parameters defined therein has been changed !!!
    calc_psds()

    print(' Read PSDs from files and plot them ')
    plot_psds()
    print("The End")
