"""
File: EEG_dataset_analysis03.py

  This file contains methods to analyze an EEG data set. Up to now
  following data sets are supported:

     Short Name | Data set
     -----------|-----------------------------
      PHYS      | Physionet MI dataset
      BCIC      | BCI competition IVa dataset
      LSMR21    | Large SMR data set

  Method main() optionally reads the specified data set and implements a kind of
  jump table to the subroutines which have implemented the selected data
  analysis method. The data analysis method is specified by variable
  'sub_command' which has to be initialized with the corresponding value at
  the beginning of main().
  Following sub_commands are supported:

    Sub command  | Short description
    -------------|-------------------------------------------------------------------------
     'TF_map'    | Calc. time freq. psd map using 'pure FFT' or 'Multitaper' method
                 | and stores it in a file
     'TF_plot'   | Reads a time freq. psd map from a file and plots it
     'Fb_power'  | Calc. and plot freq. band specific power spectral densities
     'PSD01'     | Calculate and plot power spectral density
     'MT_PSD01'  | Calc. and plot power spectral density using mne mulittaper method
     'PSDs_plot' | Plot PSDs stored in a previously measured time-frequency map
     'AMPHASE'   | Calculate and plot amplitude and phase spectra

  Starting point of any data analysis should always be the calculation
  of the 'time frequency psd map' using 'TF_map' and its inspection
  using sub_command 'TF_plot'. Afterwards the identified regions of
  interest could further be investigated using 'Fb_power' or 'PSDs_plot'.

History:
  2021-03-31: Getting started - ms (Manfred Strahnen)
  2021-08-04: Version 1.0 release - ms
  2021-08-05: AMPHASE sub_command implemented - ms
  2021-08-07: Adaptation to new ML-BCI version - ms
  2021-08-08: Start of Numpy-array to MNE-epochs conversion in order to
              use mne data analysis functions - ms
  2021-08-13: Debugging and multitaper psd implementation - ms
  2021-08-17: Some more comments
"""

import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy import signal

from config import CONFIG
from data.datasets.datasets import DATASETS
##################################################################################################
from paths import results_folder
from util.misc import makedir


def calc_amphase(ds_data, ds_tmin, ds_tmax, sampling_rate, ts_tmin, ts_samples):
    """
      Calculate mean amplitude and phase of all subjects, all trials/subject and
      channels/trial.
      HINT: I know that it does not really makes sense to calculate the mean phase
            for events which are NOT synchronized. So, please only use the
            calculated mean amplitude.

      Input parameters:
      - ds_data: Data set: 4-dim. numpy array with indices(subject, trial, channel, samples)
                 EEG amplitudes are expected to be given in Volts
      - ds_tmin: Trial slice start time of passed data set
      - ds_tmax: Trial slice stop  time of passed data set
      - sampling_rate: Sample rate with which data has been acquired
      - ts_tmin: Trial slice start time for which psd should be calculated
      - ts_samples: Number of samples of current trial slice
    """
    # Get basic parameters
    num_channels = ds_data.shape[2]
    num_trials = ds_data.shape[1]
    num_subjects = ds_data.shape[0]

    # Check boundaries
    if ts_tmin < ds_tmin or (ts_tmin + ts_samples / sampling_rate) > ds_tmax:
        print("  - Error: trial slice out of data set specific range")

    # Calc. array with time points of all samples
    t = np.zeros(num_samples)
    dt = 1 / sampling_rate
    for i in range(num_samples):
        t[i] = ds_tmin + i * dt

    # Get indexes of trial slice
    res = np.where(t >= ts_tmin)
    ts_tmin_ind = res[0][0]
    ts_tmax_ind = ts_tmin_ind + ts_samples - 1

    # Calculate power spectral density psd
    ts_samples = ts_tmax_ind - ts_tmin_ind + 1
    amp_fft_all = np.zeros(int(ts_samples / 2) + 1)
    phase_fft_all = np.zeros(int(ts_samples / 2) + 1)

    # Calculate the mean psd
    for subject in range(num_subjects):
        # sum up the psd for all trials of a subjects
        for trial in range(num_trials):
            # calculate the psd for each EEG channel and sum up all psd
            for ch in range(num_channels):
                data = ds_data[subject, trial, ch, ts_tmin_ind:ts_tmax_ind + 1]

                fourier_transform = np.fft.rfft(data)
                amp_fft = np.abs(fourier_transform)
                phase_fft = np.angle(fourier_transform)

                amp_fft_all = amp_fft_all + amp_fft
                phase_fft_all = phase_fft_all + phase_fft

    f = np.fft.rfftfreq(ts_samples, d=1. / sampling_rate)
    amp_fft_all = amp_fft_all / (num_channels * num_trials * num_subjects)
    phase_fft_all = phase_fft_all / (num_channels * num_trials * num_subjects)

    return amp_fft_all, phase_fft_all, f


##################################################################################################
def plot_amphase(amp_fft_all, phase_fft_all, ds_name, f, fmin, fmax, ts_tmin, ts_tsize):
    """
      Plot mean amplitude and phase
      Input parameters:
      - psd: 1-dim numpy array with power spectral density given in V**2/Hz
      - f:   1-dim numpy array for which power spectral density is given in psd
      - fmin: Minimum psd-frequency contained in result plot
      - fmax: Maximum psd-frequency contained in result plot
      - ts_tmin:  Trial slice offset in seconds
      - ts_tsize: Trial slice size in seconds
    """

    # Get indices of psd in range plot_fmin -> plot_fmax
    res = np.where(f >= fmin)
    fmin_ind = res[0][0]

    res = np.where(f >= fmax)
    fmax_ind = res[0][0] - 1

    plt.plot(f[fmin_ind:fmax_ind + 1], amp_fft_all[fmin_ind:fmax_ind + 1])
    legend = []
    legend.append('All trials mean')
    plt.legend(legend)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude')
    plt.title("'%s' Amplitude spectrum for: ts_tmin = %.1f s, ts_tsize = %.1f s" % (ds_name, ts_tmin, ts_tsize))
    plt.grid(True)
    plt.show()

    plt.plot(f[fmin_ind:fmax_ind + 1], phase_fft_all[fmin_ind:fmax_ind + 1])
    legend = []
    legend.append('All trials mean')
    plt.legend(legend)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Phase')
    plt.title("'%s' Phase spectrum for: ts_tmin = %.1f s, ts_tsize = %.1f s" % (ds_name, ts_tmin, ts_tsize))
    plt.grid(True)
    plt.show()
    return


##################################################################################################
def calc_psd(ds_data, ds_tmin, ds_tmax, sampling_rate, ts_tmin, ts_samples):
    """
      Calculate mean psd of all subjects, all trials/subject and channels/trial.
      Input parameters:
      - ds_data: Data set: 4-dim. numpy array with indices(subject, trial, channel, samples)
                 EEG amplitudes are expected to be given in Volts
      - ds_tmin: Trial slice start time of passed data set
      - ds_tmax: Trial slice stop  time of passed data set
      - sampling_rate: Sample rate with which data has been acquired
      - ts_tmin: Trial slice start time for which psd should be calculated
      - ts_samples: Number of samples of current trial slice
    """
    # Get basic parameters
    num_channels = ds_data.shape[2]
    num_trials = ds_data.shape[1]
    num_subjects = ds_data.shape[0]

    # Check boundaries
    if ts_tmin < ds_tmin or (ts_tmin + ts_samples / sampling_rate) > ds_tmax:
        print("  - Error: trial slice out of data set specific range")

    # Calc. array with time points of all samples
    t = np.zeros(num_samples)
    dt = 1 / sampling_rate
    for i in range(num_samples):
        t[i] = ds_tmin + i * dt

    # Get indexes of trial slice
    res = np.where(t >= ts_tmin)
    ts_tmin_ind = res[0][0]
    ts_tmax_ind = ts_tmin_ind + ts_samples - 1

    # Calculate power spectral density psd
    ts_samples = ts_tmax_ind - ts_tmin_ind + 1
    psd_all = np.zeros(int(ts_samples / 2) + 1)

    # Calculate the mean psd
    for subject in range(num_subjects):
        # sum up the psd for all trials of a subjects
        for trial in range(num_trials):
            # calculate the psd for each EEG channel and sum up all psd
            for ch in range(num_channels):
                data = ds_data[subject, trial, ch, ts_tmin_ind:ts_tmax_ind + 1]
                f, psd = signal.periodogram((data), sampling_rate)  # this method uses

                psd_all = psd_all + psd

    psd_all = psd_all / (num_channels * num_trials * num_subjects)
    return psd_all, f


##################################################################################################
def plot_psd(psd, ds_name, f, fmin, fmax, ts_tmin, ts_tsize):
    """
      Plot mean power spectral density psd
      Input parameters:
      - psd: 1-dim numpy array with power spectral density given in V**2/Hz
      - f:   1-dim numpy array for which power spectral density is given in psd
      - fmin: Minimum psd-frequency contained in result plot
      - fmax: Maximum psd-frequency contained in result plot
      - ts_tmin:  Trial slice offset in seconds
      - ts_tsize: Trial slice size in seconds
    """
    psd = np.log10(psd)

    # Get indices of psd in range plot_fmin -> plot_fmax
    res = np.where(f >= fmin)
    fmin_ind = res[0][0]

    res = np.where(f >= fmax)
    fmax_ind = res[0][0] - 1

    plt.plot(f[fmin_ind:fmax_ind + 1], psd[fmin_ind:fmax_ind + 1])
    legend = []
    legend.append('All trials mean')
    plt.legend(legend)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Log 10 of PSD in V**2/Hz')
    plt.title("'%s' Log10 PSD for: ts_tmin = %.1f s, ts_tsize = %.1f s" % (ds_name, ts_tmin, ts_tsize))
    plt.grid(True)
    plt.show()
    return


##################################################################################################
def PSDs_plot(fname, fmin, fmax, ts_offsets):
    """
    One or more PSDs which are part of the time-frequency map stored in file with name
    'fname' are plotted in a single figure. The time slice offset of each PSD which
    should be plotted is given in the list ts_offsets.
    fmin and fmax specify the frequency range of interest.
    :param fname:
    :param fmin:
    :param fmax:
    :param ts_offsets:
    :return:
    """
    print("    - Load time-freq. PSD map from file:", fname)
    with np.load(fname, allow_pickle=True) as data:
        ds_name = data['ds_name']  # name of the data set
        psd_method = data['psd_method']  # method used to calc. psd
        ts_size = data['ts_size']  # trial slice size used during measurement
        rest_ts_tmin_load = data['rest_ts_tmin']  # position of the 'rest state trial slice'
        f = data['f']
        toff = data['toff']
        psdMap = data['tf_psdMap']

    print("    - Data set:", ds_name)
    print("    - PSD method:", psd_method)
    print("    - Data set shape:", psdMap.shape)
    print("    - Trial time slice (in seconds):", ts_size)
    toff_min = toff[0]
    toff_max = toff[toff.shape[0] - 1]
    print("    - Trial slice offset range:", toff_min, '->', toff_max)

    legend = []
    for ts_offset in ts_offsets:
        print("    - Time slice offset:", ts_offset)

        # get index corresponding array index
        res = np.where(toff >= ts_offset)
        ts_offset_ind = res[0][0]

        # get PSD out of psdMap
        psd = psdMap[ts_offset_ind, :]
        psd = np.log10(psd)

        # Get indices of psd in range plot_fmin -> plot_fmax
        res = np.where(f >= fmin)
        fmin_ind = res[0][0]

        res = np.where(f >= fmax)
        fmax_ind = res[0][0] - 1

        leg = 'ts_offset=%2.1f s' % (ts_offset)
        legend.append(leg)
        plt.plot(f[fmin_ind:fmax_ind + 1], psd[fmin_ind:fmax_ind + 1])

    plt.legend(legend, prop={'size': 8})
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Log 10 of PSD in V**2/Hz')
    plt.title(
        "'%s' '%s' Log10 PSD for: ts_offset = %.1f s, ts_size = %.1f s" % (ds_name, psd_method, ts_offset, ts_size))
    plt.grid(True)
    plt.show()

    return


###############################################################################################
def calc_time_freq_FFT_PSDmap(preloaded_data, preloaded_labels, ds_name, ds_tmin, ds_tmax,
                              sampling_rate, ts_offsets, rest_ts_tmin, ts_size):
    """
      Calculates time frequency resolved mean psd (power spectral density) map for the
      data set given in 4-dim array 'preloaded_data(subject, trial, eeg_channel, sample)'.
      Samples are in the range from 'ds_tmin' to 'ds_tmax', which both are specified as times
      relative to the cue-signal. Data set has been acquired with sample rate 'sampling_rate'.
      Psd is calculated for each trial size offset defined in list 'ts_offsets'.
      Calculated psd is the mean value over all EEG channels of all trials and all subjects.
      Pure FFT method is used to calculate the PSD.

      Results are stored in a file for later post-processing. Following data is stored:
        - ds_name:      Name of the analyzed data set
        - psd_method:   Specifies the method used for psd calculation
        - ts_size:      Trial slice size in seconds for which psd has been calculated
        - rest_ts_tmin: Offset - relative to cue-signal - of the rest trial slice which
                        has been used to measured rest_power. Trial slice size is the
                        same as 'ts_size'
                        (Not needed anymore!)
        - f:            1-dim array containing the frequency values for which power
                        spectral density has been calculated.
        - toff:         1-dim array with the time slice offset for which the psd
                        has been calculated.
        - tf_psdMap:    2-dim array containing the mean psd for each frequency value
                        defined in f for each time slice defined in toff. First array
                        index is the offset, second index is frequency f.
    """
    print("    - Calculate time frequency PSD map for dataset: ", mi_ds)

    tf_psdMap = []  # two dimensional list which stores the results
    # First dim. is the time slice offset, second dim. is the frequency

    # Calculate psd for each trial time slice
    for offset in ts_offsets:
        ts_tmin = offset
        ts_samples = int(ts_size * sampling_rate)
        print("    - Trial slice with ts_tmin = %f, num_of_samples = %d" % (ts_tmin, ts_samples))
        psd_all, f = calc_psd(preloaded_data, ds_tmin, ds_tmax, sampling_rate, ts_tmin, ts_samples)
        tf_psdMap.append(psd_all)
    tf_psdMap = np.array(tf_psdMap)  # Conversion to numpy array
    print("tf_psdMap shape:", tf_psdMap.shape)

    # Save calculated data in a file for later post-processing
    fname = f'{results_folder}/EEG_data_analysis/'
    makedir(fname)
    dt = datetime.now()  # take current date/time
    fname = fname + ds_name + '_PUFFTpsdMap'
    fname = fname + '_d' + '%04d' % (dt.year)
    fname = fname + '%02d' % (dt.month)
    fname = fname + '%02d' % (dt.day)

    # File name gets an index. If a file with that index already exists
    # another index will be chosen
    findex = 0
    fname1 = fname + '-' + '%02d' % (findex)
    fname1 = fname1 + '.npz'
    while os.path.isfile(fname1):
        print("file " + fname1 + " already exists")
        findex = findex + 1
        fname1 = fname + '-' + '%02d' % (findex)
        fname1 = fname1 + '.npz'

    np.savez(fname1,
             ds_name=ds_name,
             psd_method='PUFFT',  # pure FFT method used to calculate the PSD
             ts_size=ts_size,
             rest_ts_tmin=rest_ts_tmin,
             f=np.array(f),
             toff=ts_offsets,
             tf_psdMap=tf_psdMap)
    print("  - Result saved in file", fname1)
    return


#############################################################################################
def calc_time_freq_MT_PSDmap(preloaded_data, preloaded_labels, ds_name, ds_tmin, ds_tmax,
                             sampling_rate, rest_ts_tmin, ts_size):
    """
      Calculates time frequency resolved mean psd (power spectral density) map for the
      data set given in 4-dim array 'preloaded_data(subject, trial, eeg_channel, sample)'.
      Samples are in the range from 'ds_tmin' to 'ds_tmax', which both are specified as times
      relative to the cue-signal. Data set has been acquired with sample rate 'sampling_rate'.
      Psd is calculated for equally spaced time points reaching from ds_tmin to ds_tmax.
      Calculated psd is the mean value over all EEG channels of all trials and all subjects.

      psd calculation is done using MNE's multitaper method. Results are stored in a file for later
      post-processing.

      Following data is stored in the file:
        - ds_name:      Name of the analyzed data set
        - psd_method:   Specifies the method used for psd calculation
        - ts_size:      Trial slice size in seconds for which psd has been calculated
                        (Not a valid parameter for multitaper method!)
        - rest_ts_tmin: Offset - relative to cue-signal - of the rest trial slice which
                        has been used to measured rest_power. Trial slice size is the
                        same as 'ts_size'
                        (Not needed anymore!)
        - f:            1-dim array containing the frequency values for which power
                        spectral density has been calculated.
        - toff:         1-dim array with the time slice offset for which the psd
                        has been calculated.
        - tf_psdMap:    2-dim array containing the mean psd for each frequency value
                        defined in f for each time slice defined in toff. First array
                        index is the offset, second index is frequency f.
    """

    print("    - Calculate time frequency PSD map for dataset: ", mi_ds)

    # change data set array dimensions so that it fits to MNE conventions. Dimensions
    # subjects and trials are here combined to one single dimension
    n_channels = preloaded_data.shape[2]
    n_samples = preloaded_data.shape[3]
    ds = np.reshape(preloaded_data, (-1, n_channels, n_samples))

    freqs = np.arange(1, sampling_rate / 2, 1)  # frequencies from 0 to Nyquist freq. in steps of 1 Hz
    n_cycles = freqs  # use constant t/f resolution

    decim = 6  # decimation factor of time axis used by tfr_array_multitaper
    tf_psdMap = mne.time_frequency.tfr_array_multitaper(ds, sfreq=sampling_rate, freqs=freqs, n_cycles=n_cycles,
                                                        use_fft=True, decim=decim, output='avg_power', n_jobs=8)
    tf_psdMap = np.mean(tf_psdMap, axis=0)

    # discrete time axis values
    mne_ts_offsets = np.arange(ds_tmin, ds_tmax, float(decim) / sampling_rate)
    # print("mne_ts_offsets =", mne_ts_offsets)

    # Save calculated data in a file for later post-processing
    fname = f'{results_folder}/EEG_data_analysis/'
    makedir(fname)
    dt = datetime.now()  # take current date/time
    fname = fname + ds_name + '_MUTAPpsdMap'
    fname = fname + '_d' + '%04d' % (dt.year)
    fname = fname + '%02d' % (dt.month)
    fname = fname + '%02d' % (dt.day)

    # File name gets an index. If a file with that index already exists
    # another index will be chosen
    findex = 0
    fname1 = fname + '-' + '%02d' % (findex)
    fname1 = fname1 + '.npz'
    while os.path.isfile(fname1):
        print("file " + fname1 + " already exists")
        findex = findex + 1
        fname1 = fname + '-' + '%02d' % (findex)
        fname1 = fname1 + '.npz'

    np.savez(fname1,
             ds_name=ds_name,
             psd_method='MUTAP',  # multitaper method used to calculate the PSD
             ts_size=ts_size,
             rest_ts_tmin=rest_ts_tmin,
             f=freqs,
             toff=mne_ts_offsets,
             tf_psdMap=np.transpose(tf_psdMap))
    print("  - Result saved in file", fname1)
    return


########################################################################################
def plot_time_freq_PSDmap(fname, rest_ts_tmin, REST_NORM='true', LOG10='true',
                          tmin=-1.5, tmax=3.9, fmin=2.0, fmax=78.0,
                          vmin=-0.2, vmax=0.4):
    """
      Reads in previously calculated power spectral densities from a file with
      name 'fname' and plots a 2-dim picture where the x-axis is given by
      the trial offsets defined in array 'toff' and the y-axis is given by the
      frequency values defined in array 'f'.The frequency range which should be plotted
      can be specified by 'fmin' and 'fmax', the trial slice offset ranges from 'tmin'
      to 'tmax'. Pixel color encodes the mean power spectral density at the given
      frequency and trial offset values. 'toff' and 'f' are also read from
      file 'fname'.

      Optionally power spectral density can be plotted relative to the corresponding
      density in REST state. In this case power
      power spectral density values are divided by the corresponding values in REST
      state. This kind of 'REST state normalization' can be activated by setting
      variable 'REST_NORM' to 'true'. Rest state can be
      specified by parameter 'rest_ts_tmin', which is the time slice offset of the
      trial which should be used as 'Rest state trial slice'. If rest_ts_tmin is
      specified as 'None' the value loaded from the file is taken.

      Instead of using the power spectral density
      the log10 value can optionally be displayed if LOG10 is 'true'.
    """
    # Read psd data from file
    print("    - Load psd map related data from file: ", fname)
    with np.load(fname, allow_pickle=True) as data:
        ds_name = data['ds_name']  # name of the data set
        psd_method = data['psd_method']
        ts_size = data['ts_size']  # trial slice size used during measurement
        rest_ts_tmin_load = data['rest_ts_tmin']  # position of the 'rest state trial slice'
        f = data['f']
        toff = data['toff']
        psd = data['tf_psdMap']

    if rest_ts_tmin == 'None':
        rest_ts_tmin = rest_ts_tmin_load

    print("    - Data set:", ds_name)
    print("    - PSD method:", psd_method)
    print("    - Data set shape:", psd.shape)
    print("    - Trial time slice (in seconds):", ts_size)
    print("    - Rest trial offset (in seconds):", rest_ts_tmin)
    toff_min = toff[0]
    toff_max = toff[toff.shape[0] - 1]
    print("    - Trial slice offset range:", toff_min, '->', toff_max)

    print("    - f.shape =", f.shape)
    print("    - toff.shape =", toff.shape)

    # Check boundaries
    if tmin < toff_min or tmax > toff_max:
        print("Error: tmin or tmax are out of range!", tmin, tmax)
        sys.exit()
    if fmin < f[0] or fmax > f[f.shape[0] - 1]:
        print("Error: fmin or fmax are out of range!")
        sys.exit()

    # Optionally do the 'REST state normalization'
    if REST_NORM == 'true':
        res = np.where(toff >= rest_ts_tmin)
        rest_ts_tmin_ind = res[0][0]
        print("    - rest_ts_tmin_ind = ", rest_ts_tmin_ind)
        psd_rest = np.copy(psd[rest_ts_tmin_ind, :])

        for i in range(toff.shape[0]):
            psd[i, :] = psd[i, :] / psd_rest

    # Optionally take LOG10 of power density values
    if LOG10 == 'true':
        psd = np.log10(psd)

    # Extract part of psd which should be plotted
    res = np.where(toff >= tmin)
    tmin_ind = res[0][0]
    res = np.where(toff >= tmax)
    tmax_ind = res[0][0]

    res = np.where(f >= fmin)
    fmin_ind = res[0][0]
    res = np.where(f >= fmax)
    fmax_ind = res[0][0]

    psd_plot = psd[tmin_ind:tmax_ind, fmin_ind:fmax_ind]
    print("psd_plot.shape =", psd_plot.shape)

    # Plot frequency specific energies over time slice offset
    plt.rc('font', size=8)  # default font size
    im = plt.pcolormesh(toff[tmin_ind:tmax_ind],
                        f[fmin_ind:fmax_ind],
                        np.transpose(psd[tmin_ind:tmax_ind, fmin_ind:fmax_ind]),
                        #                        vmin=vmin, vmax=vmax,
                        shading='gouraud',
                        cmap='nipy_spectral')
    #                        cmap='turbo')
    #                       cmap = 'plasma')

    plt.colorbar(im)

    # Construction of figures title
    title = np.array2string(ds_name)
    title = title + ' '
    title = title + np.array2string(psd_method) + ' '
    if LOG10 == 'true':
        title = title + 'LOG10 '
    title = title + 'PSD time-freq. map'
    if REST_NORM == 'true':
        title = title + '(with REST state normalization)'
    plt.title(title)

    plt.xlabel('Time slice offset [s]', fontsize=10)
    plt.ylabel("Frequency [Hz]")
    plt.show()
    return


##########################################################################################
def calc_plot_Fb_psd(fname, rest_ts_tmin, f_bands, tmin=-1.5, tmax=3.9,
                     REST_NORM='true', LOG10='true', SUBPLOTS='true'):
    """
      Reads in previously measure time-frequency psd map from the file with name
      'fname' and plots the mean power spectral density for each frequency band defined
      in list 'f_bands' and all trial offsets for which time-frequency psd map has been
      measured.   X-axis of the plot is the trial offset ranging from 'tmin' to 'tmax'
      and the y-axis shows the calculated mean power spectral density.

      Optionally band specific psd can be plotted relative to the corresponding
      psd value in 'REST' state. In this case the psd in ACTIVE state is divided
      by the corresponding psd in REST state before plotting. This kind of 'REST state
      normalization' can be activated by setting variable 'REST_NORM' to 'true'.
      Parameter 'rest_ts_tmin' specifies the time slice offset of the Rest state.
      With SUBPLOTS='true' band power of each frequency band is plotted in its own
      subplot. Otherwise all graphs are plotted in one plot.
    """
    # Read psd data from file
    print("    - Load psd map related data from file: ", fname)
    with np.load(fname, allow_pickle=True) as data:
        ds_name = data['ds_name']  # name of the data set
        psd_method = data['psd_method']  # method used to calc. psd
        ts_size = data['ts_size']  # trial slice size used during measurement
        rest_ts_tmin_load = data['rest_ts_tmin']  # position of the 'rest state trial slice'
        f = data['f']
        toff = data['toff']
        psd = data['tf_psdMap']

    if rest_ts_tmin == 'None':
        rest_ts_tmin = rest_ts_tmin_load

    print("    - Data set:", ds_name)
    print("    - Trial time slice size (in seconds):", ts_size)
    print("    - Rest trial offset (in seconds):", rest_ts_tmin)
    toff_min = toff[0]
    toff_max = toff[toff.shape[0] - 1]
    print("    - Min./max. trial slice offset range:", toff_min, '->', toff_max)

    # Define the numpy array containing the result
    Fb_psd = np.zeros([toff.shape[0], f_bands.shape[0]])
    Fb_psd_rest = np.zeros(f_bands.shape[0])

    # get index of rest_trial in psd
    res = np.where(toff >= rest_ts_tmin)
    rest_ts_tmin_ind = res[0][0]

    # calculate band-specific power value by adding all psd [toffset, f] for which
    # f in within that band. Additionally calculated mean power in each band.
    i = 0  # i used as frequency band index
    for bands in f_bands:
        fmin = bands[0]
        fmax = bands[1]
        print("    - F-band: %f -> %f Hz" % (fmin, fmax))
        k = 0
        for freq in f:
            if freq >= fmin and freq < fmax:
                res = np.where(f >= freq)
                j = res[0][0]
                Fb_psd[:, i] += psd[:, j]
                Fb_psd_rest[i] = Fb_psd_rest[i] + psd[rest_ts_tmin_ind, j]
                k = k + 1
        Fb_psd[:, i] /= k
        Fb_psd_rest[i] /= k

        # Optionally do the 'REST state normalization'
        if REST_NORM == 'true':
            Fb_psd[:, i] = Fb_psd[:, i] / Fb_psd_rest[i]
        i = i + 1

    # Extract desired offset range
    if tmin < toff[0] or tmax > toff[toff.shape[0] - 1]:
        print("Error: Illegal tmin or tmax values (%f -> %f s)" % (tmin, tmax))
        sys.exit()

    # Extract part of psd which should be plotted
    res = np.where(toff >= tmin)
    tmin_ind = res[0][0]
    res = np.where(toff >= tmax)
    tmax_ind = res[0][0]

    # Create arrays to store plot data
    toff_plot = np.zeros((tmax_ind - tmin_ind + 1))

    # Extract x-axis array with time slice offsets
    j = 0
    for i in range(toff.shape[0]):
        if tmin_ind <= i <= tmax_ind:
            toff_plot[j] = toff[i]
            j = j + 1

    # Extract array with plot data
    bt_power_plot = Fb_psd[tmin_ind:tmax_ind + 1, :]
    bt_power_plot = bt_power_plot.transpose()

    # Optionally take LOG10 of power density values
    if LOG10 == 'true':
        bt_power_plot = np.log10(bt_power_plot)

    title = np.array2string(ds_name)
    title = title + ' '
    title = title + np.array2string(psd_method) + ' '
    if LOG10 == 'true':
        title = title + 'LOG10 '
    title = title + 'band specific power'
    if REST_NORM == 'true':
        title = title + '(with REST state normalization)'

    if SUBPLOTS == 'true':
        # Plot frequency specific energies over time slice offset
        plt.rc('font', size=8)  # default font size
        cm = 1 / 2.54  # centimeters in inches
        xsize = 12.0 * cm
        ysize = len(f_bands) * 3.0 * cm
        fig, axs = plt.subplots(f_bands.shape[0], sharex='all', figsize=(xsize, ysize))

        fig.suptitle(title)

        for i in range(f_bands.shape[0]):
            leg = '%2.1f' % (f_bands[i, 0]) + '-' + '%2.1f' % (f_bands[i, 1]) + 'Hz band'
            axs[i].plot(toff_plot, bt_power_plot[i, :], label=leg)
            axs[i].legend(prop={'size': 8})
            axs[i].grid(True)

        plt.xlabel('Time slice offset [s]', fontsize=10)
        if REST_NORM == 'true':
            plt.ylabel("Rel. band power")
        else:
            plt.ylabel('Abs. band power [V**2]', fontsize=10)
    else:
        # Plot frequency specific energies over time slice offset
        plt.rc('font', size=8)  # default font size
        plt.title(title)

        legend = []
        for i in range(f_bands.shape[0]):
            leg = '%2.1f' % (f_bands[i, 0]) + '-' + '%2.1f' % (f_bands[i, 1]) + 'Hz band'
            legend.append(leg)
            plt.plot(toff_plot, bt_power_plot[i, :], label=leg)

        plt.legend(legend, prop={'size': 8})

        plt.xlabel('Time slice offset [s]', fontsize=10)
        if REST_NORM == 'true':
            plt.ylabel("Rel. band power")
        else:
            plt.ylabel('Abs. band power [V**2]', fontsize=10)

        plt.grid(True)

    plt.show()
    return


######################################################################################
def npArray_to_mneEpochs(preloaded_data, sampling_rate, ch_names, description, tmin):
    """
    Convert a numpy array containing EEG data of form: subject, trial, channel, times
    to a mne epochs object of form: n_epochs, n_channels, n_times

    :param preloaded_data:
    :return:
    """
    print("    - Create MNE epochs object")

    # Create the accompanying info object containing important meta data
    info = mne.create_info(ch_names, sfreq=sampling_rate)
    info['description'] = description
    info['bads'] = []  # Names of bad channels
    # print(info)

    # change data set array dimensions so that it fits to MNE concentions
    # print("preloaded_data.shape =", preloaded_data.shape)
    n_channels = preloaded_data.shape[2]
    n_samples = preloaded_data.shape[3]
    ds = np.reshape(preloaded_data, (-1, n_channels, n_samples))
    # print("ds.shape =", ds.shape)

    return mne.EpochsArray(ds, info, tmin=tmin)


##################################################################################################
def plot_mne_psd(preloaded_data, sampling_rate, ch_names, description, ds_tmin, ds_tmax, tmin, tmax):
    """
    Calculate and plot psd using mne's multitaper method

    :param preloaded_data:
    :return:
    """
    print("  - Calculate and plot psd using MNE's multitaper method")

    if tmin < ds_tmin or tmax > ds_tmax:
        print("ERROR: tmin or tmax are out of legal range")
        sys.exit()

    print("    - Start time to consider: %2.2f s" % tmin)
    print("    - End   time to consider: %2.2f s" % tmax)

    ds_epochs = npArray_to_mneEpochs(preloaded_data, CONFIG.EEG.SAMPLERATE, dataset.channels, dataset.name, ds_tmin)

    # calc and plot mean psd
    picks = 'misc'  # picks='misc' for all channels
    #    ds_epochs.plot_psd(picks=['C3', 'C4'], fmin=1, fmax=78, average=True, spatial_colors=False)
    fig = ds_epochs.plot_psd(picks=picks, fmin=1, fmax=78, tmin=tmin, tmax=tmax, average=True, spatial_colors=False)
    plt.show(block=True)
    plt.savefig(fname=f"{results_folder}/EEG_data_analysis/fig.png")

    return


###################################################################################################
if __name__ == '__main__':
    """
    subroutine main():
      Following things are done here:
      - Adjustment of configuration and dataset specific parameters, 
      - loading the data set and
      - jump into one of the data analysis routines specified by variable
        'sub_command'
    """
    mne.set_log_level('WARNING')  # Dont print MNE loading logs

    print("EEG dataset analysis started")

    # Specify what should be done
    sub_command = 'TF_map'  # Calc time freq. psd map and store it in a file (using pure FFT method)
    # sub_command = 'TF_plot'     # plot time frequency map
    #    sub_command = 'Fb_power'    # Calc. and plot freq. band specific power spectral densities
    #    sub_command = 'PSD01'       # Calc. and plot power spectral density using pure FFT method
    #    sub_command = 'MT_PSD01'    # Calc. and plot power spectral density using mne mulittaper method
    #    sub_command = 'PSDs_plot'   # Plot PSDs stored in a previously measured time-frequency map
    #    sub_command = 'AMPHASE01'   # Calc. and plot amplitude and phase specte
    #    sub_command = 'None'        # Do nothing

    # Data set specific settings
    mi_ds = 'LSMR21'  # data set's name: PHYS, BCIC or LSMR21
    psdMethod = 'MUTAP'  # 'MUTAP' -> multitaper method used for psd calc.
    # 'PUFFT' -> pure FFT method used for psd calc.
    n_class = 2  # number of classes
    ds_tmin = -2.0  # trial slice start time in s
    # BCIC:   ds_tmin = -2.0
    # PHYS:   ds_tmin = -2.0
    # LSMR21: ds_tmin = -4.0
    ds_tmax = 6.0  # trial slice stop time in s
    # BCIC:   ds_tmax = 5.5
    # PHYS:   ds_tmax = 5.8
    # LSMR21: ds_tmyx = 6.0

    # For some commands read data from following file
    #    fname = f"{results_folder}/EEG_data_analysis/BCIC_MUTAPpsdMap_d20210813-00.npz"
    #    fname = f"{results_folder}/EEG_data_analysis/BCIC_PUFFTpsdMap_d20210813-00.npz"
    #    fname = f"{results_folder}/EEG_data_analysis/PHYS_MUTAPpsdMap_d20210813-00.npz"
    #    fname = f"{results_folder}/EEG_data_analysis/PHYS_PUFFTpsdMap_d20210813-00.npz"
    # fname = f"{results_folder}/EEG_data_analysis/LSMR21_MUTAPpsdMap_d20210814-00.npz"
    # fname = f"{results_folder}/EEG_data_analysis/LSMR21_MUTAPpsdMap_d20210817-00.npz"
    # fname = f"{results_folder}/EEG_data_analysis/LSMR21_MUTAPpsdMap_d20210818-00.npz"
    # fname = f"{results_folder}/EEG_data_analysis/LSMR21_Matlab_resampled_MUTAPpsdMap_d20210819-00.npz"
    # fname = f"{results_folder}/EEG_data_analysis/LSMR21_Numpy_resampled_MUTAPpsdMap_d20210819-00.npz"
    fname = f"{results_folder}/EEG_data_analysis/LSMR21_Matlab_1000Hz_MUTAPpsdMap_d20210819-00.npz"

    # Set parameters and optionally load the dataset
    if sub_command != 'TF_plot' and sub_command != 'Fb_power' \
            and sub_command != 'PSDs_plot' and sub_command != 'None':
        """
        Preload the data set specified by 'mi_ds' into main memory
        """
        dataset = DATASETS[mi_ds]  # Initial the read-only data structure 'dataset' which
        # stores the data set specific defaults parameters

        print("  - Name of active dataset name: ", dataset.CONSTANTS.name)
        excluded = []
        available_subjects = [i for i in dataset.CONSTANTS.ALL_SUBJECTS if i not in excluded]
        used_subjects = available_subjects
        validation_subjects = []
        ch_names = dataset.CONSTANTS.CHANNELS
        print("  - Channel names: ", ch_names)

        CONFIG.EEG.set_config(dataset.CONSTANTS.CONFIG)  # Data set specific default initializaion
        CONFIG.EEG.set_times(ds_tmin, ds_tmax)

        print("  - ds_tmin, ds_tmax =", ds_tmin, ds_tmax)
        preloaded_data, preloaded_labels = dataset.load_subjects_data(used_subjects + validation_subjects, n_class,
                                                                      ch_names)
        print("  - Shape of preloaded dataset: ", preloaded_data.shape)

        num_samples = preloaded_data.shape[3]
        num_channels = preloaded_data.shape[2]
        num_trials = preloaded_data.shape[1]
        num_subjects = preloaded_data.shape[0]

        print("  - Data set %s load with following following characteristic:" % mi_ds)
        print("    - Trial size: %2.2f s -> %2.2f s" % (CONFIG.EEG.TMIN, CONFIG.EEG.TMAX))
        print("    - Sample rate:", CONFIG.EEG.SAMPLERATE)
        print("    - Number of subjects:", num_subjects)
        print("    - NUmber of EEG channels:", num_channels)
        print("    - Number of trials:", num_trials)
        print("    - Number of samples per trial:", num_samples)

    # Sub_command jump table
    ### TF_map ##########################################################################
    if sub_command == 'TF_map':
        print("  - Subcommand '%s': Calculate time-frequency psd map" % sub_command)

        # Calculate mean psd for the time slices defined in ts_offsets
        ts_tmp = []
        ts_act = ds_tmin  # Starting point for the offsets
        ts_size = 1.0  # trial slice size in seconds used for psd calculation
        rest_ts_tmin = -1.5  # trial slice offset for 'rest state' trial slice
        sampling_rate = CONFIG.EEG.SAMPLERATE
        while (ts_act < ds_tmax - ts_size):
            ts_tmp.append(ts_act)
            ts_act = ts_act + 0.1
        ts_offsets = np.array(ts_tmp)  # ts_offset is only used in case of 'PUFFT'

        if psdMethod == 'PUFFT':
            calc_time_freq_FFT_PSDmap(preloaded_data, preloaded_labels, mi_ds,
                                      ds_tmin, ds_tmax, sampling_rate, ts_offsets, rest_ts_tmin, ts_size)
        elif psdMethod == 'MUTAP':
            calc_time_freq_MT_PSDmap(preloaded_data, preloaded_labels, mi_ds,
                                     ds_tmin, ds_tmax, sampling_rate, rest_ts_tmin, ts_size)
        else:
            print("Error: Illegal psd method chosen!")

    ### TF_plot ##########################################################################
    elif sub_command == 'TF_plot':
        print("  - Subcommand '%s': Plot time-frequency map" % sub_command)
        rest_ts_tmin = -2.8
        plot_time_freq_PSDmap(fname, rest_ts_tmin, REST_NORM='false', LOG10='true',
                              tmin=-2.0, tmax=5.9,
                              fmin=1.0, fmax=80.0,
                              vmin=-0.2,  # scale to this min. value
                              vmax=0.42)  # scale to this max. value

    ### Fb_power #########################################################################
    elif sub_command == 'Fb_power':
        print("  - Subcommand '%s': Calculate freq. band specific power spectral density" % sub_command)

        f_bands_wald1 = np.array([[62.0, 87.0], [12.0, 30.0], [8.0, 12.0],
                                  [0.0, 7.0], ])  # low-, intermediate- and high-freq. bands according to [Wald-08]
        f_bands_wald2 = np.array([[62.0, 87.0],
                                  [28.0, 32.0],
                                  [16.0, 26.0],
                                  [10.0, 12.0],
                                  [8.0, 10.0],
                                  [4.0, 7.0],
                                  [0.0, 2.0], ])
        f_bands_standard = np.array([[60.0, 200.0],  # High gamma band
                                     [30.0, 60.0],  # Low gamma band
                                     [13.0, 30.0],  # Beta band
                                     [8.0, 13.0],  # Alpha or Mu band
                                     [5.0, 7.0],  # Theta
                                     [2.0, 4.0],  # Delta band
                                     [0.5, 1.5]])  # Zero band
        f_bands = f_bands_wald1
        rest_ts_tmin = -1.0
        tmin = -1.0
        tmax = 3.8
        calc_plot_Fb_psd(fname, rest_ts_tmin, f_bands, tmin=tmin, tmax=tmax,
                         REST_NORM='false', LOG10='false', SUBPLOTS='true')

    ### PSD01 ###########################################################################
    elif sub_command == 'PSD01':  # Calculate power spectral density
        print("  - Subcommand '%s': Calculate and plot power spectral density" % sub_command)
        ts_tmin = -4.0  # Trial slice start time (relative to cue signal) in s
        ts_tsize = 1.0  # Trial slice stop  time )relative to cue signal) in s
        fmin = 0.5  # Minimum frequency to plot in Hz
        fmax = 80.0  # Maximum frequency to plot in Hz
        sampling_rate = CONFIG.EEG.SAMPLERATE
        ts_samples = int(ts_tsize * sampling_rate)
        ds_name = mi_ds
        psd_all, f = calc_psd(preloaded_data, ds_tmin, ds_tmax, sampling_rate, ts_tmin, ts_samples)
        plot_psd(psd_all, ds_name, f, fmin, fmax, ts_tmin, ts_tsize)

    ### PSD_plot ########################################################################
    elif sub_command == 'PSDs_plot':
        """
        Plot PSDs which are part of a previously measured time-frequency PSD map
        """
        print("  - Subcommand '%s': Plot PSDs out of time-freq. PSD map" % sub_command)

        ts_offsets = [0.0, 2.0, 3.0]
        fmin = 0.5  # Minimum frequency to plot in Hz
        fmax = 80.0  # Maximum frequency to plot in Hz

        PSDs_plot(fname, fmin, fmax, ts_offsets)

    ### MNE_PSD01 ###########################################################################
    elif sub_command == 'MT_PSD01':  # Calculate power spectral density
        print(
            "  - Subcommand '%s': Calculate and plot power spectral density using MNE multitaper method" % sub_command)
        # Convert Numpy array to MNE-epochs data object
        tmin = -2.0  # Start time to consider for psd calculation
        tmax = -1.0  # End time to consider for psd calculation
        plot_mne_psd(preloaded_data, CONFIG.EEG.SAMPLERATE, dataset.channels, dataset.name,
                     ds_tmin, ds_tmax, tmin, tmax)

    ### AMPHASE01 ###########################################################################
    elif sub_command == 'AMPHASE01':  # Calculate power spectral density
        print("  - Subcommand '%s': Calculate and plot amplitude and phase spectra" % sub_command)
        ts_tmin = -2.0  # Trial slice start time (relative to cue signal) in s
        ts_tsize = 1.0  # Trial slice stop  time )relative to cue signal) in s
        fmin = 0.5  # Minimum frequency to plot in Hz
        fmax = 80.0  # Maximum frequency to plot in Hz
        sampling_rate = CONFIG.EEG.SAMPLERATE
        ts_samples = int(ts_tsize * sampling_rate)
        ds_name = mi_ds
        amp_fft_all, phase_fft_all, f = calc_amphase(preloaded_data, ds_tmin, ds_tmax, sampling_rate, ts_tmin,
                                                     ts_samples)
        plot_amphase(amp_fft_all, phase_fft_all, ds_name, f, fmin, fmax, ts_tmin, ts_tsize)

    else:
        print("  - Illegal subcommand '%s' specified" % sub_command)

    sys.exit()
