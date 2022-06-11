'''
File bcic_iv2a_dataset.py

Description:
  Contains class BCIC_IV2a_dataset which provides lots of methods to handle
  BCI competition IV 2a motor imagery data set.

History:
  2020-05-11: Ongoing version 0.1 implementation - ms
'''

# from matplotlib import pyplot as plt
import logging
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

# All total trials per class per n_class-Classification
from app.config import CONFIG
from app.data.datasets.bcic.bcic_dataset import BCIC
from app.paths import datasets_folder
from app.util.misc import calc_n_samples, to_idxs_of_list_str, makedir

BCIC_classes_trials = {
    "2class": {
        0: 572,  # Left
        1: 591,  # Right
    },
    "3class": {
        0: 572,  # Left
        1: 591,  # Right
        2: 574,  # Both feet
    },
    "4class": {
        0: 572,  # Left
        1: 591,  # Right
        2: 574,  # Both feet
        3: 591,  # Tongue
    },
}

'''
class: BCIC_IV2a_dataset(subject, n_classes, path)

Description:
  Holds BCIC_IV2a specific EEG time series data and the corresponding 
  classification labels for the subjects specified in list 'subjects', 
  for the number of different MI classes specified by 'n_classes'.
  Parameter 'path' specifies the directory were the original subject 
  specific data files are stored. The original subject specific
  data files and some basic ideas to handle the data were taken from:
        https://github.com/bregydoc/bcidatasetIV2a
  At that reference you will also find a description of the provided
  .npz files and their contents.
  
  This class stores EEG time series data in the 4-dim. numpy array 'pl_data'
  with following dimensions:
          dim 0: subject
          dim 1: trial
          dim 3: EEG channel
          dim 4: sample
  Trial specific labels are preloaded into the 2-dim. numpy array
  'pl_labels' with the following dimensions:
          dim 0: subject
          dim 1: trial
  
  Used class encoding for preloaded_labels:
          pl_label[x, y] =  0:  class 1 --> left hand
          pl_label[x, y] =  1:  class 2 --> right hand
          pl_label[x, y] =  2:  class 3 --> both feet
          pl_label[x, y] =  4:  class 4 --> tongue
          pl_label[x, y] = -1:  Invalid trial, please ignore it
          
  Number of trials per subject unfortunately is subject-specific. n_trials_max
  specifies the theoretically maximum number of trials per subjects. To mark
  the array entries unused because of missing trials, class label -1 has been
  definied, which indicates that the corresponding array entry in 
  pl_data should be ignored for this trial.
  
  If n_classes = 2 has been passed, only the trials belonging to class 1 and 2
  are stored in pl_data and pl_labels.
  If n_classes = 3 has been passed, only the trials belonging to class 1, 2 and 3
  are stored in pl_data and pl_labels,
  If n_classes = 4 has been passed, only the trials belonging to class 1, 2, 3 and 4
  are stored in pl_data and pl_labels,
  
  Following methods are provided to handle BCIC_IV2a dataset:
  - load_subjects_data(): Load data and labels for specified subjects and classes 
                          from the original subject specific data files into numpy
                          arrays pl_data and pl_labels. Original subject specific
                          data files and some basic ideas to handle the data were
                          taken from:
                          https://github.com/bregydoc/bcidatasetIV2a
                          
  - save_pl_dataLabels(): Save the data and labels stored in numpy arrays pl_data 
                          and pl_labels in a file
                          
  - load_pl_dataLabels(): Load arrays pl_data and pl_labels from a file which 
                          previously has been created by save_pl_dataLabels()
  - print_stats: Print statistics, which means the number of trials per class
                 stored in pl_data
'''

path = f'{datasets_folder}/BCICompetition_IV-2a/'

orgfiles_path = path + "Numpy_files/"  # Location of original subject-specific data files
pl_path = path + "pl_data/"  # Location where preloaded data is stored

class BCIC_IV2a_dataset:
    '''
    Constructor:
      Initializes the used data structures
    '''

    fs = BCIC.CONFIG.SAMPLERATE  # sampling rate: 250Hz from original paper

    # channel_idxs = to_idxs_of_list(ch_names, BCIC.CHANNELS)  # 22 EEG electrodes

    tmin = CONFIG.EEG.TMIN
    tmax = CONFIG.EEG.TMAX
    n_samples = calc_n_samples(tmin, tmax, fs)

    # Types of motor imagery
    mi_types = {769: 'left', 770: 'right', 771: 'foot', 772: 'tongue', 783: 'unknown'}

    '''
    Method: get_trials(...)
    Descpription:
      Get all trials data and corresponding labels for the specified subject.
    '''

    '''
    Method: load_subjects_data(training)
    Parameters:
      training: = 1 --> load training data   (*T.npz file)
                = 0 --> load evaluation data (*E.nps file)
      (The subjects and n_class for which data should be loaded have already been
      specified when dataset object has been created.)
    '''

    @classmethod
    def get_trials(cls, subject: int, n_class: int = 4, ch_names: List[str] = BCIC.CHANNELS, training: int = 1):
        ch_idxs = to_idxs_of_list_str(ch_names, BCIC.CHANNELS)
        n_trials_max = 6 * 12 * n_class  # 6 runs with 12 trials per class
        n_samples = calc_n_samples(CONFIG.EEG.TMIN, CONFIG.EEG.TMAX, BCIC.CONFIG.SAMPLERATE)
        fname = cls.get_subject_fname(subject, training)

        logging.info('  - Load data of subject %d from file: %s',subject, fname)
        data = np.load(fname)

        raw = data['s'].T
        events_type = data['etyp'].T
        events_position = data['epos'].T
        events_duration = data['edur'].T
        artifacts = data['artifacts'].T

        # # optional resampling + butterworth bandpass filtering
        # raw = MIDataLoader.resample_and_filter(raw)
        startrial_code = 768
        starttrial_events = events_type == startrial_code
        idxs = [i for i, x in enumerate(starttrial_events[0]) if x]

        # Originally we have 288 trial, but some have artifacts
        # trial_ind is the index of the original 288 trial dataset
        # pl_trial_ind is the index where a valid data set is stored
        # in pl_data array
        trial_ind = 0
        pl_trial_ind = 0
        subject_data, subject_labels = np.full((n_trials_max, len(ch_idxs), n_samples), -1), \
                                       np.full((n_trials_max), -1)
        # For any trial specified by its idxs valus
        for index in idxs:
            try:
                if artifacts[0, trial_ind] == 0:
                    type_e = events_type[0, index + 1]
                    class_e = cls.mi_types[type_e]

                    if (n_class == 2 and (type_e >= 769 and type_e <= 770)) or \
                            (n_class == 3 and (type_e >= 769 and type_e <= 771)) or \
                            (n_class == 4 and (type_e >= 769 and type_e <= 772)):

                        # Store the trial specific label with following class encoding:
                        # MI action   | BCIC_IV2a code | class label
                        # ------------------------------------------
                        # Left hand          769              1 -> 0
                        # Right hand         770              2 -> 1
                        # Both feet          771              3 -> 2
                        # Tongue             772              4 -> 3
                        # Unknown            783             -1 (invalid trial)

                        # Assume we have class vector like this one here:
                        # class_vec = [class1, class2, class3, class4]
                        # Then pl_labels is the index to the class to which current
                        # trials belongs.
                        subject_labels[pl_trial_ind] = type_e - 768 - 1

                        start = events_position[0, index]
                        stop = start + events_duration[0, index]
                        # Copy trial data into pl_data array
                        for i, channel in enumerate(ch_idxs):
                            trial = raw[channel, start:stop]
                            if len(trial) != 1875:
                                logging.info('get_trials(): Illegal length')

                            # Copy part of channel data into pl_data
                            start_idx = int(CONFIG.EEG.TMIN * BCIC.CONFIG.SAMPLERATE)
                            for idx in range(n_samples):
                                subject_data[pl_trial_ind, i, idx] = float(trial[start_idx + idx])

                        pl_trial_ind = pl_trial_ind + 1

                else:
                    x_temp = 0  # noop
                #                    logging.info("  - Trial %d is marked as an artifact and ignored" % (trial_ind))

                trial_ind = trial_ind + 1
            except Exception as e:
                logging.info("get_trials(): Exception occured %s", e)
                continue
        return subject_data, subject_labels

    @classmethod
    def get_raw_run_data(cls, subject: int, n_class: int, ch_names: List[str] = BCIC.CHANNELS, training: int = 1):
        """
        Loading Raw EEG Data for Live Simulation mode
        :return:
        raw_data: np.ndarray with (channel, Sample) of the run,
        trials_classes: Class Label of each Trial in the Run,
        trials_start_times: Start Times (in Sec.) of every Trial,
        trials_start_samples: Start Sample of every Trial,
        trials_samples_length: Sample Duration of every Trial
        """
        ch_idxs = to_idxs_of_list_str(ch_names, BCIC.CHANNELS)
        fname = cls.get_subject_fname(subject, training)

        logging.info('  - Load data of subject %d from file: %s', subject, fname)
        data = np.load(fname)

        raw = data['s'].T
        events_type = data['etyp'].T
        startrial_code = 768
        starttrial_events = events_type == startrial_code
        # Get indexes of Start Trial events
        idxs = [i + 1 for i, x in enumerate(starttrial_events[0]) if x]

        events_position = data['epos'].T[0]
        events_duration = data['edur'].T[0]
        events_position = events_position[idxs]
        events_duration = events_duration[idxs]
        artifacts = data['artifacts'].T

        trials_classes = cls.map_to_class_labels(events_type[0, idxs])

        trials_start_times = np.asarray([(start_pos / CONFIG.EEG.SAMPLERATE) for start_pos in events_position])
        trials_start_samples = events_position
        trials_samples_length = events_duration
        # First Trial starts after 5-6 minutes of EOG Influence Test
        # omit all Samples before first actual Trial starts
        actual_first_sample = trials_start_samples[0]
        actual_last_sample = trials_start_samples[-1] + trials_samples_length[-1]
        X = raw[ch_idxs, actual_first_sample:actual_last_sample]
        # t = 0s. is Start Time of first actual Trial (after the 5-6 min. EOG Influence Test)
        trials_start_times = trials_start_times - trials_start_times[0]
        trials_start_samples = trials_start_samples - actual_first_sample
        return X, trials_classes, trials_start_times, trials_start_samples, trials_samples_length

    @classmethod
    def map_to_class_labels(cls, events_type: np.ndarray):
        """
        Maps BCIC raw data events_types to categorical labels (0,1,2,...)
        """
        labels = []
        for event_type in events_type:
            if event_type in BCIC.event_type_to_label:
                labels.append(BCIC.event_type_to_label[event_type])
            else:
                labels.append(-1)
                # logging.info("Rejected Trial found: %s", event_type)
        return np.asarray(labels, dtype=np.int)

    @classmethod
    def get_subject_fname(cls, subject: int, training: int = 1):
        if training == 1:
            return cls.orgfiles_path + 'A0' + str(subject) + 'T.npz'
        elif training == 0:
            return cls.orgfiles_path + 'A0' + str(subject) + 'E.npz'
        else:
            logging.info('Error: Illegal parameter')
        return None

    '''
    Method: save_pl_dataLabels(self, fname)
    Description:
      Save the data and labels stored in numpy arrays self.pl_data and self.pl_labels
      in a file with filename 'fname' in the folder specified by self.pl_path
    '''

    @classmethod
    def save_pl_dataLabels(cls, subjects: List[int], n_class: int, data: np.ndarray, labels: np.ndarray, fname: str,
                           path=None):
        if path is None:
            path = cls.pl_path
        path = os.path.join(path, fname)
        logging.info("- save_pl_dataLabels: Store preprocessed data in file: %s", path)

        np.savez(path, subjects=subjects, n_classes=n_class,
                 pl_data=data, pl_labels=labels)
        logging.info("  - Data (subjects, n_classes, pl_data, pl_labels) saved in file: %s", path)

        return

    '''
    Method: load_pl_dataLabels(self, fname)
    Description:
      Load the data and labels in numpy arrays self.pl_data and self.pl_labels
      from a file with filename 'fname'. Both arrays must have been stored in that
      file using save_pl_dataLabels().
    '''

    @classmethod
    def load_pl_dataLabels(cls, fname, path=None):
        if path is None:
            path = cls.pl_path
        logging.info("- load_pl_dataLabels: Load 'pl_data' and 'pl_labels' from file: %s", (path + fname))

        # Read data from files:
        with np.load(path + fname) as data:
            subjects = data['subjects']
            n_class = data['n_classes'].item()
            pl_data = data['pl_data']
            pl_labels = data['pl_labels']

        n_trials_max = 6 * 12 * n_class  # 6 runs with 12 trials per class

        return subjects, n_class, pl_data, pl_labels, n_trials_max

    '''
    Method: calc_psds(self)
    Description:
      Calculates 'power spectral density' graphs:
      - psd_all:    Mean of the psds of all channels of all trials and all users
      - psd_classX: Mean of the psds of all channels of all classX trials and all users
      
      psds are saved in files in folder self.pl_data
      
      Have Care: Only works for n_classes = 4
    '''

    @classmethod
    def calc_psds(cls, n_class: int, subjects: List[int], data: np.ndarray, labels: np.ndarray, path=None):
        if path is None:
            path = cls.pl_path
        logging.info("- calc_psds: Calculate power spectral densities")

        if (n_class != 4):
            raise Exception("Sorry, but this method only works if n_class = 4")

        logging.info("  - Number of available subjects: %s", len(subjects))
        logging.info("  - preloaded data shape = %s", data.shape)
        logging.info("  - preloaded labels shape = %s", labels.shape)

        # Assign basic parameters
        num_samples = data.shape[3]
        num_channels = data.shape[2]
        num_trials = data.shape[1]
        num_subjects = data.shape[0]
        logging.info("  - subjects: %d, trials/subject: %d, EEG-channels/trial: %d",num_subjects, num_trials, num_channels)

        # Calculate and sum mean power spectral density psd
        # Length of psd arrays is according to the used psd algothimn
        if num_samples % 2 == 0:  # even number
            num_psd_samples = (num_samples / 2) + 1
        else:
            num_psd_samples = (num_samples + 1) / 2
        num_psd_samples = int(num_psd_samples)
        logging.info("  - num_psd_samples = %d" % num_psd_samples)

        # Create numpy arrays to store the psds
        psd_all = np.zeros(num_psd_samples)  # Mean psd of all trials
        psd_class1 = np.zeros(num_psd_samples)  # Mean psd of all 'left hand' trials
        psd_class2 = np.zeros(num_psd_samples)  # Mean psd of all 'right hand' trials
        psd_class3 = np.zeros(num_psd_samples)  # Mean psd of all 'both feet' trials
        psd_class4 = np.zeros(num_psd_samples)  # Mean psd of all 'tongue' trials

        # Start psd calculation
        num_class1_trials = 0
        num_class2_trials = 0
        num_class3_trials = 0
        num_class4_trials = 0

        for subject in range(num_subjects):
            # sum up the psd for all trials of a subjects
            for trial in range(num_trials):
                # calculate the psd for each EEG channel and sum up all psd
                for ch in range(num_channels):
                    if labels[subject, trial] != -1:
                        ch_data = data[subject, trial, ch, :]
                        f, psd = signal.periodogram(ch_data, float(CONFIG.EEG.SAMPLERATE))  # this method uses
                        # welch functions and should provide a lower noise but a
                        # a little bit poorer frequency resolution
                        psd_all = psd_all + psd
                        if labels[subject, trial] == 0:
                            num_class1_trials = num_class1_trials + 1
                            psd_class1 = psd_class1 + psd
                        elif labels[subject, trial] == 1:
                            num_class2_trials = num_class2_trials + 1
                            psd_class2 = psd_class2 + psd
                        elif labels[subject, trial] == 2:
                            num_class3_trials = num_class3_trials + 1
                            psd_class3 = psd_class3 + psd
                        elif labels[subject, trial] == 3:
                            num_class4_trials = num_class4_trials + 1
                            psd_class4 = psd_class4 + psd
                        else:
                            logging.info("ERROR: Illegal label")

        logging.info("  - Number of classX trials: %s %s %s", num_class1_trials, num_class2_trials, num_class3_trials,
              num_class4_trials)

        psd_all = psd_all / (num_class1_trials + num_class2_trials + num_class3_trials + num_class4_trials)
        psd_class1 = psd_class1 / (num_class1_trials)
        psd_class2 = psd_class2 / (num_class2_trials)
        psd_class2 = psd_class3 / (num_class3_trials)
        psd_class2 = psd_class4 / (num_class4_trials)

        makedir(path)
        # save psd's in files
        fname = os.path.join(path + '/psd_mean_')

        np.savez(fname + 'allclasses', f=f, psd=psd_all)
        np.savez(fname + 'class1', f=f, psd=psd_class1)
        np.savez(fname + 'class2', f=f, psd=psd_class2)
        np.savez(fname + 'class3', f=f, psd=psd_class3)
        np.savez(fname + 'class4', f=f, psd=psd_class4)

        logging.info("  - Data (f, psd) saved in files: ", (fname + "xxx.npz"))



########################################################################################
'''
subroutine: plot_psds

Description:
  Reads psds from files and displays them.
'''


def plot_psds(path=f'{datasets_folder}/BCICompetition_IV-2a/pl_data/', sampling_rate=250):
    logging.info("Plot psd diagram:")

    # construct psd file names
    fname = os.path.join(path + '/psd_mean_')

    # Read data from files:
    logging.info("  - Load data (f, psd) from files: ", (fname + "xxx.npz"))
    with np.load(fname + 'allclasses.npz') as data:
        f = data['f']
        psd_all = data['psd']
    with np.load(fname + 'class1.npz') as data:
        f = data['f']
        psd_class1 = data['psd']
    with np.load(fname + 'class2.npz') as data:
        f = data['f']
        psd_class2 = data['psd']
    with np.load(fname + 'class3.npz') as data:
        f = data['f']
        psd_class3 = data['psd']
    with np.load(fname + 'class4.npz') as data:
        f = data['f']
        psd_class4 = data['psd']

    # Plot parameters:
    plot_fstart = 0.5  # min frequency component shown in plot
    plot_fstop = 12.0  # max frequency component shown in plot
    LOG_PLOT = True  # enable/disable log display
    PLOT_ALL = True  # plot mean psd of all trials
    PLOT_CLASS1 = True  # plot mean psd of class1 trials
    PLOT_CLASS2 = True  # plot mean psd of class2 trials
    PLOT_CLASS3 = False  # plot mean psd of class3 trials
    PLOT_CLASS4 = False  # plot mean psd of class4 trials

    num_samples = 2 * len(psd_all)
    delta_f = float(sampling_rate / num_samples)
    plot_start_ind = int(plot_fstart / delta_f)
    plot_stop_ind = int(plot_fstop / delta_f)
    logging.info("  - delta_f = %f Hz, plot_start_ind = %d, plot_stop_ind = %d" % (delta_f, plot_start_ind, plot_stop_ind))

    # Assign plot arrays
    f_plot = f[plot_start_ind:plot_stop_ind]

    psd_all_plot = psd_all[plot_start_ind:plot_stop_ind]
    psd_class1_plot = psd_class1[plot_start_ind:plot_stop_ind]
    psd_class2_plot = psd_class2[plot_start_ind:plot_stop_ind]
    psd_class3_plot = psd_class3[plot_start_ind:plot_stop_ind]
    psd_class4_plot = psd_class4[plot_start_ind:plot_stop_ind]

    legend = []
    if PLOT_ALL == True:
        if LOG_PLOT == False:
            plt.plot(f_plot, psd_all_plot)
        else:
            plt.semilogy(f_plot, psd_all_plot)
        legend.append('ALL trials mean')
    if PLOT_CLASS1 == True:
        if LOG_PLOT == False:
            plt.plot(f_plot, psd_class1_plot)
        else:
            plt.semilogy(f_plot, psd_class1_plot)
        legend.append('LEFT HAND trials mean')
    if PLOT_CLASS2 == True:
        if LOG_PLOT == False:
            plt.plot(f_plot, psd_class2_plot)
        else:
            plt.semilogy(f_plot, psd_class2_plot)
        legend.append('RIGHT HAND trials mean')
    if PLOT_CLASS3 == True:
        if LOG_PLOT == False:
            plt.plot(f_plot, psd_class3_plot)
        else:
            plt.semilogy(f_plot, psd_class3_plot)
        legend.append('BOTH FEET trials mean')
    if PLOT_CLASS4 == True:
        if LOG_PLOT == False:
            plt.plot(f_plot, psd_class4_plot)
        else:
            plt.semilogy(f_plot, psd_class4_plot)
        legend.append('TONGUE trials mean')
    plt.legend(legend)

    bottom, top = plt.ylim()
    #    plt.ylim([1e-11, 1e-8])
    plt.xlabel('Frequency [Hz]')
    if LOG_PLOT == True:
        plt.ylabel('Log. Power spectral density [V**2/Hz]')
    else:
        plt.ylabel('Power spectral density [V**2/Hz]')
    plt.title("Power spectral density")
    plt.grid(True)
    plt.show()

    return

