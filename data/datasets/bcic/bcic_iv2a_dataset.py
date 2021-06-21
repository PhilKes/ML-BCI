'''
File bcic_iv2a_dataset.py

Description:
  Contains class BCIC_IV2a_dataset which provides lots of methods to handle
  BCI competition IV 2a motor imagery data set.

ToDo:
  - Implement n_classes specific loading
  - Implement save_pl_dataLabels(fname)
  - Implement load_pl_dataLabels(fname)
  - Implement bandpass filtering
  - Implement dataloader creation methods

History:
  2020-05-11: Ongoing version 0.1 implementation - ms
'''

# from matplotlib import pyplot as plt
import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy import signal
from config import eeg_config, datasets_folder, SHOW_PLOTS
from data.datasets.bcic.bcic_dataset import BCIC_CHANNELS
from data.data_utils import butter_bandpass_filt
from config import global_config

# All total trials per class per n_class-Classification
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


class BCIC_IV2a_dataset:
    '''
    Constructor:
      Initializes the used data structures
    '''

    def __init__(self, subjects=[1], n_classes=4, path=f'{datasets_folder}/BCICompetition_IV-2a/',
                 ch_names=BCIC_CHANNELS):
        self.path = path
        self.orgfiles_path = path + "Numpy_files/"  # Location of original subject-specific data files
        self.pl_path = path + "pl_data/"  # Location where preloaded data is stored

        # We must have at least one and at maximum 9 subjects
        if len(subjects) > 0 and len(subjects) <= 9:
            self.subjects = subjects
        else:
            print("Error in BCIC_IV2a_dataset constructor: Illegal number of subjects:", len(subjects))
            sys.exit(0)

        # Legal n_classes values are 2, 3 and 4
        if n_classes >= 2 and n_classes <= 4:
            self.n_classes = n_classes
        else:
            print("Error in BCIC_IV2a_dataset constructor: Illegal parameter n_classes: ", n_classes)
            sys.exit(0)

        self.fs = eeg_config.SAMPLERATE  # sampling rate: 250Hz from original paper
        self.n_trials_max = None  # Maximum possible number of trials
        self.n_channels = len(ch_names)  # 22 EEG electrodes

        self.tmin = eeg_config.TMIN
        self.tmax = eeg_config.TMAX
        self.n_samples = int((self.tmax - self.tmin) * self.fs)
        print("BCIC_IV2a_dataset: fs, tmin, tmax, n_samples= ", self.fs, self.tmin, self.tmax, self.n_samples)

        self.pl_data = None
        self.pl_labels = None

        # Types of motor imagery
        self.mi_types = {769: 'left', 770: 'right', 771: 'foot', 772: 'tongue', 783: 'unknown'}

    '''
    Method: print_stats()
      Analysis of pl_labels() and extraction of how many trials of each class we have
      on a per subject basis. Result is printed on the screen.
    '''

    def print_stats(self):
        print("- Some statistics of BCIC_IV2a dataset:")

        all_subjects_counts = [0, 0, 0, 0, 0, 0]

        print()
        print("  Subject | class1 | class2 | class3 | class4 | artifact | all-legal")
        print("  --------|--------|--------|--------|--------|----------|----------")
        for subject in range(len(self.subjects)):
            class_counts = [0, 0, 0, 0, 0, 0]
            for trial in range(self.n_trials_max):
                if self.pl_labels[subject, trial] == 0:
                    class_counts[0] = class_counts[0] + 1
                elif self.pl_labels[subject, trial] == 1:
                    class_counts[1] = class_counts[1] + 1
                elif self.pl_labels[subject, trial] == 2:
                    class_counts[2] = class_counts[2] + 1
                elif self.pl_labels[subject, trial] == 3:
                    class_counts[3] = class_counts[3] + 1
                elif self.pl_labels[subject, trial] == -1:
                    class_counts[4] = class_counts[4] + 1
                else:
                    print("print_stats(): Illegal class!!!", self.pl_labels[subject, trial])

            class_counts[5] = class_counts[0] + class_counts[1] + class_counts[2] \
                              + class_counts[3]

            for i in range(len(all_subjects_counts)):
                all_subjects_counts[i] = all_subjects_counts[i] + class_counts[i]

            print("    %3d   |   %3d  |   %3d  |   %3d  |   %3d  |    %3d   |    %3d" % \
                  (subject, class_counts[0], class_counts[1], class_counts[2], \
                   class_counts[3], class_counts[4], class_counts[5]))

        print("  --------|--------|--------|--------|--------|----------|----------")
        print("    All   |   %3d  |   %3d  |   %3d  |   %3d  |    %3d   |   %4d" % \
              (all_subjects_counts[0], all_subjects_counts[1], all_subjects_counts[2], \
               all_subjects_counts[3], all_subjects_counts[4], all_subjects_counts[5]))
        print()

    '''
    Method: get_trials(...)
    Descpription:
      Get all trials data and corresponding labels for the specified subject.
    '''

    def get_trials(self, subject, fname):
        data = np.load(fname)

        raw = data['s'].T
        events_type = data['etyp'].T
        events_position = data['epos'].T
        events_duration = data['edur'].T
        artifacts = data['artifacts'].T

        # optional butterworth bandpass filtering
        if global_config.FREQ_FILTER_HIGHPASS != None or global_config.FREQ_FILTER_LOWPASS != None:
            raw = butter_bandpass_filt(raw, lowcut=global_config.FREQ_FILTER_HIGHPASS, \
                                       highcut=global_config.FREQ_FILTER_LOWPASS, \
                                       fs=eeg_config.SAMPLERATE, order=7)

        startrial_code = 768
        starttrial_events = events_type == startrial_code
        idxs = [i for i, x in enumerate(starttrial_events[0]) if x]

        # Originally we have 288 trial, but some have artifacts
        # trial_ind is the index of the original 288 trial dataset
        # pl_trial_ind is the index where a valid data set is stored
        # in pl_data array
        trial_ind = 0
        pl_trial_ind = 0
        # For any trial specified by its idxs valus
        for index in idxs:
            try:
                if artifacts[0, trial_ind] == 0:
                    type_e = events_type[0, index + 1]
                    class_e = self.mi_types[type_e]

                    if (self.n_classes == 2 and (type_e >= 769 and type_e <= 770)) or \
                            (self.n_classes == 3 and (type_e >= 769 and type_e <= 771)) or \
                            (self.n_classes == 4 and (type_e >= 769 and type_e <= 772)):

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
                        # Have care!!! We have to subtract 1 because indexing starts
                        # with index 0.
                        self.pl_labels[subject - 1, pl_trial_ind] = type_e - 768 - 1

                        start = events_position[0, index]
                        stop = start + events_duration[0, index]
                        # Copy trial data into pl_data array
                        for i in range(self.n_channels):
                            channel = i
                            trial = raw[channel, start:stop]
                            if len(trial) != 1875:
                                print('get_trials(): Illegal length')

                            # Copy part of channel data into pl_data
                            start_idx = int(self.tmin * self.fs)
                            for idx in range(self.n_samples):
                                self.pl_data[subject - 1, pl_trial_ind, channel, idx] = float(trial[start_idx + idx])

                        pl_trial_ind = pl_trial_ind + 1

                else:
                    x_temp = 0  # noop
                #                    print("  - Trial %d is marked as an artifact and ignored" % (trial_ind))

                trial_ind = trial_ind + 1
            except:
                print("get_trials(): Exception occured")
                continue

        return

    '''
    Method: load_subjects_data(training)
    Parameters:
      training: = 1 --> load training data   (*T.npz file)
                = 0 --> load evaluation data (*E.nps file)
      (The subjects and n_classes for which data should be loaded have already been
      specified when dataset object has been created.)
    '''

    def load_subjects_data(self, training):
        self.n_trials_max = 6 * 12 * self.n_classes  # 6 runs with 12 trials per class

        self.pl_data = np.zeros((len(self.subjects), self.n_trials_max, self.n_channels, \
                                 self.n_samples), dtype=np.float32)
        self.pl_labels = np.full((len(self.subjects), self.n_trials_max,), -1, dtype=np.int)  # Initialize
        # preloaded_labels with -1, which indicates an invalid trial

        self.subjects.sort()
        print('- Subjects for which data will be loaded: ', self.subjects)

        for subject in self.subjects:
            if training == 1:
                fname = self.orgfiles_path + 'A0' + str(subject) + 'T.npz'
            elif training == 0:
                fname = self.orgfiles_path + 'A0' + str(subject) + 'E.npz'
            else:
                print('Error: Illegal parameter')

            print('  - Load data of subject %d from file: %s' % (subject, fname))

            self.get_trials(subject, fname)

        return self.pl_data, self.pl_labels

    '''
    Method: save_pl_dataLabels(self, fname)
    Description:
      Save the data and labels stored in numpy arrays self.pl_data and self.pl_labels
      in a file with filename 'fname' in the folder specified by self.pl_path
    '''

    def save_pl_dataLabels(self, fname):
        print("- save_pl_dataLabels: Store preprocessed data in file: ", (self.pl_path + fname))

        np.savez(self.pl_path + fname, subjects=self.subjects, n_classes=self.n_classes, \
                 pl_data=self.pl_data, pl_labels=self.pl_labels)
        print("  - Data (subjects, n_classes, pl_data, pl_labels) saved in file: ", (self.pl_path + fname))

        return

    '''
    Method: load_pl_dataLabels(self, fname)
    Description:
      Load the data and labels in numpy arrays self.pl_data and self.pl_labels
      from a file with filename 'fname'. Both arrays must have been stored in that
      file using save_pl_dataLabels().
    '''

    def load_pl_dataLabels(self, fname):
        print("- load_pl_dataLabels: Load 'pl_data' and 'pl_labels' from file: ", (self.pl_path + fname))

        # Read data from files:
        with np.load(self.pl_path + fname) as data:
            self.subjects = data['subjects']
            self.n_classes = data['n_classes']
            self.pl_data = data['pl_data']
            self.pl_labels = data['pl_labels']

        self.n_trials_max = 6 * 12 * self.n_classes  # 6 runs with 12 trials per class

        return self.pl_data, self.pl_labels

    '''
    Method: calc_psds(self)
    Description:
      Calculates 'power spectral density' graphs:
      - psd_all:    Mean of the psds of all channels of all trials and all users
      - psd_classX: Mean of the psds of all channels of all classX trials and all users
      
      psds are saved in files in folder self.pl_data
      
      Have Care: Only works for n_classes = 4
    '''

    def calc_psds(self):
        print("- calc_psds: Calculate power spectral densities")

        if (self.n_classes != 4):
            print("Sorry, but this method only works if n_classes = 4")
            return

        print("  - Number of available subjects:", len(self.subjects))
        print("  - preloaded data shape =", self.pl_data.shape)
        print("  - preloaded labels shape =", self.pl_labels.shape)

        # Assign basic parameters
        num_samples = self.pl_data.shape[3]
        num_channels = self.pl_data.shape[2]
        num_trials = self.pl_data.shape[1]
        num_subjects = self.pl_data.shape[0]
        print("  - subjects: %d, trials/subject: %d, EEG-channels/trial: %d" % (num_subjects, num_trials, num_channels))

        # Calculate and sum mean power spectral density psd
        # Length of psd arrays is according to the used psd algothimn
        if num_samples % 2 == 0:  # even number
            num_psd_samples = (num_samples / 2) + 1
        else:
            num_psd_samples = (num_samples + 1) / 2
        num_psd_samples = int(num_psd_samples)
        print("  - num_psd_samples = %d" % num_psd_samples)

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
                    if self.pl_labels[subject, trial] != -1:
                        data = self.pl_data[subject, trial, ch, :]
                        f, psd = signal.periodogram(data, float(self.fs))  # this method uses
                        # welch functions and should provide a lower noise but a
                        # a little bit poorer frequency resolution
                        psd_all = psd_all + psd
                        if self.pl_labels[subject, trial] == 0:
                            num_class1_trials = num_class1_trials + 1
                            psd_class1 = psd_class1 + psd
                        elif self.pl_labels[subject, trial] == 1:
                            num_class2_trials = num_class2_trials + 1
                            psd_class2 = psd_class2 + psd
                        elif self.pl_labels[subject, trial] == 2:
                            num_class3_trials = num_class3_trials + 1
                            psd_class3 = psd_class3 + psd
                        elif self.pl_labels[subject, trial] == 3:
                            num_class4_trials = num_class4_trials + 1
                            psd_class4 = psd_class4 + psd
                        else:
                            print("ERROR: Illegal label")

        print("  - Number of classX trials: ", num_class1_trials, num_class2_trials, num_class3_trials,
              num_class4_trials)

        psd_all = psd_all / (num_class1_trials + num_class2_trials + num_class3_trials + num_class4_trials)
        psd_class1 = psd_class1 / (num_class1_trials)
        psd_class2 = psd_class2 / (num_class2_trials)
        psd_class2 = psd_class3 / (num_class3_trials)
        psd_class2 = psd_class4 / (num_class4_trials)

        # save psd's in files
        fname = self.pl_path + '/psd_mean_'

        np.savez(fname + 'allclasses', f=f, psd=psd_all)
        np.savez(fname + 'class1', f=f, psd=psd_class1)
        np.savez(fname + 'class2', f=f, psd=psd_class2)
        np.savez(fname + 'class3', f=f, psd=psd_class3)
        np.savez(fname + 'class4', f=f, psd=psd_class4)

        print("  - Data (f, psd) saved in files: ", (fname + "xxx.npz"))


########################################################################################
'''
subroutine: plot_psds

Description:
  Reads psds from files and displays them.
'''


def plot_psds(path=f'{datasets_folder}/BCICompetition_IV-2a/pl_data/', sampling_rate=250):
    print("Plot psd diagram:")

    # construct psd file names
    fname = path + '/psd_mean_'

    # Read data from files:
    print("  - Load data (f, psd) from files: ", (fname + "xxx.npz"))
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
    print("  - delta_f = %f Hz, plot_start_ind = %d, plot_stop_ind = %d" % (delta_f, plot_start_ind, plot_stop_ind))

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
    if SHOW_PLOTS:
        plt.show()

    return


########################################################################################
if __name__ == '__main__':
    subjects = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    #    subjects = [1]
    n_classes = 4
    training = 1

    print(' Generate pl_data and pl_labels and store them in files')
    ds_w = BCIC_IV2a_dataset(subjects=subjects, n_classes=n_classes)
    preloaded_data, preloaded_labels = ds_w.load_subjects_data(training)
    ds_w.print_stats()
    ds_w.save_pl_dataLabels(fname="test1.npz")

    print('Load pl_data and pl_labels from file and calculate the psds')
    #    ds_r = BCIC_IV2a_dataset(subjects=subjects, n_classes=n_classes)
    #    ds_r.load_pl_dataLabels(fname="test1.npz")
    #    ds_r.print_stats()
    #    ds_r.calc_psds()

    print(' Plot psds ')
    plot_psds()
    print("The End")
