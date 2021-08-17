"""
File: bcic_data_loading.py

Description:
  Handles all EEG-Data loading of BCI competition IV 2a Motor Imagery Dataset.
  Software structure and architecture is taken from file 'data_loading.pc'
  developed by P. Kessler.

Author: Manfred Strahnen (based on the template given by Philipp Kessler
        file: lsmr21_data_loading.py)

History:
  2021-05-10: Getting started - ms (Manfred Strahnen
"""
from config import CONFIG
from data.MIDataLoader import MIDataLoader
from data.data_utils import butter_bandpass_filt
from data.datasets.TrialsDataset import TrialsDataset
from data.datasets.bcic.bcic_dataset import BCIC
import numpy as np

from data.datasets.bcic.bcic_iv2a_dataset import BCIC_IV2a_dataset
from data.datasets.openBCI.openBCI_dataset import OpenBCI
from machine_learning.util import get_valid_trials_per_subject
from paths import datasets_folder
from util.misc import to_idxs_of_list


class OpenBCITrialsDataset(TrialsDataset):
    """
     TrialsDataset class Implementation for openBCI Dataset
    """

    def __init__(self, subjects, used_subjects, n_class, device, preloaded_tuple,
                 ch_names=BCIC.CHANNELS, equal_trials=True):
        super().__init__(subjects, used_subjects, n_class, device, preloaded_tuple, ch_names, equal_trials)

        # max number of trials (which is the same for each subject
        # self.n_trials_max = 6 * 12 * self.n_class  # 6 runs with 12 trials per class per subject

        # number of valid trials per subject is different for each subject, because
        # some trials are marked as artifact
        self.trials_per_subject = OpenBCI.trials_per_subject  # get_valid_trials_per_subject(self.preloaded_labels, self.subjects,
        #                           self.used_subjects, self.n_trials_max)

        # Only for testing !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #        for subject_idx in range(len(self.subjects)):
        #            self.trials_per_subject[subject_idx] = self.n_trials_max

        print("trials_per_subject: ", self.trials_per_subject)


class OpenBCIDataLoader(MIDataLoader):
    """
    MI_DataLoader implementation for OpenBCI Dataset
    """
    name = OpenBCI.name
    name_short = OpenBCI.short_name
    available_subjects = OpenBCI.ALL_SUBJECTS
    folds = OpenBCI.cv_folds
    eeg_config = OpenBCI.CONFIG
    channels = OpenBCI.CHANNELS
    ds_class = OpenBCITrialsDataset

    """
    preloaded_data = [subject, trial, channel, sample_idx]
    preloaded_labels = [subject, trial]
    """

    @classmethod
    def load_subjects_data(cls, subjects, n_class, ch_names=OpenBCI.CHANNELS, equal_trials=True,
                           normalize=False, ignored_runs=[]):
        subjects.sort()
        samples = int((CONFIG.EEG.TMAX - CONFIG.EEG.TMIN) * CONFIG.EEG.SAMPLERATE)

        preloaded_data = np.zeros((len(subjects), OpenBCI.trials_per_subject, len(ch_names), samples))
        preloaded_labels = np.zeros((len(subjects), OpenBCI.trials_per_subject))
        for subject in subjects:
            dataset_path = f'{datasets_folder}/OpenBCI/Sub_1/Test_2/Session_' + str(subject) + '/Processed_data.npz'
            print("Loading Dataset " + str(subject) + " from " + dataset_path)
            data = np.load(dataset_path)
            channels = data["channels"]
            # labels = data["labels"]
            labels_start = data["labels_start"]
            # Trial indexes for labels 2  or 3
            trial_idxes = [idx for idx, trial in enumerate(labels_start) if trial[1] == 2 or trial[1] == 3]
            # create preloaded_data array
            channel_idxes = to_idxs_of_list(ch_names, OpenBCI.CHANNELS)
            for idx, trial_idx in enumerate(trial_idxes):
                preloaded_data[subject - 1, idx] = channels[channel_idxes,
                                                   labels_start[trial_idx][0]:(labels_start[trial_idx][0] + samples)]
                # Todo replace 2 when higher 2 class
                preloaded_labels[subject - 1, idx] = labels_start[trial_idx][1] - 2

        # optional butterworth bandpass filtering
        if CONFIG.FILTER.FREQ_FILTER_HIGHPASS != None or CONFIG.FILTER.FREQ_FILTER_LOWPASS != None:
            preloaded_data = butter_bandpass_filt(preloaded_data, lowcut=CONFIG.FILTER.FREQ_FILTER_HIGHPASS,
                                                  highcut=CONFIG.FILTER.FREQ_FILTER_LOWPASS,
                                                  fs=CONFIG.EEG.SAMPLERATE, order=7)
        return preloaded_data, preloaded_labels

    @classmethod
    def create_n_class_loaders_from_subject(cls, used_subject, n_class, n_test_runs, batch_size, ch_names, device):
        # TODO
        raise NotImplementedError('This method is not implemented!')

    @classmethod
    def mne_load_subject_raw(cls, subject, runs, ch_names=[], notch=False, fmin=CONFIG.FILTER.FREQ_FILTER_HIGHPASS,
                             fmax=CONFIG.FILTER.FREQ_FILTER_LOWPASS):
        # TODO
        raise NotImplementedError('This method is not implemented!')
