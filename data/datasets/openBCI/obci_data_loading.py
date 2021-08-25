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
from config import CONFIG, RESAMPLE
from data.MIDataLoader import MIDataLoader
from data.data_utils import butter_bandpass_filt
from data.datasets.TrialsDataset import TrialsDataset
# from data.datasets.bcic.bcic_dataset import BCIC
import numpy as np

from data.datasets.bcic.bcic_iv2a_dataset import BCIC_IV2a_dataset
from data.datasets.openBCI.obci_dataset import OpenBCI, OpenBCIConstants
from machine_learning.util import get_valid_trials_per_subject
from paths import datasets_folder
from util.misc import to_idxs_of_list, calc_n_samples


class OpenBCITrialsDataset(TrialsDataset):
    """
     TrialsDataset class Implementation for openBCI Dataset
    """

    def __init__(self, subjects, used_subjects, n_class, preloaded_tuple,
                 ch_names=OpenBCI.CHANNELS, equal_trials=True):
        super().__init__(subjects, used_subjects, n_class, preloaded_tuple, ch_names, equal_trials)

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
    CONSTANTS: OpenBCIConstants = OpenBCI
    ds_class = OpenBCITrialsDataset
    """
    preloaded_data = [subject, trial, channel, sample_idx]
    preloaded_labels = [subject, trial]
    """

    @classmethod
    def load_subjects_data(cls, subjects, n_class, ch_names=OpenBCI.CHANNELS, equal_trials=True, ignored_runs=[]):
        subjects.sort()
        # Final subjects_data shape (CONFIG.EEG.SAMPLES = CONFIG.SYSTEM_SAMPLERATE if in config.py RESAMPLE=True)
        subjects_data = np.zeros((len(subjects), OpenBCI.trials_per_subject, len(ch_names), CONFIG.EEG.SAMPLES))
        subjects_labels = np.zeros((len(subjects), OpenBCI.trials_per_subject))
        if RESAMPLE & (cls.CONSTANTS.CONFIG.SAMPLERATE != CONFIG.SYSTEM_SAMPLE_RATE):
            print(f"RESAMPLING from {cls.CONSTANTS.CONFIG.SAMPLERATE}Hz to {CONFIG.SYSTEM_SAMPLE_RATE}Hz")
        original_samples = calc_n_samples(CONFIG.EEG.TMIN, CONFIG.EEG.TMAX, cls.CONSTANTS.CONFIG.SAMPLERATE)
        for s_idx, subject in enumerate(subjects):
            # Loading single Subject Data with original Samplerate
            subject_data = np.full((OpenBCI.trials_per_subject, len(ch_names), original_samples),
                                   -1, dtype=np.float32)
            subject_labels = np.full(OpenBCI.trials_per_subject, -1, dtype=np.int)
            dataset_path = f'{datasets_folder}/OpenBCI/Sub_1/Test_5/Session_' + str(subject) + '/Processed_data.npz'
            print("Loading Dataset " + str(subject) + " from " + dataset_path)
            data = np.load(dataset_path)
            channels = data["channels"]
            # labels = data["labels"]
            labels_start = data["labels_start"]
            # Trial indexes in labels_start for labels 1  or 2
            trial_idxes = [idx for idx, trial in enumerate(labels_start) if trial[1] == 1 or trial[1] == 2]
            # create preloaded_data array
            channel_idxes = to_idxs_of_list(ch_names, OpenBCI.CHANNELS)
            for idx, trial_idx in enumerate(trial_idxes):
                subject_data[idx] = channels[channel_idxes,
                                    labels_start[trial_idx][0]:(labels_start[trial_idx][0] + original_samples)]
                # Trials are 1 and 2
                subject_labels[idx] = labels_start[trial_idx][1] - 1
            subject_data, subject_labels = cls.prepare_data_labels(subject_data, subject_labels)
            subjects_data[s_idx] = subject_data
            subjects_labels[s_idx] = subject_labels
        return subjects_data, subjects_labels

    @classmethod
    def create_n_class_loaders_from_subject(cls, used_subject, n_class, n_test_runs, batch_size, ch_names, device):
        # TODO
        raise NotImplementedError('This method is not implemented!')

    @classmethod
    def mne_load_subject_raw(cls, subject, runs, ch_names=[], notch=False, fmin=CONFIG.FILTER.FREQ_FILTER_HIGHPASS,
                             fmax=CONFIG.FILTER.FREQ_FILTER_LOWPASS):
        # TODO
        raise NotImplementedError('This method is not implemented!')
