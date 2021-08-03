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
import math

import numpy as np

from config import CONFIG, RESAMPLE
from data.MIDataLoader import MIDataLoader
from data.datasets.TrialsDataset import TrialsDataset
from data.datasets.bcic.bcic_dataset import BCIC

from data.datasets.bcic.bcic_iv2a_dataset import BCIC_IV2a_dataset
from machine_learning.util import get_valid_trials_per_subject, resample_eeg_data


class BCICTrialsDataset(TrialsDataset):
    """
     TrialsDataset class Implementation for BCIC Dataset
    """

    def __init__(self, subjects, used_subjects, n_class, device, preloaded_tuple,
                 ch_names=BCIC.CHANNELS, equal_trials=True):
        super().__init__(subjects, used_subjects, n_class, device, preloaded_tuple, ch_names, equal_trials)

        # max number of trials (which is the same for each subject
        self.n_trials_max = 6 * 12 * self.n_class  # 6 runs with 12 trials per class per subject

        # number of valid trials per subject is different for each subject, because
        # some trials are marked as artifact
        self.trials_per_subject = get_valid_trials_per_subject(self.preloaded_labels, self.subjects,
                                                               self.used_subjects, self.n_trials_max)

        # Only for testing !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #        for subject_idx in range(len(self.subjects)):
        #            self.trials_per_subject[subject_idx] = self.n_trials_max

        print("trials_per_subject: ", self.trials_per_subject)


class BCICDataloader(MIDataLoader):
    """
    MI_DataLoader implementation for BCIC Dataset
    """

    name = BCIC.name
    name_short = BCIC.short_name
    available_subjects = BCIC.ALL_SUBJECTS
    folds = BCIC.cv_folds
    eeg_config = BCIC.CONFIG
    channels = BCIC.CHANNELS
    ds_class = BCICTrialsDataset

    @classmethod
    def load_subjects_data(cls, subjects, n_class, ch_names=BCIC.CHANNELS, equal_trials=True,
                           normalize=False, ignored_runs=[]):
        subjects.sort()
        training = 1  # load BCIC training data set
        ds_w = BCIC_IV2a_dataset(subjects=subjects, n_class=n_class, ch_names=ch_names)
        preloaded_data, preloaded_labels = ds_w.load_subjects_data(training)
        ds_w.print_stats()
        if RESAMPLE & (cls.eeg_config.SAMPLERATE != CONFIG.SYSTEM_SAMPLE_RATE):
            print(f"RESAMPLING from {cls.eeg_config.SAMPLERATE}Hz to {CONFIG.SYSTEM_SAMPLE_RATE}Hz")
        preloaded_data = cls.check_and_resample(preloaded_data)
        return preloaded_data, preloaded_labels

    @classmethod
    def create_n_class_loaders_from_subject(cls, used_subject, n_class, n_test_runs, batch_size, ch_names, device):
        ds_w = BCIC_IV2a_dataset(subjects=[used_subject], n_class=n_class, ch_names=ch_names)
        preloaded_data, preloaded_labels = ds_w.load_subjects_data(1)
        preloaded_data = np.squeeze(preloaded_data, 0)
        preloaded_labels = np.squeeze(preloaded_labels, 0)
        preloaded_data = cls.check_and_resample(preloaded_data)
        preloaded_data = preloaded_data.reshape((preloaded_data.shape[0], 1, preloaded_data.shape[1],
                                                 preloaded_data.shape[2]))
        n_trials_max = 6 * 12 * n_class  # 6 runs with 12 trials per class per subject
        valid_trials = get_valid_trials_per_subject(np.expand_dims(preloaded_labels, 0), [used_subject],
                                                    [used_subject], n_trials_max)[0]
        # Use 80% of the subject's data as Training Data, 20% as Test Data
        training_trials_size = math.floor(4 * valid_trials / 5)
        loader_train = cls.create_loader(preloaded_data[:training_trials_size],
                                         preloaded_labels[:training_trials_size], device, batch_size)
        loader_test = cls.create_loader(preloaded_data[training_trials_size:valid_trials],
                                        preloaded_labels[training_trials_size:valid_trials], device, batch_size)
        return loader_train, loader_test

    @classmethod
    def load_live_sim_data(cls, subject, n_class, ch_names):
        pass
