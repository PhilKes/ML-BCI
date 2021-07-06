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

from config import global_config
from data.MIDataLoader import MIDataLoader
from data.datasets.TrialsDataset import TrialsDataset
from data.datasets.bcic.bcic_dataset import BCIC

from data.datasets.bcic.bcic_iv2a_dataset import BCIC_IV2a_dataset


class BCICTrialsDataset(TrialsDataset):
    """
     TrialsDataset class Implementation for BCIC Dataset
    """

    def __init__(self, subjects, n_classes, device, preloaded_tuple,
                 ch_names=BCIC.CHANNELS, equal_trials=True):

        super().__init__(subjects, n_classes, device, preloaded_tuple, ch_names, equal_trials)

        # max number of trials (which is the same for each subject
        self.n_trials_max = 6 * 12 * self.n_classes  # 6 runs with 12 trials per class per subject

        # number of valid trials per subject is different for each subject, because
        # some trials are marked as artifact
        self.trials_per_subject = [0] * len(self.subjects)
        for subject_idx in range(len(self.subjects)):
            for trial in range(self.n_trials_max):
                if self.preloaded_labels[subject_idx, trial] != -1:
                    self.trials_per_subject[subject_idx] = self.trials_per_subject[subject_idx] + 1

        # Only for testing !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #        for subject_idx in range(len(self.subjects)):
        #            self.trials_per_subject[subject_idx] = self.n_trials_max

        print("trials_per_subject: ", self.trials_per_subject)

    def load_trial(self, trial):
        # calculate subject id 'subject_idx' and subject specific trial id 'trial_idx'
        # from parameter trial
        subject_idx = None
        trial_idx = None
        trial_start = trial
        for subject in self.subjects:
            subject_idx = self.subjects.index(subject)
            if trial < self.trials_per_subject[subject_idx]:
                trial_idx = trial
                break  # exit for loop
            else:
                trial = trial - self.trials_per_subject[subject_idx]

        # print("trial , subject_idx, trial_idx: %d, %d, %d" % (trial_start, subject_idx, trial_idx))

        return self.preloaded_data[subject_idx][trial_idx], self.preloaded_labels[subject_idx][trial_idx]


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
        ds_w = BCIC_IV2a_dataset(subjects=subjects, n_classes=n_class, ch_names=ch_names)
        preloaded_data, preloaded_labels = ds_w.load_subjects_data(training)
        ds_w.print_stats()

        return preloaded_data, preloaded_labels

    @classmethod
    def create_n_class_loaders_from_subject(cls, used_subject, n_class, n_test_runs, batch_size, ch_names, device):
        # TODO
        raise NotImplementedError('This method is not implemented!')

    @classmethod
    def mne_load_subject_raw(cls, subject, runs, ch_names=[], notch=False, fmin=global_config.FREQ_FILTER_HIGHPASS,
                             fmax=global_config.FREQ_FILTER_LOWPASS):
        # TODO
        raise NotImplementedError('This method is not implemented!')
