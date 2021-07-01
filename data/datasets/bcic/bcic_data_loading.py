"""
File: bcic_data_loading.py

Description:
  Handles all EEG-Data loading of BCI competition IV 2a Motor Imagery Dataset.
  Software structure and architecture is taken from file 'data_loading.pc'
  developed by P. Kessler.

Author: Manfred Strahnen (based on the template given by Philipp Kessler
        file: phys_data_loading.py)

History:
  2021-05-10: Getting started - ms (Manfred Strahnen
"""

import torch  # noqa
from torch.utils.data import Dataset  # noqa

from config import global_config
from data.MI_DataLoader import MI_DataLoader
from data.datasets.bcic.bcic_dataset import BCIC_name, BCIC_ALL_SUBJECTS, BCIC_cv_folds, BCIC_CONFIG, BCIC_CHANNELS, \
    BCIC_short_name
from data.datasets.bcic.bcic_iv2a_dataset import BCIC_IV2a_dataset
from data.datasets.phys.phys_dataset import PHYS_CHANNELS


class BCIC_TrialsDataset(Dataset):
    """
    Class: BCIC_TrialsDataset(Dataset)

    Description:
      Dataset class which is based on torch.utils.data.Dataset. This type of class is
      required for creating a pytorch dataloader.
      Methods __len__ and __get_item__ must be implemented.
    """

    def __init__(self, subjects, n_classes, device, preloaded_tuple=None,
                 ch_names=PHYS_CHANNELS, equal_trials=True):
        """
        Method: constructor
        Parameters:
            subjects: list of subjects
        """
        self.subjects = subjects
        self.n_classes = n_classes
        self.device = device

        self.equal_trials = equal_trials
        self.preloaded_data = preloaded_tuple[0] if preloaded_tuple is not None else None
        self.preloaded_labels = preloaded_tuple[1] if preloaded_tuple is not None else None
        self.ch_names = ch_names

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

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    __len__ has to be implemented and has to return the overall number of trials.
    """

    def __len__(self):
        ds_len = 0
        for subject_idx in range(len(self.subjects)):
            ds_len = ds_len + self.trials_per_subject[subject_idx]
        # print("ds_len = ", ds_len)
        return ds_len

    """
    __get_item__ has to be implemented and has to return the trial data and label
    which corresponds to the passed index 'trial'.
    """

    # Returns a single trial as Tensor with Labels
    def __getitem__(self, trial):
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

        X, y = self.preloaded_data[subject_idx][trial_idx], self.preloaded_labels[subject_idx][trial_idx]

        # Shape of 1 Batch (list of multiple __getitem__() calls):
        # [samples (BATCH_SIZE), 1 , Channels (len(ch_names), Timepoints (641)]
        X = torch.as_tensor(X[None, ...], device=self.device, dtype=torch.float32)
        # X = TRANSFORM(X)
        return X, y


class BCIC_Dataloader(MI_DataLoader):
    name = BCIC_name
    name_short = BCIC_short_name
    available_subjects = BCIC_ALL_SUBJECTS
    folds = BCIC_cv_folds
    eeg_config = BCIC_CONFIG
    channels = BCIC_CHANNELS
    ds_class = BCIC_TrialsDataset

    @classmethod
    def load_subjects_data(cls, subjects, n_class, ch_names=PHYS_CHANNELS, equal_trials=True,
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
    def create_preloaded_loader(cls, subjects_chunk, n_class, ch_names, batch_size, device, equal_trials):
        # TODO
        raise NotImplementedError('This method is not implemented!')

    @classmethod
    def mne_load_subject_raw(cls, subject, runs, ch_names=[], notch=False, fmin=global_config.FREQ_FILTER_HIGHPASS,
                             fmax=global_config.FREQ_FILTER_LOWPASS):
        # TODO
        raise NotImplementedError('This method is not implemented!')
