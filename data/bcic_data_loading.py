"""
File: bcic_data_loading.py

Description:
  Handles all EEG-Data loading of BCI competition IV 2a Motor Imagery Dataset.
  Software structure and architecture is taken from file 'data_loading.pc'
  developed by P. Kessler.

Author: Manfred Strahnen (based on the template given by Philipp Kessler
        file: data_loading.py)

History:
  2021-05-10: Getting started - ms (Manfred Strahnen
"""

import torch  # noqa
from torch.utils.data import Dataset, DataLoader, RandomSampler  # noqa

from config import DATA_PRELOAD, BATCH_SIZE
from data.bcic_iv2a_dataset import BCIC_IV2a_dataset
from data.physionet_dataset import PHYS_ALL_SUBJECTS, PHYS_CHANNELS
from util.misc import print_subjects_ranges

"""
Function: bcic_create_loaders_from_splits(...)

Input parameters:
  splits: tuple of two arrays. First one contains the subject ids used for training,
          second one contains subject ids used for testing

Description:
  Returns Loaders of Training + Test Datasets from index splits
  for n_class classification. Optionally returns Validtion Loader containing 
  validation_subjects subject for loss calculation
"""
def bcic_create_loaders_from_splits(splits, validation_subjects, n_class, device, preloaded_data=None,
                                    preloaded_labels=None, bs=BATCH_SIZE, ch_names=PHYS_CHANNELS,
                                    equal_trials=True, used_subjects=PHYS_ALL_SUBJECTS):
    subjects_train_idxs, subjects_test_idxs = splits
    subjects_train = [used_subjects[idx] for idx in subjects_train_idxs]
    subjects_test = [used_subjects[idx] for idx in subjects_test_idxs]
    print_subjects_ranges(subjects_train, subjects_test)
    # Only pass preloaded data for subjects, not ALL_SUBJECTS
    # have to get correct idxs for subjects
    # subjects_idxs= [ALL_SUBJECTS.index(i) for i in subjects]
    loader_valid = None
    if len(validation_subjects) > 0:
        subjects_valid_idxs = [used_subjects.index(i) for i in validation_subjects]
        loader_valid = bcic_create_loader_from_subjects(validation_subjects, n_class, device,
                                                   preloaded_data[subjects_valid_idxs, :, :, :],
                                                   preloaded_labels[subjects_valid_idxs, :],
                                                   bs, ch_names, equal_trials)

    loader_train = bcic_create_loader_from_subjects(subjects_train, n_class, device,
                                               preloaded_data[subjects_train_idxs, :, :, :],
                                               preloaded_labels[subjects_train_idxs, :],
                                               bs, ch_names, equal_trials)
    loader_test = bcic_create_loader_from_subjects(subjects_test, n_class, device,
                                              preloaded_data[subjects_test_idxs, :, :, :],
                                              preloaded_labels[subjects_test_idxs, :],
                                              bs, ch_names, equal_trials)
    return loader_train, loader_test, loader_valid


# Creates DataLoader with Random Sampling from subject list
def bcic_create_loader_from_subjects(subjects, n_class, device, preloaded_data=None, preloaded_labels=None,
                                     bs=BATCH_SIZE, ch_names=PHYS_CHANNELS, equal_trials=True):
    trials_ds = bcic_trialsDataset(subjects, n_class, device,
                                   preloaded_tuple=(preloaded_data, preloaded_labels) if DATA_PRELOAD else None,
                                   ch_names=ch_names, equal_trials=equal_trials)
    # Sample the trials in random order
    sampler_train = RandomSampler(trials_ds)
    return DataLoader(trials_ds, bs, sampler=sampler_train, pin_memory=False)

"""
Subroutine:  bcic_load_subjects_data()
"""
# Loads all Subjects Data + Labels for n_class Classification
# used_runs can be passed to force to load only these runs
def bcic_load_subjects_data(subjects, n_class, ch_names=PHYS_CHANNELS, equal_trials=True,
                            normalize=False, ignored_runs=[]):
    subjects.sort()

    training = 1    # load BCIC training data set
    ds_w = BCIC_IV2a_dataset(subjects=subjects, n_classes=n_class, ch_names=ch_names)
    preloaded_data, preloaded_labels = ds_w.load_subjects_data(training)
    ds_w.print_stats()

    return preloaded_data, preloaded_labels


"""
Class: bcic_trialsDataset(Dataset)

Description:
  Dataset class which is based on torch.utils.data.Dataset. This type of class is
  required for creating a pytorch dataloader.
  Methods __len__ and __get_item__ must be implemented.
"""
class bcic_trialsDataset(Dataset):
    """
    Method: constructor
    Parameters:
        subjects: list of subjects
    """
    def __init__(self, subjects, n_classes, device, preloaded_tuple=None,
                 ch_names=PHYS_CHANNELS, equal_trials=True):
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
                    self.trials_per_subject[subject_idx] = self.trials_per_subject[subject_idx] +1

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
        #print("ds_len = ", ds_len)
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
                break               # exit for loop
            else:
                trial = trial - self.trials_per_subject[subject_idx]

        #print("trial , subject_idx, trial_idx: %d, %d, %d" % (trial_start, subject_idx, trial_idx))

        X, y = self.preloaded_data[subject_idx][trial_idx], self.preloaded_labels[subject_idx][trial_idx]

        # Shape of 1 Batch (list of multiple __getitem__() calls):
        # [samples (BATCH_SIZE), 1 , Channels (len(ch_names), Timepoints (641)]
        X = torch.as_tensor(X[None, ...], device=self.device, dtype=torch.float32)
        # X = TRANSFORM(X)
        return X, y