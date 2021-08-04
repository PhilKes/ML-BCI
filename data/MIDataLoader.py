from typing import List, Dict, Any, Callable

import torch
from torch.utils.data import RandomSampler, DataLoader, TensorDataset

from config import EEGConfig, CONFIG, RESAMPLE
import numpy as np

from data.datasets.lsmr21.lmsr_21_dataset import LSMR21
from machine_learning.util import resample_eeg_data
from util.misc import print_subjects_ranges


class MIDataLoader:
    """
    Abstract Superclass for any Motor Imagery Dataset Loader
    If a new Dataset is added, a class implementing all here
    declared static methods + attributes has to be created
    see e.g. BCIC_DataLoader or PHYS_DataLoader
    """
    name: str
    name_short: str
    available_subjects: List[int]
    folds: int
    channels: List[int]
    eeg_config: EEGConfig
    # Constructor of Dataset specific TrialsDataset subclass
    ds_class: Callable
    # Sample the trials in random order (see MIDataLoader.create_loader_from_subjects)
    sampler: Callable = RandomSampler

    # Returns Loaders of Training + Test Datasets from index splits
    # for n_class classification
    # also returns loader_valid containing validation_subjects for loss calculation if validation_subjects are present
    @classmethod
    def create_loaders_from_splits(cls, splits, validation_subjects: List[int], n_class: int, device,
                                   preloaded_data: np.ndarray = None, preloaded_labels: np.ndarray = None,
                                   bs: int = CONFIG.MI.BATCH_SIZE, ch_names: List[str] = [],
                                   equal_trials: bool = True, used_subjects: List[int] = []):
        """
        Function: create_loaders_from_splits(...)

        Input parameters:
          splits: tuple of two arrays. First one contains the subject ids used for training,
                  second one contains subject ids used for testing

        Description:
          Returns Loaders of Training + Test Datasets from index splits
          for n_class classification. Optionally returns Validation Loader containing
          validation_subjects subject for loss calculation
        """
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
            loader_valid = cls.create_loader_from_subjects(validation_subjects, used_subjects, n_class, device,
                                                           preloaded_data,
                                                           preloaded_labels,
                                                           bs, ch_names, equal_trials)
        loader_train = cls.create_loader_from_subjects(subjects_train, used_subjects, n_class, device,
                                                       preloaded_data, preloaded_labels,
                                                       bs, ch_names, equal_trials)
        loader_test = cls.create_loader_from_subjects(subjects_test, used_subjects, n_class, device,
                                                      preloaded_data,
                                                      preloaded_labels,
                                                      bs, ch_names, equal_trials)
        return loader_train, loader_test, loader_valid

    # Creates DataLoader with Random Sampling from subject list
    @classmethod
    def create_loader_from_subjects(cls, subjects, used_subjects, n_class, device, preloaded_data, preloaded_labels,
                                    bs=CONFIG.MI.BATCH_SIZE, ch_names=[], equal_trials=True) -> DataLoader:
        """
        Create Loaders for given subjects
        :return: Loader
        """
        trials_ds = cls.ds_class(subjects, used_subjects, n_class, device,
                                 preloaded_tuple=(preloaded_data, preloaded_labels),
                                 ch_names=ch_names, equal_trials=equal_trials)
        sampler = None if cls.sampler is None else cls.sampler(trials_ds)
        return DataLoader(trials_ds, bs, sampler=sampler, pin_memory=False)

    @classmethod
    def load_subjects_data(cls, subjects: List[int], n_class: int, ch_names: List[str] = [], equal_trials: bool = True,
                           normalize: bool = False, ignored_runs: List[int] = []):
        """
        Loads and returns n_class data of subjects
        Can lead to high memory usage depending on the Dataset
        :param subjects: Subjects to be loaded
        :param equal_trials: Load equal amount of Trials for each class?
        :param ignored_runs: Runs to ignore (MNE with PHYS)
        :return: preloaded_data, preloaded_labels of specified subjects
        """
        raise NotImplementedError('This method is not implemented!')

    @classmethod
    def check_and_resample(cls, data: np.ndarray):
        """
        Checks and executes resampling of given EEG Data if necessary
        :param data: original EEG Data Array
        :return: resampled EEG Data (Sample rate= CONFIG.SYSTEM_SAMPLE_RATE)
        """
        # Resample EEG Data if necessary
        if RESAMPLE & (cls.eeg_config.SAMPLERATE != CONFIG.SYSTEM_SAMPLE_RATE):
            data = resample_eeg_data(data, cls.eeg_config.SAMPLERATE,
                                     CONFIG.SYSTEM_SAMPLE_RATE,
                                     per_subject=False
                                     # per_subject=(cls.name_short == LSMR21.short_name)
                                     )
        return data

    @classmethod
    def create_loader(cls, preloaded_data, preloaded_labels, device, batch_size=CONFIG.MI.BATCH_SIZE):
        data_set = TensorDataset(torch.as_tensor(preloaded_data, device=device, dtype=torch.float32),
                                 torch.as_tensor(preloaded_labels, device=device, dtype=torch.int))
        return DataLoader(data_set, batch_size, sampler=RandomSampler(data_set), pin_memory=False)

    @classmethod
    def create_n_class_loaders_from_subject(cls, used_subject: int, n_class: int, n_test_runs: List[int],
                                            batch_size: int, ch_names: List[str], device):
        """
        Create Train/Test Loaders for a single subject
        :param n_test_runs: Which runs are used for the Test Loader
        :return: Train Loader, Test Loader
        """
        raise NotImplementedError('This method is not implemented!')

    @classmethod
    def create_preloaded_loader(cls, subjects: List[int], n_class: int, ch_names: List[str], batch_size: int, device,
                                equal_trials: bool):
        """
        Create Loader with preloaded data for given subjects
        :return: Loader of subjects' data
        """
        print(f"Preloading Subjects [{subjects[0]}-{subjects[-1]}] Data in memory")
        preloaded_data, preloaded_labels = cls.load_subjects_data(subjects, n_class, ch_names,
                                                                  equal_trials=equal_trials)
        return cls.create_loader_from_subjects(subjects, n_class, device, preloaded_data,
                                               preloaded_labels, batch_size, equal_trials=equal_trials)

    @classmethod
    def load_live_sim_data(cls, subject, n_class, ch_names):
        """
        Load all necessary Data for the Live Simulation Run of subject
        X: ndarray (channels,Samples) of single Subject's Run data
        max_sample: Maximum sample number of the Run
        slices: Trial Slices
        trials_classes: ndarray with label nr. of every Trial in the Run
        trials_start_times: ndarray with Start Times of every Trial in the Run
        trial_tdeltas: ndarray with Times of every Slice Timepoint in the Run
        """
        raise NotImplementedError('This method is not implemented!')
